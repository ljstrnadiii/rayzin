from dataclasses import replace
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import ray.data
from ray.data import ActorPoolStrategy
from ray.data.expressions import Expr, col
from shapely.geometry.base import BaseGeometry  # type: ignore[import-untyped]

from rayzin.enums import MetricType, ReaderType, SearchBackendType
from rayzin.manifest.build import build_zarr_chunk_table, compute_chunk_summary_arrow
from rayzin.manifest.cog import summarize_cog_blocks
from rayzin.manifest.filtering import filter_manifest
from rayzin.manifest.schema import MANIFEST_SCHEMA
from rayzin.metrics import add_lower_bounds_fn
from rayzin.readers.zarr_reader import ZarrVectorReader
from rayzin.search.block_searcher import BlockSearcher
from rayzin.search.heap_actor import HeapActor
from rayzin.types import COL_COUNT, COL_URL, Float32Array, SearchResults


def knn_zarr_search(
    manifest_path: str,
    query: np.ndarray,
    k: int,
    *,
    array_name: str = "embeddings",
    embedding_dim_name: str = "embedding",
    store_kwargs: dict[str, Any] | None = None,
    metric: MetricType = MetricType.EUCLIDEAN,
    backend: SearchBackendType = SearchBackendType.NUMPY,
    filter_expr: Expr | None = None,
    aoi: BaseGeometry | None = None,
    batch_size: int | None = None,
    num_cpus_per_actor: float = 1.0,
    actor_pool_size: int = 4,
) -> SearchResults:
    queries = _as_query_batch(query)
    return _knn_search(
        manifest_path,
        queries,
        k,
        reader_type=ReaderType.ZARR,
        reader_kwargs={
            "array_name": array_name,
            "store_kwargs": store_kwargs or {},
            "embedding_dim_name": embedding_dim_name,
        },
        metric=metric,
        backend=backend,
        filter_expr=filter_expr,
        aoi=aoi,
        store_kwargs=store_kwargs or {},
        batch_size=batch_size,
        num_cpus_per_actor=num_cpus_per_actor,
        actor_pool_size=actor_pool_size,
    )


def knn_cog_search(
    manifest_path: str,
    query: np.ndarray,
    k: int,
    *,
    store_kwargs: dict[str, Any] | None = None,
    normalize: bool = True,
    metric: MetricType = MetricType.EUCLIDEAN,
    backend: SearchBackendType = SearchBackendType.NUMPY,
    filter_expr: Expr | None = None,
    aoi: BaseGeometry | None = None,
    batch_size: int | None = None,
    num_cpus_per_actor: float = 1.0,
    num_gpus_per_actor: float = 0.0,
    actor_pool_size: int = 4,
    prefetch: int = 8,
    decode_threads: int = 4,
    filesystem: Any = None,
) -> SearchResults:
    """Exact top-``k`` over the COG blocks of a manifest built by ``build_manifest_from_cogs``.

    ``filter_expr`` and ``aoi`` prune blocks on their metadata and footprints before any read;
    surviving blocks are read ``prefetch`` at a time per actor. ``normalize`` must match the
    manifest's. ``SearchBackendType.TORCH`` streams blocks instead: ``prefetch`` fetch at once,
    ``decode_threads`` decompress straight into pinned buffers, and the GPU does the rest; give
    each actor a GPU with ``num_gpus_per_actor``, as for ``FAISS_GPU``. ``filesystem``, a
    ``pyarrow.fs.FileSystem``, reads the manifest, with ``manifest_path`` relative to it.
    """
    queries = _as_query_batch(query)
    if normalize:
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries = (queries / np.maximum(norms, 1e-12)).astype(np.float32)
    return _knn_search(
        manifest_path,
        queries,
        k,
        reader_type=ReaderType.COG,
        reader_kwargs={
            "normalize": normalize,
            "store_kwargs": store_kwargs or {},
            "decode_threads": decode_threads,
        },
        metric=metric,
        backend=backend,
        filter_expr=filter_expr,
        aoi=aoi,
        store_kwargs=store_kwargs or {},
        batch_size=batch_size,
        num_cpus_per_actor=num_cpus_per_actor,
        num_gpus_per_actor=num_gpus_per_actor,
        actor_pool_size=actor_pool_size,
        prefetch=prefetch,
        filesystem=filesystem,
    )


def _knn_search(
    manifest_path: str,
    query: Float32Array,
    k: int,
    *,
    reader_type: ReaderType,
    reader_kwargs: dict[str, Any],
    metric: MetricType,
    backend: SearchBackendType,
    filter_expr: Expr | None,
    aoi: BaseGeometry | None,
    store_kwargs: dict[str, Any],
    batch_size: int | None,
    num_cpus_per_actor: float,
    actor_pool_size: int,
    num_gpus_per_actor: float = 0.0,
    prefetch: int = 1,
    filesystem: Any = None,
) -> SearchResults:
    if metric != MetricType.EUCLIDEAN:
        msg = "Search pruning currently supports only the euclidean metric."
        raise NotImplementedError(msg)

    heap_actor = HeapActor.remote(  # type: ignore[attr-defined]
        nq=query.shape[0],
        k=k,
        metric_type=metric.value,
        backend_type=backend.value,
    )

    summary = (
        filter_manifest(
            ray.data.read_parquet(manifest_path, filesystem=filesystem).filter(
                expr=col(COL_COUNT) > 0
            ),
            filter_expr=filter_expr,
            aoi=aoi,
            store_kwargs=store_kwargs,
        )
        .map_batches(
            add_lower_bounds_fn,  # type: ignore[arg-type]
            fn_kwargs={"queries": query, "metric_type": metric.value},
            batch_format="pyarrow",
            udf_modifying_row_count=False,
        )
        .map_batches(
            BlockSearcher,
            fn_constructor_kwargs={
                "queries": query,
                "k": k,
                "metric_type": metric.value,
                "reader_type": reader_type.value,
                "reader_kwargs": reader_kwargs,
                "backend_type": backend.value,
                "heap_actor": heap_actor,
                "prefetch": prefetch,
            },
            batch_size=batch_size,
            batch_format="pyarrow",
            udf_modifying_row_count=True,
            compute=ActorPoolStrategy(min_size=1, max_size=actor_pool_size),
            num_cpus=num_cpus_per_actor,
            num_gpus=num_gpus_per_actor,
        )
        .materialize()
        .to_pandas()
    )
    results: SearchResults = ray.get(heap_actor.results.remote())
    return replace(
        results,
        stats={
            "blocks_after_pushdown": int(summary["rows_seen"].sum()),
            "blocks_searched": int(summary["rows_searched"].sum()),
            "vectors_searched": int(summary["vectors_searched"].sum()),
            "query_evaluations": int(summary["query_evaluations"].sum()),
        },
    )


def build_manifest(
    source: str | list[str],
    output_path: str,
    *,
    n_blocks: int = 256,
    embedding_dim_name: str = "embedding",
) -> None:
    if isinstance(source, list):
        build_manifest_from_cogs(source, output_path)
    else:
        build_manifest_from_zarr(
            source,
            output_path,
            n_blocks=n_blocks,
            embedding_dim_name=embedding_dim_name,
        )


def build_manifest_from_zarr(
    store_url: str,
    output_path: str,
    *,
    array_name: str = "embeddings",
    embedding_dim_name: str = "embedding",
    store_kwargs: dict[str, Any] | None = None,
    n_blocks: int = 256,
) -> None:
    reader = ZarrVectorReader(
        array_name=array_name,
        store_kwargs=store_kwargs,
        embedding_dim_name=embedding_dim_name,
    )
    chunk_table = build_zarr_chunk_table(
        store_url,
        array_name=array_name,
        store_kwargs=store_kwargs or {},
        embedding_dim_name=embedding_dim_name,
    )
    if chunk_table.num_rows == 0:
        ray.data.from_arrow(MANIFEST_SCHEMA.empty_table()).write_parquet(
            output_path,
            mode=ray.data.SaveMode.OVERWRITE,
        )
        return

    (
        ray.data.from_arrow(chunk_table, override_num_blocks=max(1, n_blocks))
        .map_batches(
            compute_chunk_summary_arrow,  # type: ignore[arg-type]
            fn_kwargs={"reader": reader},
            batch_format="pyarrow",
            udf_modifying_row_count=False,
        )
        .write_parquet(output_path, mode=ray.data.SaveMode.OVERWRITE)
    )


def build_manifest_from_cogs(
    cog_urls: list[str],
    output_path: str,
    *,
    metadata: dict[str, list[Any]] | None = None,
    store_kwargs: dict[str, Any] | None = None,
    normalize: bool = True,
    files_per_task: int = 8,
    files_in_flight: int = 4,
    filesystem: Any = None,
) -> int:
    """Summarise every block of each COG not yet in the manifest at ``output_path``.

    One row per block: its slice, count, centroid, radius and lon/lat footprint, plus the
    ``metadata`` columns, one value per URL, e.g. acquisition time, for ``filter_expr``. A URL
    already in the manifest is skipped and new rows are appended as new parquet files, so a
    growing collection is only ever scanned once. ``filesystem``, a ``pyarrow.fs.FileSystem``,
    holds the manifest, with ``output_path`` relative to it. Returns how many files were added.
    """
    urls = list(cog_urls)
    indexed = manifest_urls(output_path, filesystem)
    new = [index for index, url in enumerate(urls) if url not in indexed]
    if not new:
        return 0
    columns = {
        COL_URL: [urls[index] for index in new],
        **{name: [values[index] for index in new] for name, values in (metadata or {}).items()},
    }
    (
        ray.data.from_arrow(pa.table(columns), override_num_blocks=-(-len(new) // files_per_task))
        .map_batches(
            summarize_cog_blocks,  # type: ignore[arg-type]
            batch_size=None,
            batch_format="pyarrow",
            fn_kwargs={
                "normalize": normalize,
                "store_kwargs": store_kwargs or {},
                "files_in_flight": files_in_flight,
            },
            udf_modifying_row_count=True,
        )
        .write_parquet(output_path, filesystem=filesystem, mode=ray.data.SaveMode.APPEND)
    )
    return len(new)


def manifest_urls(path: str, filesystem: Any = None) -> set[str]:
    """Every URL a manifest already indexes; empty when there is no manifest yet."""
    import pyarrow.dataset as ds  # type: ignore[import-untyped]
    import pyarrow.fs as pafs  # type: ignore[import-untyped]

    if filesystem is None:
        filesystem, root = pafs.FileSystem.from_uri(path)
    else:
        root = path
    if filesystem.get_file_info(root).type == pafs.FileType.NotFound:
        return set()
    table = ds.dataset(root, filesystem=filesystem, format="parquet").to_table(columns=[COL_URL])
    return set(table.column(COL_URL).unique().to_pylist())


def _as_query_batch(query: np.ndarray) -> Float32Array:
    queries = np.asarray(query, dtype=np.float32)
    if queries.ndim != 2:
        msg = f"Expected query to have shape (nq, d), got {queries.shape!r}."
        raise ValueError(msg)
    return queries
