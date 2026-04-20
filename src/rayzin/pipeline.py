from typing import Any

import numpy as np
import ray.data
from ray.data import ActorPoolStrategy
from ray.data.expressions import Expr
from shapely.geometry.base import BaseGeometry  # type: ignore[import-untyped]

from rayzin.enums import MetricType, ReaderType, SearchBackendType
from rayzin.manifest.build import build_zarr_chunk_table, compute_chunk_summary_arrow
from rayzin.manifest.filtering import filter_manifest
from rayzin.metrics import add_lower_bounds_fn
from rayzin.readers.zarr_reader import ZarrVectorReader
from rayzin.schema import MANIFEST_SCHEMA
from rayzin.search.block_searcher import BlockSearcher
from rayzin.search.heap_actor import HeapActor
from rayzin.selectors import Selector
from rayzin.types import Float32Array, SearchResults


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
    open_kwargs: dict[str, Any] | None = None,
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
        reader_type=ReaderType.COG,
        reader_kwargs=open_kwargs or {},
        metric=metric,
        backend=backend,
        filter_expr=filter_expr,
        aoi=aoi,
        store_kwargs=open_kwargs or {},
        batch_size=batch_size,
        num_cpus_per_actor=num_cpus_per_actor,
        actor_pool_size=actor_pool_size,
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
) -> SearchResults:
    if reader_type != ReaderType.ZARR:
        msg = "COG search is not implemented yet."
        raise NotImplementedError(msg)
    if metric != MetricType.EUCLIDEAN:
        msg = "Search pruning currently supports only the euclidean metric."
        raise NotImplementedError(msg)

    heap_actor = HeapActor.remote(  # type: ignore[attr-defined]
        nq=query.shape[0],
        k=k,
        metric_type=metric.value,
        backend_type=backend.value,
    )

    (
        filter_manifest(
            ray.data.read_parquet(manifest_path),
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
            },
            batch_size=batch_size,
            batch_format="pyarrow",
            udf_modifying_row_count=True,
            compute=ActorPoolStrategy(min_size=1, max_size=actor_pool_size),
            num_cpus=num_cpus_per_actor,
        )
        .materialize()
    )

    return ray.get(heap_actor.results.remote())  # type: ignore[no-any-return]


def build_manifest(
    source: str | list[str],
    output_path: str,
    *,
    n_blocks: int = 256,
    embedding_dim_name: str = "embedding",
    selectors: Selector | None = None,
) -> None:
    if isinstance(source, list):
        if selectors is not None:
            msg = "Selectors are only supported for Zarr manifest generation."
            raise NotImplementedError(msg)
        build_manifest_from_cogs(source, output_path, n_blocks=n_blocks)
    else:
        build_manifest_from_zarr(
            source,
            output_path,
            n_blocks=n_blocks,
            embedding_dim_name=embedding_dim_name,
            selectors=selectors,
        )


def build_manifest_from_zarr(
    store_url: str,
    output_path: str,
    *,
    array_name: str = "embeddings",
    embedding_dim_name: str = "embedding",
    store_kwargs: dict[str, Any] | None = None,
    selectors: Selector | None = None,
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
        selectors=selectors,
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
    open_kwargs: dict[str, Any] | None = None,
    n_blocks: int = 256,
) -> None:
    # TODO: building a manifest for COGs will require scanning all COGs and we should store the
    # bounds per tile in the COG potentially. Or have an expanding index where we index all the
    # individual cogs but expand their tiles at task time...
    # TODO: We should probably support stac-geoparquet directly as well.
    del cog_urls, output_path, open_kwargs, n_blocks
    msg = "COG manifest generation is not implemented yet."
    raise NotImplementedError(msg)


def _as_query_batch(query: np.ndarray) -> Float32Array:
    queries = np.asarray(query, dtype=np.float32)
    if queries.ndim != 2:
        msg = f"Expected query to have shape (nq, d), got {queries.shape!r}."
        raise ValueError(msg)
    return queries
