import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import ray
from shapely.geometry.base import BaseGeometry  # type: ignore[import-untyped]

from rayzin.enums import MetricType, ReaderType, SearchBackendType
from rayzin.manifest.filtering import rows_intersecting
from rayzin.metrics import _add_lower_bounds
from rayzin.search.block_searcher import BlockSearcher
from rayzin.search.heap_actor import HeapActor
from rayzin.types import COL_CENTROID, COL_COUNT, COL_RADIUS, SearchResults

FileSystem = Any


class ShardSearcher:
    """One actor's share of a COG manifest, loaded once, searched as often as asked."""

    def __init__(
        self,
        files: list[str],
        filesystem: FileSystem | Callable[[], FileSystem],
        *,
        prune: bool,
        prefetch: int,
        reader_kwargs: dict[str, Any],
        backend_type: str,
        metric_type: str,
        reader_type: str = ReaderType.COG.value,
    ) -> None:
        import pyarrow.dataset as ds  # type: ignore[import-untyped]

        dataset = ds.dataset(files, filesystem=_resolved(filesystem), format="parquet")
        columns = (
            None
            if prune
            else [n for n in dataset.schema.names if n not in (COL_CENTROID, COL_RADIUS)]
        )
        self._table: pa.Table = dataset.to_table(columns=columns, filter=pc.field(COL_COUNT) > 0)
        self._prune = prune
        self._metric = metric_type
        self._searcher = BlockSearcher(
            queries=np.zeros((1, 1), dtype=np.float32),
            k=1,
            metric_type=metric_type,
            reader_type=reader_type,
            reader_kwargs=reader_kwargs,
            backend_type=backend_type,
            heap_actor=None,
            prefetch=prefetch,
        )

    def rows(self) -> int:
        return int(self._table.num_rows)

    def search(
        self,
        queries: np.ndarray,
        k: int,
        heap_actor: Any,
        where: pc.Expression | None,
        aoi: BaseGeometry | None,
    ) -> dict[str, float]:
        started = time.perf_counter()
        table = self._table if where is None else self._table.filter(where)
        if aoi is not None:
            table = rows_intersecting(table, aoi)
        if self._prune:
            table = _add_lower_bounds(table, queries, self._metric)
        prepared = time.perf_counter()
        self._searcher.reset(queries, k, heap_actor)
        summary = self._searcher(table).to_pylist()[0]
        return {
            **summary,
            "prepare_seconds": prepared - started,
            "search_seconds": time.perf_counter() - prepared,
        }


class KnnSearcher:
    """Exact KNN over a manifest, loaded once and searched many times.

    ``knn_cog_search`` and ``knn_parquet_search`` start actors, read the manifest and build a
    heap on every call.
    This keeps all of that alive between searches: the manifest's files are split across
    ``actors`` actors, one per GPU with ``num_gpus_per_actor=1``, each holding its share with a
    warm reader, backend and block stream, and one heap actor merges their top ``k``. Call
    ``close`` to release them.

    ``filter_expr`` is a ``pyarrow.compute.Expression`` over manifest columns and ``aoi`` a
    lon/lat geometry tested against block footprints. ``filesystem`` may be a zero-argument
    callable, run on each actor, for filesystems that do not pickle. The other arguments mean
    what they do for ``knn_cog_search``. ``reader`` picks the manifest's kind: ``COG`` for
    rasters, ``PARQUET`` for vectors stored as rows, with ``reader_kwargs`` such as ``column``,
    ``row_filter`` and ``row_filter_columns`` passed to ``ParquetVectorReader``.
    """

    def __init__(
        self,
        manifest_path: str,
        *,
        actors: int = 8,
        num_cpus_per_actor: float = 1.0,
        num_gpus_per_actor: float = 0.0,
        backend: SearchBackendType = SearchBackendType.TORCH,
        normalize: bool = True,
        prefetch: int = 32,
        decode_threads: int = 4,
        store_kwargs: dict[str, Any] | None = None,
        filesystem: FileSystem | Callable[[], FileSystem] = None,
        prune: bool = True,
        reader: ReaderType = ReaderType.COG,
        reader_kwargs: dict[str, Any] | None = None,
    ) -> None:
        import pyarrow.dataset as ds
        import pyarrow.fs as pafs  # type: ignore[import-untyped]

        started = time.perf_counter()
        self._normalize = normalize
        local = _resolved(filesystem)
        if local is None:
            local, manifest_path = pafs.FileSystem.from_uri(manifest_path)
            filesystem = local
        files = sorted(ds.dataset(manifest_path, filesystem=local, format="parquet").files)
        shard = ray.remote(num_cpus=num_cpus_per_actor, num_gpus=num_gpus_per_actor)(ShardSearcher)
        self._shards: list[Any] = [
            shard.remote(
                files[index::actors],
                filesystem,
                prune=prune,
                prefetch=prefetch,
                reader_kwargs={
                    "normalize": normalize,
                    "store_kwargs": store_kwargs or {},
                    **({"decode_threads": decode_threads} if reader == ReaderType.COG else {}),
                    **(reader_kwargs or {}),
                },
                backend_type=backend.value,
                metric_type=MetricType.EUCLIDEAN.value,
                reader_type=reader.value,
            )
            for index in range(min(actors, len(files)))
        ]
        self._heap = HeapActor.options(num_cpus=0).remote(  # type: ignore[attr-defined]
            nq=1, k=1, metric_type=MetricType.EUCLIDEAN.value, backend_type=backend.value
        )
        self.blocks = sum(ray.get([s.rows.remote() for s in self._shards]))
        self.startup_seconds = time.perf_counter() - started

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        filter_expr: pc.Expression | None = None,
        aoi: BaseGeometry | None = None,
    ) -> SearchResults:
        queries = np.asarray(query, dtype=np.float32)
        if queries.ndim != 2:
            msg = f"Expected query to have shape (nq, d), got {queries.shape!r}."
            raise ValueError(msg)
        if self._normalize:
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            queries = (queries / np.maximum(norms, 1e-12)).astype(np.float32)
        ray.get(self._heap.reset.remote(len(queries), k))
        shared = ray.put(queries)
        summaries = ray.get(
            [s.search.remote(shared, k, self._heap, filter_expr, aoi) for s in self._shards]
        )
        results: SearchResults = ray.get(self._heap.results.remote())
        totals = {
            name: int(sum(summary[column] for summary in summaries))
            for name, column in (
                ("blocks_after_pushdown", "rows_seen"),
                ("blocks_searched", "rows_searched"),
                ("vectors_searched", "vectors_searched"),
                ("query_evaluations", "query_evaluations"),
            )
        }
        slowest = {
            name: round(max(summary[name] for summary in summaries), 3)
            for name in ("prepare_seconds", "search_seconds")
        }
        return replace(results, stats={**totals, **slowest})

    def close(self) -> None:
        for actor in (*self._shards, self._heap):
            ray.kill(actor)


def _resolved(filesystem: FileSystem | Callable[[], FileSystem]) -> FileSystem:
    import pyarrow.fs as pafs

    if callable(filesystem) and not isinstance(filesystem, pafs.FileSystem):
        return filesystem()
    return filesystem
