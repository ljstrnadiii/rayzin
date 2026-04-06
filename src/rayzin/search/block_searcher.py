from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import ray

from rayzin.enums import MetricType, ReaderType, SearchBackendType
from rayzin.manifest.schema import (
    BLOCK_SEARCH_SUMMARY_SCHEMA,
    BlockSearchSummaryTable,
    LowerBoundTable,
)
from rayzin.readers import make_reader
from rayzin.search.backends import make_search_backend
from rayzin.types import (
    COL_CENTROID,
    COL_COUNT,
    COL_DIM,
    COL_LOWER_BOUND,
    COL_RADIUS,
    COL_SLICE,
    COL_START,
    COL_STOP,
    COL_URL,
    ChunkRecord,
    ChunkRef,
    DimSlice,
    Float32Array,
    IndexSlice,
    LowerBoundRow,
    SearchResults,
)


class BlockSearcher:
    """Ray Data actor: processes one full block of manifest rows and returns batch summary."""

    def __init__(
        self,
        query: Float32Array,
        k: int,
        metric_type: str,
        reader_type: str,
        reader_kwargs: dict[str, Any],
        backend_type: str,
        heap_actor: Any,
    ) -> None:
        metric = MetricType(metric_type)
        self.query = np.asarray(query, dtype=np.float32)
        self.k = k
        self.reader = make_reader(ReaderType(reader_type), **reader_kwargs)
        self.backend = make_search_backend(SearchBackendType(backend_type), metric)
        self.heap = self.backend.create_heap(self.k)
        self.heap_actor = heap_actor
        self.global_tau = float("inf")

    def __call__(self, batch: LowerBoundTable) -> BlockSearchSummaryTable:
        return self._search_batch(batch)

    def _search_batch(self, batch: LowerBoundTable) -> BlockSearchSummaryTable:
        self._refresh_global_tau()
        rows = _manifest_rows(batch)
        rows.sort(key=lambda row: row[COL_LOWER_BOUND])
        added_chunks: list[ChunkRef] = []
        added_offsets: list[int] = []
        added_distances: list[float] = []
        rows_searched = 0

        for row in rows:
            if row[COL_LOWER_BOUND] >= min(self.heap.tau, self.global_tau):
                break

            rows_searched += 1
            chunk = _chunk_record(row)
            chunk_ref = _chunk_ref(chunk)
            vectors, _shape = self.reader.read(chunk)
            distances, local_indices = self.backend.search(vectors, self.query, self.k)
            new_offsets, new_distances = self.heap.add(distances, chunk_ref, local_indices)
            if len(new_offsets) == 0:
                continue

            added_chunks.extend([chunk_ref] * len(new_offsets))
            added_offsets.extend(int(offset) for offset in new_offsets.tolist())
            added_distances.extend(float(distance) for distance in new_distances.tolist())

        if added_offsets:
            merged_tau = float(
                ray.get(
                    self.heap_actor.add_results.remote(
                        SearchResults(
                            chunks=added_chunks,
                            offsets=added_offsets,
                            distances=added_distances,
                        )
                    )
                )
            )
            self.global_tau = min(self.global_tau, merged_tau)
        return pa.Table.from_pydict(
            {
                "rows_seen": [len(rows)],
                "rows_searched": [rows_searched],
                "results_added": [len(added_offsets)],
            },
            schema=BLOCK_SEARCH_SUMMARY_SCHEMA,
        )

    def _refresh_global_tau(self) -> None:
        self.global_tau = min(
            self.global_tau,
            float(ray.get(self.heap_actor.tau.remote())),
        )


def _manifest_rows(batch: LowerBoundTable) -> list[LowerBoundRow]:
    urls = batch.column(COL_URL).to_pylist()
    slices = batch.column(COL_SLICE).to_pylist()
    counts = batch.column(COL_COUNT).to_pylist()
    centroids = batch.column(COL_CENTROID).to_pylist()
    radii = batch.column(COL_RADIUS).to_pylist()
    lower_bounds = batch.column(COL_LOWER_BOUND).to_pylist()

    return [
        LowerBoundRow(
            url=str(urls[i]),
            slice=_coerce_index_slice(slices[i]),
            count=int(counts[i]),
            centroid=np.asarray(centroids[i], dtype=np.float32),
            radius=float(radii[i]),
            lower_bound=float(lower_bounds[i]),
        )
        for i in range(batch.num_rows)
    ]


def _chunk_record(row: LowerBoundRow) -> ChunkRecord:
    return ChunkRecord(
        url=row[COL_URL],
        slice=row[COL_SLICE],
        count=row[COL_COUNT],
        centroid=row[COL_CENTROID],
        radius=row[COL_RADIUS],
    )


def _chunk_ref(chunk: ChunkRecord) -> ChunkRef:
    return ChunkRef(
        url=chunk[COL_URL],
        slice=chunk[COL_SLICE],
    )


def _coerce_index_slice(raw: Any) -> IndexSlice:
    if not isinstance(raw, list):
        msg = f"Expected slice to be a list, got {type(raw).__name__}."
        raise TypeError(msg)

    parts: IndexSlice = []
    for part in raw:
        if not isinstance(part, dict):
            msg = f"Expected each slice entry to be a dict, got {type(part).__name__}."
            raise TypeError(msg)
        parts.append(
            DimSlice(
                dim=str(part[COL_DIM]),
                start=int(part[COL_START]),
                stop=int(part[COL_STOP]),
            )
        )
    return parts
