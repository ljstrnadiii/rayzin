from collections.abc import Iterator
from itertools import islice
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
    COL_LOWER_BOUNDS,
    COL_MIN_LOWER_BOUND,
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
        queries: Float32Array,
        k: int,
        metric_type: str,
        reader_type: str,
        reader_kwargs: dict[str, Any],
        backend_type: str,
        heap_actor: Any,
        prefetch: int = 1,
    ) -> None:
        metric = MetricType(metric_type)
        self.prefetch = max(1, prefetch)
        self.reader = make_reader(ReaderType(reader_type), **reader_kwargs)
        self.backend = make_search_backend(
            SearchBackendType(backend_type), metric, normalize=reader_kwargs.get("normalize", True)
        )
        self._stream: Any = None
        self.reset(queries, k, heap_actor)

    def reset(self, queries: Float32Array, k: int, heap_actor: Any) -> None:
        """Take a new query batch, keeping the reader, backend and block stream warm."""
        self.queries = np.asarray(queries, dtype=np.float32)
        if self.queries.ndim != 2:
            msg = f"Expected queries to have shape (nq, d), got {self.queries.shape!r}."
            raise ValueError(msg)
        self.nq = self.queries.shape[0]
        self.k = k
        self.heap = self.backend.create_heap(self.nq, self.k)
        self.heap_actor = heap_actor
        self.global_tau = np.full(self.nq, float("inf"), dtype=np.float32)

    def __call__(self, batch: LowerBoundTable) -> BlockSearchSummaryTable:
        return self._search_batch(batch)

    def _search_batch(self, batch: LowerBoundTable) -> BlockSearchSummaryTable:
        self._refresh_global_tau()
        rows = _manifest_rows(batch, self.nq)
        rows.sort(key=lambda row: row[COL_MIN_LOWER_BOUND])
        added_query_ids: list[int] = []
        added_chunks: list[ChunkRef] = []
        added_offsets: list[int] = []
        added_distances: list[float] = []
        rows_searched = 0
        query_evaluations = 0

        for row in rows:
            if len(row[COL_LOWER_BOUNDS]) != self.nq:
                msg = (
                    "Expected one lower bound per query, got "
                    f"{len(row[COL_LOWER_BOUNDS])} bounds for {self.nq} queries."
                )
                raise ValueError(msg)

        vectors_searched = 0
        reads = self._streamed(rows) if self._streams() else self._grouped(rows)
        for row, payload in reads:
            effective_tau = np.minimum(self.heap.tau, self.global_tau)
            active_mask = np.asarray(row[COL_LOWER_BOUNDS] < effective_tau, dtype=bool)
            if not np.any(active_mask):
                continue
            rows_searched += 1
            vectors_searched += row[COL_COUNT]
            active_query_ids = np.asarray(np.flatnonzero(active_mask), dtype=np.int64)
            query_evaluations += int(len(active_query_ids))
            chunk_ref = _chunk_ref(_chunk_record(row))
            if self._streams():
                distances, offsets = self.backend.search_raw_device(  # type: ignore[attr-defined]
                    payload, self.queries, self.k
                )
                self.heap.add_device(active_mask, distances, offsets, chunk_ref)  # type: ignore[attr-defined]
                continue
            vectors, _shape = payload
            distances, offsets = self.backend.search(vectors, self.queries[active_mask], self.k)
            new_results = self.heap.add_result_subset(
                active_query_ids, distances, chunk_ref, offsets
            )
            added_query_ids.extend(new_results.query_ids)
            added_chunks.extend(new_results.chunks)
            added_offsets.extend(new_results.offsets)
            added_distances.extend(new_results.distances)

        drain = getattr(self.heap, "drain", None)
        if drain is not None:
            drained = drain()
            added_query_ids, added_chunks = drained.query_ids, drained.chunks
            added_offsets, added_distances = drained.offsets, drained.distances

        if added_offsets:
            merged_tau = np.asarray(
                ray.get(
                    self.heap_actor.add_results.remote(
                        SearchResults(
                            query_ids=added_query_ids,
                            chunks=added_chunks,
                            offsets=added_offsets,
                            distances=added_distances,
                        )
                    )
                ),
                dtype=np.float32,
            )
            if merged_tau.shape != (self.nq,):
                msg = f"Expected global tau to have shape ({self.nq},), got {merged_tau.shape!r}."
                raise ValueError(msg)
            self.global_tau = np.minimum(self.global_tau, merged_tau)
        return pa.Table.from_pydict(
            {
                "rows_seen": [len(rows)],
                "rows_searched": [rows_searched],
                "query_evaluations": [query_evaluations],
                "results_added": [len(added_offsets)],
                "vectors_searched": [vectors_searched],
            },
            schema=BLOCK_SEARCH_SUMMARY_SCHEMA,
        )

    def _streams(self) -> bool:
        """Whether blocks stream to the device: COG blocks, and a backend that scores them raw."""
        from rayzin.readers.cog_reader import CogVectorReader

        return hasattr(self.backend, "search_raw_device") and isinstance(
            self.reader, CogVectorReader
        )

    def _candidates(self, rows: list[LowerBoundRow]) -> Iterator[LowerBoundRow]:
        """Rows, in bound order, that the bounds known when each is reached cannot rule out."""
        for row in rows:
            effective_tau = np.minimum(self.heap.tau, self.global_tau)
            if row[COL_MIN_LOWER_BOUND] >= float(np.max(effective_tau)):
                return
            if np.any(row[COL_LOWER_BOUNDS] < effective_tau):
                yield row

    def _grouped(self, rows: list[LowerBoundRow]) -> Iterator[tuple[LowerBoundRow, Any]]:
        """Read ``prefetch`` candidates at a time, then hand them over one by one."""
        candidates = self._candidates(rows)
        while group := list(islice(candidates, self.prefetch)):
            chunks = [_chunk_record(row) for row in group]
            read_many = getattr(self.reader, "read_many", None)
            fetched = read_many(chunks) if read_many else [self.reader.read(c) for c in chunks]
            yield from zip(group, fetched, strict=True)

    def _streamed(self, rows: list[LowerBoundRow]) -> Iterator[tuple[LowerBoundRow, Any]]:
        """Keep ``prefetch`` candidates fetching and decoding while earlier ones are scored."""
        if self._stream is None:
            from rayzin.readers.cog_reader import RawBlockStream

            self._stream = RawBlockStream(
                self.reader,  # type: ignore[arg-type]
                self.backend.allocate,  # type: ignore[attr-defined]
                in_flight=self.prefetch,
                threads=self.reader.decode_threads,  # type: ignore[attr-defined]
            )
        stream = self._stream
        candidates = self._candidates(rows)
        waiting: dict[int, LowerBoundRow] = {}
        submitted = 0
        while True:
            while stream.pending < self.prefetch and (row := next(candidates, None)) is not None:
                waiting[submitted] = row
                origin = {part[COL_DIM]: part[COL_START] for part in row[COL_SLICE]}
                stream.submit(submitted, row[COL_URL], origin["y"], origin["x"])
                submitted += 1
            if not stream.pending:
                return
            done, block = stream.next()
            yield waiting.pop(done), block
            stream.release(block)

    def _refresh_global_tau(self) -> None:
        remote_tau = np.asarray(ray.get(self.heap_actor.tau.remote()), dtype=np.float32)
        if remote_tau.shape != (self.nq,):
            msg = f"Expected global tau to have shape ({self.nq},), got {remote_tau.shape!r}."
            raise ValueError(msg)
        self.global_tau = np.minimum(self.global_tau, remote_tau)


def _manifest_rows(batch: LowerBoundTable, nq: int) -> list[LowerBoundRow]:
    """Rows to search; without bounds columns, pruning is off and every query is a candidate."""
    rows = batch.num_rows
    names = batch.column_names
    urls = batch.column(COL_URL).to_pylist()
    slices = batch.column(COL_SLICE).to_pylist()
    counts = batch.column(COL_COUNT).to_pylist()
    if COL_LOWER_BOUNDS in names:
        lower_bounds = _matrix(batch.column(COL_LOWER_BOUNDS), rows)
        min_lower_bounds = np.asarray(batch.column(COL_MIN_LOWER_BOUND), dtype=np.float32)
    else:
        lower_bounds = np.zeros((rows, nq), dtype=np.float32)
        min_lower_bounds = np.zeros(rows, dtype=np.float32)
    centroids = (
        _matrix(batch.column(COL_CENTROID), rows)
        if COL_CENTROID in names
        else np.zeros((rows, 0), dtype=np.float32)
    )
    radii = (
        np.asarray(batch.column(COL_RADIUS), dtype=np.float32)
        if COL_RADIUS in names
        else np.zeros(rows, dtype=np.float32)
    )
    return [
        LowerBoundRow(
            url=str(urls[i]),
            slice=_coerce_index_slice(slices[i]),
            count=int(counts[i]),
            centroid=centroids[i],
            radius=float(radii[i]),
            lower_bounds=lower_bounds[i],
            min_lower_bound=float(min_lower_bounds[i]),
        )
        for i in range(rows)
    ]


def _matrix(column: Any, rows: int) -> np.ndarray:
    """A list column of equal-length rows as one ``(rows, width)`` float32 array."""
    values = pa.concat_arrays(column.chunks) if isinstance(column, pa.ChunkedArray) else column
    flat = np.asarray(values.flatten().to_numpy(zero_copy_only=False), dtype=np.float32)
    return flat.reshape(rows, -1) if rows else flat.reshape(0, 0)


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
