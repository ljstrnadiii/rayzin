import heapq

import numpy as np

from rayzin.enums import MetricType
from rayzin.metrics import make_metric
from rayzin.search.backends.protocols import SearchResultHeap
from rayzin.types import ChunkRef, Float32Array, Int64Array, SearchResults


class NumpyResultHeap(SearchResultHeap):
    """Top-k heaps over a batch of query vectors."""

    def __init__(self, nq: int, k: int) -> None:
        self._nq = nq
        self._k = k
        self._counter = 0
        self._entries: list[list[tuple[float, int, ChunkRef, int]]] = [[] for _ in range(nq)]

    @property
    def tau(self) -> Float32Array:
        return np.asarray(
            [
                -entries[0][0] if len(entries) == self._k else float("inf")
                for entries in self._entries
            ],
            dtype=np.float32,
        )

    def add_result_subset(
        self,
        query_ids: Int64Array,
        distances: Float32Array,
        chunk: ChunkRef,
        offsets: Int64Array,
    ) -> SearchResults:
        query_index_array = np.asarray(query_ids, dtype=np.int64)
        distance_matrix = np.asarray(distances, dtype=np.float32)
        offset_matrix = np.asarray(offsets, dtype=np.int64)
        if distance_matrix.shape != offset_matrix.shape:
            msg = (
                "Expected distances and offsets to have the same shape, got "
                f"{distance_matrix.shape!r} and {offset_matrix.shape!r}."
            )
            raise ValueError(msg)
        if distance_matrix.ndim != 2:
            msg = (
                "Expected distances to have shape (n_active, k_eff), got "
                f"{distance_matrix.shape!r}."
            )
            raise ValueError(msg)
        if distance_matrix.shape[0] != len(query_index_array):
            msg = (
                "Expected one distance row per active query, got "
                f"{distance_matrix.shape[0]} rows for {len(query_index_array)} query ids."
            )
            raise ValueError(msg)

        added_query_ids: list[int] = []
        added_chunks: list[ChunkRef] = []
        added_distances: list[float] = []
        added_offsets: list[int] = []
        tau = self.tau
        for query_position, query_id in enumerate(query_index_array.tolist()):
            entries = self._entries[int(query_id)]
            for distance, offset in zip(
                distance_matrix[query_position].tolist(),
                offset_matrix[query_position].tolist(),
            ):
                entry = (-float(distance), self._counter, chunk, int(offset))
                self._counter += 1
                if len(entries) < self._k:
                    heapq.heappush(entries, entry)
                    tau[int(query_id)] = -entries[0][0] if len(entries) == self._k else float("inf")
                    added_query_ids.append(int(query_id))
                    added_chunks.append(chunk)
                    added_offsets.append(int(offset))
                    added_distances.append(float(distance))
                elif float(distance) < float(tau[int(query_id)]):
                    heapq.heapreplace(entries, entry)
                    tau[int(query_id)] = -entries[0][0]
                    added_query_ids.append(int(query_id))
                    added_chunks.append(chunk)
                    added_offsets.append(int(offset))
                    added_distances.append(float(distance))
        return SearchResults(
            query_ids=added_query_ids,
            chunks=added_chunks,
            offsets=added_offsets,
            distances=added_distances,
        )

    def clear(self) -> None:
        self._counter = 0
        self._entries = [[] for _ in range(self._nq)]

    def add_results(self, results: SearchResults) -> int:
        for query_id, chunk, offset, distance in zip(
            results.query_ids,
            results.chunks,
            results.offsets,
            results.distances,
        ):
            self.add_result_subset(
                np.asarray([query_id], dtype=np.int64),
                np.asarray([[distance]], dtype=np.float32),
                chunk,
                np.asarray([[offset]], dtype=np.int64),
            )
        return len(results.offsets)

    def results(self) -> SearchResults:
        query_ids: list[int] = []
        chunks: list[ChunkRef] = []
        offsets: list[int] = []
        distances: list[float] = []
        for query_id, entries in enumerate(self._entries):
            sorted_entries = sorted(entries, key=lambda entry: entry[0], reverse=True)
            query_ids.extend([query_id] * len(sorted_entries))
            chunks.extend(entry[2] for entry in sorted_entries)
            offsets.extend(entry[3] for entry in sorted_entries)
            distances.extend(-entry[0] for entry in sorted_entries)
        return SearchResults(
            query_ids=query_ids,
            chunks=chunks,
            offsets=offsets,
            distances=distances,
        )


class NumpySearchBackend:
    def __init__(self, metric_type: MetricType = MetricType.EUCLIDEAN) -> None:
        self._metric = make_metric(metric_type)

    def create_heap(self, nq: int, k: int) -> SearchResultHeap:
        return NumpyResultHeap(nq, k)

    def search(
        self, vectors: Float32Array, queries: Float32Array, k: int
    ) -> tuple[Float32Array, Int64Array]:
        query_matrix = np.asarray(queries, dtype=np.float32)
        if query_matrix.ndim != 2:
            msg = f"Expected queries to have shape (nq, d), got {query_matrix.shape!r}."
            raise ValueError(msg)

        pairwise = np.asarray(self._metric.pairwise(vectors, query_matrix), dtype=np.float32)
        if len(vectors) == 0:
            return (
                np.empty((len(query_matrix), 0), dtype=np.float32),
                np.empty((len(query_matrix), 0), dtype=np.int64),
            )

        k = min(k, pairwise.shape[1])
        if k == 0:
            return (
                np.empty((len(query_matrix), 0), dtype=np.float32),
                np.empty((len(query_matrix), 0), dtype=np.int64),
            )
        indices = np.argpartition(pairwise, kth=k - 1, axis=1)[:, :k]
        selected_distances = np.take_along_axis(pairwise, indices, axis=1)
        order = np.argsort(selected_distances, axis=1)
        sorted_indices = np.take_along_axis(indices, order, axis=1)
        sorted_distances = np.take_along_axis(selected_distances, order, axis=1)
        return (
            np.asarray(sorted_distances, dtype=np.float32),
            np.asarray(sorted_indices, dtype=np.int64),
        )

    def radius_search(
        self, vectors: Float32Array, query: Float32Array, radius: float
    ) -> tuple[Float32Array, Int64Array]:
        distances = self._metric.pairwise(vectors, query)
        indices = np.asarray(np.where(distances <= radius)[0], dtype=np.int64)
        return np.asarray(distances[indices], dtype=np.float32), indices
