import heapq

import numpy as np

from rayzin.enums import MetricType
from rayzin.metrics import make_metric
from rayzin.search.backends.protocols import SearchResultHeap
from rayzin.types import ChunkRef, Float32Array, Int64Array, SearchResults


class NumpyResultHeap(SearchResultHeap):
    """Max-heap of size k over (distance, chunk, offset) triples."""

    def __init__(self, k: int) -> None:
        self._k = k
        self._counter = 0
        self._entries: list[tuple[float, int, ChunkRef, int]] = []

    @property
    def tau(self) -> float:
        return -self._entries[0][0] if len(self._entries) == self._k else float("inf")

    def add(
        self,
        distances: np.ndarray,
        chunk: ChunkRef,
        offsets: np.ndarray,
    ) -> tuple[Int64Array, Float32Array]:
        added_offsets: list[int] = []
        added_distances: list[float] = []
        for distance, offset in zip(distances.tolist(), offsets.tolist()):
            entry = (-float(distance), self._counter, chunk, int(offset))
            self._counter += 1
            if len(self._entries) < self._k:
                heapq.heappush(self._entries, entry)
                added_offsets.append(int(offset))
                added_distances.append(float(distance))
            elif float(distance) < self.tau:
                heapq.heapreplace(self._entries, entry)
                added_offsets.append(int(offset))
                added_distances.append(float(distance))
        return (
            np.asarray(added_offsets, dtype=np.int64),
            np.asarray(added_distances, dtype=np.float32),
        )

    def clear(self) -> None:
        self._counter = 0
        self._entries.clear()

    def add_results(self, results: SearchResults) -> int:
        for chunk, offset, distance in zip(
            results.chunks,
            results.offsets,
            results.distances,
        ):
            self.add(
                np.asarray([distance], dtype=np.float32),
                chunk,
                np.asarray([offset], dtype=np.int64),
            )
        return len(results.offsets)

    def results(self) -> SearchResults:
        sorted_entries = sorted(self._entries, key=lambda entry: entry[0], reverse=True)
        return SearchResults(
            chunks=[entry[2] for entry in sorted_entries],
            offsets=[entry[3] for entry in sorted_entries],
            distances=[-entry[0] for entry in sorted_entries],
        )


class NumpySearchBackend:
    def __init__(self, metric_type: MetricType = MetricType.EUCLIDEAN) -> None:
        self._metric = make_metric(metric_type)

    def create_heap(self, k: int) -> SearchResultHeap:
        return NumpyResultHeap(k)

    def search(
        self, vectors: Float32Array, query: Float32Array, k: int
    ) -> tuple[Float32Array, Int64Array]:
        distances = self._metric.pairwise(vectors, query)
        if len(distances) == 0:
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        k = min(k, len(distances))
        indices = np.argpartition(distances, k - 1)[:k]
        order = np.argsort(distances[indices])
        sorted_indices = np.asarray(indices[order], dtype=np.int64)
        return np.asarray(distances[sorted_indices], dtype=np.float32), sorted_indices

    def radius_search(
        self, vectors: Float32Array, query: Float32Array, radius: float
    ) -> tuple[Float32Array, Int64Array]:
        distances = self._metric.pairwise(vectors, query)
        indices = np.asarray(np.where(distances <= radius)[0], dtype=np.int64)
        return np.asarray(distances[indices], dtype=np.float32), indices
