from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np

from rayzin.enums import MetricType
from rayzin.search.backends.protocols import SearchResultHeap
from rayzin.types import (
    COL_DIM,
    COL_SLICE,
    COL_START,
    COL_STOP,
    ChunkRef,
    Float32Array,
    Int64Array,
    SearchResults,
)

ChunkKey = tuple[str, tuple[tuple[str, int, int], ...]]


class FaissResultHeap(SearchResultHeap):
    """Top-k heap backed by faiss.ResultHeap for distance selection."""

    def __init__(self, k: int) -> None:
        self._k = k
        self._heap: Any
        self._chunk_refs: list[ChunkRef]
        self._chunk_index_by_key: dict[ChunkKey, int]
        self._result_metadata: dict[int, tuple[int, int]]
        self._next_result_id: int
        self._size: int
        self.clear()

    @property
    def tau(self) -> float:
        return float(self._heap.D[0, 0]) if self._size == self._k else float("inf")

    def add(
        self,
        distances: np.ndarray,
        chunk: ChunkRef,
        offsets: np.ndarray,
    ) -> tuple[Int64Array, Float32Array]:
        if len(distances) == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
            )

        chunk_index = self._chunk_index(chunk)
        added_offsets: list[int] = []
        added_distances: list[float] = []
        for distance, offset in zip(distances.tolist(), offsets.tolist()):
            result_id = self._next_result_id
            self._result_metadata[result_id] = (chunk_index, int(offset))
            self._next_result_id += 1

            self._heap.add_result(
                np.asarray([[distance]], dtype=np.float32),
                np.asarray([[result_id]], dtype=np.int64),
            )
            self._size = min(self._k, self._size + 1)
            live_ids = self._live_ids()
            if result_id in live_ids:
                added_offsets.append(int(offset))
                added_distances.append(float(distance))
            self._prune_metadata(live_ids)
        return (
            np.asarray(added_offsets, dtype=np.int64),
            np.asarray(added_distances, dtype=np.float32),
        )

    def clear(self) -> None:
        self._heap = faiss.ResultHeap(1, self._k, keep_max=False)
        self._chunk_refs = []
        self._chunk_index_by_key = {}
        self._result_metadata = {}
        self._next_result_id = 0
        self._size = 0

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
        self._heap.finalize()
        heap_ids = np.asarray(self._heap.I[0], dtype=np.int64)
        heap_distances = np.asarray(self._heap.D[0], dtype=np.float32)

        chunks: list[ChunkRef] = []
        offsets: list[int] = []
        distances: list[float] = []
        for result_id, distance in zip(heap_ids.tolist(), heap_distances.tolist()):
            if result_id < 0:
                continue
            chunk_index, offset = self._result_metadata[int(result_id)]
            chunks.append(self._chunk_refs[chunk_index])
            offsets.append(offset)
            distances.append(float(distance))
        return SearchResults(chunks=chunks, offsets=offsets, distances=distances)

    def _chunk_index(self, chunk: ChunkRef) -> int:
        key = _chunk_key(chunk)
        existing = self._chunk_index_by_key.get(key)
        if existing is not None:
            return existing

        index = len(self._chunk_refs)
        self._chunk_refs.append(chunk)
        self._chunk_index_by_key[key] = index
        return index

    def _live_ids(self) -> set[int]:
        return {
            int(result_id)
            for result_id in np.asarray(self._heap.I[0], dtype=np.int64).tolist()
            if int(result_id) >= 0
        }

    def _prune_metadata(self, live_ids: set[int] | None = None) -> None:
        if live_ids is None:
            live_ids = self._live_ids()
        self._result_metadata = {
            result_id: metadata
            for result_id, metadata in self._result_metadata.items()
            if result_id in live_ids
        }


class FaissSearchBackend:
    """KNN backend backed by FAISS."""

    def __init__(
        self,
        metric_type: MetricType = MetricType.EUCLIDEAN,
        use_gpu: bool = False,
    ) -> None:
        self._metric_type = metric_type
        self._use_gpu = use_gpu

    def create_heap(self, k: int) -> SearchResultHeap:
        return FaissResultHeap(k)

    def search(
        self, vectors: Float32Array, query: Float32Array, k: int
    ) -> tuple[Float32Array, Int64Array]:
        if len(vectors) == 0:
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        vectors32, query32 = _prepare_inputs(vectors, query, self._metric_type)
        index = _build_index(vectors32.shape[1], self._metric_type, self._use_gpu)
        index.add(vectors32)
        distances, indices = index.search(query32, min(k, len(vectors32)))
        flat_distances = np.asarray(distances[0], dtype=np.float32)
        if self._metric_type == MetricType.COSINE:
            flat_distances = np.asarray(1.0 - flat_distances, dtype=np.float32)
        return flat_distances, np.asarray(indices[0], dtype=np.int64)

    def radius_search(
        self, vectors: Float32Array, query: Float32Array, radius: float
    ) -> tuple[Float32Array, Int64Array]:
        if len(vectors) == 0:
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        vectors32, query32 = _prepare_inputs(vectors, query, self._metric_type)
        index = _build_index(vectors32.shape[1], self._metric_type, self._use_gpu)
        index.add(vectors32)
        threshold = radius if self._metric_type == MetricType.EUCLIDEAN else 1.0 - radius
        limits, distances, indices = index.range_search(query32, threshold)
        del limits
        flat_distances = np.asarray(distances, dtype=np.float32)
        if self._metric_type == MetricType.COSINE:
            flat_distances = np.asarray(1.0 - flat_distances, dtype=np.float32)
        return flat_distances, np.asarray(indices, dtype=np.int64)


def _build_index(
    dimension: int,
    metric_type: MetricType,
    use_gpu: bool,
) -> Any:
    if metric_type == MetricType.EUCLIDEAN:
        index = faiss.IndexFlatL2(dimension)
    else:
        index = faiss.IndexFlatIP(dimension)

    if not use_gpu:
        return index
    if not hasattr(faiss, "StandardGpuResources"):
        msg = "This FAISS build does not include GPU support."
        raise NotImplementedError(msg)
    resources = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(resources, 0, index)


def _prepare_inputs(
    vectors: Float32Array,
    query: Float32Array,
    metric_type: MetricType,
) -> tuple[Float32Array, Float32Array]:
    vectors32 = np.asarray(vectors, dtype=np.float32, order="C").copy()
    query32 = np.asarray(query, dtype=np.float32, order="C").reshape(1, -1).copy()
    if metric_type == MetricType.COSINE:
        faiss.normalize_L2(vectors32)
        faiss.normalize_L2(query32)
    return vectors32, query32


def _chunk_key(chunk: ChunkRef) -> ChunkKey:
    return (
        chunk["url"],
        tuple(
            (
                part[COL_DIM],
                int(part[COL_START]),
                int(part[COL_STOP]),
            )
            for part in chunk[COL_SLICE]
        ),
    )
