from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from rayzin.types import ChunkRef, Float32Array, Int64Array

if TYPE_CHECKING:
    from rayzin.types import SearchResults


@runtime_checkable
class SearchResultHeap(Protocol):
    @property
    def tau(self) -> float: ...

    def add(
        self, distances: np.ndarray, chunk: ChunkRef, offsets: np.ndarray
    ) -> tuple[Int64Array, Float32Array]: ...
    def add_results(self, results: "SearchResults") -> int: ...
    def clear(self) -> None: ...
    def results(self) -> "SearchResults": ...


@runtime_checkable
class SearchBackend(Protocol):
    def create_heap(self, k: int) -> SearchResultHeap:
        """Create a heap suitable for merging chunk-local search results."""
        ...

    def search(
        self, vectors: Float32Array, query: Float32Array, k: int
    ) -> tuple[Float32Array, Int64Array]:
        """Return (distances, local_indices) of the k nearest vectors, sorted ascending."""
        ...

    def radius_search(
        self, vectors: Float32Array, query: Float32Array, radius: float
    ) -> tuple[Float32Array, Int64Array]:
        """Return (distances, local_indices) for all vectors within radius of query."""
        ...
