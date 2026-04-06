from typing import Protocol, runtime_checkable

from rayzin.types import ChunkRecord, Float32Array


@runtime_checkable
class VectorReader(Protocol):
    def read(self, chunk: ChunkRecord) -> tuple[Float32Array, tuple[int, ...]]:
        """
        Read all vectors for a chunk.

        Returns
        -------
        vectors: float32 ndarray of shape (N, d), where N = product of chunk spatial dims
        original_shape: tuple of ints giving the pre-flatten index shape for the chunk
        """
        ...
