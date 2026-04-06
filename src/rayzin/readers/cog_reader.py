from typing import Any

from rayzin.types import ChunkRecord, Float32Array


class CogVectorReader:
    """Reads embedding vectors from Cloud-Optimised GeoTIFF files.

    Each COG stores embeddings as bands; one file = one chunk.
    Spatial dims are (y, x); band dim maps to the embedding dimension d.
    """

    def __init__(self, **open_kwargs: Any) -> None:
        self._open_kwargs = open_kwargs

    def read(self, chunk: ChunkRecord) -> tuple[Float32Array, tuple[int, ...]]:
        raise NotImplementedError
