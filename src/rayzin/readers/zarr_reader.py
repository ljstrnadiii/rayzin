from typing import Any

import numpy as np
import zarr

from rayzin.readers.zarr_layout import dimension_names, embedding_axis, index_axis_names
from rayzin.types import COL_SLICE, COL_START, COL_STOP, ChunkRecord, Float32Array


class ZarrVectorReader:
    """Reads embedding vectors from a Zarr group.

    Expected array layout: group[array_name] has one axis named
    ``embedding_dim_name`` in ``_ARRAY_DIMENSIONS`` and any number of index
    axes.
    The reader flattens all index dims into a single axis.
    """

    def __init__(
        self,
        array_name: str = "embeddings",
        store_kwargs: dict[str, Any] | None = None,
        embedding_dim_name: str = "embedding",
    ) -> None:
        self._array_name = array_name
        self._store_kwargs: dict[str, Any] = store_kwargs or {}
        self._embedding_dim_name = embedding_dim_name
        self._cache: dict[str, zarr.Array] = {}  # type: ignore[type-arg]

    def _open(self, url: str) -> zarr.Array:  # type: ignore[type-arg]
        if url not in self._cache:
            group = zarr.open_group(
                store=url,
                mode="r",
                storage_options=self._store_kwargs or None,
            )
            arr = group[self._array_name]
            assert isinstance(arr, zarr.Array)
            self._cache[url] = arr
        return self._cache[url]

    def read(self, chunk: ChunkRecord) -> tuple[Float32Array, tuple[int, ...]]:
        arr = self._open(chunk["url"])
        if arr.ndim < 2:
            msg = (
                "Expected at least one index axis plus the embedding axis, "
                f"got {arr.ndim} dimensions."
            )
            raise ValueError(msg)

        chunk_slice = chunk[COL_SLICE]
        all_axis_names = dimension_names(arr)
        axis_names = index_axis_names(arr, embedding_dim_name=self._embedding_dim_name)
        embedding_index = embedding_axis(arr, embedding_dim_name=self._embedding_dim_name)
        parts_by_dim = {part["dim"]: part for part in chunk_slice}
        if len(parts_by_dim) != len(chunk_slice):
            msg = f"Chunk {chunk['url']} contains duplicate slice dimensions."
            raise ValueError(msg)

        missing_dims = [name for name in axis_names if name not in parts_by_dim]
        extra_dims = [part["dim"] for part in chunk_slice if part["dim"] not in axis_names]
        if missing_dims or extra_dims:
            msg = (
                f"Chunk slice dimensions do not match array {chunk['url']}: "
                f"missing={missing_dims}, extra={extra_dims}."
            )
            raise ValueError(msg)

        indexer = tuple(
            slice(None)
            if axis_index == embedding_index
            else slice(
                parts_by_dim[axis_name][COL_START],
                parts_by_dim[axis_name][COL_STOP],
            )
            for axis_index, axis_name in enumerate(all_axis_names)
        )
        raw = np.asarray(arr[indexer], dtype=np.float32)
        raw = np.moveaxis(raw, embedding_index, -1)
        spatial_shape = tuple(
            int(parts_by_dim[axis_name][COL_STOP] - parts_by_dim[axis_name][COL_START])
            for axis_name in axis_names
        )
        vectors = raw.reshape(-1, raw.shape[-1])
        return vectors, spatial_shape
