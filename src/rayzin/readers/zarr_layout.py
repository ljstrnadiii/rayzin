from collections.abc import Sequence

import zarr
import zarr.core
import zarr.core.metadata


def dimension_names(
    array: zarr.Array[zarr.core.metadata.ArrayV3Metadata],
) -> tuple[str, ...]:
    raw_names = array.attrs.get("_ARRAY_DIMENSIONS")
    if raw_names is None:
        msg = (
            "Expected _ARRAY_DIMENSIONS metadata including an embedding axis. "
            "Refusing to guess which array axis holds embeddings."
        )
        raise ValueError(msg)

    if not isinstance(raw_names, Sequence) or isinstance(raw_names, str):
        msg = f"Expected _ARRAY_DIMENSIONS to be a sequence, got {raw_names!r}."
        raise TypeError(msg)

    names = tuple(str(name) for name in raw_names)
    if len(names) != array.ndim:
        msg = (
            "Expected _ARRAY_DIMENSIONS to describe every array axis, got "
            f"{len(names)} names for {array.ndim} dimensions."
        )
        raise ValueError(msg)
    return names


def embedding_axis(
    array: zarr.Array[zarr.core.metadata.ArrayV3Metadata],
    *,
    embedding_dim_name: str,
) -> int:
    names = dimension_names(array)
    matches = [index for index, name in enumerate(names) if name == embedding_dim_name]
    if len(matches) != 1:
        msg = (
            f"Expected exactly one {embedding_dim_name!r} axis in _ARRAY_DIMENSIONS, "
            f"got {matches!r}."
        )
        raise ValueError(msg)
    return matches[0]


def index_axis_names(
    array: zarr.Array[zarr.core.metadata.ArrayV3Metadata],
    *,
    embedding_dim_name: str,
) -> tuple[str, ...]:
    names = dimension_names(array)
    embedding_index = embedding_axis(array, embedding_dim_name=embedding_dim_name)
    return tuple(name for index, name in enumerate(names) if index != embedding_index)


def index_axis_positions(
    array: zarr.Array[zarr.core.metadata.ArrayV3Metadata],
    *,
    embedding_dim_name: str,
) -> tuple[int, ...]:
    embedding_index = embedding_axis(array, embedding_dim_name=embedding_dim_name)
    return tuple(index for index in range(array.ndim) if index != embedding_index)
