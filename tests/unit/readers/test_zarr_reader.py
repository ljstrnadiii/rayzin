from pathlib import Path

import numpy as np
import zarr

from rayzin.readers.zarr_reader import ZarrVectorReader
from rayzin.types import ChunkRecord


def test_zarr_reader_supports_arrays_without_time_axis(tmp_path: Path) -> None:
    store_path = str(tmp_path / "source.zarr")
    vectors = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)

    root = zarr.open_group(store_path, mode="w")
    array = root.create_array(
        "embeddings",
        shape=vectors.shape,
        dtype=vectors.dtype,
        chunks=(2, 2, 3),
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["y", "x", "embedding"]
    array[:] = vectors

    reader = ZarrVectorReader()
    chunk = ChunkRecord(
        url=store_path,
        slice=[
            {"dim": "y", "start": 1, "stop": 3},
            {"dim": "x", "start": 2, "stop": 4},
        ],
        count=0,
        centroid=np.empty(0, dtype=np.float32),
        radius=0.0,
    )

    flat, spatial_shape = reader.read(chunk)
    expected = vectors[1:3, 2:4, :].reshape(-1, 3)

    assert spatial_shape == (2, 2)
    np.testing.assert_allclose(flat, expected)


def test_zarr_reader_supports_embedding_axis_not_last(tmp_path: Path) -> None:
    store_path = str(tmp_path / "source_embedding_first.zarr")
    vectors = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)

    root = zarr.open_group(store_path, mode="w")
    array = root.create_array(
        "embeddings",
        shape=vectors.shape,
        dtype=vectors.dtype,
        chunks=(3, 2, 2),
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["feature", "y", "x"]
    array[:] = vectors

    reader = ZarrVectorReader(embedding_dim_name="feature")
    chunk = ChunkRecord(
        url=store_path,
        slice=[
            {"dim": "y", "start": 1, "stop": 3},
            {"dim": "x", "start": 2, "stop": 4},
        ],
        count=0,
        centroid=np.empty(0, dtype=np.float32),
        radius=0.0,
    )

    flat, spatial_shape = reader.read(chunk)
    expected = np.moveaxis(vectors[:, 1:3, 2:4], 0, -1).reshape(-1, 3)

    assert spatial_shape == (2, 2)
    np.testing.assert_allclose(flat, expected)
