from pathlib import Path

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import zarr

from rayzin.pipeline import build_manifest_from_zarr
from rayzin.schema import MANIFEST_SCHEMA


def test_build_manifest_from_zarr_writes_chunk_summaries(
    ray_session: None,
    tmp_path: Path,
) -> None:
    store_path = str(tmp_path / "source.zarr")
    manifest_path = str(tmp_path / "manifest.parquet")
    vectors = np.arange(1 * 4 * 4 * 3, dtype=np.float32).reshape(1, 4, 4, 3)

    root = zarr.open_group(store_path, mode="w")
    array = root.create_array(
        "embeddings",
        shape=vectors.shape,
        dtype=vectors.dtype,
        chunks=(1, 2, 2, 3),
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x", "embedding"]
    array[:] = vectors

    build_manifest_from_zarr(store_path, manifest_path, n_blocks=2)

    table = pq.read_table(manifest_path)
    assert table.schema.equals(MANIFEST_SCHEMA)

    rows = sorted(
        table.to_pylist(),
        key=lambda row: tuple((part["dim"], part["start"]) for part in row["slice"]),
    )
    first_chunk = vectors[0, 0:2, 0:2, :].reshape(-1, 3)
    expected_centroid = first_chunk.mean(axis=0)
    expected_radius = float(np.linalg.norm(first_chunk - expected_centroid[None, :], axis=1).max())

    assert len(rows) == 4
    assert rows[0]["slice"] == [
        {"dim": "time", "start": 0, "stop": 1},
        {"dim": "y", "start": 0, "stop": 2},
        {"dim": "x", "start": 0, "stop": 2},
    ]
    assert rows[0]["count"] == 4
    assert rows[0]["centroid"] == pytest.approx(expected_centroid.tolist())
    assert rows[0]["radius"] == pytest.approx(expected_radius)


def test_build_manifest_from_zarr_supports_embedding_axis_not_last(
    ray_session: None,
    tmp_path: Path,
) -> None:
    store_path = str(tmp_path / "source_embedding_middle.zarr")
    manifest_path = str(tmp_path / "manifest_embedding_middle.parquet")
    vectors = np.arange(1 * 3 * 4 * 4, dtype=np.float32).reshape(1, 3, 4, 4)

    root = zarr.open_group(store_path, mode="w")
    array = root.create_array(
        "embeddings",
        shape=vectors.shape,
        dtype=vectors.dtype,
        chunks=(1, 3, 2, 2),
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "feature", "y", "x"]
    array[:] = vectors

    build_manifest_from_zarr(
        store_path,
        manifest_path,
        n_blocks=2,
        embedding_dim_name="feature",
    )

    table = pq.read_table(manifest_path)
    assert table.schema.equals(MANIFEST_SCHEMA)

    rows = sorted(
        table.to_pylist(),
        key=lambda row: tuple((part["dim"], part["start"]) for part in row["slice"]),
    )
    first_chunk = np.moveaxis(vectors[0:1, :, 0:2, 0:2], 1, -1).reshape(-1, 3)
    expected_centroid = first_chunk.mean(axis=0)
    expected_radius = float(np.linalg.norm(first_chunk - expected_centroid[None, :], axis=1).max())

    assert len(rows) == 4
    assert rows[0]["slice"] == [
        {"dim": "time", "start": 0, "stop": 1},
        {"dim": "y", "start": 0, "stop": 2},
        {"dim": "x", "start": 0, "stop": 2},
    ]
    assert rows[0]["count"] == 4
    assert rows[0]["centroid"] == pytest.approx(expected_centroid.tolist())
    assert rows[0]["radius"] == pytest.approx(expected_radius)
