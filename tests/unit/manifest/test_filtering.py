from pathlib import Path

import numpy as np
import pytest
import ray.data
import zarr
from ray.data.expressions import col

from rayzin.manifest.filtering import filter_manifest
from rayzin.pipeline import build_manifest_from_zarr

shapely = pytest.importorskip("shapely")
from shapely.geometry import box  # type: ignore[import-untyped]  # noqa: E402


@pytest.fixture
def geospatial_manifest(ray_session: None, tmp_path: Path) -> tuple[str, str]:
    store_path = str(tmp_path / "source.zarr")
    manifest_path = str(tmp_path / "manifest.parquet")
    vectors = np.arange(1 * 4 * 4 * 3, dtype=np.float32).reshape(1, 4, 4, 3)

    root = zarr.open_group(store_path, mode="w")
    spatial_ref = root.create_group("spatial_ref")
    spatial_ref.attrs["GeoTransform"] = "0 1 0 4 0 -1"
    array = root.create_array(
        "embeddings",
        shape=vectors.shape,
        dtype=vectors.dtype,
        chunks=(1, 2, 2, 3),
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x", "embedding"]
    array[:] = vectors

    build_manifest_from_zarr(store_path, manifest_path, n_blocks=2)
    return store_path, manifest_path


def test_filter_manifest_applies_row_predicate(geospatial_manifest: tuple[str, str]) -> None:
    dataset = ray.data.from_items(
        [
            {"value": 1},
            {"value": 2},
            {"value": 3},
        ]
    )

    rows = filter_manifest(
        dataset,
        filter_expr=col("value") >= 2,
    ).take_all()

    assert [int(row["value"]) for row in rows] == [2, 3]


def test_filter_manifest_intersects_aoi(geospatial_manifest: tuple[str, str]) -> None:
    _, manifest_path = geospatial_manifest
    aoi = box(0.5, 2.5, 1.5, 3.5)

    rows = filter_manifest(ray.data.read_parquet(manifest_path), aoi=aoi).take_all()

    assert len(rows) == 1
    assert rows[0]["slice"] == [
        {"dim": "time", "start": 0, "stop": 1},
        {"dim": "y", "start": 0, "stop": 2},
        {"dim": "x", "start": 0, "stop": 2},
    ]
