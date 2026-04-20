from pathlib import Path

import numpy as np
import pytest
import zarr
from pyproj import CRS

from rayzin.manifest.spatial import (
    chunk_polygon,
    parse_geotransform,
    read_zarr_crs,
    selector_from_aoi,
)
from rayzin.types import ChunkRef

shapely = pytest.importorskip("shapely")
from shapely.geometry import box  # type: ignore[import-untyped]  # noqa: E402


def _require_proj_data() -> None:
    try:
        CRS.from_user_input("EPSG:4326")
    except Exception:
        pytest.skip("pyproj is installed without usable PROJ data")


def test_chunk_polygon_uses_geotransform() -> None:
    transform = parse_geotransform("0 1 0 4 0 -1")
    chunk = ChunkRef(
        url="store.zarr",
        slice=[
            {"dim": "time", "start": 0, "stop": 1},
            {"dim": "y", "start": 0, "stop": 2},
            {"dim": "x", "start": 0, "stop": 2},
        ],
    )

    geometry = chunk_polygon(chunk, transform)

    assert geometry.equals(box(0.0, 2.0, 2.0, 4.0))


def test_selector_from_aoi_returns_pixel_bounds(tmp_path: Path) -> None:
    _require_proj_data()
    store_path = str(tmp_path / "source_selector_aoi.zarr")
    root = zarr.open_group(store_path, mode="w")
    array = root.create_array(
        "embeddings",
        shape=(1, 4, 4, 2),
        dtype=np.float32,
        chunks=(1, 2, 2, 2),
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x", "embedding"]
    spatial_ref = root.create_group("spatial_ref")
    spatial_ref.attrs["GeoTransform"] = "0 1 0 4 0 -1"
    spatial_ref.attrs["crs_wkt"] = "EPSG:4326"

    selector = selector_from_aoi(
        store_path,
        box(1.0, 1.0, 3.0, 3.0),
        aoi_crs=CRS.from_user_input("EPSG:4326"),
    )

    assert selector == {"x": slice(1, 3), "y": slice(1, 3)}


def test_read_zarr_crs_prefers_spatial_ref_metadata(tmp_path: Path) -> None:
    _require_proj_data()
    store_path = str(tmp_path / "source_selector_crs.zarr")
    root = zarr.open_group(store_path, mode="w")
    array = root.create_array(
        "embeddings",
        shape=(1, 4, 4, 2),
        dtype=np.float32,
        chunks=(1, 2, 2, 2),
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x", "embedding"]
    spatial_ref = root.create_group("spatial_ref")
    spatial_ref.attrs["GeoTransform"] = "0 1 0 4 0 -1"
    spatial_ref.attrs["crs_wkt"] = "EPSG:4326"

    assert read_zarr_crs(store_path) == CRS.from_user_input("EPSG:4326")


def test_read_zarr_crs_prefers_array_proj_metadata_over_group(tmp_path: Path) -> None:
    _require_proj_data()
    store_path = str(tmp_path / "source_selector_array_proj_code.zarr")
    root = zarr.open_group(store_path, mode="w")
    root.attrs["proj:code"] = "EPSG:3857"
    array = root.create_array(
        "embeddings",
        shape=(1, 4, 4, 2),
        dtype=np.float32,
        chunks=(1, 2, 2, 2),
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x", "embedding"]
    array.attrs["proj:code"] = "EPSG:4326"

    assert read_zarr_crs(store_path) == CRS.from_user_input("EPSG:4326")


def test_read_zarr_crs_supports_projjson(tmp_path: Path) -> None:
    _require_proj_data()
    store_path = str(tmp_path / "source_selector_projjson.zarr")
    root = zarr.open_group(store_path, mode="w")
    array = root.create_array(
        "embeddings",
        shape=(1, 4, 4, 2),
        dtype=np.float32,
        chunks=(1, 2, 2, 2),
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x", "embedding"]
    array.attrs["proj:projjson"] = CRS.from_user_input("EPSG:4326").to_json_dict()

    assert read_zarr_crs(store_path) == CRS.from_user_input("EPSG:4326")


def test_selector_from_aoi_requires_explicit_aoi_crs(tmp_path: Path) -> None:
    store_path = str(tmp_path / "source_selector_requires_crs.zarr")
    root = zarr.open_group(store_path, mode="w")
    array = root.create_array(
        "embeddings",
        shape=(1, 4, 4, 2),
        dtype=np.float32,
        chunks=(1, 2, 2, 2),
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x", "embedding"]
    spatial_ref = root.create_group("spatial_ref")
    spatial_ref.attrs["GeoTransform"] = "0 1 0 4 0 -1"
    spatial_ref.attrs["crs_wkt"] = "EPSG:4326"

    with pytest.raises(ValueError, match="explicit aoi_crs"):
        selector_from_aoi(store_path, box(1.0, 1.0, 3.0, 3.0))


def test_selector_from_aoi_reprojects_when_crs_differs(tmp_path: Path) -> None:
    pyproj = pytest.importorskip("pyproj")
    _require_proj_data()
    store_path = str(tmp_path / "source_selector_reproject.zarr")
    root = zarr.open_group(store_path, mode="w")
    array = root.create_array(
        "embeddings",
        shape=(1, 4, 4, 2),
        dtype=np.float32,
        chunks=(1, 2, 2, 2),
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x", "embedding"]
    spatial_ref = root.create_group("spatial_ref")
    spatial_ref.attrs["GeoTransform"] = "0 1 0 4 0 -1"
    spatial_ref.attrs["crs_wkt"] = "EPSG:4326"

    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    minx, miny = transformer.transform(1.0, 1.0)
    maxx, maxy = transformer.transform(3.0, 3.0)
    selector = selector_from_aoi(
        store_path,
        box(minx, miny, maxx, maxy),
        aoi_crs=CRS.from_user_input("EPSG:3857"),
    )

    assert selector == {"x": slice(1, 3), "y": slice(1, 3)}
