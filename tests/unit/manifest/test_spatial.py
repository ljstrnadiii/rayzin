import pytest

from rayzin.manifest.spatial import chunk_polygon, parse_geotransform
from rayzin.types import ChunkRef

shapely = pytest.importorskip("shapely")
from shapely.geometry import box  # type: ignore[import-untyped]  # noqa: E402


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
