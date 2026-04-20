import math
from collections.abc import Mapping, Sequence
from typing import Any

import pyproj
import zarr
from pyproj import CRS
from shapely.geometry.base import BaseGeometry  # type: ignore[import-untyped]
from shapely.ops import transform as shapely_transform  # type: ignore[import-untyped]

from rayzin.readers.zarr_layout import dimension_names
from rayzin.types import (
    COL_DIM,
    COL_SLICE,
    COL_START,
    COL_STOP,
    COL_URL,
    ChunkRef,
    DimSlice,
)

GeoTransform = tuple[float, float, float, float, float, float]


def read_zarr_geotransform(
    store_url: str,
    *,
    store_kwargs: dict[str, Any] | None = None,
) -> GeoTransform:
    # TODO: also support spatial zarr conventions.
    group = zarr.open_group(
        store=store_url,
        mode="r",
        storage_options=store_kwargs or None,
    )
    if "spatial_ref" not in group:
        msg = f"Zarr store {store_url} is missing spatial_ref."
        raise ValueError(msg)
    spatial_ref = group["spatial_ref"]

    raw_transform = spatial_ref.attrs.get("GeoTransform")
    if raw_transform is None:
        msg = f"Zarr store {store_url} is missing spatial_ref.GeoTransform."
        raise ValueError(msg)

    return parse_geotransform(raw_transform)


def read_zarr_crs(
    store_url: str,
    *,
    array_name: str = "embeddings",
    store_kwargs: dict[str, Any] | None = None,
) -> CRS:
    group = zarr.open_group(
        store=store_url,
        mode="r",
        storage_options=store_kwargs or None,
    )
    array = group[array_name]
    assert isinstance(array, zarr.Array)

    for attrs in (array.attrs, group.attrs):
        raw_crs = _crs_from_attrs(attrs)
        if raw_crs is not None:
            return _coerce_crs(raw_crs)

    if "spatial_ref" in group:
        spatial_ref = group["spatial_ref"]
        spatial_ref_crs: Any = spatial_ref.attrs.get("crs_wkt") or spatial_ref.attrs.get(
            "spatial_ref"
        )
        if spatial_ref_crs is not None:
            return _coerce_crs(spatial_ref_crs)

    msg = (
        f"Zarr store {store_url} is missing CRS metadata. "
        "Expected one of crs, proj:code, proj:wkt2, proj:projjson, "
        "spatial_ref.crs_wkt, or spatial_ref.spatial_ref."
    )
    raise ValueError(msg)


def parse_geotransform(raw_transform: Any) -> GeoTransform:
    parts: Sequence[Any]
    if isinstance(raw_transform, str):
        parts = raw_transform.split()
    elif isinstance(raw_transform, Sequence):
        parts = raw_transform
    else:
        msg = f"Unsupported GeoTransform value: {raw_transform!r}"
        raise TypeError(msg)

    if len(parts) != 6:
        msg = f"Expected 6 GeoTransform coefficients, got {len(parts)}."
        raise ValueError(msg)

    origin_x = _coerce_float(parts[0])
    pixel_width = _coerce_float(parts[1])
    row_rotation = _coerce_float(parts[2])
    origin_y = _coerce_float(parts[3])
    column_rotation = _coerce_float(parts[4])
    pixel_height = _coerce_float(parts[5])
    return origin_x, pixel_width, row_rotation, origin_y, column_rotation, pixel_height


def chunk_polygon(chunk: ChunkRef, transform: GeoTransform) -> BaseGeometry:
    from shapely.geometry import Polygon  # type: ignore[import-untyped]

    x_start, x_stop = _dim_interval(chunk["slice"], "x")
    y_start, y_stop = _dim_interval(chunk["slice"], "y")
    x0, y0 = pixel_to_world(transform, x_start, y_start)
    x1, y1 = pixel_to_world(transform, x_stop, y_start)
    x2, y2 = pixel_to_world(transform, x_stop, y_stop)
    x3, y3 = pixel_to_world(transform, x_start, y_stop)
    return Polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)])


def pixel_to_world(transform: GeoTransform, x: int, y: int) -> tuple[float, float]:
    origin_x, pixel_width, row_rotation, origin_y, column_rotation, pixel_height = transform
    world_x = origin_x + (x * pixel_width) + (y * row_rotation)
    world_y = origin_y + (x * column_rotation) + (y * pixel_height)
    return world_x, world_y


def validate_aoi_geometry(aoi: BaseGeometry) -> BaseGeometry:
    if isinstance(aoi, BaseGeometry):
        if aoi.is_empty:
            msg = "AOI geometry is empty."
            raise ValueError(msg)
        return aoi

    msg = "AOI must be a shapely BaseGeometry. " f"Got {type(aoi).__name__}."
    raise TypeError(msg)


def selector_from_aoi(
    store_url: str,
    aoi: BaseGeometry,
    *,
    array_name: str = "embeddings",
    store_kwargs: dict[str, Any] | None = None,
    aoi_crs: CRS | None = None,
    x_dim: str = "x",
    y_dim: str = "y",
) -> dict[str, slice]:
    geometry = validate_aoi_geometry(aoi)
    group = zarr.open_group(
        store=store_url,
        mode="r",
        storage_options=store_kwargs or None,
    )
    array = group[array_name]
    assert isinstance(array, zarr.Array)

    dim_names = dimension_names(array)
    if x_dim not in dim_names or y_dim not in dim_names:
        msg = (
            "AOI selectors require spatial dimensions to be present in the array. "
            f"Got x_dim={x_dim!r}, y_dim={y_dim!r}, available={dim_names!r}."
        )
        raise ValueError(msg)

    if aoi_crs is None:
        msg = "selector_from_aoi requires an explicit aoi_crs for CRS-aware AOI selection."
        raise ValueError(msg)

    mosaic_crs = read_zarr_crs(
        store_url,
        array_name=array_name,
        store_kwargs=store_kwargs,
    )

    geometry = _transform_geometry_to_mosaic_crs(
        geometry,
        source_crs=aoi_crs,
        target_crs=mosaic_crs,
    )
    transform = read_zarr_geotransform(store_url, store_kwargs=store_kwargs)
    x_axis = dim_names.index(x_dim)
    y_axis = dim_names.index(y_dim)
    x_size = int(array.shape[x_axis])
    y_size = int(array.shape[y_axis])

    minx, miny, maxx, maxy = geometry.bounds
    corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
    pixel_coords = [
        _world_to_pixel(transform, x=world_x, y=world_y) for world_x, world_y in corners
    ]
    cols = [col for col, _row in pixel_coords]
    rows = [row for _col, row in pixel_coords]

    x_start = max(0, math.floor(min(cols)))
    x_stop = min(x_size, math.ceil(max(cols)))
    y_start = max(0, math.floor(min(rows)))
    y_stop = min(y_size, math.ceil(max(rows)))
    if x_start >= x_stop or y_start >= y_stop:
        msg = "AOI does not intersect the raster extent."
        raise ValueError(msg)

    return {
        x_dim: slice(x_start, x_stop),
        y_dim: slice(y_start, y_stop),
    }


def chunk_from_row(row: dict[str, Any]) -> ChunkRef:
    return ChunkRef(
        url=str(row[COL_URL]),
        slice=_coerce_slice(row[COL_SLICE]),
    )


def _dim_interval(parts: list[DimSlice], dim: str) -> tuple[int, int]:
    for part in parts:
        if part[COL_DIM] == dim:
            return int(part[COL_START]), int(part[COL_STOP])
    msg = f"Chunk is missing the required {dim!r} dimension."
    raise KeyError(msg)


def _coerce_slice(raw: Any) -> list[DimSlice]:
    if not isinstance(raw, list):
        msg = f"Expected slice to be a list, got {type(raw).__name__}."
        raise TypeError(msg)

    parts: list[DimSlice] = []
    for part in raw:
        if not isinstance(part, dict):
            msg = f"Expected each slice entry to be a dict, got {type(part).__name__}."
            raise TypeError(msg)
        parts.append(
            DimSlice(
                dim=str(part[COL_DIM]),
                start=_coerce_int(part[COL_START]),
                stop=_coerce_int(part[COL_STOP]),
            )
        )
    return parts


def _coerce_float(value: Any) -> float:
    return float(str(value))


def _coerce_int(value: Any) -> int:
    return int(str(value))


def _crs_from_attrs(attrs: Mapping[str, Any]) -> str | dict[str, Any] | None:
    for key in ("crs", "proj:code", "proj:wkt2", "proj:projjson"):
        raw = attrs.get(key)
        if raw is None:
            continue
        if key == "proj:projjson":
            if isinstance(raw, dict):
                return raw
            msg = f"Expected 'proj:projjson' metadata to be a dict, got {type(raw).__name__}."
            raise TypeError(msg)
        return str(raw)
    return None


def _world_to_pixel(transform: GeoTransform, *, x: float, y: float) -> tuple[float, float]:
    origin_x, pixel_width, row_rotation, origin_y, column_rotation, pixel_height = transform
    if row_rotation != 0.0 or column_rotation != 0.0:
        msg = "AOI selectors do not support rotated geotransforms."
        raise NotImplementedError(msg)
    if pixel_width == 0.0 or pixel_height == 0.0:
        msg = "GeoTransform must have non-zero pixel size."
        raise ValueError(msg)

    col = (x - origin_x) / pixel_width
    row = (y - origin_y) / pixel_height
    return float(col), float(row)


def _transform_geometry_to_mosaic_crs(
    geometry: BaseGeometry,
    *,
    source_crs: CRS,
    target_crs: CRS,
) -> BaseGeometry:
    if source_crs == target_crs:
        return geometry
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return shapely_transform(transformer.transform, geometry)


def _coerce_crs(raw_crs: Any) -> CRS:
    try:
        return CRS.from_user_input(raw_crs)
    except Exception as exc:
        raise ValueError(
            "Unable to resolve CRS information for AOI selection. "
            "Check that pyproj is installed with valid PROJ data "
            "and that both CRS values are valid."
        ) from exc
