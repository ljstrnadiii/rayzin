from collections.abc import Sequence
from typing import Any

import zarr
from shapely.geometry.base import BaseGeometry  # type: ignore[import-untyped]

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
