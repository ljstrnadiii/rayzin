from typing import Any

import ray.data
from ray.data.expressions import Expr
from shapely.geometry.base import BaseGeometry  # type: ignore[import-untyped]

from rayzin.manifest.spatial import (
    GeoTransform,
    chunk_from_row,
    chunk_polygon,
    read_zarr_geotransform,
    validate_aoi_geometry,
)


def filter_manifest(
    dataset: ray.data.Dataset,
    *,
    filter_expr: Expr | None = None,
    aoi: BaseGeometry | None = None,
    store_kwargs: dict[str, Any] | None = None,
) -> ray.data.Dataset:
    filtered = dataset
    if isinstance(filter_expr, Expr):
        filtered = filtered.filter(expr=filter_expr)
    if aoi is not None:
        filtered = filtered.filter(
            ChunkIntersectsAOI,
            fn_constructor_kwargs={
                "aoi": aoi,
                "store_kwargs": store_kwargs or {},
            },
        )
    return filtered


class ChunkIntersectsAOI:
    # TODO: we can get away with opening tha zarr store once since we can extract the geotransform
    # and analytically check for intersection given a chunks key.
    # TODO: this should probably be renamed to something zarr specific and we should probable
    # support cog by forcing manifest to be geoparquet and adding geometry up front per cog tile.
    def __init__(
        self,
        aoi: BaseGeometry,
        store_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._aoi = validate_aoi_geometry(aoi)
        self._store_kwargs = store_kwargs or {}
        self._transform_cache: dict[str, GeoTransform] = {}

    def __call__(self, row: dict[str, Any]) -> bool:
        chunk = chunk_from_row(row)
        transform = self._transform_cache.get(chunk["url"])
        if transform is None:
            transform = read_zarr_geotransform(
                chunk["url"],
                store_kwargs=self._store_kwargs,
            )
            self._transform_cache[chunk["url"]] = transform
        return bool(chunk_polygon(chunk, transform).intersects(self._aoi))
