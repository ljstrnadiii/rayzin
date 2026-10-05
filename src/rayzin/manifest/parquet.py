from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from rayzin.manifest.cog import vector_summary
from rayzin.manifest.schema import INDEX_SLICE_TYPE
from rayzin.readers.parquet_reader import ParquetVectorReader, row_group_slice
from rayzin.types import (
    COL_CENTROID,
    COL_COUNT,
    COL_GEOMETRY,
    COL_RADIUS,
    COL_SLICE,
    COL_URL,
)


def summarize_parquet_row_groups(
    batch: pa.Table,
    *,
    column: str,
    geometry: str | None,
    normalize: bool,
    store_kwargs: dict[str, Any],
    files_in_flight: int,
) -> pa.Table:
    """One manifest row per row group of every Parquet file in ``batch``, keeping its columns.

    Each row carries the row group's slice, count, centroid and radius over its valid vectors
    and, when the files have a lon/lat ``geometry`` column, the row group's bounding box as WKB,
    so an AOI filter never opens a file.
    """
    import shapely  # type: ignore[import-untyped]

    reader = ParquetVectorReader(column=column, normalize=normalize, store_kwargs=store_kwargs)

    def one_file(url: str) -> list[tuple[int, int, np.ndarray, float, bytes | None]]:
        summaries = []
        for group in range(reader.row_groups(url)):
            vectors, valid = reader.read_row_group(url, group)
            count, centroid, radius = vector_summary(vectors, valid)
            footprint = None
            if geometry is not None:
                shapes = reader.read_rows(url, group, range(len(vectors)), [geometry])
                footprint = shapely.to_wkb(shapely.box(*geometry_bounds(shapes.column(geometry))))
            summaries.append((group, count, centroid, radius, footprint))
        return summaries

    urls = batch.column(COL_URL).to_pylist()
    with ThreadPoolExecutor(files_in_flight) as pool:
        per_file = list(pool.map(one_file, urls))

    rows: list[int] = []
    slices, counts, centroids, radii, footprints = [], [], [], [], []
    for index, summaries in enumerate(per_file):
        for group, count, centroid, radius, footprint in summaries:
            rows.append(index)
            slices.append(row_group_slice(group))
            counts.append(count)
            centroids.append(centroid)
            radii.append(radius)
            footprints.append(footprint)
    dimension = len(centroids[0]) if centroids else 0
    flat = np.concatenate(centroids).astype(np.float32) if centroids else np.empty(0, np.float32)
    extra = [name for name in batch.column_names if name != COL_URL]
    taken = batch.take(pa.array(rows, type=pa.int64()))
    columns = [
        taken.column(COL_URL),
        pa.array(slices, type=INDEX_SLICE_TYPE),
        pa.array(counts, type=pa.int32()),
        pa.ListArray.from_arrays(
            pa.array(np.arange(len(centroids) + 1, dtype=np.int32) * dimension), pa.array(flat)
        ),
        pa.array(radii, type=pa.float32()),
    ]
    names = [COL_URL, COL_SLICE, COL_COUNT, COL_CENTROID, COL_RADIUS]
    if geometry is not None:
        columns.append(pa.array(footprints, type=pa.binary()))
        names.append(COL_GEOMETRY)
    return pa.Table.from_arrays(
        [*columns, *(taken.column(name) for name in extra)], names=[*names, *extra]
    )


def geometry_bounds(column: Any) -> tuple[float, float, float, float]:
    """``(minx, miny, maxx, maxy)`` of a WKB or native GeoArrow geometry column."""
    import shapely

    values = pa.concat_arrays(column.chunks) if isinstance(column, pa.ChunkedArray) else column
    if pa.types.is_binary(values.type) or pa.types.is_large_binary(values.type):
        bounds = shapely.total_bounds(shapely.from_wkb(values.to_numpy(zero_copy_only=False)))
        return tuple(float(v) for v in bounds)  # type: ignore[return-value]
    while not pa.types.is_struct(values.type):
        values = values.flatten()
    xs = np.asarray(values.field("x"), dtype=np.float64)
    ys = np.asarray(values.field("y"), dtype=np.float64)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
