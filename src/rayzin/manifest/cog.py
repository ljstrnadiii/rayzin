import asyncio
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from rayzin.manifest.schema import INDEX_SLICE_TYPE
from rayzin.readers.cog_reader import Block, CogVectorReader, block_slice
from rayzin.types import (
    COL_CENTROID,
    COL_COUNT,
    COL_EPSG,
    COL_GEOMETRY,
    COL_RADIUS,
    COL_SLICE,
    COL_URL,
)

LONLAT = 4326


def summarize_cog_blocks(
    batch: pa.Table,
    *,
    normalize: bool,
    store_kwargs: dict[str, Any],
    files_in_flight: int,
) -> pa.Table:
    """One manifest row per block of every COG in ``batch``, keeping its other columns.

    Each row carries the block's slice, count, centroid and radius over its valid vectors, and
    its footprint as lon/lat WKB, so an AOI filter never opens a file. A block with no valid
    vector keeps a row with count 0, so the file counts as indexed.
    """
    reader = CogVectorReader(normalize=normalize, store_kwargs=store_kwargs)
    urls = batch.column(COL_URL).to_pylist()
    gate = asyncio.Semaphore(files_in_flight)

    async def one_file(url: str) -> list[tuple[Block, int, np.ndarray, float]]:
        async with gate:
            blocks = await reader.fetch_file(url)
            return [(block, *_summary(block)) for block in blocks]

    async def every_file() -> list[list[tuple[Block, int, np.ndarray, float]]]:
        return list(await asyncio.gather(*(one_file(url) for url in urls)))

    rows: list[int] = []
    slices, counts, centroids, radii, footprints, epsgs = [], [], [], [], [], []
    for index, summaries in enumerate(reader.run(every_file())):
        for block, count, centroid, radius in summaries:
            rows.append(index)
            slices.append(block_slice(block.layout, block.row, block.column))
            counts.append(count)
            centroids.append(centroid)
            radii.append(radius)
            footprints.append(_footprint(block))
            epsgs.append(block.layout.epsg)
    dimension = len(centroids[0]) if centroids else 0
    flat = np.concatenate(centroids).astype(np.float32) if centroids else np.empty(0, np.float32)
    extra = [name for name in batch.column_names if name != COL_URL]
    taken = batch.take(pa.array(rows, type=pa.int64()))
    return pa.Table.from_arrays(
        [
            taken.column(COL_URL),
            pa.array(slices, type=INDEX_SLICE_TYPE),
            pa.array(counts, type=pa.int32()),
            pa.ListArray.from_arrays(
                pa.array(np.arange(len(centroids) + 1, dtype=np.int32) * dimension),
                pa.array(flat),
            ),
            pa.array(radii, type=pa.float32()),
            pa.array(footprints, type=pa.binary()),
            pa.array(epsgs, type=pa.int32()),
            *(taken.column(name) for name in extra),
        ],
        names=[COL_URL, COL_SLICE, COL_COUNT, COL_CENTROID, COL_RADIUS, COL_GEOMETRY, COL_EPSG, *extra],
    )


def _summary(block: Block) -> tuple[int, np.ndarray, float]:
    kept = block.vectors[block.valid]
    if not len(kept):
        return 0, np.zeros(block.vectors.shape[1], np.float32), 0.0
    centroid = kept.mean(axis=0, dtype=np.float32)
    squared = (
        np.einsum("ij,ij->i", kept, kept) - 2 * kept @ centroid + float(centroid @ centroid)
    )
    return int(len(kept)), centroid, float(np.sqrt(max(float(squared.max()), 0.0)))


def _footprint(block: Block) -> bytes:
    import shapely  # type: ignore[import-untyped]

    a, _, c, _, e, f = block.layout.transform
    y0, y1, x0, x1 = block.layout.window(block.row, block.column)
    xs = np.array([x0, x1, x1, x0], dtype=np.float64) * a + c
    ys = np.array([y0, y0, y1, y1], dtype=np.float64) * e + f
    epsg = block.layout.epsg
    if epsg is not None and epsg != LONLAT:
        xs, ys = _to_lonlat(epsg).transform(xs, ys)
    return shapely.to_wkb(shapely.Polygon(zip(xs, ys, strict=True)))  # type: ignore[no-any-return]


_TRANSFORMERS: dict[int, Any] = {}


def _to_lonlat(epsg: int) -> Any:
    if epsg not in _TRANSFORMERS:
        from pyproj import Transformer

        _TRANSFORMERS[epsg] = Transformer.from_crs(epsg, LONLAT, always_xy=True)
    return _TRANSFORMERS[epsg]
