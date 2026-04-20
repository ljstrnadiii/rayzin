from collections.abc import Iterator
from itertools import product
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import zarr

from rayzin.readers.protocol import VectorReader
from rayzin.readers.zarr_layout import index_axis_names
from rayzin.schema import CHUNK_SCHEMA, MANIFEST_SCHEMA, ChunkTable, ManifestTable
from rayzin.selectors import Selector, resolve_selector_intervals
from rayzin.types import (
    COL_DIM,
    COL_SLICE,
    COL_START,
    COL_STOP,
    COL_URL,
    ChunkRecord,
    ChunkRef,
    DimSlice,
    IndexSlice,
)


def build_zarr_chunk_table(
    store_url: str,
    *,
    array_name: str,
    store_kwargs: dict[str, Any],
    embedding_dim_name: str = "embedding",
    selectors: Selector | None = None,
) -> ChunkTable:
    return pa.Table.from_pylist(
        list(
            iter_zarr_chunk_slices(
                store_url,
                array_name=array_name,
                store_kwargs=store_kwargs,
                embedding_dim_name=embedding_dim_name,
                selectors=selectors,
            )
        ),
        schema=CHUNK_SCHEMA,
    )


def iter_zarr_chunk_slices(
    store_url: str,
    *,
    array_name: str,
    store_kwargs: dict[str, Any],
    embedding_dim_name: str,
    selectors: Selector | None = None,
) -> Iterator[ChunkRef]:
    group = zarr.open_group(
        store=store_url,
        mode="r",
        storage_options=store_kwargs or None,
    )
    array = group[array_name]
    assert isinstance(array, zarr.Array)

    if array.ndim < 2:
        msg = (
            "Expected at least one index axis plus the embedding axis, "
            f"got {array.ndim} dimensions."
        )
        raise ValueError(msg)

    axis_names = index_axis_names(array, embedding_dim_name=embedding_dim_name)
    dim_intervals = resolve_selector_intervals(
        group,
        array,
        selectors=selectors,
        embedding_dim_name=embedding_dim_name,
    )

    for intervals in product(*(dim_intervals[axis_name] for axis_name in axis_names)):
        yield ChunkRef(
            url=store_url,
            slice=[
                DimSlice(
                    dim=axis_name,
                    start=start,
                    stop=stop,
                )
                for axis_name, (start, stop) in zip(
                    axis_names,
                    intervals,
                )
            ],
        )


def compute_chunk_summary_arrow(batch: ChunkTable, reader: VectorReader) -> ManifestTable:
    return _compute_chunk_summary(batch, reader)


def _compute_chunk_summary(batch: ChunkTable, reader: VectorReader) -> ManifestTable:
    rows = _chunk_slice_rows(batch)
    centroids: list[list[float]] = []
    radii: list[float] = []
    counts: list[int] = []

    for row in rows:
        chunk = _chunk_record_from_slice(row)
        vectors, _shape = reader.read(chunk)
        if vectors.shape[0] == 0:
            msg = f"Chunk {chunk[COL_URL]} produced no vectors."
            raise ValueError(msg)

        centroid = vectors.mean(axis=0).astype(np.float32)
        distances = np.linalg.norm(vectors - centroid[None, :], axis=1)
        centroids.append(centroid.tolist())
        radii.append(float(distances.max()))
        counts.append(int(vectors.shape[0]))

    return pa.Table.from_arrays(
        [
            batch.column(COL_URL),
            batch.column(COL_SLICE),
            pa.array(counts, type=pa.int32()),
            pa.array(centroids, type=pa.list_(pa.float32())),
            pa.array(radii, type=pa.float32()),
        ],
        schema=MANIFEST_SCHEMA,
    )


def _chunk_slice_rows(batch: ChunkTable) -> list[ChunkRef]:
    urls = batch.column(COL_URL).to_pylist()
    slices = batch.column(COL_SLICE).to_pylist()
    return [
        ChunkRef(
            url=str(urls[i]),
            slice=_coerce_index_slice(slices[i]),
        )
        for i in range(batch.num_rows)
    ]


def _chunk_record_from_slice(chunk: ChunkRef) -> ChunkRecord:
    return ChunkRecord(
        url=chunk[COL_URL],
        slice=chunk[COL_SLICE],
        count=0,
        centroid=np.empty(0, dtype=np.float32),
        radius=0.0,
    )


def _coerce_index_slice(raw: Any) -> IndexSlice:
    if not isinstance(raw, list):
        msg = f"Expected slice to be a list, got {type(raw).__name__}."
        raise TypeError(msg)

    parts: IndexSlice = []
    for part in raw:
        if not isinstance(part, dict):
            msg = f"Expected each slice entry to be a dict, got {type(part).__name__}."
            raise TypeError(msg)
        parts.append(
            DimSlice(
                dim=str(part[COL_DIM]),
                start=int(part[COL_START]),
                stop=int(part[COL_STOP]),
            )
        )
    return parts
