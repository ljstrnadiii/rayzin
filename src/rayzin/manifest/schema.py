from typing import Annotated, TypeAlias

import pyarrow as pa  # type: ignore[import-untyped]

from rayzin.types import (
    COL_CENTROID,
    COL_CHUNK_ID,
    COL_COUNT,
    COL_DIM,
    COL_DISTANCE,
    COL_LOWER_BOUND,
    COL_OFFSET,
    COL_RADIUS,
    COL_SLICE,
    COL_START,
    COL_STOP,
    COL_URL,
)

INDEX_SLICE_TYPE: pa.ListType = pa.list_(
    pa.struct(
        [
            pa.field(COL_DIM, pa.string()),
            pa.field(COL_START, pa.int64()),
            pa.field(COL_STOP, pa.int64()),
        ]
    )
)

CHUNK_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field(COL_URL, pa.string()),
        pa.field(COL_SLICE, INDEX_SLICE_TYPE),
    ]
)

MANIFEST_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field(COL_URL, pa.string()),
        pa.field(COL_SLICE, INDEX_SLICE_TYPE),
        pa.field(COL_COUNT, pa.int32()),
        pa.field(COL_CENTROID, pa.list_(pa.float32())),
        pa.field(COL_RADIUS, pa.float32()),
    ]
)

LOWER_BOUND_SCHEMA: pa.Schema = MANIFEST_SCHEMA.append(pa.field(COL_LOWER_BOUND, pa.float32()))

SEARCH_RESULT_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field(COL_CHUNK_ID, pa.string()),
        pa.field(COL_URL, pa.string()),
        pa.field(COL_SLICE, INDEX_SLICE_TYPE),
        pa.field(COL_OFFSET, pa.int64()),
        pa.field(COL_DISTANCE, pa.float32()),
    ]
)

BLOCK_SEARCH_SUMMARY_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("rows_seen", pa.int64()),
        pa.field("rows_searched", pa.int64()),
        pa.field("results_added", pa.int64()),
    ]
)

ChunkTable: TypeAlias = Annotated[pa.Table, CHUNK_SCHEMA]
ManifestTable: TypeAlias = Annotated[pa.Table, MANIFEST_SCHEMA]
LowerBoundTable: TypeAlias = Annotated[pa.Table, LOWER_BOUND_SCHEMA]
SearchResultTable: TypeAlias = Annotated[pa.Table, SEARCH_RESULT_SCHEMA]
BlockSearchSummaryTable: TypeAlias = Annotated[pa.Table, BLOCK_SEARCH_SUMMARY_SCHEMA]
