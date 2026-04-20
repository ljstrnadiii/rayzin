import pyarrow as pa  # type: ignore[import-untyped]

from rayzin.schema import SEARCH_RESULT_SCHEMA, SearchResultTable
from rayzin.types import (
    COL_CHUNK_ID,
    COL_DIM,
    COL_DISTANCE,
    COL_OFFSET,
    COL_QUERY_ID,
    COL_SLICE,
    COL_START,
    COL_STOP,
    COL_URL,
    ChunkRef,
    SearchResults,
)


def format_chunk_id(chunk: ChunkRef) -> str:
    encoded_slice = "/".join(
        f"{part[COL_DIM]}={part[COL_START]}:{part[COL_STOP]}" for part in chunk[COL_SLICE]
    )
    return f"{chunk[COL_URL]}#{encoded_slice}"


def search_result_table(results: SearchResults) -> SearchResultTable:
    return pa.Table.from_pydict(
        {
            COL_QUERY_ID: results.query_ids,
            COL_CHUNK_ID: [format_chunk_id(chunk) for chunk in results.chunks],
            COL_URL: [chunk[COL_URL] for chunk in results.chunks],
            COL_SLICE: [chunk[COL_SLICE] for chunk in results.chunks],
            COL_OFFSET: results.offsets,
            COL_DISTANCE: results.distances,
        },
        schema=SEARCH_RESULT_SCHEMA,
    )
