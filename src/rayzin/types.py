from dataclasses import dataclass
from typing import Final, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

# Manifest column names
COL_URL: Final = "url"
COL_SLICE: Final = "slice"
COL_DIM: Final = "dim"
COL_START: Final = "start"
COL_STOP: Final = "stop"
COL_COUNT: Final = "count"
COL_CENTROID: Final = "centroid"
COL_RADIUS: Final = "radius"
COL_LOWER_BOUND: Final = "lower_bound"

# Result column names
COL_CHUNK_ID: Final = "chunk_id"
COL_OFFSET: Final = "offset"
COL_DISTANCE: Final = "distance"

MANIFEST_VERSION: Final = 1

Float32Array = npt.NDArray[np.float32]
Int64Array = npt.NDArray[np.int64]


class DimSlice(TypedDict):
    dim: str
    start: int
    stop: int


IndexSlice: TypeAlias = list[DimSlice]


class ChunkRef(TypedDict):
    url: str
    slice: IndexSlice


class ChunkRecord(ChunkRef):
    count: int
    centroid: Float32Array
    radius: float


class LowerBoundRow(ChunkRecord):
    lower_bound: float


@dataclass(frozen=True)
class SearchResults:
    chunks: list[ChunkRef]
    offsets: list[int]
    distances: list[float]

    def to_rows(self) -> list[dict[str, object]]:
        return [
            {
                COL_CHUNK_ID: f"{chunk[COL_URL]}#"
                + "/".join(
                    f"{part[COL_DIM]}={part[COL_START]}:{part[COL_STOP]}"
                    for part in chunk[COL_SLICE]
                ),
                COL_URL: chunk[COL_URL],
                COL_SLICE: chunk[COL_SLICE],
                COL_OFFSET: offset,
                COL_DISTANCE: distance,
            }
            for chunk, offset, distance in zip(self.chunks, self.offsets, self.distances)
        ]
