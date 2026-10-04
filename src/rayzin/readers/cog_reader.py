import asyncio
import zlib
from collections.abc import Coroutine, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar
from urllib.parse import urlparse

import numpy as np

from rayzin.types import COL_DIM, COL_SLICE, COL_START, COL_STOP, ChunkRecord, Float32Array

FAR: float = 1e3
NO_COMPRESSION = 1
DEFLATE = (8, 32946)
ZSTD = 50000
CHUNKY = 1
NO_PREDICTOR = 1
IEEE_FLOAT = 3

T = TypeVar("T")


@dataclass(frozen=True)
class CogLayout:
    """What a reader needs to know about a COG's full-resolution image to decode its blocks."""

    height: int
    width: int
    bands: int
    block_height: int
    block_width: int
    dtype: np.dtype[Any]
    compression: int
    transform: tuple[float, float, float, float, float, float]
    epsg: int | None

    @property
    def blocks_down(self) -> int:
        return -(-self.height // self.block_height)

    @property
    def blocks_across(self) -> int:
        return -(-self.width // self.block_width)

    def window(self, row: int, column: int) -> tuple[int, int, int, int]:
        """``(y0, y1, x0, x1)`` of a block, clipped to the image."""
        y0, x0 = row * self.block_height, column * self.block_width
        return (
            y0,
            min(y0 + self.block_height, self.height),
            x0,
            min(x0 + self.block_width, self.width),
        )


@dataclass(frozen=True)
class Block:
    """One decoded COG block: unit vectors where ``valid``, its window and its file's layout."""

    url: str
    row: int
    column: int
    vectors: Float32Array
    valid: np.ndarray
    shape: tuple[int, int]
    layout: CogLayout


class CogVectorReader:
    """Reads embedding vectors from pixel-interleaved COGs, one internal block per chunk.

    A COG stores embeddings as bands, so a block holds whole vectors for its cells and is one
    range request. Blocks are fetched with async-tiff and decoded here, which covers float16:
    deflate, zstd or no compression, no predictor. Cells with any non-finite value are no-data:
    they come back as a far-away vector, so offsets stay row-major within the block and no-data
    never ranks. With ``normalize``, vectors are unit length, so squared L2 distance is
    ``2 - 2 * cosine``.
    """

    def __init__(
        self,
        normalize: bool = True,
        store_kwargs: dict[str, Any] | None = None,
        decode_threads: int = 8,
    ) -> None:
        self._normalize = normalize
        self._store_kwargs = store_kwargs or {}
        self._stores: dict[str, Any] = {}
        self._files: dict[str, tuple[Any, CogLayout]] = {}
        self._pool = ThreadPoolExecutor(decode_threads)
        self._loop: asyncio.AbstractEventLoop | None = None

    def read(self, chunk: ChunkRecord) -> tuple[Float32Array, tuple[int, ...]]:
        [(vectors, shape)] = self.read_many([chunk])
        return vectors, shape

    def read_many(
        self, chunks: Sequence[ChunkRecord]
    ) -> list[tuple[Float32Array, tuple[int, ...]]]:
        blocks = self.run(
            self.fetch_blocks([(chunk["url"], *_origin_of(chunk)) for chunk in chunks])
        )
        out: list[tuple[Float32Array, tuple[int, ...]]] = []
        for block in blocks:
            vectors = block.vectors
            vectors[~block.valid] = FAR
            out.append((vectors, block.shape))
        return out

    def run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coroutine)

    async def layout(self, url: str) -> CogLayout:
        return (await self._open(url))[1]

    async def fetch_blocks(self, origins: Sequence[tuple[str, int, int]]) -> list[Block]:
        """Fetch and decode the blocks at ``(url, y0, x0)`` pixel origins, one batch per file."""
        by_url: dict[str, list[int]] = {}
        for index, (url, _, _) in enumerate(origins):
            by_url.setdefault(url, []).append(index)
        decoded: dict[int, Block] = {}

        async def one_file(url: str, indices: list[int]) -> None:
            layout = await self.layout(url)
            cells = [
                (origins[i][1] // layout.block_height, origins[i][2] // layout.block_width)
                for i in indices
            ]
            for i, block in zip(indices, await self.fetch_file(url, cells), strict=True):
                decoded[i] = block

        await asyncio.gather(*(one_file(url, indices) for url, indices in by_url.items()))
        return [decoded[i] for i in range(len(origins))]

    async def fetch_file(
        self, url: str, cells: Sequence[tuple[int, int]] | None = None
    ) -> list[Block]:
        """Fetch and decode ``(row, column)`` blocks of one file, every block by default."""
        tiff, layout = await self._open(url)
        if cells is None:
            cells = [(r, c) for r in range(layout.blocks_down) for c in range(layout.blocks_across)]
        tiles = await tiff.fetch_tiles([(column, row) for row, column in cells], 0)
        loop = asyncio.get_running_loop()
        return list(
            await asyncio.gather(
                *(
                    loop.run_in_executor(
                        self._pool,
                        self._decode,
                        url,
                        row,
                        column,
                        layout,
                        bytes(tile.compressed_bytes),
                    )
                    for (row, column), tile in zip(cells, tiles, strict=True)
                )
            )
        )

    def _decode(self, url: str, row: int, column: int, layout: CogLayout, data: bytes) -> Block:
        pixels = decode_block(data, layout)
        y0, y1, x0, x1 = layout.window(row, column)
        cells = pixels[: y1 - y0, : x1 - x0].reshape(-1, layout.bands).astype(np.float32)
        valid: np.ndarray = np.asarray(np.isfinite(cells).all(axis=1))
        if self._normalize:
            norms = np.linalg.norm(cells, axis=1, keepdims=True)
            np.divide(cells, np.maximum(norms, 1e-12), out=cells, where=valid[:, None])
        return Block(url, row, column, cells, valid, (y1 - y0, x1 - x0), layout)

    async def _open(self, url: str) -> tuple[Any, CogLayout]:
        if url not in self._files:
            import async_tiff

            store, path = self._store_and_path(url)
            tiff = await async_tiff.TIFF.open(path, store=store)
            self._files[url] = (tiff, cog_layout(tiff.ifds[0]))
        return self._files[url]

    def _store_and_path(self, url: str) -> tuple[Any, str]:
        from obstore.store import LocalStore, from_url

        parsed = urlparse(url)
        if parsed.scheme in ("", "file"):
            return self._stores.setdefault("/", LocalStore("/")), parsed.path.lstrip("/")
        root = f"{parsed.scheme}://{parsed.netloc}"
        if root not in self._stores:
            self._stores[root] = from_url(root, **self._store_kwargs)
        return self._stores[root], parsed.path.lstrip("/")


def cog_layout(ifd: Any) -> CogLayout:
    if int(ifd.planar_configuration) != CHUNKY:
        msg = "Only pixel-interleaved COGs hold whole vectors per block."
        raise NotImplementedError(msg)
    if int(ifd.predictor or NO_PREDICTOR) != NO_PREDICTOR:
        msg = f"COG predictor {int(ifd.predictor)} is not supported; write with no predictor."
        raise NotImplementedError(msg)
    bits, kind = int(ifd.bits_per_sample[0]), int(ifd.sample_format[0])
    if kind != IEEE_FLOAT:
        msg = f"Expected floating-point embeddings, got sample format {kind}."
        raise NotImplementedError(msg)
    scale, tiepoint = ifd.model_pixel_scale, ifd.model_tiepoint
    keys = ifd.geo_key_directory
    return CogLayout(
        height=int(ifd.image_height),
        width=int(ifd.image_width),
        bands=int(ifd.samples_per_pixel),
        block_height=int(ifd.tile_height),
        block_width=int(ifd.tile_width),
        dtype=np.dtype(f"<f{bits // 8}"),
        compression=int(ifd.compression),
        transform=(
            float(scale[0]),
            0.0,
            float(tiepoint[3] - tiepoint[0] * scale[0]),
            0.0,
            -float(scale[1]),
            float(tiepoint[4] + tiepoint[1] * scale[1]),
        ),
        epsg=None if keys is None else (keys.projected_type or keys.geographic_type),
    )


def decode_block(data: bytes, layout: CogLayout) -> np.ndarray:
    """Decode one block to ``(block_height, block_width, bands)``."""
    size = layout.block_height * layout.block_width * layout.bands * layout.dtype.itemsize
    if layout.compression == ZSTD:
        import zstandard

        raw = zstandard.ZstdDecompressor().decompress(data, max_output_size=size)
    elif layout.compression in DEFLATE:
        raw = zlib.decompress(data)
    elif layout.compression == NO_COMPRESSION:
        raw = data
    else:
        msg = f"COG compression {layout.compression} is not supported."
        raise NotImplementedError(msg)
    return np.frombuffer(raw, dtype=layout.dtype).reshape(
        layout.block_height, layout.block_width, layout.bands
    )


def block_slice(layout: CogLayout, row: int, column: int) -> list[dict[str, Any]]:
    y0, y1, x0, x1 = layout.window(row, column)
    return [
        {COL_DIM: "y", COL_START: y0, COL_STOP: y1},
        {COL_DIM: "x", COL_START: x0, COL_STOP: x1},
    ]


def _origin_of(chunk: ChunkRecord) -> tuple[int, int]:
    bounds = {part[COL_DIM]: part[COL_START] for part in chunk[COL_SLICE]}
    return bounds["y"], bounds["x"]
