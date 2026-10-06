import asyncio
import io
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from rayzin.types import COL_DIM, COL_SLICE, COL_START, ChunkRecord, Float32Array

FAR: float = 1e3
ROW_GROUP = "row_group"
ROW_INDEX = "__rayzin_row"


class ObjectFile(io.RawIOBase):
    """An obstore readable file as the seekable binary file pyarrow reads Parquet from."""

    def __init__(self, reader: Any) -> None:
        self._reader = reader

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        data = self._reader.read(len(buffer))
        memoryview(buffer)[: len(data)] = memoryview(data)
        return len(data)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return int(self._reader.seek(offset, whence))

    def tell(self) -> int:
        return int(self._reader.tell())


def sparse_file(size: int, ranges: Sequence[tuple[int, Any]]) -> pa.BufferReader:
    """A file of ``size`` bytes holding only the fetched ``ranges``, as pyarrow reads Parquet.

    The buffer is allocated but only the ranges are written, so the rest never takes memory, and
    pyarrow reads it natively: a Python file object would take the GIL for every small read.
    Callers decode with ``use_threads=False``: they decode on threads of their own, and Ray
    sizes Arrow's pool by ``OMP_NUM_THREADS``, one thread in an actor with one CPU.
    """
    buffer = np.empty(size, dtype=np.uint8)
    for start, data in ranges:
        view = np.frombuffer(memoryview(data), dtype=np.uint8)
        buffer[start : start + len(view)] = view
    return pa.BufferReader(pa.py_buffer(buffer))


@dataclass(frozen=True)
class FileInfo:
    """What streaming a file's row groups needs: its size, footer and vector layout."""

    size: int
    footer: pq.FileMetaData
    dtype: np.dtype[Any]
    width: int


@dataclass(frozen=True)
class EncodedRowGroup:
    """A row group's column chunks, fetched but not decoded."""

    reader: "ParquetVectorReader"
    info: FileInfo
    group: int
    ranges: list[tuple[int, Any]]

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.info.dtype

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.info.footer.row_group(self.group).num_rows, 1, self.info.width

    @property
    def window(self) -> tuple[int, int]:
        return self.shape[0], 1

    def decode_into(self, out: np.ndarray) -> None:
        """Vectors in their stored dtype, with invalid or filtered-out rows as NaN."""
        file = pq.ParquetFile(sparse_file(self.info.size, self.ranges), metadata=self.info.footer)
        table = self.reader.read_columns(file, self.group)
        values = table.column(self.reader.column).combine_chunks()
        rows, _, width = self.shape
        cells = out.view(self.dtype).reshape(rows, width)
        flat = values.values.slice(values.offset * width, rows * width)
        cells[:] = flat.to_numpy(zero_copy_only=False).reshape(rows, width)
        cells[~self.reader.passes(table, values)] = np.nan


class ParquetVectorReader:
    """Reads embedding vectors from a fixed-size list column of Parquet files, a row group a chunk.

    A chunk is one row group, named by its ``row_group`` slice, so offsets are row positions
    within it and map straight back to the row a vector came from. Only ``column`` and the
    ``row_filter_columns`` are read. Rows failing ``row_filter``, e.g. ``pc.field("score") >
    0.1``, or with any non-finite value are no-data: they come back as a far-away vector, so
    offsets stay aligned and no-data never ranks. With ``normalize``, vectors are unit length.

    A row group is read with one ranged request per run of adjacent column chunks it needs, at
    offsets from its file's footer, which is read once per file. ``fetch_encoded`` does the
    same for ``RawBlockStream``.
    """

    def __init__(
        self,
        column: str = "embedding",
        normalize: bool = True,
        store_kwargs: dict[str, Any] | None = None,
        row_filter: pc.Expression | None = None,
        row_filter_columns: Sequence[str] = (),
        threads: int = 8,
    ) -> None:
        self.column = column
        self.decode_threads = threads
        self._normalize = normalize
        self._store_kwargs = store_kwargs or {}
        self._row_filter = row_filter
        self._filter_columns = list(row_filter_columns)
        self._stores: dict[str, Any] = {}
        self._files: dict[str, FileInfo] = {}
        self._infos: dict[str, asyncio.Future[FileInfo]] = {}
        self._pool = ThreadPoolExecutor(threads)

    def read(self, chunk: ChunkRecord) -> tuple[Float32Array, tuple[int, ...]]:
        [(vectors, shape)] = self.read_many([chunk])
        return vectors, shape

    def read_many(
        self, chunks: Sequence[ChunkRecord]
    ) -> list[tuple[Float32Array, tuple[int, ...]]]:
        def one(chunk: ChunkRecord) -> tuple[Float32Array, tuple[int, ...]]:
            vectors, valid = self.read_row_group(chunk["url"], _row_group(chunk))
            vectors[~valid] = FAR
            return vectors, (len(vectors),)

        return list(self._pool.map(one, chunks))

    def row_groups(self, url: str) -> int:
        return int(self._file_info(url).footer.num_row_groups)

    def read_row_group(self, url: str, group: int) -> tuple[Float32Array, np.ndarray]:
        """Unit vectors of one row group and which rows hold a valid, unfiltered vector."""
        table = self.read_columns(self._row_group_file(url, group, self._columns()), group)
        values = table.column(self.column).combine_chunks()
        width = values.type.list_size
        vectors = np.asarray(
            values.values.to_numpy(zero_copy_only=False), dtype=np.float32
        ).reshape(len(values), width)
        valid = np.isfinite(vectors).all(axis=1) & self.passes(table, values)
        if self._normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            np.divide(vectors, np.maximum(norms, 1e-12), out=vectors, where=valid[:, None])
        return vectors, valid

    def read_columns(self, file: pq.ParquetFile, group: int) -> pa.Table:
        return file.read_row_group(group, columns=self._columns(), use_threads=False)

    def passes(self, table: pa.Table, values: pa.FixedSizeListArray) -> np.ndarray:
        """Which rows hold a vector and pass ``row_filter``."""
        valid = ~np.asarray(values.is_null().to_numpy(zero_copy_only=False))
        if self._row_filter is not None:
            indexed = table.append_column(ROW_INDEX, pa.array(np.arange(len(table))))
            kept = indexed.filter(self._row_filter).column(ROW_INDEX).to_numpy()
            passes = np.zeros(len(table), dtype=bool)
            passes[kept] = True
            valid &= passes
        return valid

    async def fetch_encoded(self, chunk: ChunkRecord) -> EncodedRowGroup:
        """A chunk's row group, fetched but not decoded."""
        import obstore

        url, group = chunk["url"], _row_group(chunk)
        info = await self._info(url)
        starts, ends = _spans(info.footer, group, self._columns())
        store, path = self._store_and_path(url)
        parts = await obstore.get_ranges_async(store, path, starts=starts, ends=ends)
        return EncodedRowGroup(self, info, group, list(zip(starts, parts, strict=True)))

    def read_rows(
        self, url: str, group: int, rows: Sequence[int], columns: Sequence[str]
    ) -> pa.Table:
        """Other columns of some rows of a row group, e.g. a hit's geometry and score."""
        file = self._row_group_file(url, group, columns)
        table = file.read_row_group(group, columns=list(columns), use_threads=False)
        return table.take(pa.array(list(rows), type=pa.int64()))

    def _columns(self) -> list[str]:
        return [self.column, *self._filter_columns]

    def _row_group_file(self, url: str, group: int, columns: Sequence[str]) -> pq.ParquetFile:
        """A file holding just the column chunks of ``group`` that ``columns`` need."""
        import obstore

        info = self._file_info(url)
        starts, ends = _spans(info.footer, group, columns)
        parts = obstore.get_ranges(*self._store_and_path(url), starts=starts, ends=ends)
        ranges = list(zip(starts, parts, strict=True))
        return pq.ParquetFile(sparse_file(info.size, ranges), metadata=info.footer)

    async def _info(self, url: str) -> FileInfo:
        """``_file_info`` without blocking, fetched once however many row groups ask at once."""
        if url not in self._infos:
            self._infos[url] = asyncio.ensure_future(asyncio.to_thread(self._file_info, url))
        try:
            return await self._infos[url]
        except Exception:
            self._infos.pop(url, None)
            raise

    def _file_info(self, url: str) -> FileInfo:
        if url not in self._files:
            import obstore

            handle = obstore.open_reader(*self._store_and_path(url))
            footer = pq.ParquetFile(ObjectFile(handle)).metadata
            field = footer.schema.to_arrow_schema().field(self.column).type
            self._files[url] = FileInfo(
                int(handle.size),
                footer,
                np.dtype(field.value_type.to_pandas_dtype()),
                field.list_size,
            )
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


def _spans(
    footer: pq.FileMetaData, group: int, columns: Sequence[str]
) -> tuple[list[int], list[int]]:
    """Byte ranges of a row group's chunks of ``columns``, adjacent chunks merged, since
    pyarrow reads adjacent chunks in one go."""
    meta = footer.row_group(group)
    spans = []
    for index in range(meta.num_columns):
        chunk = meta.column(index)
        if chunk.path_in_schema.split(".")[0] in columns:
            start = (
                chunk.dictionary_page_offset
                if chunk.has_dictionary_page
                else chunk.data_page_offset
            )
            spans.append((start, start + chunk.total_compressed_size))
    starts: list[int] = []
    ends: list[int] = []
    for start, end in sorted(spans):
        if ends and start <= ends[-1]:
            ends[-1] = max(ends[-1], end)
        else:
            starts.append(start)
            ends.append(end)
    return starts, ends


def row_group_slice(group: int) -> list[dict[str, Any]]:
    return [{COL_DIM: ROW_GROUP, COL_START: group, "stop": group + 1}]


def _row_group(chunk: ChunkRecord) -> int:
    return int({part[COL_DIM]: part[COL_START] for part in chunk[COL_SLICE]}[ROW_GROUP])
