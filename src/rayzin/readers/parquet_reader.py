import io
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
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


class ParquetVectorReader:
    """Reads embedding vectors from a fixed-size list column of Parquet files, a row group a chunk.

    A chunk is one row group, named by its ``row_group`` slice, so offsets are row positions
    within it and map straight back to the row a vector came from. Only ``column`` and the
    ``row_filter_columns`` are read. Rows failing ``row_filter``, e.g. ``pc.field("score") >
    0.1``, or with any non-finite value are no-data: they come back as a far-away vector, so
    offsets stay aligned and no-data never ranks. With ``normalize``, vectors are unit length.
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
        self._column = column
        self._normalize = normalize
        self._store_kwargs = store_kwargs or {}
        self._row_filter = row_filter
        self._filter_columns = list(row_filter_columns)
        self._stores: dict[str, Any] = {}
        self._footers: dict[str, pq.FileMetaData] = {}
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
        return int(self._footer(url).num_row_groups)

    def read_row_group(self, url: str, group: int) -> tuple[Float32Array, np.ndarray]:
        """Unit vectors of one row group and which rows hold a valid, unfiltered vector."""
        table = self._file(url).read_row_group(group, columns=[self._column, *self._filter_columns])
        values = table.column(self._column).combine_chunks()
        width = values.type.list_size
        vectors = np.asarray(
            values.values.to_numpy(zero_copy_only=False), dtype=np.float32
        ).reshape(len(values), width)
        valid = np.isfinite(vectors).all(axis=1) & ~np.asarray(values.is_null())
        if self._row_filter is not None:
            indexed = table.append_column(ROW_INDEX, pa.array(np.arange(len(table))))
            kept = indexed.filter(self._row_filter).column(ROW_INDEX).to_numpy()
            passes = np.zeros(len(table), dtype=bool)
            passes[kept] = True
            valid &= passes
        if self._normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            np.divide(vectors, np.maximum(norms, 1e-12), out=vectors, where=valid[:, None])
        return vectors, valid

    def read_rows(
        self, url: str, group: int, rows: Sequence[int], columns: Sequence[str]
    ) -> pa.Table:
        """Other columns of some rows of a row group, e.g. a hit's geometry and score."""
        table = self._file(url).read_row_group(group, columns=list(columns))
        return table.take(pa.array(list(rows), type=pa.int64()))

    def _file(self, url: str) -> pq.ParquetFile:
        """A handle of its own for each read; one file object read from several threads at once
        crashes pyarrow. The footer is read once per file and reused."""
        import obstore
        from obstore.store import LocalStore, from_url

        parsed = urlparse(url)
        if parsed.scheme in ("", "file"):
            root, path = "/", parsed.path.lstrip("/")
            store = self._stores.setdefault(root, LocalStore("/"))
        else:
            root, path = f"{parsed.scheme}://{parsed.netloc}", parsed.path.lstrip("/")
            if root not in self._stores:
                self._stores[root] = from_url(root, **self._store_kwargs)
            store = self._stores[root]
        handle = ObjectFile(obstore.open_reader(store, path))
        footer = self._footers.get(url)
        file = pq.ParquetFile(handle, metadata=footer)
        if footer is None:
            self._footers[url] = file.metadata
        return file

    def _footer(self, url: str) -> pq.FileMetaData:
        if url not in self._footers:
            self._file(url)
        return self._footers[url]


def row_group_slice(group: int) -> list[dict[str, Any]]:
    return [{COL_DIM: ROW_GROUP, COL_START: group, "stop": group + 1}]


def _row_group(chunk: ChunkRecord) -> int:
    return int({part[COL_DIM]: part[COL_START] for part in chunk[COL_SLICE]}[ROW_GROUP])
