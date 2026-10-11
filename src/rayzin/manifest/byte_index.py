"""Metadata-only byte index for existing Parquet row groups and COG tiles."""

import asyncio
import hashlib
import json
from dataclasses import asdict
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from rayzin.manifest.catalog import Catalog
from rayzin.readers.cog_reader import CogLayout, block_slice, cog_layout
from rayzin.readers.parquet_reader import row_group_slice


def _bucket(url: str) -> int:
    return hashlib.blake2s(url.encode(), digest_size=2).digest()[0]


def _store_path(url: str, stores: dict[str, Any], options: dict[str, Any]) -> tuple[Any, str]:
    from obstore.store import LocalStore, from_url

    parsed = urlparse(url)
    if parsed.scheme in ("", "file"):
        return stores.setdefault("/", LocalStore("/")), parsed.path.lstrip("/")
    root = f"{parsed.scheme}://{parsed.netloc}"
    if root not in stores:
        stores[root] = from_url(root, **options)
    return stores[root], parsed.path.lstrip("/")


async def _cog_refs(
    url: str, stores: dict[str, Any], options: dict[str, Any]
) -> tuple[list[dict[str, Any]], bytes]:
    import async_tiff

    store, path = _store_path(url, stores, options)
    tiff = await async_tiff.TIFF.open(path, store=store)
    ifd = tiff.ifds[0]
    layout = cog_layout(ifd)
    offsets, lengths = ifd.tile_offsets, ifd.tile_byte_counts
    if offsets is None or lengths is None:
        raise ValueError("Expected a tiled TIFF")
    if len(offsets) != layout.blocks_down * layout.blocks_across:
        raise ValueError("Unexpected number of tile offsets")
    payload = {**asdict(layout), "dtype": layout.dtype.str}
    refs = []
    for idx, (offset, length) in enumerate(zip(offsets, lengths, strict=True)):
        row, col = divmod(idx, layout.blocks_across)
        if not length:
            continue
        refs.append({
            "url": url, "bucket": _bucket(url), "kind": "cog",
            "slice": block_slice(layout, row, col),
            "row_group": -1, "tile_row": row, "tile_col": col,
            "spans": [{"column": "", "offset": int(offset), "length": int(length)}],
        })
    return refs, json.dumps(payload).encode()


def _parquet_refs(
    url: str, filesystem: Any
) -> tuple[list[dict[str, Any]], bytes, int]:
    import pyarrow.fs as pafs

    if filesystem is None:
        fs, path = pafs.FileSystem.from_uri(url)
    else:
        fs, path = filesystem, url
    metadata = pq.read_metadata(path, filesystem=fs)
    size = fs.get_file_info(path).size
    sink = pa.BufferOutputStream()
    metadata.write_metadata_file(sink)
    refs = []
    for group in range(metadata.num_row_groups):
        rg = metadata.row_group(group)
        spans = []
        for n in range(rg.num_columns):
            col = rg.column(n)
            start = (
                col.dictionary_page_offset if col.has_dictionary_page
                else col.data_page_offset
            )
            spans.append({
                "column": col.path_in_schema,
                "offset": int(start), "length": int(col.total_compressed_size),
            })
        refs.append({
            "url": url, "bucket": _bucket(url), "kind": "parquet",
            "slice": row_group_slice(group), "row_group": group,
            "tile_row": -1, "tile_col": -1, "spans": spans,
        })
    return refs, sink.getvalue().to_pybytes(), int(size)


def build_byte_index(
    urls: list[str],
    root: str,
    *,
    kind: str,
    metadata: dict[str, list[Any]] | None = None,
    partition_by: tuple[str, ...] = ("bucket",),
    filesystem: Any = None,
    store_kwargs: dict[str, Any] | None = None,
    files_per_batch: int = 32,
) -> int:
    """Index only metadata; source payload bytes are never downloaded or decoded."""
    if kind not in ("parquet", "cog"):
        raise ValueError(f"Unsupported source kind: {kind}")
    if files_per_batch < 1:
        raise ValueError("files_per_batch must be positive")
    if any(len(values) != len(urls) for values in (metadata or {}).values()):
        raise ValueError("Metadata values must match URLs")
    if (set(metadata or {}) & {
        "url", "bucket", "kind", "slice", "row_group", "tile_row", "tile_col",
        "spans", "decoder", "file_size"
    }):
        raise ValueError("Metadata overrides a physical index field")
    catalog = Catalog(root, partition_by=partition_by, filesystem=filesystem)
    catalog.fs.create_dir(catalog._path("decoders"), recursive=True)
    stores: dict[str, Any] = {}
    for start in range(0, len(urls), files_per_batch):
        batch = []
        for i in range(start, min(start + files_per_batch, len(urls))):
            url = urls[i]
            if kind == "parquet":
                refs, decoder, size = _parquet_refs(url, None)
            else:
                refs, decoder = asyncio.run(_cog_refs(url, stores, store_kwargs or {}))
                store, path = _store_path(url, stores, store_kwargs or {})
                size = int(store.head(path)["size"])
            decoder_path = f"decoders/{uuid4().hex}.bin"
            with catalog.fs.open_output_stream(catalog._path(decoder_path)) as out:
                out.write(decoder)
            for ref in refs:
                ref.update({
                    "decoder": decoder_path,
                    "file_size": size,
                    **{name: values[i] for name, values in (metadata or {}).items()},
                })
            batch.extend(refs)
        if batch:
            catalog.append(pa.Table.from_pylist(batch))
    return len(urls)


def _merge_spans(spans: list[dict[str, Any]], columns: list[str]) -> list[tuple[int, int]]:
    selected = sorted(
        (int(s["offset"]), int(s["offset"]) + int(s["length"]))
        for s in spans
        if s["column"].split(".")[0] in columns
    )
    merged: list[tuple[int, int]] = []
    for start, end in selected:
        if merged and start <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(end, merged[-1][1])
        else:
            merged.append((start, end))
    return merged


class ByteIndex:
    """Loads only one URL bucket at a time, and caches decoder metadata per source."""

    def __init__(
        self, root: str, *, partition_by: tuple[str, ...] = ("bucket",),
        filesystem: Any = None
    ) -> None:
        catalog = Catalog(root, partition_by=partition_by, filesystem=filesystem)
        self.fs = catalog.fs
        self.base = catalog.base
        self.dataset = catalog.dataset()
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self._decoder: dict[str, bytes] = {}

    def rows(self, url: str) -> list[dict[str, Any]]:
        if url not in self._rows:
            predicate = (pc.field("bucket") == _bucket(url)) & (pc.field("url") == url)
            self._rows[url] = self.dataset.to_table(filter=predicate).to_pylist()
            if not self._rows[url]:
                raise KeyError(f"Source not indexed: {url}")
        return self._rows[url]

    def decoder(self, url: str) -> bytes:
        if url not in self._decoder:
            path = self.rows(url)[0]["decoder"]
            with self.fs.open_input_file(f"{self.base}/{path}") as source:
                self._decoder[url] = source.read()
        return self._decoder[url]

    def parquet(
        self, url: str, group: int, columns: list[str]
    ) -> tuple[Any, list[tuple[int, int]]]:
        from rayzin.readers.parquet_reader import FileInfo

        row = next(ref for ref in self.rows(url) if ref["row_group"] == group)
        footer = pq.read_metadata(pa.BufferReader(self.decoder(url)))
        field = footer.schema.to_arrow_schema().field(columns[0]).type
        info = FileInfo(
            row["file_size"], footer,
            np.dtype(field.value_type.to_pandas_dtype()), field.list_size,
        )
        return info, _merge_spans(row["spans"], columns)

    def cog_layout(self, url: str) -> CogLayout:
        raw = json.loads(self.decoder(url))
        raw["dtype"] = np.dtype(raw["dtype"])
        raw["transform"] = tuple(raw["transform"])
        return CogLayout(**raw)

    def cog(self, url: str, row: int, col: int) -> tuple[CogLayout, int, int]:
        ref = next(
            item for item in self.rows(url)
            if item["tile_row"] == row and item["tile_col"] == col
        )
        span = ref["spans"][0]
        return self.cog_layout(url), int(span["offset"]), int(span["length"])
