from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from rayzin.manifest.byte_index import ByteIndex, build_byte_index
from rayzin.readers.cog_reader import CogVectorReader
from rayzin.readers.parquet_reader import ParquetVectorReader, row_group_slice


def test_indexed_parquet_never_opens_source_footer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = str(tmp_path / "source.parquet")
    root = str(tmp_path / "index")
    values = np.arange(48, dtype=np.float32).reshape(12, 4)
    table = pa.table({
        "embedding": pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), 4),
        "score": pa.array(np.arange(12), type=pa.int32()),
    })
    pq.write_table(table, url, row_group_size=4, compression="zstd")
    assert build_byte_index([url], root, kind="parquet", metadata={"year": [2026]}) == 1

    index = ByteIndex(root)
    assert len(index.rows(url)) == 3
    assert index.dataset.to_table(
        filter=(pc.field("bucket") == index.rows(url)[0]["bucket"]) &
        (pc.field("year") == 2026)
    ).num_rows == 3

    import obstore

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Source footer was opened")

    monkeypatch.setattr(obstore, "open_reader", forbidden)
    reader = ParquetVectorReader(normalize=False, byte_index_path=root)
    for group in range(3):
        found, valid = reader.read_row_group(url, group)
        np.testing.assert_array_equal(found, values[group * 4: (group + 1) * 4])
        assert valid.all()
    chunk = {"url": url, "slice": row_group_slice(1)}
    encoded = reader.run(reader.fetch_encoded(chunk))  # type: ignore[arg-type]
    assert encoded.shape == (4, 1, 4)


def test_cog_tile_direct_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine

    url = str(tmp_path / "source.tif")
    root = str(tmp_path / "index")
    values = np.arange(8 * 32 * 32, dtype=np.float32).reshape(8, 32, 32)
    with rasterio.open(
        url, "w", driver="GTiff", width=32, height=32, count=8,
        dtype="float32", tiled=True, blockxsize=16, blockysize=16,
        interleave="pixel", compress="deflate", crs="EPSG:4326",
        transform=Affine(0.001, 0, 10, 0, -0.001, 20)
    ) as dest:
        dest.write(values)
    build_byte_index([url], root, kind="cog")
    index = ByteIndex(root)
    assert len(index.rows(url)) == 4
    assert index.rows(url)[0]["spans"][0]["length"] > 0

    import async_tiff

    async def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Source TIFF header was opened")

    monkeypatch.setattr(async_tiff.TIFF, "open", forbidden)
    indexed = CogVectorReader(normalize=False, byte_index_path=root)
    chunk = {
        "url": url, "slice": [
            {"dim": "y", "start": 16, "stop": 32},
            {"dim": "x", "start": 0, "stop": 16},
        ]
    }
    out, shape = indexed.read(chunk)  # type: ignore[arg-type]
    assert shape == (16, 16)
    np.testing.assert_array_equal(
        out, values[:, 16:32, :16].transpose(1, 2, 0).reshape(-1, 8)
    )
    raw = indexed.run(indexed.fetch_encoded(chunk))  # type: ignore[arg-type]
    assert raw.shape == (16, 16, 8)
