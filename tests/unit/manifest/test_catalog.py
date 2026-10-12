from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pytest

from rayzin.manifest.catalog import Catalog, CommitConflict, LocalStore


def test_one_snapshot_covers_all_indexes_and_compaction(tmp_path: Path) -> None:
    catalog = Catalog(str(tmp_path))
    layouts: dict[str, tuple[str, ...]] = {
        part: ("year",) for part in ("chunks", "bytes", "files")
    }

    def batch(year: int) -> dict[str, pa.Table]:
        return {
            "chunks": pa.table({"year": [year], "url": [f"{year}.parquet"], "count": [10]}),
            "bytes": pa.table({"year": [year], "url": [f"{year}.parquet"], "offset": [32]}),
            "files": pa.table({"year": [year], "url": [f"{year}.parquet"], "size": [1024]}),
        }

    catalog.append_many(batch(2024), partition_by=layouts)
    catalog.append_many(batch(2026), partition_by=layouts)
    old, _ = catalog.snapshot()
    assert set(old["components"]) == {"chunks", "bytes", "files"}
    assert all(len(value["files"]) == 2 for value in old["components"].values())
    assert len(catalog.files("bytes", partitions={"year": 2026})) == 1
    assert catalog.scan(
        "chunks", filter=pc.field("year") == 2026, columns=["url"]
    ).column("url").to_pylist() == ["2026.parquet"]

    original = {kind: catalog.files(kind) for kind in layouts}
    replacements = catalog.compact()
    assert set(replacements) == set(layouts)
    for kind in layouts:
        assert not set(original[kind]).intersection(catalog.files(kind))
        assert sorted(catalog.scan(kind, columns=["url"]).column("url").to_pylist()) == [
            "2024.parquet", "2026.parquet"
        ]
        assert catalog.dataset(kind, snapshot=old).count_rows() == 2
        assert all(Path(path).exists() for path in original[kind])


def test_cas_rebases_additions_across_components(tmp_path: Path) -> None:
    store = LocalStore(str(tmp_path))
    a = Catalog(str(tmp_path), store=store)
    b = Catalog(str(tmp_path), store=store)
    a.commit({"chunks": ["segments/chunks/a.parquet"]})
    prior, token = b.snapshot()
    a.commit({"bytes": ["segments/bytes/b.parquet"]})
    assert prior["components"]["chunks"]["files"] == ["segments/chunks/a.parquet"]
    with pytest.raises(CommitConflict):
        store.write("HEAD.json", b"{}", token)
    b.commit({"files": ["segments/files/c.parquet"]})
    assert set(a.snapshot()[0]["components"]) == {"chunks", "bytes", "files"}
    with pytest.raises(CommitConflict, match="compaction"):
        b.commit(
            {"chunks": ["segments/chunks/replace.parquet"]},
            remove={"chunks": ["segments/chunks/absent.parquet"]},
        )


def test_empty_and_bad_path_guards(tmp_path: Path) -> None:
    catalog = Catalog(str(tmp_path))
    assert catalog.files() == []
    assert catalog.append(pa.table({"value": pa.array([], type=pa.int64())})) == []
    with pytest.raises(ValueError):
        catalog.commit({"chunks": ["../escape.parquet"]})
