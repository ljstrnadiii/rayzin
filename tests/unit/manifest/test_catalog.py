from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pytest

from rayzin.manifest.catalog import Catalog, CommitConflict, LocalStore


def test_local_snapshots_partition_filter_and_compaction(tmp_path: Path) -> None:
    root = str(tmp_path / "catalog")
    catalog = Catalog(root, partition_by=("year",))
    first = catalog.append(pa.table({"year": [2024, 2025], "url": ["a", "b"]}))
    second = catalog.append(pa.table({"year": [2025, 2026], "url": ["c", "d"]}))
    assert first and second
    assert len(catalog.files()) >= 2
    filtered = catalog.scan(filter=pc.field("year") == 2025, columns=["url"])
    assert sorted(filtered.column("url").to_pylist()) == ["b", "c"]

    old_paths = set(catalog.files())
    compacted = catalog.compact(rows_per_file=100)
    assert compacted and not old_paths.intersection(catalog.files())
    assert sorted(catalog.scan(columns=["url"]).column("url").to_pylist()) == ["a", "b", "c", "d"]
    assert all((tmp_path / "catalog" / p).exists() for p in first + second)


def test_compare_and_swap_prevents_lost_updates(tmp_path: Path) -> None:
    store = LocalStore(str(tmp_path))
    a = Catalog(str(tmp_path), store=store)
    b = Catalog(str(tmp_path), store=store)
    a.commit(["segments/a.parquet"])
    stale, token = b.snapshot()
    a.commit(["segments/b.parquet"])
    assert stale["files"] == ["segments/a.parquet"]
    with pytest.raises(CommitConflict):
        store.write("HEAD.json", b"{}", token)
    b.commit(["segments/c.parquet"])
    assert a.snapshot()[0]["files"] == [
        "segments/a.parquet",
        "segments/b.parquet",
        "segments/c.parquet",
    ]
    with pytest.raises(CommitConflict, match="compaction"):
        b.commit(["segments/replaced.parquet"], remove=["segments/gone.parquet"])


def test_empty_and_bad_path_guards(tmp_path: Path) -> None:
    catalog = Catalog(str(tmp_path))
    assert catalog.files() == []
    assert catalog.append(pa.table({"value": pa.array([], type=pa.int64())})) == []
    with pytest.raises(ValueError):
        catalog.commit(["../escape.parquet"])
