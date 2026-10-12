"""Immutable Parquet manifests with a conditional S3/GCS snapshot pointer."""

import hashlib
import json
import os
import posixpath
from typing import Any, Protocol
from urllib.parse import urlparse
from uuid import uuid4

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.dataset as ds  # type: ignore[import-untyped]
import pyarrow.fs as pafs  # type: ignore[import-untyped]


class CommitConflict(RuntimeError):
    pass


class ObjectStore(Protocol):
    def read(self, key: str) -> tuple[bytes | None, str | int | None]: ...

    def write(self, key: str, value: bytes, expected: str | int | None) -> None: ...


class S3Store:
    def __init__(self, bucket: str, client: Any = None) -> None:
        if client is None:
            import boto3  # type: ignore[import-not-found]

            client = boto3.client("s3")
        self.bucket = bucket
        self.client = client

    def read(self, key: str) -> tuple[bytes | None, str | None]:
        from botocore.exceptions import ClientError  # type: ignore[import-not-found]

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "404", "NotFound"):
                return None, None
            raise
        return response["Body"].read(), response["ETag"]

    def write(self, key: str, value: bytes, expected: str | int | None) -> None:
        from botocore.exceptions import ClientError

        condition = {"IfNoneMatch": "*"} if expected is None else {"IfMatch": str(expected)}
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=value, **condition)
        except ClientError as exc:
            if exc.response["Error"]["Code"] in (
                "PreconditionFailed",
                "ConditionalRequestConflict",
                "412",
                "409",
            ):
                raise CommitConflict(key) from exc
            raise


class GCSStore:
    def __init__(self, bucket: str, client: Any = None) -> None:
        from google.cloud import storage  # type: ignore[import-untyped]

        self.bucket = (client or storage.Client()).bucket(bucket)

    def read(self, key: str) -> tuple[bytes | None, int | None]:
        from google.api_core.exceptions import (  # type: ignore[import-untyped]
            NotFound,
            PreconditionFailed,
        )

        for _ in range(3):
            blob = self.bucket.blob(key)
            try:
                blob.reload()
                generation = int(blob.generation)
                return blob.download_as_bytes(if_generation_match=generation), generation
            except NotFound:
                return None, None
            except PreconditionFailed:
                continue
        raise CommitConflict(f"Concurrent reads of {key} did not stabilize")

    def write(self, key: str, value: bytes, expected: str | int | None) -> None:
        from google.api_core.exceptions import Conflict, PreconditionFailed

        try:
            self.bucket.blob(key).upload_from_string(
                value,
                content_type="application/json",
                if_generation_match=0 if expected is None else int(expected),
            )
        except (PreconditionFailed, Conflict) as exc:
            raise CommitConflict(key) from exc


class LocalStore:
    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def read(self, key: str) -> tuple[bytes | None, str | None]:
        try:
            with open(os.path.join(self.root, key), "rb") as source:
                data = source.read()
        except FileNotFoundError:
            return None, None
        return data, hashlib.sha256(data).hexdigest()

    def write(self, key: str, value: bytes, expected: str | int | None) -> None:
        import fcntl

        path = os.path.join(self.root, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lock_path = os.path.join(self.root, ".catalog.lock")
        with open(lock_path, "a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _, current = self.read(key)
            if current != expected:
                raise CommitConflict(key)
            temporary = f"{path}.{uuid4().hex}.tmp"
            try:
                with open(temporary, "xb") as target:
                    target.write(value)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)


COMPONENTS = ("chunks", "bytes", "files")


class Catalog:
    """One atomic pointer for the chunk index, byte index, and file registry."""

    def __init__(
        self,
        root: str,
        *,
        store: ObjectStore | None = None,
        filesystem: Any = None,
    ) -> None:
        self.root = root.rstrip("/")
        self.fs: Any
        self.base: str
        if filesystem is None:
            self.fs, self.base = pafs.FileSystem.from_uri(self.root)
        else:
            self.fs = filesystem
            self.base = pafs.FileSystem.from_uri(self.root)[1] if "://" in self.root else self.root
        parsed = urlparse(self.root)
        self.prefix = parsed.path.strip("/") if parsed.scheme in ("s3", "gs", "gcs") else ""
        if store is None:
            if parsed.scheme == "s3":
                store = S3Store(parsed.netloc)
            elif parsed.scheme in ("gs", "gcs"):
                store = GCSStore(parsed.netloc)
            else:
                store = LocalStore(self.base)
        self.store = store

    def _key(self, relative: str) -> str:
        return posixpath.join(self.prefix, relative)

    def _path(self, relative: str) -> str:
        return posixpath.join(self.base, relative)

    def snapshot(self) -> tuple[dict[str, Any], str | int | None]:
        head, version = self.store.read(self._key("HEAD.json"))
        if head is None:
            return {"version": 2, "components": {}}, None
        target = json.loads(head)["snapshot"]
        body, _ = self.store.read(self._key(target))
        if body is None:
            raise FileNotFoundError(target)
        result: dict[str, Any] = json.loads(body)
        if result.get("version") != 2:
            raise ValueError("Unsupported catalog snapshot version")
        return result, version

    def files(
        self,
        component: str = "chunks",
        *,
        partitions: dict[str, int | str] | None = None,
        snapshot: dict[str, Any] | None = None,
    ) -> list[str]:
        if component not in COMPONENTS:
            raise ValueError(f"Unknown catalog component: {component}")
        current = snapshot if snapshot is not None else self.snapshot()[0]
        state = current["components"].get(component, {"files": [], "partition_by": []})
        names = state["files"]
        if partitions:
            if not set(partitions).issubset(state["partition_by"]):
                raise ValueError("Cannot prune on an unpartitioned field")
            names = [
                name
                for name in names
                if all(f"{key}={value}" in name.split("/") for key, value in partitions.items())
            ]
        return [self._path(name) for name in names]

    def dataset(
        self,
        component: str = "chunks",
        *,
        partitions: dict[str, int | str] | None = None,
        snapshot: dict[str, Any] | None = None,
    ) -> ds.Dataset:
        files = self.files(component, partitions=partitions, snapshot=snapshot)
        if not files:
            raise ValueError(f"No {component} fragments match the selected partitions")
        return ds.dataset(
            files,
            filesystem=self.fs,
            format="parquet",
            partitioning="hive",
            partition_base_dir=self.base,
        )

    def scan(
        self,
        component: str = "chunks",
        *,
        filter: Any = None,
        columns: list[str] | None = None,
        partitions: dict[str, int | str] | None = None,
        snapshot: dict[str, Any] | None = None,
    ) -> pa.Table:
        return self.dataset(component, partitions=partitions, snapshot=snapshot).to_table(
            filter=filter, columns=columns
        )

    def commit(
        self,
        add: dict[str, list[str]],
        *,
        remove: dict[str, list[str]] | None = None,
        partition_by: dict[str, tuple[str, ...]] | None = None,
        retries: int = 8,
    ) -> None:
        """Prepare outside the CAS window; rebase additions on every retry."""
        remove = remove or {}
        partition_by = partition_by or {}
        if not (set(add) | set(remove) | set(partition_by)).issubset(COMPONENTS):
            raise ValueError("Unknown catalog component")
        paths = [name for group in (*add.values(), *remove.values()) for name in group]
        if any(name.startswith("/") or ".." in name.split("/") for name in paths):
            raise ValueError("Expected paths relative to the catalog root")
        for _ in range(retries):
            previous, token = self.snapshot()
            states = {
                key: {"files": list(value["files"]), "partition_by": list(value["partition_by"])}
                for key, value in previous["components"].items()
            }
            for component in set(add) | set(remove):
                state = states.setdefault(component, {"files": [], "partition_by": []})
                expected = list(partition_by.get(component, state["partition_by"]))
                if state["files"] and state["partition_by"] != expected:
                    raise ValueError(f"Partition layout changed for {component}")
                state["partition_by"] = expected
                dropped = set(remove.get(component, []))
                if not dropped.issubset(state["files"]):
                    raise CommitConflict(
                        f"Input fragments were replaced by another compaction: {component}"
                    )
                state["files"] = list(
                    dict.fromkeys(
                        [name for name in state["files"] if name not in dropped]
                        + add.get(component, [])
                    )
                )
            target = f"snapshots/{uuid4().hex}.json"
            self.store.write(
                self._key(target),
                json.dumps({"version": 2, "components": states}).encode(),
                None,
            )
            try:
                self.store.write(
                    self._key("HEAD.json"), json.dumps({"snapshot": target}).encode(), token
                )
                return
            except CommitConflict:
                continue
        raise CommitConflict("Too many concurrent catalog commits")

    def stage(
        self,
        table: pa.Table | ds.Dataset,
        component: str,
        *,
        partition_by: tuple[str, ...] = (),
        rows_per_file: int = 131072,
    ) -> list[str]:
        if component not in COMPONENTS:
            raise ValueError(f"Unknown catalog component: {component}")
        if isinstance(table, pa.Table) and table.num_rows == 0:
            return []
        target = f"segments/{component}/{uuid4().hex}"
        ds.write_dataset(
            table,
            self._path(target),
            filesystem=self.fs,
            format="parquet",
            partitioning=list(partition_by) or None,
            partitioning_flavor="hive" if partition_by else None,
            basename_template="part-{i}.parquet",
            max_rows_per_file=rows_per_file,
            max_rows_per_group=rows_per_file,
            max_open_files=64,
        )
        return self._fragments(target)

    def append_many(
        self,
        tables: dict[str, pa.Table],
        *,
        partition_by: dict[str, tuple[str, ...]] | None = None,
        rows_per_file: int = 131072,
    ) -> dict[str, list[str]]:
        partition_by = partition_by or {}
        previous, _ = self.snapshot()
        paths: dict[str, list[str]] = {}
        layouts: dict[str, tuple[str, ...]] = {}
        for component, table in tables.items():
            existing = previous["components"].get(component, {})
            layout = partition_by.get(component, tuple(existing.get("partition_by", ())))
            layouts[component] = layout
            paths[component] = self.stage(
                table, component, partition_by=layout, rows_per_file=rows_per_file
            )
        if any(paths.values()):
            self.commit(paths, partition_by=layouts)
        return paths

    def append(
        self,
        table: pa.Table,
        *,
        component: str = "chunks",
        partition_by: tuple[str, ...] | None = None,
        rows_per_file: int = 131072,
    ) -> list[str]:
        layouts = {} if partition_by is None else {component: partition_by}
        return self.append_many(
            {component: table}, partition_by=layouts, rows_per_file=rows_per_file
        )[component]

    def compact(
        self,
        *,
        components: tuple[str, ...] = COMPONENTS,
        rows_per_file: int = 131072,
    ) -> dict[str, list[str]]:
        """Rewrite only metadata segments; commit all replacements atomically."""
        previous, _ = self.snapshot()
        replacements: dict[str, list[str]] = {}
        removed: dict[str, list[str]] = {}
        for component in components:
            state = previous["components"].get(component, {})
            originals: list[str] = state.get("files", [])
            if len(originals) < 2:
                continue
            dataset = self.dataset(component, snapshot=previous)
            replacements[component] = self.stage(
                dataset,
                component,
                partition_by=tuple(state["partition_by"]),
                rows_per_file=rows_per_file,
            )
            removed[component] = originals
        if replacements:
            self.commit(replacements, remove=removed)
        return replacements

    def _fragments(self, target: str) -> list[str]:
        info = self.fs.get_file_info(pafs.FileSelector(self._path(target), recursive=True))
        return sorted(
            posixpath.relpath(item.path, self.base)
            for item in info
            if item.type == pafs.FileType.File and item.path.endswith(".parquet")
        )


def catalog_at(path: str, filesystem: Any = None) -> Catalog | None:
    """Detect a catalog without requiring an optional S3/GCS commit client."""
    fs, base = (
        pafs.FileSystem.from_uri(path)
        if filesystem is None
        else (filesystem, pafs.FileSystem.from_uri(path)[1] if "://" in path else path)
    )
    if fs.get_file_info(posixpath.join(base, "HEAD.json")).type != pafs.FileType.File:
        return None
    return Catalog(path, filesystem=fs)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compact immutable Rayzin index fragments")
    parser.add_argument("root", help="Catalog root, local, s3:// or gs://")
    parser.add_argument("--components", nargs="*", default=list(COMPONENTS))
    parser.add_argument("--rows-per-file", type=int, default=131072)
    args = parser.parse_args()
    Catalog(args.root).compact(components=tuple(args.components), rows_per_file=args.rows_per_file)


if __name__ == "__main__":
    main()
