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
                "PreconditionFailed", "ConditionalRequestConflict", "412", "409"
            ):
                raise CommitConflict(key) from exc
            raise


class GCSStore:
    def __init__(self, bucket: str, client: Any = None) -> None:
        from google.cloud import storage  # type: ignore[import-untyped]

        self.bucket = (client or storage.Client()).bucket(bucket)

    def read(self, key: str) -> tuple[bytes | None, int | None]:
        from google.api_core.exceptions import NotFound, PreconditionFailed  # type: ignore[import-untyped]

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
        from google.api_core.exceptions import Conflict, PreconditionFailed  # type: ignore[import-untyped]

        try:
            self.bucket.blob(key).upload_from_string(
                value, content_type="application/json",
                if_generation_match=0 if expected is None else int(expected)
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


class Catalog:
    """No service: immutable fragments and snapshots, one conditional HEAD update."""

    def __init__(
        self,
        root: str,
        *,
        partition_by: tuple[str, ...] = (),
        store: ObjectStore | None = None,
        filesystem: Any = None,
    ) -> None:
        self.root = root.rstrip("/")
        self.partition_by = tuple(partition_by)
        self.fs: Any
        self.base: str
        self.fs, self.base = (
            pafs.FileSystem.from_uri(self.root) if filesystem is None
            else (filesystem, self.root)
        )
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
            return {"files": [], "partition_by": list(self.partition_by)}, None
        target = json.loads(head)["snapshot"]
        body, _ = self.store.read(self._key(target))
        if body is None:
            raise FileNotFoundError(target)
        result: dict[str, Any] = json.loads(body)
        if tuple(result["partition_by"]) != self.partition_by:
            raise ValueError("Catalog partition_by differs from the committed snapshot")
        return result, version

    def files(self) -> list[str]:
        snapshot, _ = self.snapshot()
        return [self._path(name) for name in snapshot["files"]]

    def dataset(self) -> ds.Dataset:
        files = self.files()
        if not files:
            raise ValueError("Catalog has no fragments")
        return ds.dataset(
            files,
            filesystem=self.fs,
            format="parquet",
            partitioning="hive",
            partition_base_dir=self.base,
        )

    def scan(self, filter: Any = None, columns: list[str] | None = None) -> pa.Table:
        return self.dataset().to_table(filter=filter, columns=columns)

    def commit(
        self, add: list[str], *, remove: list[str] | None = None, retries: int = 8
    ) -> None:
        removed = set(remove or [])
        if any(path.startswith("/") or ".." in path.split("/") for path in [*add, *removed]):
            raise ValueError("Expected paths relative to the catalog root")
        for _ in range(retries):
            previous, token = self.snapshot()
            if not removed.issubset(previous["files"]):
                raise CommitConflict("Input fragments were replaced by another compaction")
            files = list(dict.fromkeys(
                [file for file in previous["files"] if file not in removed] + add
            ))
            target = f"snapshots/{uuid4().hex}.json"
            body = json.dumps(
                {"version": 1, "files": files, "partition_by": list(self.partition_by)}
            ).encode()
            self.store.write(self._key(target), body, None)
            try:
                self.store.write(
                    self._key("HEAD.json"), json.dumps({"snapshot": target}).encode(), token
                )
                return
            except CommitConflict:
                continue
        raise CommitConflict("Too many concurrent catalog commits")

    def append(self, table: pa.Table, *, rows_per_file: int = 131072) -> list[str]:
        if table.num_rows == 0:
            return []
        target = f"segments/{uuid4().hex}"
        ds.write_dataset(
            table,
            self._path(target),
            filesystem=self.fs,
            format="parquet",
            partitioning=list(self.partition_by) or None,
            partitioning_flavor="hive" if self.partition_by else None,
            basename_template="part-{i}.parquet",
            max_rows_per_file=rows_per_file,
            max_rows_per_group=rows_per_file,
            existing_data_behavior="error",
        )
        fragments = self._fragments(target)
        self.commit(fragments)
        return fragments

    def compact(self, *, rows_per_file: int = 131072) -> list[str]:
        original, _ = self.snapshot()
        if len(original["files"]) < 2:
            return []
        target = f"segments/{uuid4().hex}"
        ds.write_dataset(
            self.dataset(),
            self._path(target),
            filesystem=self.fs,
            format="parquet",
            partitioning=list(self.partition_by) or None,
            partitioning_flavor="hive" if self.partition_by else None,
            basename_template="part-{i}.parquet",
            max_rows_per_file=rows_per_file,
            max_rows_per_group=rows_per_file,
            max_open_files=16,
        )
        fragments = self._fragments(target)
        self.commit(fragments, remove=original["files"])
        return fragments

    def _fragments(self, target: str) -> list[str]:
        info = self.fs.get_file_info(pafs.FileSelector(self._path(target), recursive=True))
        return sorted(
            posixpath.relpath(item.path, self.base)
            for item in info
            if item.type == pafs.FileType.File and item.path.endswith(".parquet")
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compact immutable Rayzin manifest fragments")
    parser.add_argument("root", help="Manifest root, local, s3:// or gs://")
    parser.add_argument("--partition-by", nargs="*", default=["bucket"])
    parser.add_argument("--rows-per-file", type=int, default=131072)
    args = parser.parse_args()
    catalog = Catalog(args.root, partition_by=tuple(args.partition_by))
    catalog.compact(rows_per_file=args.rows_per_file)


if __name__ == "__main__":
    main()
