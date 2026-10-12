"""Public entrypoint for Rayzin's chunk index and snapshot catalog."""

from rayzin.manifest.byte_index import ByteIndex, build_byte_index
from rayzin.manifest.catalog import Catalog, CommitConflict
from rayzin.manifest.schema import MANIFEST_SCHEMA as CHUNK_INDEX_SCHEMA
from rayzin.pipeline import index_cogs, index_parquet

__all__ = [
    "ByteIndex",
    "Catalog",
    "CHUNK_INDEX_SCHEMA",
    "CommitConflict",
    "build_byte_index",
    "index_cogs",
    "index_parquet",
]


if __name__ == "__main__":
    from rayzin.manifest.catalog import main

    main()
