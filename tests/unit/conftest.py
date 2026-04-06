from collections.abc import Generator
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import ray
import zarr

from rayzin.manifest.schema import MANIFEST_SCHEMA


@pytest.fixture(scope="session")
def ray_session() -> Generator[None, None, None]:
    ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def zarr_manifest(tmp_path: Path) -> tuple[str, str, np.ndarray]:
    rng = np.random.default_rng(42)
    embedding_dim, height, width = 8, 40, 40
    chunk_height, chunk_width = 10, 10

    store_path = str(tmp_path / "test.zarr")
    root = zarr.open_group(store_path, mode="w")
    vectors = rng.standard_normal((1, height, width, embedding_dim)).astype(np.float32)
    array = root.create_array(
        "embeddings",
        shape=vectors.shape,
        dtype=vectors.dtype,
        chunks=(1, chunk_height, chunk_width, embedding_dim),
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x", "embedding"]
    array[:] = vectors

    rows = []
    for y_start in range(0, height, chunk_height):
        for x_start in range(0, width, chunk_width):
            chunk = vectors[0, y_start : y_start + chunk_height, x_start : x_start + chunk_width, :]
            flat = chunk.reshape(-1, embedding_dim)
            centroid = flat.mean(axis=0)
            radius = float(np.linalg.norm(flat - centroid[None, :], axis=1).max())
            rows.append(
                {
                    "url": store_path,
                    "slice": [
                        {"dim": "time", "start": 0, "stop": 1},
                        {"dim": "y", "start": y_start, "stop": y_start + chunk_height},
                        {"dim": "x", "start": x_start, "stop": x_start + chunk_width},
                    ],
                    "count": chunk_height * chunk_width,
                    "centroid": centroid.tolist(),
                    "radius": radius,
                }
            )

    manifest_path = str(tmp_path / "manifest.parquet")
    table = pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA)
    pq.write_table(table, manifest_path)

    query = rng.standard_normal(embedding_dim).astype(np.float32)
    return store_path, manifest_path, query
