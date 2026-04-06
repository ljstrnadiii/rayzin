from pathlib import Path
from typing import cast

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import zarr
from ray.data.expressions import col

from rayzin.enums import MetricType, SearchBackendType
from rayzin.manifest.schema import MANIFEST_SCHEMA
from rayzin.pipeline import build_manifest_from_zarr, knn_zarr_search
from rayzin.types import (
    COL_CHUNK_ID,
    COL_DISTANCE,
    COL_OFFSET,
    COL_SLICE,
    COL_URL,
)


def test_knn_zarr_returns_k_results(
    ray_session: None,
    zarr_manifest: tuple[str, str, np.ndarray],
) -> None:
    _, manifest_path, query = zarr_manifest
    results = knn_zarr_search(
        manifest_path,
        query,
        k=5,
        batch_size=2,
        actor_pool_size=2,
    )
    assert len(results.chunks) == 5
    assert len(results.offsets) == 5
    assert len(results.distances) == 5


def test_knn_zarr_results_sorted(
    ray_session: None,
    zarr_manifest: tuple[str, str, np.ndarray],
) -> None:
    _, manifest_path, query = zarr_manifest
    results = knn_zarr_search(manifest_path, query, k=5, batch_size=2, actor_pool_size=2)
    assert results.distances == sorted(results.distances)


def test_knn_zarr_supports_batch_size(
    ray_session: None,
    zarr_manifest: tuple[str, str, np.ndarray],
) -> None:
    _, manifest_path, query = zarr_manifest
    results = knn_zarr_search(
        manifest_path,
        query,
        k=5,
        batch_size=1,
        actor_pool_size=2,
    )
    assert len(results.distances) == 5
    assert results.distances == sorted(results.distances)


def test_knn_zarr_results_include_chunk_metadata(
    ray_session: None,
    zarr_manifest: tuple[str, str, np.ndarray],
) -> None:
    _, manifest_path, query = zarr_manifest
    results = knn_zarr_search(
        manifest_path,
        query,
        k=1,
        batch_size=2,
        actor_pool_size=2,
    )
    rows = results.to_rows()
    row = rows[0]
    assert COL_CHUNK_ID in row
    assert str(row[COL_URL]) == str(row[COL_CHUNK_ID]).split("#", maxsplit=1)[0]
    assert COL_SLICE in row
    assert COL_OFFSET in row
    assert COL_DISTANCE in row


def test_knn_zarr_rejects_unsupported_metric(zarr_manifest: tuple[str, str, np.ndarray]) -> None:
    _, manifest_path, query = zarr_manifest
    with pytest.raises(NotImplementedError, match="pruning"):
        knn_zarr_search(manifest_path, query, k=1, metric=MetricType.COSINE)


def test_knn_zarr_filter_expr_limits_manifest_rows(
    ray_session: None,
    zarr_manifest: tuple[str, str, np.ndarray],
) -> None:
    _, manifest_path, query = zarr_manifest
    manifest_rows = pq.read_table(manifest_path).to_pylist()
    radii = [float(row["radius"]) for row in manifest_rows]
    assert len(set(radii)) > 1
    threshold = (min(radii) + max(radii)) / 2.0

    results = knn_zarr_search(
        manifest_path,
        query,
        k=5,
        filter_expr=col("radius") > threshold,
        batch_size=2,
        actor_pool_size=2,
    )
    rows = results.to_rows()

    assert rows
    filtered_radii = [
        float(row["radius"]) for row in manifest_rows if float(row["radius"]) > threshold
    ]
    assert len(rows) <= len(filtered_radii)


def test_knn_zarr_faiss_backend_matches_numpy(
    ray_session: None,
    zarr_manifest: tuple[str, str, np.ndarray],
) -> None:
    _, manifest_path, query = zarr_manifest
    numpy_rows = knn_zarr_search(
        manifest_path,
        query,
        k=5,
        backend=SearchBackendType.NUMPY,
        batch_size=2,
        actor_pool_size=2,
    ).to_rows()
    faiss_rows = knn_zarr_search(
        manifest_path,
        query,
        k=5,
        backend=SearchBackendType.FAISS_CPU,
        batch_size=2,
        actor_pool_size=2,
    ).to_rows()

    assert [row["chunk_id"] for row in faiss_rows] == [row["chunk_id"] for row in numpy_rows]
    assert [row["offset"] for row in faiss_rows] == [row["offset"] for row in numpy_rows]
    faiss_distances = np.array(
        [cast(float, row[COL_DISTANCE]) for row in faiss_rows],
        dtype=float,
    )
    numpy_distances = np.array(
        [cast(float, row[COL_DISTANCE]) for row in numpy_rows],
        dtype=float,
    )
    np.testing.assert_allclose(
        faiss_distances,
        numpy_distances,
        atol=1e-5,
    )


def test_knn_zarr_supports_custom_embedding_dim_name(
    ray_session: None,
    tmp_path: Path,
) -> None:
    store_path = str(tmp_path / "custom_embedding_name.zarr")
    manifest_path = str(tmp_path / "custom_embedding_name.parquet")
    vectors = np.arange(1 * 3 * 4 * 4, dtype=np.float32).reshape(1, 3, 4, 4)

    root = zarr.open_group(store_path, mode="w")
    array = root.create_array(
        "embeddings",
        shape=vectors.shape,
        dtype=vectors.dtype,
        chunks=(1, 3, 2, 2),
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "feature", "y", "x"]
    array[:] = vectors

    build_manifest_from_zarr(
        store_path,
        manifest_path,
        n_blocks=2,
        embedding_dim_name="feature",
    )

    table = pq.read_table(manifest_path)
    assert table.schema.equals(MANIFEST_SCHEMA)

    results = knn_zarr_search(
        manifest_path,
        np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        k=3,
        batch_size=1,
        actor_pool_size=1,
        embedding_dim_name="feature",
    )
    rows = results.to_rows()

    distances = [cast(float, row[COL_DISTANCE]) for row in rows]
    assert len(rows) == 3
    assert distances == sorted(distances)
