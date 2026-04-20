import faiss  # type: ignore[import-untyped]
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from rayzin.metrics import CosineMetric, EuclideanMetric, add_lower_bounds_fn
from rayzin.schema import MANIFEST_SCHEMA
from rayzin.types import COL_LOWER_BOUNDS, COL_MIN_LOWER_BOUND


@pytest.fixture
def euclidean_metric() -> EuclideanMetric:
    return EuclideanMetric()


@pytest.fixture
def cosine_metric() -> CosineMetric:
    return CosineMetric()


def test_lower_bound_zero_when_query_inside_ball(euclidean_metric: EuclideanMetric) -> None:
    centroid = np.zeros(4, dtype=np.float32)
    query = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    radius = 1.0
    assert euclidean_metric.lower_bound(query, centroid, radius) == 0.0


def test_lower_bound_positive_when_query_outside_ball(euclidean_metric: EuclideanMetric) -> None:
    centroid = np.zeros(4, dtype=np.float32)
    query = np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32)
    radius = 1.0
    lower_bound = euclidean_metric.lower_bound(query, centroid, radius)
    assert lower_bound == pytest.approx(4.0, abs=1e-5)


def test_add_lower_bounds_matches_faiss_l2_pruning() -> None:
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    radii = np.array([0.1, 0.1], dtype=np.float32)
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    batch = pa.Table.from_pydict(
        {
            "url": ["a", "b"],
            "slice": [
                [{"dim": "x", "start": 0, "stop": 1}],
                [{"dim": "x", "start": 1, "stop": 2}],
            ],
            "count": [1, 1],
            "centroid": centroids.tolist(),
            "radius": radii.tolist(),
        },
        schema=MANIFEST_SCHEMA,
    )

    lower_bounds = np.asarray(
        add_lower_bounds_fn(batch, queries=queries, metric_type="euclidean")
        .column(COL_LOWER_BOUNDS)
        .to_pylist(),
        dtype=np.float32,
    )
    min_lower_bounds = np.asarray(
        add_lower_bounds_fn(batch, queries=queries, metric_type="euclidean")
        .column(COL_MIN_LOWER_BOUND)
        .to_pylist(),
        dtype=np.float32,
    )
    index = faiss.IndexFlatL2(2)
    index.add(centroids)
    distances, indices = index.search(queries, len(centroids))
    expected_distances = np.empty((len(queries), len(centroids)), dtype=np.float32)
    row_indices = np.arange(len(queries))[:, None]
    expected_distances[row_indices, indices] = distances
    expected_lower_bounds = np.square(
        np.maximum(0.0, np.sqrt(expected_distances.T) - radii[:, None]),
    )

    np.testing.assert_allclose(lower_bounds, expected_lower_bounds, atol=1e-6)
    np.testing.assert_allclose(min_lower_bounds, expected_lower_bounds.min(axis=1), atol=1e-6)


def test_euclidean_lower_bounds_matches_scalar_lower_bound(
    euclidean_metric: EuclideanMetric,
) -> None:
    centroids = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    radii = np.array([1.0, 0.5], dtype=np.float32)
    queries = np.array([[0.0, 0.0], [6.0, 8.0]], dtype=np.float32)

    lower_bounds = euclidean_metric.lower_bounds(queries, centroids, radii)
    expected = np.asarray(
        [
            [euclidean_metric.lower_bound(query, centroid, float(radius)) for query in queries]
            for centroid, radius in zip(centroids, radii)
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(lower_bounds, expected, atol=1e-6)


def test_cosine_lower_bounds_matches_scalar_lower_bound(
    cosine_metric: CosineMetric,
) -> None:
    centroids = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    radii = np.array([0.1, 0.2], dtype=np.float32)
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    lower_bounds = cosine_metric.lower_bounds(queries, centroids, radii)
    expected = np.asarray(
        [
            [cosine_metric.lower_bound(query, centroid, float(radius)) for query in queries]
            for centroid, radius in zip(centroids, radii)
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(lower_bounds, expected, atol=1e-6)


def test_euclidean_pairwise_matches_faiss_l2(euclidean_metric: EuclideanMetric) -> None:
    vectors = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    query = np.array([0.0, 0.0], dtype=np.float32)
    pairwise = euclidean_metric.pairwise(vectors, query)

    index = faiss.IndexFlatL2(2)
    index.add(vectors)
    distances, indices = index.search(query[None, :], len(vectors))
    expected = np.empty(len(vectors), dtype=np.float32)
    expected[indices[0]] = distances[0]

    np.testing.assert_allclose(pairwise, expected, atol=1e-6)


def test_cosine_pairwise_matches_faiss_inner_product(cosine_metric: CosineMetric) -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)
    pairwise = cosine_metric.pairwise(vectors, query)

    normalized_vectors = vectors.copy()
    normalized_query = query[None, :].copy()
    faiss.normalize_L2(normalized_vectors)
    faiss.normalize_L2(normalized_query)
    index = faiss.IndexFlatIP(2)
    index.add(normalized_vectors)
    scores, indices = index.search(normalized_query, len(vectors))
    expected = np.empty(len(vectors), dtype=np.float32)
    expected[indices[0]] = 1.0 - scores[0]

    np.testing.assert_allclose(pairwise, expected, atol=1e-6)
