import numpy as np
import pytest

from rayzin.enums import MetricType
from rayzin.search.backends import FaissSearchBackend, NumpySearchBackend


@pytest.fixture
def backend() -> NumpySearchBackend:
    return NumpySearchBackend()


def test_search_returns_k_results(backend: NumpySearchBackend) -> None:
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((100, 16)).astype(np.float32)
    query = rng.standard_normal(16).astype(np.float32)
    distances, indices = backend.search(vectors, query, k=5)
    assert len(distances) == 5
    assert len(indices) == 5


def test_search_sorted_ascending(backend: NumpySearchBackend) -> None:
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((50, 8)).astype(np.float32)
    query = rng.standard_normal(8).astype(np.float32)
    distances, _ = backend.search(vectors, query, k=10)
    assert list(distances) == sorted(distances)


def test_search_finds_exact_nearest(backend: NumpySearchBackend) -> None:
    vectors = np.eye(4, dtype=np.float32)
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, indices = backend.search(vectors, query, k=1)
    assert indices[0] == 0


def test_radius_search_returns_all_within_radius(backend: NumpySearchBackend) -> None:
    vectors = np.array([[0.0], [1.0], [2.0], [5.0]], dtype=np.float32)
    query = np.array([0.0], dtype=np.float32)
    distances, indices = backend.radius_search(vectors, query, radius=4.0)
    assert set(indices.tolist()) == {0, 1, 2}
    assert all(distance <= 4.0 for distance in distances)


def test_faiss_backend_matches_numpy_for_l2() -> None:
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((32, 8)).astype(np.float32)
    query = rng.standard_normal(8).astype(np.float32)

    numpy_distances, numpy_indices = NumpySearchBackend(metric_type=MetricType.EUCLIDEAN).search(
        vectors,
        query,
        k=5,
    )
    faiss_distances, faiss_indices = FaissSearchBackend(metric_type=MetricType.EUCLIDEAN).search(
        vectors,
        query,
        k=5,
    )

    np.testing.assert_allclose(faiss_distances, numpy_distances, atol=1e-5)
    np.testing.assert_array_equal(faiss_indices, numpy_indices)


def test_faiss_backend_matches_numpy_for_cosine() -> None:
    rng = np.random.default_rng(17)
    vectors = rng.standard_normal((32, 8)).astype(np.float32)
    query = rng.standard_normal(8).astype(np.float32)

    numpy_distances, numpy_indices = NumpySearchBackend(metric_type=MetricType.COSINE).search(
        vectors,
        query,
        k=5,
    )
    faiss_distances, faiss_indices = FaissSearchBackend(metric_type=MetricType.COSINE).search(
        vectors,
        query,
        k=5,
    )

    np.testing.assert_allclose(faiss_distances, numpy_distances, atol=1e-5)
    np.testing.assert_array_equal(faiss_indices, numpy_indices)


def test_faiss_radius_search_matches_numpy_for_l2() -> None:
    rng = np.random.default_rng(21)
    vectors = rng.standard_normal((48, 8)).astype(np.float32)
    query = rng.standard_normal(8).astype(np.float32)
    radius = np.float32(6.0)

    numpy_backend = NumpySearchBackend(metric_type=MetricType.EUCLIDEAN)
    faiss_backend = FaissSearchBackend(metric_type=MetricType.EUCLIDEAN)

    numpy_distances, numpy_indices = numpy_backend.radius_search(
        vectors,
        query,
        radius=float(radius),
    )
    faiss_distances, faiss_indices = faiss_backend.radius_search(
        vectors,
        query,
        radius=float(radius),
    )

    np.testing.assert_allclose(np.sort(faiss_distances), np.sort(numpy_distances), atol=1e-5)
    np.testing.assert_array_equal(np.sort(faiss_indices), np.sort(numpy_indices))
