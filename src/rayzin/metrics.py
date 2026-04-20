from typing import Protocol, runtime_checkable

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from rayzin.enums import MetricType
from rayzin.schema import LOWER_BOUND_SCHEMA, LowerBoundTable, ManifestTable
from rayzin.types import (
    COL_CENTROID,
    COL_COUNT,
    COL_RADIUS,
    COL_SLICE,
    COL_URL,
    Float32Array,
)


@runtime_checkable
class Metric(Protocol):
    def distance(self, a: Float32Array, b: Float32Array) -> float: ...
    def pairwise(self, vectors: Float32Array, query: Float32Array) -> Float32Array: ...
    def lower_bound(self, query: Float32Array, centroid: Float32Array, radius: float) -> float: ...


class EuclideanMetric:
    """Squared L2 distance, matching FAISS IndexFlatL2 semantics."""

    def distance(self, a: Float32Array, b: Float32Array) -> float:
        delta = np.asarray(a - b, dtype=np.float32)
        return float(np.dot(delta, delta))

    def pairwise(self, vectors: Float32Array, query: Float32Array) -> Float32Array:
        query_array = np.asarray(query, dtype=np.float32)
        if query_array.ndim == 1:
            deltas = np.asarray(vectors - query_array[None, :], dtype=np.float32)
            return np.asarray(np.sum(deltas * deltas, axis=1, dtype=np.float32), dtype=np.float32)

        deltas = np.asarray(vectors[None, :, :] - query_array[:, None, :], dtype=np.float32)
        return np.asarray(np.sum(deltas * deltas, axis=2, dtype=np.float32), dtype=np.float32)

    def lower_bound(self, query: Float32Array, centroid: Float32Array, radius: float) -> float:
        centroid_distance = float(np.linalg.norm(query - centroid))
        return float(max(0.0, centroid_distance - radius) ** 2)


class CosineMetric:
    """Cosine distance implemented as 1 - inner_product(normalized(a), normalized(b))."""

    def distance(self, a: Float32Array, b: Float32Array) -> float:
        return float(1.0 - np.dot(_normalize(a), _normalize(b)))

    def pairwise(self, vectors: Float32Array, query: Float32Array) -> Float32Array:
        normalized_vectors = _normalize(vectors)
        normalized_query = _normalize(query)
        if normalized_query.ndim == 1:
            scores = normalized_vectors @ normalized_query
            return np.asarray(1.0 - scores, dtype=np.float32)

        scores = normalized_query @ normalized_vectors.T
        return np.asarray(1.0 - scores, dtype=np.float32)

    def lower_bound(self, query: Float32Array, centroid: Float32Array, radius: float) -> float:
        return max(0.0, self.distance(query, centroid) - radius)


EUCLIDEAN = EuclideanMetric()
COSINE = CosineMetric()


def make_metric(metric_type: MetricType) -> Metric:
    if metric_type == MetricType.EUCLIDEAN:
        return EuclideanMetric()
    if metric_type == MetricType.COSINE:
        return CosineMetric()
    raise ValueError(metric_type)


def add_lower_bounds_fn(
    batch: ManifestTable,
    queries: Float32Array,
    metric_type: str,
) -> LowerBoundTable:
    return _add_lower_bounds(batch, queries=queries, metric_type=metric_type)


def _add_lower_bounds(
    batch: ManifestTable,
    queries: Float32Array,
    metric_type: str,
) -> LowerBoundTable:
    query_matrix = np.asarray(queries, dtype=np.float32)
    if query_matrix.ndim != 2:
        msg = f"Expected queries to have shape (nq, d), got {query_matrix.shape!r}."
        raise ValueError(msg)

    centroids = np.asarray(batch.column(COL_CENTROID).to_pylist(), dtype=np.float32)
    radii = np.asarray(batch.column(COL_RADIUS).to_pylist(), dtype=np.float32)
    lower_bounds = _lower_bounds(
        query_matrix,
        centroids,
        radii,
        metric_type=MetricType(metric_type),
    )
    min_lower_bounds = np.asarray(np.min(lower_bounds, axis=1), dtype=np.float32)

    return pa.Table.from_arrays(
        [
            batch.column(COL_URL),
            batch.column(COL_SLICE),
            batch.column(COL_COUNT),
            batch.column(COL_CENTROID),
            batch.column(COL_RADIUS),
            pa.array(lower_bounds.tolist(), type=pa.list_(pa.float32())),
            pa.array(min_lower_bounds.tolist(), type=pa.float32()),
        ],
        schema=LOWER_BOUND_SCHEMA,
    )


def _lower_bounds(
    queries: Float32Array,
    centroids: Float32Array,
    radii: Float32Array,
    *,
    metric_type: MetricType,
) -> Float32Array:
    if metric_type == MetricType.EUCLIDEAN:
        deltas = np.asarray(
            centroids[:, None, :] - queries[None, :, :],
            dtype=np.float32,
        )
        centroid_distances = np.linalg.norm(deltas, axis=2)
        return np.asarray(
            np.square(np.maximum(0.0, centroid_distances - radii[:, None])),
            dtype=np.float32,
        )

    metric = make_metric(metric_type)
    distances = np.asarray(metric.pairwise(centroids, queries), dtype=np.float32).T
    if distances.ndim != 2:
        msg = f"Expected pairwise distances to have shape (n_rows, nq), got {distances.shape!r}."
        raise ValueError(msg)
    return np.asarray(np.maximum(0.0, distances - radii[:, None]), dtype=np.float32)


def _normalize(vectors: Float32Array) -> Float32Array:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        denominator = np.float32(np.linalg.norm(array))
        if denominator < np.finfo(np.float32).eps:
            denominator = np.float32(np.finfo(np.float32).eps)
        return np.asarray(array / denominator, dtype=np.float32)

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, np.finfo(np.float32).eps)
    return np.asarray(array / safe_norms, dtype=np.float32)
