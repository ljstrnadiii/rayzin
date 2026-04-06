from typing import Any

from rayzin.enums import MetricType, ReaderType, SearchBackendType
from rayzin.manifest import MANIFEST_SCHEMA, filter_manifest
from rayzin.metrics import COSINE, EUCLIDEAN, CosineMetric, EuclideanMetric, Metric
from rayzin.pipeline import (
    build_manifest,
    build_manifest_from_zarr,
    knn_zarr_search,
)
from rayzin.readers import VectorReader, ZarrVectorReader
from rayzin.search import BlockSearcher, SearchBackend, make_search_backend

__all__ = [
    # pipeline — search
    "knn_zarr_search",
    # pipeline — manifest build
    "build_manifest",
    "build_manifest_from_zarr",
    # enums
    "MetricType",
    "ReaderType",
    "SearchBackendType",
    # manifest
    "MANIFEST_SCHEMA",
    "filter_manifest",
    # metrics
    "Metric",
    "EuclideanMetric",
    "CosineMetric",
    "EUCLIDEAN",
    "COSINE",
    # readers
    "VectorReader",
    "ZarrVectorReader",
    # search
    "SearchBackend",
    "make_search_backend",
    "NumpySearchBackend",
    "FaissSearchBackend",
    "BlockSearcher",
]


def __getattr__(name: str) -> Any:
    if name in {"NumpySearchBackend", "FaissSearchBackend"}:
        from rayzin.search import __getattr__ as search_getattr

        return search_getattr(name)
    raise AttributeError(name)
