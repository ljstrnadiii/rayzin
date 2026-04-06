from typing import Any

from rayzin.enums import MetricType, SearchBackendType
from rayzin.search.backends.protocols import SearchBackend, SearchResultHeap


def make_search_backend(
    backend_type: SearchBackendType,
    metric_type: MetricType,
) -> SearchBackend:
    if backend_type == SearchBackendType.NUMPY:
        from rayzin.search.backends.numpy import NumpySearchBackend

        return NumpySearchBackend(metric_type=metric_type)
    if backend_type in (SearchBackendType.FAISS_CPU, SearchBackendType.FAISS_GPU):
        from rayzin.search.backends.faiss import FaissSearchBackend

        return FaissSearchBackend(
            metric_type=metric_type,
            use_gpu=(backend_type == SearchBackendType.FAISS_GPU),
        )
    raise ValueError(backend_type)


def __getattr__(name: str) -> Any:
    if name == "NumpySearchBackend":
        from rayzin.search.backends.numpy import NumpySearchBackend

        return NumpySearchBackend
    if name == "FaissSearchBackend":
        from rayzin.search.backends.faiss import FaissSearchBackend

        return FaissSearchBackend
    raise AttributeError(name)


__all__ = [
    "SearchBackend",
    "SearchResultHeap",
    "NumpySearchBackend",
    "FaissSearchBackend",
    "make_search_backend",
]
