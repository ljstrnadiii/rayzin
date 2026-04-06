import ray

from rayzin.enums import MetricType, SearchBackendType
from rayzin.search.backends import make_search_backend
from rayzin.types import SearchResults


@ray.remote
class HeapActor:
    def __init__(self, k: int, metric_type: str, backend_type: str) -> None:
        backend = make_search_backend(
            SearchBackendType(backend_type),
            MetricType(metric_type),
        )
        self._heap = backend.create_heap(k)

    def add_results(self, results: SearchResults) -> float:
        self._heap.add_results(results)
        return self._heap.tau

    def results(self) -> SearchResults:
        return self._heap.results()

    def tau(self) -> float:
        return self._heap.tau
