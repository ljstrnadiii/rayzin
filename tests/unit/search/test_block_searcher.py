import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
import ray

from rayzin.manifest.schema import LOWER_BOUND_SCHEMA
from rayzin.search.backends.numpy import NumpyResultHeap
from rayzin.search.block_searcher import BlockSearcher
from rayzin.types import ChunkRecord, ChunkRef, SearchResults


class _FakeReader:
    def read(self, chunk: ChunkRecord) -> tuple[np.ndarray, tuple[int, ...]]:
        x_start = next(part["start"] for part in chunk["slice"] if part["dim"] == "x")
        return np.asarray([[float(x_start)]], dtype=np.float32), (1,)


class _FakeBackend:
    def create_heap(self, k: int) -> NumpyResultHeap:
        return NumpyResultHeap(k)

    def search(
        self,
        vectors: np.ndarray,
        query: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        del query, k
        return vectors[:, 0].astype(np.float32), np.asarray([0], dtype=np.int64)


def test_block_searcher_accumulates_heap_across_batches(
    ray_session: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rayzin.search.block_searcher.make_reader",
        lambda *args, **kwargs: _FakeReader(),
    )
    monkeypatch.setattr(
        "rayzin.search.block_searcher.make_search_backend",
        lambda *args, **kwargs: _FakeBackend(),
    )
    heap_actor = _CaptureHeapActor.remote()  # type: ignore[attr-defined]

    searcher = BlockSearcher(
        query=np.zeros(1, dtype=np.float32),
        k=1,
        metric_type="euclidean",
        reader_type="zarr",
        reader_kwargs={},
        backend_type="numpy",
        heap_actor=heap_actor,
    )

    first = searcher(_batch_for_x_start(20))
    second = searcher(_batch_for_x_start(10))
    third = searcher(_batch_for_x_start(15, lower_bound=15.0))

    assert first.to_pylist() == [{"rows_seen": 1, "rows_searched": 1, "results_added": 1}]
    assert second.to_pylist() == [{"rows_seen": 1, "rows_searched": 1, "results_added": 1}]
    assert third.to_pylist() == [{"rows_seen": 1, "rows_searched": 0, "results_added": 0}]
    calls = ray.get(heap_actor.calls_made.remote())
    assert len(calls) == 2
    assert calls[0][1] == [0]
    assert calls[0][2] == [20.0]
    assert calls[1][1] == [0]
    assert calls[1][2] == [10.0]


def test_block_searcher_returns_empty_batch_when_no_result_beats_tau(
    ray_session: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rayzin.search.block_searcher.make_reader",
        lambda *args, **kwargs: _FakeReader(),
    )
    monkeypatch.setattr(
        "rayzin.search.block_searcher.make_search_backend",
        lambda *args, **kwargs: _FakeBackend(),
    )
    heap_actor = _CaptureHeapActor.remote()  # type: ignore[attr-defined]

    searcher = BlockSearcher(
        query=np.zeros(1, dtype=np.float32),
        k=1,
        metric_type="euclidean",
        reader_type="zarr",
        reader_kwargs={},
        backend_type="numpy",
        heap_actor=heap_actor,
    )

    first = searcher(_batch_for_x_start(10))
    second = searcher(_batch_for_x_start(20))

    assert first.to_pylist() == [{"rows_seen": 1, "rows_searched": 1, "results_added": 1}]
    assert second.to_pylist() == [{"rows_seen": 1, "rows_searched": 1, "results_added": 0}]
    calls = ray.get(heap_actor.calls_made.remote())
    assert len(calls) == 1
    assert calls[0][1] == [0]
    assert calls[0][2] == [10.0]


@ray.remote
class _CaptureHeapActor:
    def __init__(self) -> None:
        self.calls: list[tuple[list[ChunkRef], list[int], list[float]]] = []
        self._tau = float("inf")

    def add_results(self, results: SearchResults) -> float:
        self.calls.append((results.chunks, results.offsets, results.distances))
        if results.distances:
            self._tau = min(self._tau, max(results.distances))
        return self._tau

    def tau(self) -> float:
        return self._tau

    def calls_made(
        self,
    ) -> list[tuple[list[ChunkRef], list[int], list[float]]]:
        return self.calls


def test_block_searcher_pushes_results_to_heap_actor(
    ray_session: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rayzin.search.block_searcher.make_reader",
        lambda *args, **kwargs: _FakeReader(),
    )
    monkeypatch.setattr(
        "rayzin.search.block_searcher.make_search_backend",
        lambda *args, **kwargs: _FakeBackend(),
    )

    heap_actor = _CaptureHeapActor.remote()  # type: ignore[attr-defined]
    searcher = BlockSearcher(
        query=np.zeros(1, dtype=np.float32),
        k=1,
        metric_type="euclidean",
        reader_type="zarr",
        reader_kwargs={},
        backend_type="numpy",
        heap_actor=heap_actor,
    )

    result = searcher(_batch_for_x_start(10))

    assert result.to_pylist() == [{"rows_seen": 1, "rows_searched": 1, "results_added": 1}]
    calls = ray.get(heap_actor.calls_made.remote())
    assert len(calls) == 1
    chunks, offsets, distances = calls[0]
    assert chunks == [
        {
            "url": "store.zarr",
            "slice": [
                {"dim": "time", "start": 0, "stop": 1},
                {"dim": "y", "start": 0, "stop": 1},
                {"dim": "x", "start": 10, "stop": 11},
            ],
        }
    ]
    assert offsets == [0]
    assert distances == [10.0]


def test_block_searcher_uses_cached_global_tau_for_pruning(
    ray_session: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rayzin.search.block_searcher.make_reader",
        lambda *args, **kwargs: _FakeReader(),
    )
    monkeypatch.setattr(
        "rayzin.search.block_searcher.make_search_backend",
        lambda *args, **kwargs: _FakeBackend(),
    )

    heap_actor = _CaptureHeapActor.remote()  # type: ignore[attr-defined]
    other_searcher = BlockSearcher(
        query=np.zeros(1, dtype=np.float32),
        k=1,
        metric_type="euclidean",
        reader_type="zarr",
        reader_kwargs={},
        backend_type="numpy",
        heap_actor=heap_actor,
    )
    searcher = BlockSearcher(
        query=np.zeros(1, dtype=np.float32),
        k=1,
        metric_type="euclidean",
        reader_type="zarr",
        reader_kwargs={},
        backend_type="numpy",
        heap_actor=heap_actor,
    )

    first = other_searcher(_batch_for_x_start(5))
    second = searcher(
        pa.Table.from_pydict(
            {
                "url": ["store.zarr", "store.zarr"],
                "slice": [
                    [
                        {"dim": "time", "start": 0, "stop": 1},
                        {"dim": "y", "start": 0, "stop": 1},
                        {"dim": "x", "start": 6, "stop": 7},
                    ],
                    [
                        {"dim": "time", "start": 0, "stop": 1},
                        {"dim": "y", "start": 0, "stop": 1},
                        {"dim": "x", "start": 8, "stop": 9},
                    ],
                ],
                "count": [1, 1],
                "centroid": [[0.0], [0.0]],
                "radius": [0.0, 0.0],
                "lower_bound": [6.0, 8.0],
            },
            schema=LOWER_BOUND_SCHEMA,
        )
    )

    assert first.to_pylist() == [{"rows_seen": 1, "rows_searched": 1, "results_added": 1}]
    assert second.to_pylist() == [{"rows_seen": 2, "rows_searched": 0, "results_added": 0}]


def _batch_for_x_start(x_start: int, lower_bound: float = 0.0) -> pa.Table:
    return pa.Table.from_pydict(
        {
            "url": ["store.zarr"],
            "slice": [
                [
                    {"dim": "time", "start": 0, "stop": 1},
                    {"dim": "y", "start": 0, "stop": 1},
                    {"dim": "x", "start": x_start, "stop": x_start + 1},
                ]
            ],
            "count": [1],
            "centroid": [[0.0]],
            "radius": [0.0],
            "lower_bound": [lower_bound],
        },
        schema=LOWER_BOUND_SCHEMA,
    )
