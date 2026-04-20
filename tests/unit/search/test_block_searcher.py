import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pytest
import ray

from rayzin.schema import LOWER_BOUND_SCHEMA
from rayzin.search.backends.numpy import NumpyResultHeap
from rayzin.search.block_searcher import BlockSearcher
from rayzin.types import ChunkRecord, ChunkRef, SearchResults


class _FakeReader:
    def read(self, chunk: ChunkRecord) -> tuple[np.ndarray, tuple[int, ...]]:
        x_start = next(part["start"] for part in chunk["slice"] if part["dim"] == "x")
        return np.asarray([[float(x_start)]], dtype=np.float32), (1,)


class _FakeBackend:
    def create_heap(self, nq: int, k: int) -> NumpyResultHeap:
        return NumpyResultHeap(nq, k)

    def search(
        self,
        vectors: np.ndarray,
        queries: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        del k
        nq = len(queries)
        return (
            np.repeat(vectors[:, 0][None, :], nq, axis=0).astype(np.float32),
            np.repeat(np.asarray([[0]], dtype=np.int64), nq, axis=0),
        )


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
        queries=np.zeros((1, 1), dtype=np.float32),
        k=1,
        metric_type="euclidean",
        reader_type="zarr",
        reader_kwargs={},
        backend_type="numpy",
        heap_actor=heap_actor,
    )

    first = searcher(_batch_for_x_start(20))
    second = searcher(_batch_for_x_start(10))
    third = searcher(_batch_for_x_start(15, min_lower_bound=15.0, lower_bounds=[15.0]))

    assert first.to_pylist() == [
        {"rows_seen": 1, "rows_searched": 1, "query_evaluations": 1, "results_added": 1}
    ]
    assert second.to_pylist() == [
        {"rows_seen": 1, "rows_searched": 1, "query_evaluations": 1, "results_added": 1}
    ]
    assert third.to_pylist() == [
        {"rows_seen": 1, "rows_searched": 0, "query_evaluations": 0, "results_added": 0}
    ]
    calls = ray.get(heap_actor.calls_made.remote())
    assert len(calls) == 2
    assert calls[0][2] == [0]
    assert calls[0][3] == [20.0]
    assert calls[1][2] == [0]
    assert calls[1][3] == [10.0]


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
        queries=np.zeros((1, 1), dtype=np.float32),
        k=1,
        metric_type="euclidean",
        reader_type="zarr",
        reader_kwargs={},
        backend_type="numpy",
        heap_actor=heap_actor,
    )

    first = searcher(_batch_for_x_start(10))
    second = searcher(_batch_for_x_start(20))

    assert first.to_pylist() == [
        {"rows_seen": 1, "rows_searched": 1, "query_evaluations": 1, "results_added": 1}
    ]
    assert second.to_pylist() == [
        {"rows_seen": 1, "rows_searched": 1, "query_evaluations": 1, "results_added": 0}
    ]
    calls = ray.get(heap_actor.calls_made.remote())
    assert len(calls) == 1
    assert calls[0][2] == [0]
    assert calls[0][3] == [10.0]


@ray.remote
class _CaptureHeapActor:
    def __init__(self) -> None:
        self.calls: list[tuple[list[int], list[ChunkRef], list[int], list[float]]] = []
        self._tau = np.asarray([float("inf")], dtype=np.float32)

    def add_results(self, results: SearchResults) -> np.ndarray:
        self.calls.append((results.query_ids, results.chunks, results.offsets, results.distances))
        if results.distances:
            self._tau = np.minimum(
                self._tau,
                np.asarray([max(results.distances)], dtype=np.float32),
            )
        return self._tau

    def tau(self) -> np.ndarray:
        return self._tau

    def calls_made(
        self,
    ) -> list[tuple[list[int], list[ChunkRef], list[int], list[float]]]:
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
        queries=np.zeros((1, 1), dtype=np.float32),
        k=1,
        metric_type="euclidean",
        reader_type="zarr",
        reader_kwargs={},
        backend_type="numpy",
        heap_actor=heap_actor,
    )

    result = searcher(_batch_for_x_start(10))

    assert result.to_pylist() == [
        {"rows_seen": 1, "rows_searched": 1, "query_evaluations": 1, "results_added": 1}
    ]
    calls = ray.get(heap_actor.calls_made.remote())
    assert len(calls) == 1
    query_ids, chunks, offsets, distances = calls[0]
    assert query_ids == [0]
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
        queries=np.zeros((1, 1), dtype=np.float32),
        k=1,
        metric_type="euclidean",
        reader_type="zarr",
        reader_kwargs={},
        backend_type="numpy",
        heap_actor=heap_actor,
    )
    searcher = BlockSearcher(
        queries=np.zeros((1, 1), dtype=np.float32),
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
                "lower_bounds": [[6.0], [8.0]],
                "min_lower_bound": [6.0, 8.0],
            },
            schema=LOWER_BOUND_SCHEMA,
        )
    )

    assert first.to_pylist() == [
        {"rows_seen": 1, "rows_searched": 1, "query_evaluations": 1, "results_added": 1}
    ]
    assert second.to_pylist() == [
        {"rows_seen": 2, "rows_searched": 0, "query_evaluations": 0, "results_added": 0}
    ]


def _batch_for_x_start(
    x_start: int,
    *,
    min_lower_bound: float = 0.0,
    lower_bounds: list[float] | None = None,
) -> pa.Table:
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
            "lower_bounds": [lower_bounds or [min_lower_bound]],
            "min_lower_bound": [min_lower_bound],
        },
        schema=LOWER_BOUND_SCHEMA,
    )
