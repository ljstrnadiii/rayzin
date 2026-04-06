import numpy as np

from rayzin.search.backends.numpy import NumpyResultHeap
from rayzin.types import ChunkRef


def test_result_heap_returns_sorted_distances() -> None:
    heap = NumpyResultHeap(1, 3)
    chunk = ChunkRef(
        url="chunk.zarr",
        slice=[
            {"dim": "y", "start": 0, "stop": 10},
            {"dim": "x", "start": 0, "stop": 10},
        ],
    )

    heap.add_result_subset(
        np.array([0], dtype=np.int64),
        np.array([[1.0, 2.0, 0.5]], dtype=np.float32),
        chunk,
        np.array([[0, 1, 2]], dtype=np.int64),
    )

    results = heap.results()
    assert results.query_ids == [0, 0, 0]
    assert results.chunks == [chunk, chunk, chunk]
    assert results.offsets == [2, 0, 1]
    assert results.distances == [0.5, 1.0, 2.0]


def test_result_heap_keeps_top_k_across_multiple_adds() -> None:
    heap = NumpyResultHeap(1, 2)
    first_chunk = ChunkRef(
        url="first.zarr",
        slice=[
            {"dim": "y", "start": 0, "stop": 10},
            {"dim": "x", "start": 0, "stop": 10},
        ],
    )
    second_chunk = ChunkRef(
        url="second.zarr",
        slice=[
            {"dim": "y", "start": 10, "stop": 20},
            {"dim": "x", "start": 10, "stop": 20},
        ],
    )

    heap.add_result_subset(
        np.array([0], dtype=np.int64),
        np.array([[4.0, 1.5]], dtype=np.float32),
        first_chunk,
        np.array([[0, 1]], dtype=np.int64),
    )
    heap.add_result_subset(
        np.array([0], dtype=np.int64),
        np.array([[0.75, 2.0]], dtype=np.float32),
        second_chunk,
        np.array([[2, 3]], dtype=np.int64),
    )

    results = heap.results()
    assert results.query_ids == [0, 0]
    assert results.chunks == [second_chunk, first_chunk]
    assert results.offsets == [2, 1]
    assert results.distances == [0.75, 1.5]
