from pathlib import Path
from typing import cast

import numpy as np
import pytest
import zarr
from zarr.core.metadata import ArrayV3Metadata

from rayzin.selectors import (
    Selector,
    _coalesce_indices_to_intervals,
    _resolve_dim_intervals,
    _split_interval_on_chunk_boundaries,
    resolve_selector_intervals,
)


def test_split_interval_on_chunk_boundaries_splits_partial_edges() -> None:
    intervals = _split_interval_on_chunk_boundaries(1, 7, 3)

    assert intervals == [(1, 3), (3, 6), (6, 7)]


def test_coalesce_indices_to_intervals_merges_and_splits_on_chunks() -> None:
    intervals = _coalesce_indices_to_intervals([0, 1, 2, 4, 5, 8], chunk_size=3)

    assert intervals == [(0, 3), (4, 6), (8, 9)]


def test_resolve_dim_intervals_supports_negative_integer_index() -> None:
    intervals = _resolve_dim_intervals(
        dim_size=5,
        chunk_size=2,
        selector=-1,
        dim_name="time",
    )

    assert intervals == [(4, 5)]


def test_resolve_dim_intervals_supports_stepped_integer_slice() -> None:
    intervals = _resolve_dim_intervals(
        dim_size=8,
        chunk_size=3,
        selector=slice(0, 8, 2),
        dim_name="time",
    )

    assert intervals == [(0, 1), (2, 3), (4, 5), (6, 7)]


def test_resolve_dim_intervals_rejects_out_of_bounds_integer() -> None:
    with pytest.raises(IndexError, match="out of bounds"):
        _resolve_dim_intervals(
            dim_size=3,
            chunk_size=2,
            selector=3,
            dim_name="time",
        )


def test_resolve_selector_intervals_rejects_unknown_dims(tmp_path: Path) -> None:
    group, array = _create_test_group(tmp_path)

    with pytest.raises(ValueError, match="unknown dimensions"):
        resolve_selector_intervals(
            group,
            array,
            selectors={"band": 0},
            embedding_dim_name="embedding",
        )


def test_resolve_selector_intervals_defaults_to_full_chunked_extent(tmp_path: Path) -> None:
    group, array = _create_test_group(tmp_path)

    intervals = resolve_selector_intervals(
        group,
        array,
        selectors=None,
        embedding_dim_name="embedding",
    )

    assert intervals == {
        "time": [(0, 2), (2, 4)],
        "y": [(0, 2), (2, 4)],
        "x": [(0, 2), (2, 4)],
    }


def test_resolve_dim_intervals_rejects_non_integer_slice() -> None:
    with pytest.raises(TypeError, match="integer indices only"):
        _resolve_dim_intervals(
            dim_size=4,
            chunk_size=2,
            selector=slice("2023", "2024"),
            dim_name="time",
        )


def test_resolve_selector_intervals_rejects_non_integer_selector(tmp_path: Path) -> None:
    group, array = _create_test_group(tmp_path)

    with pytest.raises(TypeError, match="integer index or integer slice"):
        resolve_selector_intervals(
            group,
            array,
            selectors=cast(Selector, {"time": "2023"}),
            embedding_dim_name="embedding",
        )


def _create_test_group(tmp_path: Path) -> tuple[zarr.Group, zarr.Array[ArrayV3Metadata]]:
    store_path = str(tmp_path / "selector_test.zarr")
    group = zarr.open_group(store_path, mode="w")
    array = group.create_array(
        "embeddings",
        shape=(4, 4, 4, 2),
        dtype=np.float32,
        chunks=(2, 2, 2, 2),
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x", "embedding"]
    array[:] = np.arange(4 * 4 * 4 * 2, dtype=np.float32).reshape(4, 4, 4, 2)
    time = group.create_array("time", shape=(4,), dtype="U10")
    time[:] = np.asarray(["2022-01-01", "2023-01-01", "2023-06-01", "2024-01-01"])
    return group, array
