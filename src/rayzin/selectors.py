from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import TypeAlias

import zarr
from zarr.core.metadata import ArrayV3Metadata

from rayzin.readers.zarr_layout import index_axis_names, index_axis_positions

SelectorValue: TypeAlias = int | slice
Selector: TypeAlias = Mapping[str, SelectorValue]
Interval: TypeAlias = tuple[int, int]


def resolve_selector_intervals(
    group: zarr.Group,
    array: zarr.Array[ArrayV3Metadata],
    *,
    selectors: Selector | None,
    embedding_dim_name: str,
) -> dict[str, list[Interval]]:
    axis_names = index_axis_names(array, embedding_dim_name=embedding_dim_name)
    axis_positions = index_axis_positions(array, embedding_dim_name=embedding_dim_name)
    requested = dict(selectors or {})
    unknown_dims = sorted(set(requested) - set(axis_names))
    if unknown_dims:
        msg = (
            "Selectors must target array index dimensions only. "
            f"Got unknown dimensions: {unknown_dims!r}; "
            f"available dimensions: {axis_names!r}."
        )
        raise ValueError(msg)

    intervals: dict[str, list[Interval]] = {}
    for dim_name, axis in zip(axis_names, axis_positions):
        intervals[dim_name] = _resolve_dim_intervals(
            dim_size=int(array.shape[axis]),
            chunk_size=int(array.chunks[axis]),
            selector=requested.get(dim_name),
            dim_name=dim_name,
        )
    return intervals


def _resolve_dim_intervals(
    *,
    dim_size: int,
    chunk_size: int,
    selector: SelectorValue | None,
    dim_name: str,
) -> list[Interval]:
    if selector is None:
        return _split_interval_on_chunk_boundaries(0, dim_size, chunk_size)

    if isinstance(selector, Integral) and not isinstance(selector, bool):
        index_value = int(selector)
        index = index_value if index_value >= 0 else dim_size + index_value
        if index < 0 or index >= dim_size:
            raise IndexError(f"Selector index out of bounds for dimension {dim_name!r}: {selector}")
        return [(int(index), int(index + 1))]

    if isinstance(selector, slice):
        if not all(
            _is_integer_index_part(part) for part in (selector.start, selector.stop, selector.step)
        ):
            msg = (
                f"Selector for dimension {dim_name!r} must use integer indices only. "
                f"Got {selector!r}."
            )
            raise TypeError(msg)
        start, stop, step = selector.indices(dim_size)
        if step == 1:
            return _split_interval_on_chunk_boundaries(start, stop, chunk_size)
        return _coalesce_indices_to_intervals(list(range(start, stop, step)), chunk_size)

    msg = (
        f"Selector for dimension {dim_name!r} must be an integer index or integer slice. "
        f"Got {type(selector).__name__}."
    )
    raise TypeError(msg)


def _is_integer_index_part(value: object | None) -> bool:
    return value is None or (isinstance(value, Integral) and not isinstance(value, bool))


def _coalesce_indices_to_intervals(indices: list[int], chunk_size: int) -> list[Interval]:
    if not indices:
        return []

    unique_indices = sorted(set(indices))
    merged: list[Interval] = []
    start = unique_indices[0]
    stop = start + 1
    for index in unique_indices[1:]:
        if index == stop:
            stop += 1
            continue
        merged.extend(_split_interval_on_chunk_boundaries(start, stop, chunk_size))
        start = index
        stop = index + 1
    merged.extend(_split_interval_on_chunk_boundaries(start, stop, chunk_size))
    return merged


def _split_interval_on_chunk_boundaries(start: int, stop: int, chunk_size: int) -> list[Interval]:
    if start >= stop:
        return []

    intervals: list[Interval] = []
    current = start
    while current < stop:
        next_boundary = ((current // chunk_size) + 1) * chunk_size
        next_stop = min(stop, next_boundary)
        intervals.append((int(current), int(next_stop)))
        current = next_stop
    return intervals
