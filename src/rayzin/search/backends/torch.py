from typing import Any

import numpy as np

from rayzin.enums import MetricType
from rayzin.readers.cog_reader import RawBlock
from rayzin.search.backends.numpy import NumpyResultHeap
from rayzin.search.backends.protocols import SearchResultHeap
from rayzin.types import Float32Array, Int64Array

NODATA_DISTANCE = 1e6


class TorchSearchBackend:
    """Exact squared-L2 search with torch, on the GPU when there is one.

    ``search_raw`` takes a block still in its stored dtype, e.g. float16, from a pinned host
    buffer: it is copied to the device asynchronously and cropped, masked, normalized, scored
    and reduced to the top ``k`` there, so the host only decompresses. Cells with a non-finite
    value score ``NODATA_DISTANCE``.
    """

    def __init__(self, metric_type: MetricType, normalize: bool = True) -> None:
        if metric_type != MetricType.EUCLIDEAN:
            msg = "The torch backend scores squared L2 only."
            raise NotImplementedError(msg)
        import torch  # type: ignore[import-not-found,unused-ignore]

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._normalize = normalize
        self._queries: tuple[int, Any, Any] | None = None

    def create_heap(self, nq: int, k: int) -> SearchResultHeap:
        return NumpyResultHeap(nq, k)

    def allocate(self, nbytes: int) -> tuple[Any, np.ndarray]:
        buffer = self._torch.empty(
            nbytes, dtype=self._torch.uint8, pin_memory=self._device.type == "cuda"
        )
        return buffer, buffer.numpy()

    def search(
        self, vectors: Float32Array, queries: Float32Array, k: int
    ) -> tuple[Float32Array, Int64Array]:
        x = self._torch.from_numpy(np.ascontiguousarray(vectors, np.float32)).to(self._device)
        return self._top_k(x, queries, k)

    def search_raw(
        self, block: RawBlock, queries: Float32Array, k: int
    ) -> tuple[Float32Array, Int64Array]:
        torch = self._torch
        layout = block.layout
        y0, y1, x0, x1 = layout.window(block.row, block.column)
        raw = block.slot[0][: block.nbytes].to(self._device, non_blocking=True)
        cells = (
            raw.view(_torch_dtype(torch, layout.dtype))
            .view(layout.block_height, layout.block_width, layout.bands)[: y1 - y0, : x1 - x0]
            .reshape(-1, layout.bands)
            .float()
        )
        return self._top_k(cells, queries, k)

    def radius_search(
        self, vectors: Float32Array, query: Float32Array, radius: float
    ) -> tuple[Float32Array, Int64Array]:
        distances, indices = self.search(vectors, query[None, :], len(vectors))
        keep = distances[0] <= radius
        return distances[0][keep], indices[0][keep]

    def _top_k(self, x: Any, queries: Float32Array, k: int) -> tuple[Float32Array, Int64Array]:
        torch = self._torch
        q = self._device_queries(queries)
        valid = torch.isfinite(x).all(dim=1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self._normalize:
            x = torch.nn.functional.normalize(x, dim=1)
        squared = (x * x).sum(dim=1, keepdim=True) + (q * q).sum(dim=1)[None, :] - 2 * (x @ q.T)
        squared = squared.clamp_min_(0.0).masked_fill_(~valid[:, None], NODATA_DISTANCE)
        distances, indices = squared.topk(min(k, len(x)), dim=0, largest=False)
        return (
            distances.T.contiguous().cpu().numpy().astype(np.float32),
            indices.T.contiguous().cpu().numpy().astype(np.int64),
        )

    def _device_queries(self, queries: Float32Array) -> Any:
        key = id(queries)
        if self._queries is None or self._queries[0] != key or self._queries[1] is not queries:
            tensor = self._torch.from_numpy(np.ascontiguousarray(queries, np.float32))
            self._queries = (key, queries, tensor.to(self._device))
        return self._queries[2]


def _torch_dtype(torch: Any, dtype: np.dtype[Any]) -> Any:
    return {2: torch.float16, 4: torch.float32, 8: torch.float64}[dtype.itemsize]
