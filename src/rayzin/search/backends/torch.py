from typing import Any

import numpy as np

from rayzin.enums import MetricType
from rayzin.readers.cog_reader import RawBlock
from rayzin.types import ChunkRef, Float32Array, Int64Array, SearchResults

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

    def create_heap(self, nq: int, k: int) -> "TorchResultHeap":
        return TorchResultHeap(nq, k, self._torch, self._device)

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
        distances, indices = self.search_raw_device(block, queries, k)
        return _host(distances, np.float32), _host(indices, np.int64)

    def search_raw_device(self, block: RawBlock, queries: Float32Array, k: int) -> tuple[Any, Any]:
        """``(nq, kk)`` distances and offsets of a raw block's top ``k``, left on the device."""
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
        return self._top_k_device(cells, queries, k)

    def radius_search(
        self, vectors: Float32Array, query: Float32Array, radius: float
    ) -> tuple[Float32Array, Int64Array]:
        distances, indices = self.search(vectors, query[None, :], len(vectors))
        keep = distances[0] <= radius
        return distances[0][keep], indices[0][keep]

    def _top_k(self, x: Any, queries: Float32Array, k: int) -> tuple[Float32Array, Int64Array]:
        distances, indices = self._top_k_device(x, queries, k)
        return _host(distances, np.float32), _host(indices, np.int64)

    def _top_k_device(self, x: Any, queries: Float32Array, k: int) -> tuple[Any, Any]:
        torch = self._torch
        q = self._device_queries(queries)
        valid = torch.isfinite(x).all(dim=1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self._normalize:
            x = torch.nn.functional.normalize(x, dim=1)
        squared = (x * x).sum(dim=1, keepdim=True) + (q * q).sum(dim=1)[None, :] - 2 * (x @ q.T)
        squared = squared.clamp_min_(0.0).masked_fill_(~valid[:, None], NODATA_DISTANCE)
        distances, indices = squared.topk(min(k, len(x)), dim=0, largest=False)
        return distances.T, indices.T

    def _device_queries(self, queries: Float32Array) -> Any:
        key = id(queries)
        if self._queries is None or self._queries[0] != key or self._queries[1] is not queries:
            tensor = self._torch.from_numpy(np.ascontiguousarray(queries, np.float32))
            self._queries = (key, queries, tensor.to(self._device))
        return self._queries[2]


def _host(tensor: Any, dtype: type) -> Any:
    return tensor.contiguous().cpu().numpy().astype(dtype)


def _torch_dtype(torch: Any, dtype: np.dtype[Any]) -> Any:
    return {2: torch.float16, 4: torch.float32, 8: torch.float64}[dtype.itemsize]


class TorchResultHeap:
    """Per-query top ``k`` held as ``(nq, k)`` tensors, merged a whole block at a time.

    A block's top ``k`` joins the heap with one ``topk`` over the concatenation, so a batch of
    thousands of queries costs no Python per candidate. ``drain`` hands the heap over and
    empties it, so each block's candidates leave an actor once.
    """

    def __init__(self, nq: int, k: int, torch: Any, device: Any) -> None:
        self._torch = torch
        self._device = device
        self._nq = nq
        self._k = k
        self.clear()

    @property
    def tau(self) -> Float32Array:
        return np.asarray(self._distances[:, -1].cpu().numpy(), dtype=np.float32)

    def clear(self) -> None:
        torch = self._torch
        shape = (self._nq, self._k)
        self._distances = torch.full(shape, float("inf"), device=self._device)
        self._blocks = torch.full(shape, -1, dtype=torch.int64, device=self._device)
        self._offsets = torch.zeros(shape, dtype=torch.int64, device=self._device)
        self._chunks: list[ChunkRef] = []
        self._chunk_ids: dict[tuple[Any, ...], int] = {}

    def add_device(self, active: np.ndarray, distances: Any, offsets: Any, chunk: ChunkRef) -> None:
        """Merge one block's ``(nq, kk)`` device top ``k``; inactive queries are left out."""
        torch = self._torch
        block = torch.full_like(offsets, self._chunk_id(chunk))
        distances = distances.masked_fill(
            ~torch.from_numpy(np.asarray(active, dtype=bool)).to(self._device)[:, None],
            float("inf"),
        )
        self._merge(distances, block, offsets)

    def add_result_subset(
        self, query_ids: Int64Array, distances: Float32Array, chunk: ChunkRef, offsets: Int64Array
    ) -> SearchResults:
        torch = self._torch
        width = distances.shape[1] if distances.ndim == 2 else 0
        full_d = torch.full((self._nq, width), float("inf"), device=self._device)
        full_o = torch.zeros((self._nq, width), dtype=torch.int64, device=self._device)
        rows = torch.from_numpy(np.asarray(query_ids, dtype=np.int64)).to(self._device)
        full_d[rows] = torch.from_numpy(np.asarray(distances, np.float32)).to(self._device)
        full_o[rows] = torch.from_numpy(np.asarray(offsets, np.int64)).to(self._device)
        self._merge(full_d, torch.full_like(full_o, self._chunk_id(chunk)), full_o)
        return SearchResults(query_ids=[], chunks=[], offsets=[], distances=[])

    def add_results(self, results: SearchResults) -> int:
        """Merge another heap's results; at most ``k`` per query, as ``results`` returns them."""
        if not results.offsets:
            return 0
        torch = self._torch
        queries = np.asarray(results.query_ids, dtype=np.int64)
        order = np.argsort(queries, kind="stable")
        queries = queries[order]
        starts = np.searchsorted(queries, queries, side="left")
        ranks = np.arange(len(queries)) - starts
        width = int(np.max(ranks)) + 1
        ids = np.asarray([self._chunk_id(results.chunks[i]) for i in order], dtype=np.int64)
        d = np.full((self._nq, width), np.inf, dtype=np.float32)
        b = np.full((self._nq, width), -1, dtype=np.int64)
        o = np.zeros((self._nq, width), dtype=np.int64)
        d[queries, ranks] = np.asarray(results.distances, dtype=np.float32)[order]
        b[queries, ranks] = ids
        o[queries, ranks] = np.asarray(results.offsets, dtype=np.int64)[order]

        def device(array: np.ndarray) -> Any:
            return torch.from_numpy(array).to(self._device)

        self._merge(device(d), device(b), device(o))
        return len(results.offsets)

    def results(self) -> SearchResults:
        distances = self._distances.cpu().numpy()
        blocks = self._blocks.cpu().numpy()
        offsets = self._offsets.cpu().numpy()
        keep = np.isfinite(distances) & (blocks >= 0)
        queries, ranks = np.nonzero(keep)
        return SearchResults(
            query_ids=queries.tolist(),
            chunks=[self._chunks[i] for i in blocks[queries, ranks].tolist()],
            offsets=offsets[queries, ranks].tolist(),
            distances=distances[queries, ranks].tolist(),
        )

    def drain(self) -> SearchResults:
        results = self.results()
        self.clear()
        return results

    def _merge(self, distances: Any, blocks: Any, offsets: Any) -> None:
        torch = self._torch
        all_d = torch.cat([self._distances, distances], dim=1)
        best, position = all_d.topk(self._k, dim=1, largest=False)
        self._blocks = torch.cat([self._blocks, blocks], dim=1).gather(1, position)
        self._offsets = torch.cat([self._offsets, offsets], dim=1).gather(1, position)
        self._distances = best

    def _chunk_id(self, chunk: ChunkRef) -> int:
        key = (chunk["url"], *((p["dim"], p["start"], p["stop"]) for p in chunk["slice"]))
        if key not in self._chunk_ids:
            self._chunk_ids[key] = len(self._chunks)
            self._chunks.append(chunk)
        return self._chunk_ids[key]
