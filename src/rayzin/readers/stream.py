import asyncio
import queue
import threading
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from rayzin.types import ChunkRecord


class EncodedBlock(Protocol):
    """A fetched chunk, still encoded: what it decodes to, and how to decode it into a buffer."""

    @property
    def dtype(self) -> np.dtype[Any]: ...

    @property
    def shape(self) -> tuple[int, int, int]: ...

    @property
    def window(self) -> tuple[int, int]: ...

    def decode_into(self, out: np.ndarray) -> None: ...


class StreamingReader(Protocol):
    decode_threads: int

    def fetch_encoded(self, chunk: ChunkRecord) -> Coroutine[Any, Any, EncodedBlock]: ...


@dataclass(frozen=True)
class RawBlock:
    """Cells decoded in their stored dtype into a host buffer the consumer hands back.

    The buffer holds a ``(height, width, bands)`` grid of which the top-left ``window`` cells
    are real: a COG block at an image edge is padded, a row group is ``(rows, 1, bands)``.
    """

    slot: tuple[Any, np.ndarray]
    dtype: np.dtype[Any]
    shape: tuple[int, int, int]
    window: tuple[int, int]

    @property
    def nbytes(self) -> int:
        return decoded_nbytes(self.dtype, self.shape)


def decoded_nbytes(dtype: np.dtype[Any], shape: tuple[int, int, int]) -> int:
    height, width, bands = shape
    return height * width * bands * dtype.itemsize


class RawBlockStream:
    """Fetches and decodes chunks into reusable host buffers, as many at once as allowed.

    ``in_flight`` bounds chunks fetched but not yet decoded, ``threads`` decode at once, and a
    decoded block holds its buffer until ``release``, so memory stays bounded. ``allocate``
    makes a buffer of at least n bytes, e.g. pinned memory, and returns it with a writable
    uint8 view. Blocks come back in the order they finish.
    """

    def __init__(
        self,
        reader: StreamingReader,
        allocate: Callable[[int], tuple[Any, np.ndarray]],
        *,
        in_flight: int,
        threads: int,
    ) -> None:
        self._reader = reader
        self._allocate = allocate
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._gate = asyncio.Semaphore(in_flight)
        self._pool = ThreadPoolExecutor(threads)
        self._free: queue.Queue[tuple[Any, np.ndarray] | None] = queue.Queue()
        for _ in range(threads + 2):
            self._free.put(None)
        self._ready: queue.Queue[tuple[Any, RawBlock | None, BaseException | None]] = queue.Queue()
        self.pending = 0

    def submit(self, key: Any, chunk: ChunkRecord) -> None:
        self.pending += 1
        asyncio.run_coroutine_threadsafe(self._fetch(key, chunk), self._loop)

    def next(self) -> tuple[Any, RawBlock]:
        key, block, error = self._ready.get()
        self.pending -= 1
        if error is not None:
            raise error
        assert block is not None
        return key, block

    def release(self, block: RawBlock) -> None:
        self._free.put(block.slot)

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._pool.shutdown()

    async def _fetch(self, key: Any, chunk: ChunkRecord) -> None:
        await self._gate.acquire()
        try:
            encoded = await self._reader.fetch_encoded(chunk)
        except Exception as error:
            self._gate.release()
            self._ready.put((key, None, error))
            return
        self._pool.submit(self._decode, key, encoded)

    def _decode(self, key: Any, encoded: EncodedBlock) -> None:
        slot = self._free.get()
        try:
            nbytes = decoded_nbytes(encoded.dtype, encoded.shape)
            if slot is None or slot[1].nbytes < nbytes:
                slot = self._allocate(nbytes)
            encoded.decode_into(slot[1][:nbytes])
            block = RawBlock(slot, encoded.dtype, encoded.shape, encoded.window)
            self._ready.put((key, block, None))
        except Exception as error:
            self._free.put(slot)
            self._ready.put((key, None, error))
        finally:
            self._loop.call_soon_threadsafe(self._gate.release)
