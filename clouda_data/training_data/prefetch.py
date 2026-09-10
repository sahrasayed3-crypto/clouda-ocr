"""Bounded prefetch over a batch iterator.

Runs the upstream batch iterator on a bounded background thread so the
consumer (trainer) can process the current batch while the next ones are
prepared. Guarantees:

- bounded memory (at most ``depth`` prepared batches);
- deterministic delivery order (FIFO, mirrors upstream order);
- exceptions from upstream propagate to the consumer, then shutdown;
- clean shutdown on consumer exit / ``close()`` / context manager;
- no runaway threads (daemon thread + sentinel + join with timeout).
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Iterable

_SENTINEL: Any = object()


class PrefetchIterator:
    """Thread-based bounded prefetch wrapper (CPU-safe, single consumer)."""

    def __init__(self, iterable: Iterable[Any], depth: int = 2) -> None:
        if depth < 1:
            raise ValueError("prefetch depth must be >= 1")
        self._iterator = iter(iterable)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=depth)
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._started = False

    def _produce(self) -> None:
        try:
            for item in self._iterator:
                while True:
                    if self._closed.is_set():
                        return
                    try:
                        self._queue.put(item, timeout=0.05)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # propagate later to the consumer
            self._error = exc
        finally:
            while True:
                if self._closed.is_set():
                    return
                try:
                    self._queue.put(_SENTINEL, timeout=0.05)
                    return
                except queue.Full:
                    continue

    def __iter__(self) -> "PrefetchIterator":
        return self

    def _ensure_started(self) -> None:
        with self._lock:
            if not self._started:
                self._started = True
                self._thread = threading.Thread(
                    target=self._produce, name="clouda-prefetch", daemon=True
                )
                self._thread.start()

    def __next__(self) -> Any:
        self._ensure_started()
        while True:
            if self._closed.is_set():
                raise StopIteration
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                if self._error is not None:
                    error = self._error
                    self.close()
                    raise error
                raise StopIteration
            return item

    def close(self) -> None:
        self._closed.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                # Producer stuck on put(); drain one slot so it can observe
                # the closed flag and exit.
                try:
                    self._queue.get_nowait()
                    thread.join(timeout=1.0)
                except queue.Empty:
                    pass

    def __enter__(self) -> "PrefetchIterator":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def prefetch(
    iterable: Iterable[Any],
    *,
    config: Any = None,
    depth: int = 2,
) -> PrefetchIterator:
    """Wrap an iterable with bounded prefetch (depth from config or arg)."""

    if config is not None:
        from clouda_data.training_data.models import PrefetchConfig

        if isinstance(config, PrefetchConfig):
            depth = config.depth
    return PrefetchIterator(iterable, depth=depth)
