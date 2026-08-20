# -*- coding: utf-8 -*-

"""Bounded single-thread prefetch for already-verified batch loaders."""

from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Callable, Iterable, Iterator


_POLL_SECONDS = 0.05
_CLOSE_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class _Loaded:
    batch_id: Any
    batch: Any


@dataclass(frozen=True)
class _Failure:
    batch_id: Any
    exception: BaseException
    traceback: TracebackType | None


class _End:
    pass


_END = _End()


class VerifiedBatchPrefetcher(Iterator[tuple[Any, Any]]):
    """Load verified batches in order on one bounded background thread.

    ``load_fn`` remains the authority for SHA verification, decoding,
    processing, and collation. This class never calls sampler commit methods,
    optimizer methods, or any training-state transition.
    """

    def __init__(
        self,
        batch_id_iterable: Iterable[Any],
        load_fn: Callable[[Any], Any],
        max_prefetch: int = 1,
    ) -> None:
        if not callable(load_fn):
            raise TypeError("load_fn must be callable")
        if type(max_prefetch) is not int or max_prefetch <= 0:
            raise ValueError("max_prefetch must be a positive integer")
        self.max_prefetch = max_prefetch
        self._owner_pid = os.getpid()
        self._source = iter(batch_id_iterable)
        self._load_fn = load_fn
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max_prefetch)
        self._slots = threading.BoundedSemaphore(max_prefetch)
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._started = False
        self._closed = False
        self._terminal = False

    @property
    def worker_alive(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def __enter__(self) -> "VerifiedBatchPrefetcher":
        self._check_process()
        self._start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def __iter__(self) -> "VerifiedBatchPrefetcher":
        self._check_process()
        self._start()
        return self

    def __next__(self) -> tuple[Any, Any]:
        self._check_process()
        if self._terminal:
            raise StopIteration
        if self._closed:
            raise RuntimeError("VerifiedBatchPrefetcher is closed")
        self._start()
        item = self._queue.get()
        if isinstance(item, (_Loaded, _Failure)):
            self._slots.release()
        if isinstance(item, _Loaded):
            return item.batch_id, item.batch
        if isinstance(item, _Failure):
            self._terminal = True
            self.close()
            raise item.exception.with_traceback(item.traceback)
        if item is _END:
            self._terminal = True
            self.close()
            raise StopIteration
        self._terminal = True
        self.close()
        raise RuntimeError("VerifiedBatchPrefetcher received an invalid queue item")

    def close(self) -> None:
        self._check_process()
        with self._close_lock:
            if self._closed and not self.worker_alive:
                return
            self._closed = True
            self._cancel.set()
            self._drain_queue()
            thread = self._thread
            if thread is None:
                return
            thread.join(timeout=_CLOSE_TIMEOUT_SECONDS)
            if thread.is_alive():
                raise RuntimeError(
                    "VerifiedBatchPrefetcher loader did not stop within "
                    f"{_CLOSE_TIMEOUT_SECONDS:g}s"
                )

    def _check_process(self) -> None:
        current_pid = os.getpid()
        if current_pid != self._owner_pid:
            raise RuntimeError(
                "VerifiedBatchPrefetcher cannot be reused after fork; create "
                "a new prefetcher in the child process"
            )

    def _start(self) -> None:
        if self._closed:
            raise RuntimeError("VerifiedBatchPrefetcher is closed")
        with self._start_lock:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name=f"verified-batch-prefetch-{id(self):x}",
                daemon=False,
            )
            self._thread.start()

    def _run(self) -> None:
        while not self._cancel.is_set():
            if not self._acquire_slot():
                return
            try:
                batch_id = next(self._source)
            except StopIteration:
                self._slots.release()
                self._put(_END)
                return
            except BaseException as exc:
                failure = _Failure(None, exc, exc.__traceback__)
                if not self._put(failure):
                    self._slots.release()
                return

            try:
                batch = self._load_fn(batch_id)
            except BaseException as exc:
                failure = _Failure(batch_id, exc, exc.__traceback__)
                if not self._put(failure):
                    self._slots.release()
                return
            if not self._put(_Loaded(batch_id, batch)):
                self._slots.release()
                return

    def _acquire_slot(self) -> bool:
        while not self._cancel.is_set():
            if self._slots.acquire(timeout=_POLL_SECONDS):
                if self._cancel.is_set():
                    self._slots.release()
                    return False
                return True
        return False

    def _put(self, item: object) -> bool:
        while not self._cancel.is_set():
            try:
                self._queue.put(item, timeout=_POLL_SECONDS)
                return True
            except queue.Full:
                continue
        return False

    def _drain_queue(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, (_Loaded, _Failure)):
                self._slots.release()


__all__ = ["VerifiedBatchPrefetcher"]
