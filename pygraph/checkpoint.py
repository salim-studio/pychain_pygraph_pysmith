"""Checkpoint — MemorySaver (مطابق لـ langgraph.checkpoint.memory)."""
from __future__ import annotations

import copy
from threading import RLock


class Checkpoint:
    __slots__ = ("thread_id", "step", "state", "next_nodes")

    def __init__(self, thread_id: str, step: int, state: dict, next_nodes: tuple = ()):
        self.thread_id = thread_id
        self.step = step
        self.state = state
        self.next_nodes = next_nodes


class MemorySaver:
    """حفظ محلي سريع thread-safe: thread_id -> قائمة لقطات."""

    def __init__(self):
        self._store: dict[str, list[dict]] = {}
        self._lock = RLock()

    def put(self, thread_id: str, state: dict, step: int):
        snap = copy.deepcopy(state)
        with self._lock:
            self._store.setdefault(thread_id, []).append({"step": step, "state": snap})

    def get(self, thread_id: str) -> dict | None:
        with self._lock:
            hist = self._store.get(thread_id)
            if not hist:
                return None
            return copy.deepcopy(hist[-1]["state"])

    def history(self, thread_id: str) -> list[dict]:
        with self._lock:
            return copy.deepcopy(self._store.get(thread_id, []))

    def clear(self, thread_id: str | None = None):
        with self._lock:
            if thread_id is None:
                self._store.clear()
            else:
                self._store.pop(thread_id, None)


# توافق اسمي
MemorySaverCheckpoint = MemorySaver
InMemorySaver = MemorySaver

__all__ = ["MemorySaver", "Checkpoint"]
