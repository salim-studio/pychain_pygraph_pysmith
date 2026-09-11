"""Tracing — @traceable + Run (مطابق لـ langsmith.run_helpers، محلي أولاً وسريع)."""
from __future__ import annotations

import contextvars
import functools
import inspect
import json
import os
import queue
import threading
import time
import uuid

_ENABLED = os.environ.get("PYSMITH_TRACING", "true").lower() not in ("0", "false", "off", "no")
_STORE = os.environ.get("PYSMITH_STORE", ".pysmith_runs.jsonl")
_current: contextvars.ContextVar = contextvars.ContextVar("pysmith_run", default=None)

_Q: queue.Queue = queue.Queue()
_FLUSH_N = int(os.environ.get("PYSMITH_BATCH", "50"))
_worker_started = False
_worker_lock = threading.Lock()


def is_enabled() -> bool:
    return _ENABLED


def set_enabled(v: bool):
    global _ENABLED
    _ENABLED = bool(v)


def current_run():
    return _current.get()


def _event(kind: str, name: str, data=None):
    # خطاف خفيف لـ pychain (لا يفعل شيئاً إن تعطّل التتبع)
    if not _ENABLED:
        return
    run = _current.get()
    if run is not None:
        try:
            run._events.append({"kind": kind, "name": name, "t": time.time()})
        except Exception:
            pass


class Run:
    __slots__ = ("id", "name", "run_type", "inputs", "outputs", "error",
                 "start", "end", "parent_id", "tags", "metadata", "children",
                 "feedback", "_events")

    def __init__(self, name: str, inputs=None, run_type: str = "chain",
                 tags: list | None = None, metadata: dict | None = None, parent_id=None):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.run_type = run_type
        self.inputs = inputs
        self.outputs = None
        self.error = None
        self.start = time.time()
        self.end = 0.0
        self.parent_id = parent_id
        self.tags = tags or []
        self.metadata = metadata or {}
        self.children: list[Run] = []
        self.feedback: list[dict] = []
        self._events: list[dict] = []

    @property
    def latency(self) -> float:
        return (self.end or time.time()) - self.start

    def dict(self) -> dict:
        return {"id": self.id, "name": self.name, "run_type": self.run_type,
                "inputs": _safe(self.inputs), "outputs": _safe(self.outputs),
                "error": str(self.error) if self.error else None,
                "latency": round(self.latency, 4), "parent_id": self.parent_id,
                "tags": self.tags, "metadata": self.metadata,
                "children": [c.dict() for c in self.children],
                "feedback": self.feedback}


def _safe(x, depth: int = 0):
    if depth > 3 or x is None or isinstance(x, (str, int, float, bool)):
        return x if not isinstance(x, str) or len(x) < 2000 else x[:2000]
    if isinstance(x, dict):
        return {str(k)[:80]: _safe(v, depth + 1) for k, v in list(x.items())[:20]}
    if isinstance(x, (list, tuple)):
        return [_safe(v, depth + 1) for v in list(x)[:20]]
    if hasattr(x, "content"):
        return _safe(str(getattr(x, "content"))[:2000], depth + 1)
    try:
        s = str(x)
        return s[:2000]
    except Exception:
        return "?"


def _ensure_worker():
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        _worker_started = True
        t = threading.Thread(target=_worker, daemon=True)
        t.start()


def _worker():
    buf: list[dict] = []
    while True:
        try:
            item = _Q.get(timeout=2.0)
            if item is None:  # إغلاق
                _flush(buf)
                return
            buf.append(item)
            while len(buf) < _FLUSH_N:
                try:
                    buf.append(_Q.get_nowait())
                except queue.Empty:
                    break
            if len(buf) >= _FLUSH_N or True:
                _flush(buf)
                buf = []
        except queue.Empty:
            if buf:
                _flush(buf)
                buf = []


def _flush(buf: list[dict]):
    if not buf:
        return
    try:
        with open(_STORE, "a", encoding="utf-8") as f:
            for r in buf:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
    # إرسال شبكي best-effort إن وُجد endpoint (لا يحجب أبداً)
    ep = os.environ.get("PYSMITH_ENDPOINT", "")
    if ep:
        try:
            import urllib.request
            body = json.dumps(buf, default=str).encode()
            req = urllib.request.Request(ep, data=body, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            pass


def _log_run(run: Run):
    if not _ENABLED:
        return
    _ensure_worker()
    try:
        _Q.put_nowait(run.dict())
    except Exception:
        pass


def flush():
    if not _ENABLED:
        return
    try:
        while not _Q.empty():
            time.sleep(0.01)
        time.sleep(0.05)
    except Exception:
        pass


def traceable(func=None, *, name: str | None = None, run_type: str = "chain",
              tags: list | None = None, metadata: dict | None = None):
    """مزخرف مطابق لـ @traceable في langsmith. صفر تكلفة تقريباً عند التعطيل."""
    def deco(fn):
        fname = name or fn.__name__
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def aw(*a, **k):
                if not _ENABLED:
                    return await fn(*a, **k)
                parent = _current.get()
                run = Run(fname, {"args": _safe(a), "kwargs": _safe(k)},
                          run_type, tags, metadata, parent.id if parent else None)
                tok = _current.set(run)
                try:
                    out = await fn(*a, **k)
                    run.outputs = out
                    return out
                except Exception as e:
                    run.error = e
                    raise
                finally:
                    run.end = time.time()
                    _current.reset(tok)
                    if parent is not None:
                        parent.children.append(run)
                    else:
                        _log_run(run)
            return aw

        @functools.wraps(fn)
        def w(*a, **k):
            if not _ENABLED:
                return fn(*a, **k)
            parent = _current.get()
            run = Run(fname, {"args": _safe(a), "kwargs": _safe(k)},
                      run_type, tags, metadata, parent.id if parent else None)
            tok = _current.set(run)
            try:
                out = fn(*a, **k)
                run.outputs = out
                return out
            except Exception as e:
                run.error = e
                raise
            finally:
                run.end = time.time()
                _current.reset(tok)
                if parent is not None:
                    parent.children.append(run)
                else:
                    _log_run(run)
        return w
    return deco(func) if func else deco


# اسم توافقي
trace = traceable

__all__ = ["Run", "traceable", "trace", "current_run", "flush", "is_enabled", "set_enabled"]
