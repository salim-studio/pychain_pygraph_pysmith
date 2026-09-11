"""Runnables (LCEL) — Runnable / Lambda / Parallel / Passthrough / Branch + عامل | سريع."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor


def _coerce_runnable(x):
    from typing import TYPE_CHECKING as _TC  # noqa
    # import متأخر لتفادي الترتيب الدائري
    _Runnable = globals().get("Runnable")
    _Adapter = globals().get("_InvokeAdapter")
    _Lambda = globals().get("RunnableLambda")
    _Parallel = globals().get("RunnableParallel")
    if _Runnable is not None and isinstance(x, _Runnable):
        return x
    # duck-type: parsers / retrievers / أي كائن له invoke (توافق langchain)
    if hasattr(x, "invoke") and callable(getattr(x, "invoke")):
        if _Adapter is not None and not isinstance(x, _Runnable):  # type: ignore
            return _Adapter(x)
        return x
    if _Parallel is not None and isinstance(x, dict):
        return _Parallel(x)
    if callable(x):
        return _Lambda(x)
    raise TypeError(f"Cannot coerce to Runnable: {x!r}")


class Runnable:
    """الأساس — يدعم invoke/batch/stream + | و .pipe."""

    def invoke(self, inputs, config: dict | None = None):
        raise NotImplementedError

    def batch(self, inputs_list: list, config: dict | None = None, max_workers: int = 8) -> list:
        # تنفيذ متوازٍ حقيقي (ThreadPool) — أسرع من التسلسل في langchain للـ IO
        if len(inputs_list) <= 1:
            return [self.invoke(x, config) for x in inputs_list]
        with ThreadPoolExecutor(max_workers=min(max_workers, len(inputs_list))) as ex:
            return list(ex.map(lambda x: self.invoke(x, config), inputs_list))

    async def ainvoke(self, inputs, config: dict | None = None):
        return await asyncio.to_thread(self.invoke, inputs, config)

    async def abatch(self, inputs_list: list, config: dict | None = None) -> list:
        return await asyncio.gather(*[self.ainvoke(x, config) for x in inputs_list])

    def stream(self, inputs, config: dict | None = None):
        yield self.invoke(inputs, config)

    async def astream(self, inputs, config: dict | None = None):
        yield await self.ainvoke(inputs, config)

    def pipe(self, *others) -> "RunnableSequence":
        seq: list[Runnable] = [self]
        for o in others:
            seq.append(_coerce_runnable(o))
        return RunnableSequence(seq)

    def __or__(self, other) -> "RunnableSequence":
        right = _coerce_runnable(other)
        if isinstance(self, RunnableSequence):
            if isinstance(right, RunnableSequence):
                return RunnableSequence(self.steps + right.steps)
            return RunnableSequence(self.steps + [right])
        if isinstance(right, RunnableSequence):
            return RunnableSequence([self] + right.steps)
        return RunnableSequence([self, right])

    def __ror__(self, other) -> "RunnableSequence":
        return _coerce_runnable(other).__or__(self)


class RunnableLambda(Runnable):
    __slots__ = ("func", "name")

    def __init__(self, func, name: str | None = None):
        self.func = func
        self.name = name or getattr(func, "__name__", "lambda")

    def invoke(self, inputs, config=None):
        try:
            from pysmith import tracing as _t
            run = _t.current_run()
            if run is not None:
                _t._event("runnable", self.name, inputs)
        except Exception:
            pass
        return self.func(inputs)

    def __repr__(self):
        return f"RunnableLambda({self.name})"


class RunnableSequence(Runnable):
    __slots__ = ("steps",)

    def __init__(self, steps: list):
        self.steps = [_coerce_runnable(s) for s in steps]

    def invoke(self, inputs, config=None):
        val = inputs
        for s in self.steps:
            val = s.invoke(val, config)
        return val

    def stream(self, inputs, config=None):
        # بث المراحل الوسيطة إن دعمت
        val = inputs
        for i, s in enumerate(self.steps):
            if i == len(self.steps) - 1 and hasattr(s, "stream"):
                yield from s.stream(val, config)
            else:
                val = s.invoke(val, config)
        if not hasattr(self.steps[-1], "stream") or True:
            pass

    def __repr__(self):
        return " | ".join(getattr(s, "name", type(s).__name__) for s in self.steps)


class RunnableParallel(Runnable):
    """مطابق لـ RunnableParallel(dict) — ينفذ الفروع بالتوازي."""

    __slots__ = ("branches",)

    def __init__(self, branches: dict):
        self.branches = {k: _coerce_runnable(v) for k, v in branches.items()}

    def invoke(self, inputs, config=None):
        keys = list(self.branches)
        if len(keys) == 1:
            return {keys[0]: self.branches[keys[0]].invoke(inputs, config)}
        with ThreadPoolExecutor(max_workers=min(32, len(keys))) as ex:
            futs = {k: ex.submit(self.branches[k].invoke, inputs, config) for k in keys}
            return {k: futs[k].result() for k in keys}


class RunnablePassthrough(Runnable):
    __slots__ = ()

    def invoke(self, inputs, config=None):
        return inputs


class RunnableBranch(Runnable):
    __slots__ = ("branches", "default")

    def __init__(self, *branches, default=None):
        # branches: [(cond, runnable), ...]
        self.branches = [(c, _coerce_runnable(r)) for c, r in branches]
        self.default = _coerce_runnable(default) if default is not None else RunnablePassthrough()

    def invoke(self, inputs, config=None):
        for cond, r in self.branches:
            if cond(inputs):
                return r.invoke(inputs, config)
        return self.default.invoke(inputs, config)


class RunnableWithFallbacks(Runnable):
    __slots__ = ("primary", "fallbacks")

    def __init__(self, primary, fallbacks: list):
        self.primary = _coerce_runnable(primary)
        self.fallbacks = [_coerce_runnable(f) for f in fallbacks]

    def invoke(self, inputs, config=None):
        err = None
        try:
            return self.primary.invoke(inputs, config)
        except Exception as e:
            err = e
        for f in self.fallbacks:
            try:
                return f.invoke(inputs, config)
            except Exception as e:
                err = e
        raise err


def chain(func=None, *, name: str | None = None):
    """مزخرف @chain مثل langchain_core — يحول الدالة إلى RunnableLambda."""
    def deco(fn):
        return RunnableLambda(fn, name=name or fn.__name__)
    return deco(func) if func else deco


class _InvokeAdapter(Runnable):
    """مغلف لأي كائن له invoke (parsers/retrievers) ليقبل عامل |."""
    __slots__ = ("inner",)

    def __init__(self, inner):
        self.inner = inner

    def invoke(self, inputs, config=None):
        try:
            return self.inner.invoke(inputs, config)
        except TypeError:
            return self.inner.invoke(inputs)


__all__ = ["Runnable", "RunnableLambda", "RunnableSequence", "RunnableParallel",
           "RunnablePassthrough", "RunnableBranch", "RunnableWithFallbacks", "chain"]
