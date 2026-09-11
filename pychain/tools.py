"""Tools — @tool / Tool / StructuredTool (مطابق لـ langchain.tools)."""
from __future__ import annotations

import inspect


class Tool:
    __slots__ = ("name", "description", "func", "args_schema")

    def __init__(self, name: str, func, description: str = "", args_schema=None):
        self.name = name
        self.func = func
        self.description = description or (func.__doc__ or "")
        self.args_schema = args_schema

    def invoke(self, inputs) -> str:
        if isinstance(inputs, dict):
            try:
                return str(self.func(**inputs))
            except TypeError:
                pass
            # tool بوسيط واحد
            if len(inputs) == 1:
                return str(self.func(next(iter(inputs.values()))))
            return str(self.func(inputs))
        return str(self.func(inputs))

    run = invoke

    def __repr__(self):
        return f"Tool({self.name})"


StructuredTool = Tool


def tool(func=None, *, name: str | None = None, description: str | None = None):
    """مزخرف مطابق لـ @tool في langchain."""
    def deco(fn):
        sig = inspect.signature(fn)
        desc = description or (fn.__doc__ or "").strip()
        t = Tool(name or fn.__name__, fn, desc)
        t.args_schema = sig
        return t
    if func is not None:
        return deco(func)
    return deco


__all__ = ["Tool", "StructuredTool", "tool"]
