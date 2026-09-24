# Copyright (c) 2026 salim-slimani — MIT License
"""pygraph — بديل langgraph الأسرع (نفس الواجهة).

مثال:
    from pygraph import StateGraph, START, END, MemorySaver
    g = StateGraph(dict)
    g.add_node("a", lambda s: {"x": s.get("x", 0) + 1})
    g.add_node("b", lambda s: {"y": s["x"] * 2})
    g.add_edge(START, "a"); g.add_edge("a", "b"); g.add_edge("b", END)
    app = g.compile(checkpointer=MemorySaver())
    print(app.invoke({"x": 0}))

مصادر السرعة:
- حالة dict مباشرة بدون نسخ عميق إلا عند checkpoint
- fan-out متوازٍ عبر ThreadPool و fast-path للسلاسل الخطية
- reducers اختيارية بدون pydantic
"""
from __future__ import annotations

__version__ = "0.1.0"

from .graph import (END, START, Command, CompiledGraph, MessageGraph, Send,
                    StateGraph, add_messages)
from .checkpoint import Checkpoint, MemorySaver

__all__ = ["StateGraph", "MessageGraph", "CompiledGraph", "Command", "Send",
           "START", "END", "add_messages", "MemorySaver", "Checkpoint"]
