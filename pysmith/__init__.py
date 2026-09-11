"""pysmith — بديل langsmith الأسرع (تتبع محلي أولاً + تقييم متوازٍ).

مثال:
    from pysmith import traceable, Client, evaluate

    @traceable(name="qa")
    def answer(q: str) -> str:
        return "الإجابة: " + q

    c = Client()
    c.create_dataset("qa-v1")
    c.create_examples("qa-v1", [{"inputs": {"q": "مرحبا"}, "outputs": {"a": "مرحبا"}}])
    print(evaluate(lambda inp: {"a": inp["q"]}, "qa-v1"))

التحكم:
- PYSMITH_TRACING=false لتعطيل التتبع (تكلفة ~صفر)
- PYSMITH_STORE=.pysmith_runs.jsonl مسار التخزين
- PYSMITH_ENDPOINT=http://... لإرسال best-effort في الخلفية
"""
from __future__ import annotations

__version__ = "0.1.0"

from .tracing import (Run, current_run, flush, is_enabled, set_enabled, trace,
                      traceable)
from .client import Client
from .evaluate import ExperimentResults, contains, evaluate, exact_match

__all__ = ["traceable", "trace", "Run", "current_run", "flush",
           "is_enabled", "set_enabled", "Client",
           "evaluate", "ExperimentResults", "exact_match", "contains"]
