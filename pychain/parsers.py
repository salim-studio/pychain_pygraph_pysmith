"""Output parsers — Str / JSON / Pydantic-lite (بدون اعتماد ثقيل)."""
from __future__ import annotations

import json
import re

from .runnables import Runnable


class StrOutputParser(Runnable):
    def invoke(self, x, config=None) -> str:
        if hasattr(x, "content"):
            return str(x.content)
        if isinstance(x, dict) and len(x) == 1:
            return str(next(iter(x.values())))
        return str(x)

    def __or__(self, other):
        from .runnables import _coerce_runnable, RunnableLambda, RunnableSequence
        return RunnableSequence([RunnableLambda(self.invoke, name="StrOutputParser"), _coerce_runnable(other)])


class JsonOutputParser(Runnable):
    def invoke(self, x, config=None):
        text = x.content if hasattr(x, "content") else (next(iter(x.values())) if isinstance(x, dict) and len(x) == 1 else str(x))
        text = str(text).strip()
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.S)
            if m:
                return json.loads(m.group(1))
            i, j = text.find("{"), text.rfind("}")
            if i != -1 and j > i:
                return json.loads(text[i:j + 1])
            raise ValueError(f"Cannot parse JSON: {text[:200]}")


class PydanticOutputParser(Runnable):
    """متوافق شكلياً مع langchain.output_parsers.PydanticOutputParser باستخدام dataclass/pydantic إن توفر."""

    __slots__ = ("schema",)

    def __init__(self, pydantic_object):
        self.schema = pydantic_object

    def get_format_instructions(self) -> str:
        fields = getattr(self.schema, "model_fields", None) or {}
        if fields:
            inner = ", ".join(f'"{k}"' for k in fields)
            return f"أعد JSON صالحاً بالمفاتيح: {inner}"
        ann = getattr(self.schema, "__annotations__", {})
        return f"أعد JSON بالمفاتيح: {list(ann)}"

    def invoke(self, x, config=None):
        data = JsonOutputParser().invoke(x)
        try:
            if hasattr(self.schema, "model_validate"):
                return self.schema.model_validate(data)
            return self.schema(**data)
        except Exception:
            return data


__all__ = ["StrOutputParser", "JsonOutputParser", "PydanticOutputParser"]
