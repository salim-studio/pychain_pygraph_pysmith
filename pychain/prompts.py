"""Prompts — PromptTemplate / ChatPromptTemplate / FewShot (أسرع: cache للمتغيرات + format واحد)."""
from __future__ import annotations

import re
from string import Formatter

from .messages import (AIMessage, BaseMessage, ChatMessage, HumanMessage,
                       SystemMessage, convert_to_messages)

_FORMAT_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _extract_vars(template: str) -> list[str]:
    # أسرع من Formatter().parse في الحلقات الساخنة + يحافظ على الترتيب بدون تكرار
    seen: list[str] = []
    seen_set: set[str] = set()
    for m in _FORMAT_RE.finditer(template):
        v = m.group(1)
        if v not in seen_set:
            seen_set.add(v)
            seen.append(v)
    return seen


class PromptTemplate:
    """مطابق لـ langchain.prompts.PromptTemplate."""

    __slots__ = ("template", "input_variables", "_formatter")

    def __init__(self, template: str, input_variables: list[str] | None = None):
        self.template = template
        self.input_variables = input_variables if input_variables is not None else _extract_vars(template)

    @classmethod
    def from_template(cls, template: str) -> "PromptTemplate":
        return cls(template)

    def format(self, **kwargs) -> str:
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            missing = [v for v in self.input_variables if v not in kwargs]
            raise KeyError(f"Missing prompt variables: {missing or e}") from e

    def invoke(self, inputs: dict) -> str:
        return self.format(**inputs)

    def partial(self, **kwargs) -> "PromptTemplate":
        # دمج جزئي سريع بدون نسخ زائد
        new_template = self.template
        for k, v in kwargs.items():
            new_template = new_template.replace("{" + k + "}", str(v))
        remaining = [x for x in self.input_variables if x not in kwargs]
        return PromptTemplate(new_template, remaining)

    # LCEL: prompt | model
    def __or__(self, other):
        from .runnables import RunnableLambda, _coerce_runnable
        right = _coerce_runnable(other)
        left = RunnableLambda(lambda x: self.format(**(x if isinstance(x, dict) else {"input": x})),
                              name="PromptTemplate")
        return left | right

    def __repr__(self):
        return f"PromptTemplate(vars={self.input_variables})"


class ChatPromptTemplate:
    """مطابق لـ ChatPromptTemplate مع دعم from_messages / from_template."""

    __slots__ = ("messages", "input_variables")

    def __init__(self, messages: list):
        self.messages = list(messages)
        vars_: list[str] = []
        seen: set[str] = set()
        for m in self.messages:
            if isinstance(m, PromptTemplate):
                for v in m.input_variables:
                    if v not in seen:
                        seen.add(v)
                        vars_.append(v)
            elif isinstance(m, str):
                for v in _extract_vars(m):
                    if v not in seen:
                        seen.add(v)
                        vars_.append(v)
            elif isinstance(m, (tuple, list)) and len(m) == 2:
                for v in _extract_vars(str(m[1])):
                    if v not in seen:
                        seen.add(v)
                        vars_.append(v)
        self.input_variables = vars_

    @classmethod
    def from_messages(cls, messages: list) -> "ChatPromptTemplate":
        return cls(messages)

    @classmethod
    def from_template(cls, template: str) -> "ChatPromptTemplate":
        return cls([("human", template)])

    def format_messages(self, **kwargs) -> list[BaseMessage]:
        out: list[BaseMessage] = []
        for m in self.messages:
            if isinstance(m, BaseMessage):
                out.append(m)
            elif isinstance(m, PromptTemplate):
                out.append(HumanMessage(m.format(**kwargs)))
            elif isinstance(m, str):
                out.append(HumanMessage(m.format(**kwargs)))
            elif isinstance(m, (tuple, list)) and len(m) == 2:
                role, tmpl = m
                text = str(tmpl).format(**kwargs) if kwargs else str(tmpl)
                r = str(role).lower()
                if r in ("human", "user"):
                    out.append(HumanMessage(text))
                elif r in ("ai", "assistant"):
                    out.append(AIMessage(text))
                elif r == "system":
                    out.append(SystemMessage(text))
                else:
                    out.append(ChatMessage(text, role=str(role)))
            else:
                out.append(HumanMessage(str(m)))
        return out

    def invoke(self, inputs: dict) -> list[BaseMessage]:
        return self.format_messages(**inputs)

    def partial(self, **kwargs) -> "ChatPromptTemplate":
        new_msgs = []
        for m in self.messages:
            if isinstance(m, (tuple, list)) and len(m) == 2:
                role, tmpl = m
                t = str(tmpl)
                for k, v in kwargs.items():
                    t = t.replace("{" + k + "}", str(v))
                new_msgs.append((role, t))
            else:
                new_msgs.append(m)
        return ChatPromptTemplate(new_msgs)

    def __or__(self, other):
        from .runnables import RunnableLambda, _coerce_runnable
        right = _coerce_runnable(other)
        left = RunnableLambda(lambda x: self.format_messages(**(x if isinstance(x, dict) else {})), name="ChatPrompt")
        return left | right


class FewShotPromptTemplate(PromptTemplate):
    __slots__ = ("examples", "example_prompt", "prefix", "suffix", "separator")

    def __init__(self, examples: list[dict], example_prompt: PromptTemplate,
                 prefix: str = "", suffix: str = "", separator: str = "\n\n",
                 input_variables: list[str] | None = None):
        self.examples = examples
        self.example_prompt = example_prompt
        self.prefix = prefix
        self.suffix = suffix
        self.separator = separator
        template = prefix + separator + "SUFFIX"
        super().__init__(template, input_variables or _extract_vars(suffix))

    def format(self, **kwargs) -> str:
        ex = self.separator.join(self.example_prompt.format(**e) for e in self.examples)
        parts = [p for p in [self.prefix, ex, self.suffix.format(**kwargs)] if p]
        return self.separator.join(parts)


__all__ = ["PromptTemplate", "ChatPromptTemplate", "FewShotPromptTemplate"]
