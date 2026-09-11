"""Memory — Buffer / Window / Summary (مطابق لـ langchain.memory)."""
from __future__ import annotations

from collections import deque

from .messages import AIMessage, BaseMessage, HumanMessage, get_buffer_string


class ConversationBufferMemory:
    __slots__ = ("messages", "return_messages", "memory_key", "input_key")

    def __init__(self, return_messages: bool = False, memory_key: str = "history", input_key: str = "input"):
        self.messages: list[BaseMessage] = []
        self.return_messages = return_messages
        self.memory_key = memory_key
        self.input_key = input_key

    def save_context(self, inputs: dict, outputs: dict):
        iv = inputs.get(self.input_key, next(iter(inputs.values()), ""))
        ov = outputs.get("output", next(iter(outputs.values()), ""))
        self.messages.append(HumanMessage(str(iv)))
        self.messages.append(AIMessage(str(ov)))

    def load_memory_variables(self, inputs: dict | None = None) -> dict:
        if self.return_messages:
            return {self.memory_key: list(self.messages)}
        return {self.memory_key: get_buffer_string(self.messages)}

    def clear(self):
        self.messages.clear()


class ConversationBufferWindowMemory(ConversationBufferMemory):
    __slots__ = ("k",)

    def __init__(self, k: int = 5, **kw):
        super().__init__(**kw)
        self.k = k

    def load_memory_variables(self, inputs=None) -> dict:
        msgs = self.messages[-(self.k * 2):]
        if self.return_messages:
            return {self.memory_key: list(msgs)}
        return {self.memory_key: get_buffer_string(msgs)}


class ConversationSummaryMemory:
    """ملخص سريع بدون LLM ثقيل: يحتفظ بآخر N ثم يلخص القديم بقص ذكي."""

    __slots__ = ("llm", "messages", "summary", "max_messages", "memory_key")

    def __init__(self, llm=None, max_messages: int = 20, memory_key: str = "history"):
        self.llm = llm
        self.messages: list[BaseMessage] = []
        self.summary = ""
        self.max_messages = max_messages
        self.memory_key = memory_key

    def save_context(self, inputs: dict, outputs: dict):
        iv = next(iter(inputs.values()), "")
        ov = outputs.get("output", next(iter(outputs.values()), ""))
        self.messages.append(HumanMessage(str(iv)))
        self.messages.append(AIMessage(str(ov)))
        if len(self.messages) > self.max_messages:
            old = self.messages[:len(self.messages) - self.max_messages + 2]
            self.messages = self.messages[len(old):]
            if self.llm is not None:
                try:
                    out = self.llm.invoke(f"لخص بإيجاز:\n{get_buffer_string(old)}\nالملخص السابق: {self.summary}")
                    self.summary = out.content if hasattr(out, "content") else str(out)
                except Exception:
                    self.summary += " " + get_buffer_string(old)[:500]
            else:
                # تلخيص استخراجي سريع: أول 500 حرف
                self.summary += " " + get_buffer_string(old)[:500]

    def load_memory_variables(self, inputs=None) -> dict:
        text = (self.summary + "\n" + get_buffer_string(self.messages)).strip()
        return {self.memory_key: text}

    def clear(self):
        self.messages.clear()
        self.summary = ""


__all__ = ["ConversationBufferMemory", "ConversationBufferWindowMemory", "ConversationSummaryMemory"]
