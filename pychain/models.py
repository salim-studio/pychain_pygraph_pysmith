"""Models — BaseChatModel / Fake / Echo / OpenAI / Ollama (مطابق لسطح langchain)."""
from __future__ import annotations

import os
from collections.abc import Iterator

from .messages import AIMessage, BaseMessage, HumanMessage, convert_to_messages, get_buffer_string
from .runnables import Runnable


class BaseChatModel(Runnable):
    """واجهة ChatModel الدنيا: invoke(messages)->AIMessage + batch/stream مجانية من Runnable."""

    def invoke(self, inputs, config: dict | None = None) -> AIMessage:
        msgs = convert_to_messages(inputs) if not isinstance(inputs, str) else [HumanMessage(inputs)]
        content = self._generate(msgs, config=config or {})
        if isinstance(content, AIMessage):
            return content
        return AIMessage(str(content))

    def _generate(self, messages: list[BaseMessage], config: dict) -> str | AIMessage:
        raise NotImplementedError

    def stream(self, inputs, config=None) -> Iterator[AIMessage]:
        msg = self.invoke(inputs, config)
        # بث كلمات (chunking) بدون تكلفة إضافية
        text = msg.content
        step = max(1, len(text) // 8)
        for i in range(0, len(text), step):
            yield AIMessage(text[i:i + step])

    def __or__(self, other):
        from .runnables import _coerce_runnable
        right = _coerce_runnable(other)
        # fast-path: model | parser
        from .runnables import RunnableSequence, RunnableLambda
        left = RunnableLambda(lambda x: self.invoke(x), name=type(self).__name__)
        return RunnableSequence([left, right])


class FakeListChatModel(BaseChatModel):
    """للاختبارات: يعيد ردوداً جاهزة بالتناوب (مثل langchain FakeListChatModel)."""

    __slots__ = ("responses", "_i")

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self._i = 0

    def _generate(self, messages, config):
        r = self.responses[self._i % len(self.responses)]
        self._i += 1
        return r


class EchoChatModel(BaseChatModel):
    """صدى سريع: يعيد آخر رسالة بشرية مع بادئة — مفيد للـ benchmarks."""

    __slots__ = ("prefix",)

    def __init__(self, prefix: str = "Echo: "):
        self.prefix = prefix

    def _generate(self, messages, config):
        last = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last = m.content
                break
        else:
            last = messages[-1].content if messages else ""
        return self.prefix + last


class OpenAIChatModel(BaseChatModel):
    """متوافق مع ChatOpenAI (urllib فقط — بدون مكتبة openai). يدعم temperature/model/api_key."""

    __slots__ = ("model", "api_key", "base_url", "temperature", "timeout", "_cache")

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None,
                 base_url: str = "https://api.openai.com/v1", temperature: float = 0.0,
                 timeout: int = 60):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self._cache: dict[str, str] = {}

    def _generate(self, messages, config):
        import json
        import urllib.request
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY missing — استخدم EchoChatModel/FakeListChatModel محلياً")
        key = get_buffer_string(messages) + f"|{self.model}|{self.temperature}"
        if key in self._cache:
            return self._cache[key]
        payload = {"model": self.model, "temperature": self.temperature,
                   "messages": [{"role": ("assistant" if isinstance(m, AIMessage) else
                                          "system" if type(m).__name__ == "SystemMessage" else "user"),
                                 "content": m.content} for m in messages]}
        req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Authorization": f"Bearer {self.api_key}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode())
        text = data["choices"][0]["message"]["content"]
        self._cache[key] = text
        return text


class OllamaChatModel(BaseChatModel):
    """دعم Ollama المحلي (http://localhost:11434) — سريع ومجاني."""

    __slots__ = ("model", "base_url", "timeout")

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434", timeout: int = 120):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _generate(self, messages, config):
        import json
        import urllib.request
        prompt = get_buffer_string(messages)
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(f"{self.base_url}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode()).get("response", "")


# أسماء توافقية مع langchain
ChatOpenAI = OpenAIChatModel
ChatOllama = OllamaChatModel
FakeChatModel = FakeListChatModel

__all__ = ["BaseChatModel", "FakeListChatModel", "EchoChatModel", "OpenAIChatModel",
           "OllamaChatModel", "ChatOpenAI", "ChatOllama", "FakeChatModel"]
