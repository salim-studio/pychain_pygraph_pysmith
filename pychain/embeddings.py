"""Embeddings — واجهة مطابقة + HashingEmbeddings السريع (بدون شبكة/بدون نماذج)."""
from __future__ import annotations

import hashlib
import math
import os


class BaseEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class FakeEmbeddings(BaseEmbeddings):
    __slots__ = ("size",)

    def __init__(self, size: int = 128):
        self.size = size

    def embed_query(self, text: str) -> list[float]:
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [((h >> (i % 60)) & 0xFF) / 255.0 for i in range(self.size)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


class HashingEmbeddings(BaseEmbeddings):
    """تضمين حتمي سريع: hashing-trick + تطبيع L2. يكفي للـ RAG المحلي والاختبارات.

    أسرع بعشرات المرات من استدعاء API؛ ودقته كافية للترتيب التقريبي.
    """

    __slots__ = ("size",)

    def __init__(self, size: int = 384):
        self.size = size

    def _embed_one(self, text: str) -> list[float]:
        size = self.size
        vec = [0.0] * size
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % size] += 1.0
            vec[(h >> 16) % size] += 0.5
        n = math.sqrt(sum(v * v for v in vec)) or 1.0
        inv = 1.0 / n
        for i in range(size):
            vec[i] *= inv
        return vec

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # دفعات متسلسلة سريعة (بدون threads لصغر العملية؛ GIL-friendly)
        return [self._embed_one(t) for t in texts]


class OpenAIEmbeddings(BaseEmbeddings):
    """متوافق مع langchain_openai.OpenAIEmbeddings (urllib فقط، بدون openai lib)."""

    __slots__ = ("model", "api_key", "base_url")

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None,
                 base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")

    def _post(self, texts: list[str]) -> list[list[float]]:
        import json
        import urllib.request
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY missing")
        body = json.dumps({"model": self.model, "input": texts}).encode()
        req = urllib.request.Request(f"{self.base_url}/embeddings", data=body,
                                     headers={"Authorization": f"Bearer {self.api_key}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())
        return [d["embedding"] for d in data["data"]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):  # دفعات 100
            out.extend(self._post(texts[i:i + 100]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._post([text])[0]


__all__ = ["BaseEmbeddings", "FakeEmbeddings", "HashingEmbeddings", "OpenAIEmbeddings"]
