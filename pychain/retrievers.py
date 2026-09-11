"""Retrievers."""
from __future__ import annotations

from .documents import Document


class BaseRetriever:
    def invoke(self, query: str) -> list[Document]:
        raise NotImplementedError

    def batch(self, queries: list[str]) -> list[list[Document]]:
        return [self.invoke(q) for q in queries]

    def __or__(self, other):
        from .runnables import _coerce_runnable, RunnableLambda, RunnableSequence
        left = RunnableLambda(lambda x: self.invoke(x if isinstance(x, str) else str(x)), name="Retriever")
        return RunnableSequence([left, _coerce_runnable(other)])


class VectorStoreRetriever(BaseRetriever):
    __slots__ = ("store", "k")

    def __init__(self, store, k: int = 4):
        self.store = store
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        if isinstance(query, dict):
            query = query.get("input", query.get("question", str(query)))
        return self.store.similarity_search(str(query), k=self.k)


__all__ = ["BaseRetriever", "VectorStoreRetriever"]
