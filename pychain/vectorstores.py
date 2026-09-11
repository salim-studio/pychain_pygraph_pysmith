"""VectorStores — InMemoryVectorStore ببحث cosine متجه (numpy إن توفر) + ثبات سريع."""
from __future__ import annotations

import heapq
import math

try:  # numpy اختياري لكنه يسرّع 10-50x
    import numpy as _np
    _HAS_NP = True
except Exception:  # pragma: no cover
    _np = None  # type: ignore
    _HAS_NP = False

from .documents import Document


def _cosine_lists(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class InMemoryVectorStore:
    """مطابق لسطح FAISS/Chroma الشائع: add_* / similarity_search / as_retriever."""

    __slots__ = ("embedding", "_docs", "_vecs", "_mat", "_norms")

    def __init__(self, embedding=None):
        from .embeddings import HashingEmbeddings
        self.embedding = embedding or HashingEmbeddings()
        self._docs: list[Document] = []
        self._vecs: list[list[float]] = []
        self._mat = None
        self._norms = None

    # -- بناء --
    @classmethod
    def from_texts(cls, texts: list[str], embedding=None, metadatas: list[dict] | None = None) -> "InMemoryVectorStore":
        vs = cls(embedding)
        vs.add_texts(texts, metadatas)
        return vs

    @classmethod
    def from_documents(cls, documents: list[Document], embedding=None) -> "InMemoryVectorStore":
        vs = cls(embedding)
        vs.add_documents(documents)
        return vs

    def _rebuild_matrix(self):
        if _HAS_NP and self._vecs:
            self._mat = _np.asarray(self._vecs, dtype=_np.float32)
            n = _np.linalg.norm(self._mat, axis=1, keepdims=True)
            n[n == 0] = 1.0
            self._mat = self._mat / n
        else:
            self._mat = None

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        vecs = self.embedding.embed_documents(texts)
        ids: list[str] = []
        for i, (t, v) in enumerate(zip(texts, vecs)):
            idx = len(self._docs)
            sid = f"doc-{idx}"
            self._docs.append(Document(page_content=t, metadata=dict(metadatas[i]) if metadatas else {}, id=sid))
            self._vecs.append(v)
            ids.append(sid)
        self._rebuild_matrix()
        return ids

    def add_documents(self, documents: list[Document]) -> list[str]:
        return self.add_texts([d.page_content for d in documents],
                              [d.metadata for d in documents])

    def __len__(self):
        return len(self._docs)

    # -- بحث --
    def similarity_search_with_score(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        if not self._docs:
            return []
        q = self.embedding.embed_query(query)
        k = min(k, len(self._docs))
        if _HAS_NP and self._mat is not None:
            qv = _np.asarray(q, dtype=_np.float32)
            n = float(_np.linalg.norm(qv)) or 1.0
            qv = qv / n
            scores = self._mat @ qv  # (N,)
            if k >= len(scores):
                idx = _np.argsort(-scores)
            else:
                part = _np.argpartition(-scores, k - 1)[:k]
                idx = part[_np.argsort(-scores[part])]
            return [(self._docs[int(i)], float(scores[int(i)])) for i in idx]
        scored = [(_cosine_lists(q, v), i) for i, v in enumerate(self._vecs)]
        top = heapq.nlargest(k, scored)
        return [(self._docs[i], s) for s, i in top]

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return [d for d, _ in self.similarity_search_with_score(query, k)]

    def max_marginal_relevance_search(self, query: str, k: int = 4, fetch_k: int = 20,
                                      lambda_mult: float = 0.5) -> list[Document]:
        cands = self.similarity_search_with_score(query, min(fetch_k, len(self._docs)))
        if not cands:
            return []
        selected: list[Document] = [cands[0][0]]
        sel_idx = [0]
        q = self.embedding.embed_query(query)
        cand_vecs = [self._vecs[self._docs.index(d)] for d, _ in cands]
        while len(selected) < min(k, len(cands)):
            best_i, best_s = -1, -1e18
            for i in range(len(cands)):
                if i in sel_idx:
                    continue
                rel = cands[i][1]
                div = max(_cosine_lists(cand_vecs[i], cand_vecs[j]) for j in sel_idx)
                score = lambda_mult * rel - (1 - lambda_mult) * div
                if score > best_s:
                    best_s, best_i = score, i
            sel_idx.append(best_i)
            selected.append(cands[best_i][0])
        return selected

    def as_retriever(self, k: int = 4):
        from .retrievers import VectorStoreRetriever
        return VectorStoreRetriever(self, k=k)

    # -- ثبات --
    def persist(self, path: str):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"docs": [{"page_content": d.page_content, "metadata": d.metadata} for d in self._docs],
                       "vecs": self._vecs}, f)

    @classmethod
    def load(cls, path: str, embedding=None) -> "InMemoryVectorStore":
        import json
        vs = cls(embedding)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for d, v in zip(data["docs"], data["vecs"]):
            vs._docs.append(Document(d["page_content"], d.get("metadata", {})))
            vs._vecs.append(v)
        vs._rebuild_matrix()
        return vs


# أسماء توافقية
FAISS = InMemoryVectorStore
Chroma = InMemoryVectorStore
VectorStore = InMemoryVectorStore

__all__ = ["InMemoryVectorStore", "FAISS", "Chroma", "VectorStore"]
