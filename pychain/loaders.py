"""Loaders — Text / Directory (مطابق لسطح langchain.document_loaders)."""
from __future__ import annotations

import os

from .documents import Document


class TextLoader:
    __slots__ = ("path", "encoding")

    def __init__(self, path: str, encoding: str = "utf-8"):
        self.path = path
        self.encoding = encoding

    def load(self) -> list[Document]:
        with open(self.path, encoding=self.encoding) as f:
            return [Document(f.read(), {"source": self.path})]


class DirectoryLoader:
    __slots__ = ("path", "glob", "loader_cls")

    def __init__(self, path: str, glob: str = "*.txt", loader_cls=TextLoader):
        self.path = path
        self.glob = glob
        self.loader_cls = loader_cls

    def load(self) -> list[Document]:
        import glob as _glob
        out: list[Document] = []
        for p in _glob.glob(os.path.join(self.path, self.glob)):
            try:
                out.extend(self.loader_cls(p).load())
            except Exception:
                continue
        return out


__all__ = ["TextLoader", "DirectoryLoader"]
