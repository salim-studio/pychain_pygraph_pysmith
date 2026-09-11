"""Document — مطابق لـ langchain_core.documents."""
from __future__ import annotations


class Document:
    __slots__ = ("page_content", "metadata", "id")

    def __init__(self, page_content: str, metadata: dict | None = None, id: str | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}
        self.id = id

    def __repr__(self):
        return f"Document(len={len(self.page_content)}, metadata={self.metadata})"

    def __eq__(self, other):
        return (isinstance(other, Document) and self.page_content == other.page_content
                and self.metadata == other.metadata)

    def dict(self):
        return {"page_content": self.page_content, "metadata": self.metadata}


__all__ = ["Document"]
