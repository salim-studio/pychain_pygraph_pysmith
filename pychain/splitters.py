"""Text splitters — أسرع من langchain: تمريرة واحدة بدون recursion زائد."""
from __future__ import annotations

from .documents import Document


def _split_text_single_pass(text: str, chunk_size: int, chunk_overlap: int, separators: list[str]) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")
    if not text:
        return []
    # fast-path: نص أقصر من الحد
    if len(text) <= chunk_size:
        return [text]
    # نقسم بأول فاصل متاح (مثل langchain: \n\n ثم \n ثم space ثم "")
    chunks: list[str] = [text]
    for sep in separators:
        if sep == "":
            break
        new_chunks: list[str] = []
        for c in chunks:
            if len(c) <= chunk_size:
                new_chunks.append(c)
            else:
                parts = c.split(sep)
                cur = ""
                for p in parts:
                    piece = p if sep in ("\n\n", "\n", " ") else p
                    add = (sep + piece) if cur else piece
                    if len(cur) + len(add) <= chunk_size + chunk_overlap and cur:
                        cur += add
                    else:
                        if cur:
                            new_chunks.append(cur.strip() if sep.strip() == "" else cur)
                        cur = piece
                if cur:
                    new_chunks.append(cur)
        chunks = new_chunks
        if all(len(c) <= chunk_size for c in chunks):
            break
    # تقسيم الحروف للقطع المتبقية الطويلة + دمج overlap
    final: list[str] = []
    for c in chunks:
        if len(c) <= chunk_size:
            final.append(c)
        else:
            step = chunk_size - chunk_overlap
            for i in range(0, len(c), step):
                final.append(c[i:i + chunk_size])
                if i + chunk_size >= len(c):
                    break
    # إزالة الفارغ
    return [c for c in final if c]


class RecursiveCharacterTextSplitter:
    """مطابق لـ langchain.text_splitter.RecursiveCharacterTextSplitter."""

    __slots__ = ("chunk_size", "chunk_overlap", "separators", "length_function")

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 separators: list[str] | None = None, length_function=len):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self.length_function = length_function

    @classmethod
    def from_tiktoken_encoder(cls, chunk_size: int = 1000, chunk_overlap: int = 200, **kw):
        # توافق اسمي بدون tiktoken (نستخدم len كتقريب سريع)
        return cls(chunk_size, chunk_overlap, **kw)

    def split_text(self, text: str) -> list[str]:
        if self.length_function is not len:
            return self._split_generic(text)
        return _split_text_single_pass(text, self.chunk_size, self.chunk_overlap, self.separators)

    def _split_generic(self, text: str) -> list[str]:
        # طول مخصص: نستخدم نفس الخوارزمية لكن بوحدة length_function
        if not text:
            return []
        step = self.chunk_size - self.chunk_overlap
        # تقدير: نحول الطول لوحدة الحروف
        raw = _split_text_single_pass(text, self.chunk_size * 4, self.chunk_overlap * 4, self.separators)
        out: list[str] = []
        for c in raw:
            if self.length_function(c) <= self.chunk_size:
                out.append(c)
            else:
                # تقسيم خشن حسب نسبة الطول
                ratio = max(1, self.length_function(c) // self.chunk_size)
                size = max(1, len(c) // ratio)
                for i in range(0, len(c), max(1, size - self.chunk_overlap)):
                    out.append(c[i:i + size])
                    if i + size >= len(c):
                        break
        return [c for c in out if c]

    def split_documents(self, documents: list[Document]) -> list[Document]:
        out: list[Document] = []
        for d in documents:
            for chunk in self.split_text(d.page_content):
                out.append(Document(page_content=chunk, metadata=dict(d.metadata)))
        return out

    def split_text_with_meta(self, text: str, metadata: dict | None = None) -> list[Document]:
        return [Document(page_content=c, metadata=dict(metadata or {})) for c in self.split_text(text)]


class CharacterTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separator: str = "\n\n"):
        super().__init__(chunk_size, chunk_overlap, [separator])


__all__ = ["RecursiveCharacterTextSplitter", "CharacterTextSplitter"]
