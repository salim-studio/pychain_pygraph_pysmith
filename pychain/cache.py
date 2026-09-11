"""Cache — InMemory (LRU) + SQLite (مطابق لـ langchain.cache)."""
from __future__ import annotations

from collections import OrderedDict
from threading import RLock


class InMemoryCache:
    __slots__ = ("_d", "_lock", "maxsize")

    def __init__(self, maxsize: int = 1000):
        self._d: OrderedDict[str, str] = OrderedDict()
        self._lock = RLock()
        self.maxsize = maxsize

    def _key(self, prompt: str, llm_string: str = "") -> str:
        return f"{llm_string}||{prompt}"

    def lookup(self, prompt: str, llm_string: str = "") -> str | None:
        k = self._key(prompt, llm_string)
        with self._lock:
            if k in self._d:
                self._d.move_to_end(k)
                return self._d[k]
        return None

    def update(self, prompt: str, llm_string: str, value: str):
        k = self._key(prompt, llm_string)
        with self._lock:
            self._d[k] = value
            self._d.move_to_end(k)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def clear(self):
        with self._lock:
            self._d.clear()


class SQLiteCache(InMemoryCache):
    """واجهة مطابقة مع ثبات SQLite + طبقة LRU أمامية سريعة."""

    __slots__ = ("path", "_conn")

    def __init__(self, path: str = ".pychain_cache.db", maxsize: int = 1000):
        super().__init__(maxsize)
        self.path = path
        import sqlite3
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self._conn.commit()

    def lookup(self, prompt: str, llm_string: str = "") -> str | None:
        hit = super().lookup(prompt, llm_string)
        if hit is not None:
            return hit
        cur = self._conn.execute("SELECT v FROM cache WHERE k=?", (self._key(prompt, llm_string),))
        row = cur.fetchone()
        if row:
            super().update(prompt, llm_string, row[0])
            return row[0]
        return None

    def update(self, prompt: str, llm_string: str, value: str):
        super().update(prompt, llm_string, value)
        self._conn.execute("INSERT OR REPLACE INTO cache (k,v) VALUES (?,?)",
                           (self._key(prompt, llm_string), value))
        self._conn.commit()


__all__ = ["InMemoryCache", "SQLiteCache"]
