"""Client — مطابق لسطح langsmith.Client (محلي أولاً: JSON + بحث سريع)."""
from __future__ import annotations

import json
import os
import time
import uuid
from threading import RLock


class Client:
    """Client محلي: datasets / examples / runs / feedback — بدون شبكة.

    التخزين: .pysmith/datasets.json + runs في .pysmith_runs.jsonl (عبر tracing).
    إن وُجد PYSMITH_API_KEY و PYSMITH_ENDPOINT تُرسل best-effort في الخلفية.
    """

    def __init__(self, base_dir: str = ".pysmith", api_key: str | None = None,
                 endpoint: str | None = None):
        self.base_dir = base_dir
        self.api_key = api_key or os.environ.get("PYSMITH_API_KEY", "")
        self.endpoint = (endpoint or os.environ.get("PYSMITH_ENDPOINT", "")).rstrip("/")
        self._lock = RLock()
        os.makedirs(base_dir, exist_ok=True)
        self._ds_path = os.path.join(base_dir, "datasets.json")
        if not os.path.exists(self._ds_path):
            with open(self._ds_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    # -- datasets --
    def _load_ds(self) -> dict:
        with open(self._ds_path, encoding="utf-8") as f:
            return json.load(f)

    def _save_ds(self, d: dict):
        tmp = self._ds_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self._ds_path)

    def create_dataset(self, name: str, description: str = "") -> dict:
        with self._lock:
            d = self._load_ds()
            if name not in d:
                d[name] = {"id": uuid.uuid4().hex[:12], "name": name,
                           "description": description, "examples": [],
                           "created_at": time.time()}
                self._save_ds(d)
            return d[name]

    def list_datasets(self) -> list[dict]:
        return list(self._load_ds().values())

    def create_examples(self, dataset_name: str, examples: list[dict]):
        """examples: [{inputs: {...}, outputs: {...}, metadata?}]"""
        with self._lock:
            d = self._load_ds()
            ds = d.get(dataset_name) or self.create_dataset(dataset_name)
            # إعادة تحميل بعد الإنشاء المحتمل
            d = self._load_ds()
            ds = d[dataset_name]
            for e in examples:
                ds["examples"].append({"id": uuid.uuid4().hex[:12], **e})
            self._save_ds(d)

    create_example = create_examples

    def list_examples(self, dataset_name: str) -> list[dict]:
        return self._load_ds().get(dataset_name, {}).get("examples", [])

    # -- runs / feedback (محلي) --
    def list_runs(self, limit: int = 50) -> list[dict]:
        from . import tracing as _t
        path = os.environ.get("PYSMITH_STORE", ".pysmith_runs.jsonl")
        if not os.path.exists(path):
            return []
        out: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out[-limit:]

    def create_feedback(self, run_id: str, key: str, score: float, comment: str = "") -> dict:
        fb = {"run_id": run_id, "key": key, "score": score, "comment": comment, "t": time.time()}
        path = os.path.join(self.base_dir, "feedback.jsonl")
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(fb, ensure_ascii=False) + "\n")
        return fb

    def flush(self):
        from .tracing import flush as _flush
        _flush()


__all__ = ["Client"]
