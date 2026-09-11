"""Evaluate — مطابق لـ langsmith.evaluate (متوازٍ + مقاييس زمنية)."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor


class ExperimentResults:
    __slots__ = ("experiment", "results", "summary")

    def __init__(self, experiment: str, results: list[dict]):
        self.experiment = experiment
        self.results = results
        lat = [r["latency"] for r in results]
        scores: dict[str, list[float]] = {}
        for r in results:
            for k, v in (r.get("scores") or {}).items():
                if isinstance(v, (int, float)):
                    scores.setdefault(k, []).append(float(v))
        self.summary = {
            "n": len(results),
            "errors": sum(1 for r in results if r.get("error")),
            "avg_latency": round(sum(lat) / len(lat), 4) if lat else 0,
            "scores": {k: round(sum(v) / len(v), 4) for k, v in scores.items()},
        }

    def __repr__(self):
        return f"ExperimentResults({self.experiment}, {self.summary})"


def _run_one(target, evaluators, example: dict) -> dict:
    t0 = time.time()
    try:
        out = target(example.get("inputs", example))
        scores: dict = {}
        for ev in evaluators or []:
            try:
                r = ev(example, out) if callable(ev) else None
                if isinstance(r, dict):
                    scores.update(r)
                elif isinstance(r, (int, float)):
                    scores[getattr(ev, "__name__", "score")] = r
            except Exception as e:
                scores[getattr(ev, "__name__", "eval") + "_error"] = str(e)[:200]
        return {"example_id": example.get("id"), "outputs": out, "scores": scores,
                "latency": round(time.time() - t0, 4), "error": None}
    except Exception as e:
        return {"example_id": example.get("id"), "outputs": None, "scores": {},
                "latency": round(time.time() - t0, 4), "error": str(e)[:500]}


def evaluate(target, dataset=None, evaluators: list | None = None,
             experiment: str = "exp-1", max_workers: int = 8,
             client=None, **kw) -> ExperimentResults:
    """target: دالة(inputs)->outputs. dataset: اسم أو قائمة أمثلة."""
    if isinstance(dataset, str):
        from .client import Client as _C
        c = client or _C()
        examples = c.list_examples(dataset)
    elif isinstance(dataset, list):
        examples = dataset
    elif hasattr(dataset, "examples"):
        examples = dataset.examples
    else:
        examples = list(dataset or [])
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, max(1, len(examples))))) as ex:
        results = list(ex.map(lambda e: _run_one(target, evaluators, e), examples))
    return ExperimentResults(experiment, results)


# مقيّمات جاهزة
def exact_match(example: dict, outputs) -> dict:
    exp = example.get("outputs")
    if isinstance(exp, dict):
        exp = next(iter(exp.values()), "")
    got = outputs.get("output", outputs) if isinstance(outputs, dict) else outputs
    return {"exact_match": 1.0 if str(got).strip() == str(exp).strip() else 0.0}


def contains(expected_key: str = "answer"):
    def _ev(example: dict, outputs) -> dict:
        exp = (example.get("outputs") or {}).get(expected_key, "") if isinstance(example.get("outputs"), dict) else example.get("outputs", "")
        got = str(outputs)
        return {"contains": 1.0 if str(exp)[:40] in got or str(exp).strip() in got else 0.0}
    _ev.__name__ = "contains"
    return _ev


__all__ = ["evaluate", "ExperimentResults", "exact_match", "contains"]
