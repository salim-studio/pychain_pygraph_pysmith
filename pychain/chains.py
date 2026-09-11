"""Chains — LLMChain / Stuff / MapReduce / Refine / Sequential / RetrievalQA."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .documents import Document


def _render(template, inputs: dict) -> str:
    if hasattr(template, "format"):
        try:
            return template.format(**inputs)
        except Exception:
            return template.format(**{k: inputs.get(k, "") for k in getattr(template, "input_variables", inputs)})
    return str(template).format(**inputs)


def _to_messages(prompt_value) -> list:
    from .messages import convert_to_messages
    if isinstance(prompt_value, str):
        return convert_to_messages(prompt_value)
    if isinstance(prompt_value, list):
        return prompt_value
    return convert_to_messages(prompt_value)


class LLMChain:
    """مطابق لـ langchain.chains.LLMChain."""

    __slots__ = ("prompt", "llm", "output_key")

    def __init__(self, prompt, llm, output_key: str = "text"):
        self.prompt = prompt
        self.llm = llm
        self.output_key = output_key

    def invoke(self, inputs: dict) -> dict:
        if hasattr(self.prompt, "format_messages"):
            msgs = self.prompt.format_messages(**inputs)
        elif hasattr(self.prompt, "format"):
            msgs = _render(self.prompt, inputs)
        else:
            msgs = str(self.prompt)
        out = self.llm.invoke(msgs)
        text = out.content if hasattr(out, "content") else str(out)
        return {self.output_key: text}

    def run(self, *args, **kwargs) -> str:
        if args and not kwargs:
            kwargs = {"input": args[0]}
        if len(getattr(self.prompt, "input_variables", [])) == 1 and len(kwargs) == 1:
            pass
        return self.invoke(kwargs)[self.output_key]

    def batch(self, inputs_list: list[dict], max_workers: int = 8) -> list[dict]:
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(inputs_list)))) as ex:
            return list(ex.map(self.invoke, inputs_list))

    def __or__(self, other):
        from .runnables import RunnableLambda, _coerce_runnable
        left = RunnableLambda(lambda x: self.invoke(x if isinstance(x, dict) else {"input": x}), name="LLMChain")
        return left | _coerce_runnable(other)


class StuffDocumentsChain:
    __slots__ = ("llm", "prompt", "separator")

    def __init__(self, llm, prompt, separator: str = "\n\n"):
        self.llm = llm
        self.prompt = prompt
        self.separator = separator

    def invoke(self, inputs: dict) -> dict:
        docs: list[Document] = inputs.get("input_documents", inputs.get("documents", []))
        context = self.separator.join(d.page_content for d in docs)
        merged = dict(inputs)
        merged["context"] = context
        if hasattr(self.prompt, "format_messages"):
            msgs = self.prompt.format_messages(**{k: v for k, v in merged.items() if isinstance(v, str)})
        else:
            msgs = _render(self.prompt, {**merged, "context": context})
        out = self.llm.invoke(msgs)
        return {"output_text": out.content if hasattr(out, "content") else str(out)}


class MapReduceDocumentsChain:
    __slots__ = ("map_chain", "reduce_chain", "max_workers")

    def __init__(self, map_chain: LLMChain, reduce_chain: LLMChain, max_workers: int = 8):
        self.map_chain = map_chain
        self.reduce_chain = reduce_chain
        self.max_workers = max_workers

    def invoke(self, inputs: dict) -> dict:
        docs: list[Document] = inputs.get("input_documents", [])
        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(docs)))) as ex:
            mapped = list(ex.map(lambda d: self.map_chain.invoke({"input": d.page_content, **inputs})["text"]
                                 if "text" in self.map_chain.invoke.__code__.co_names else None, docs)) \
                if False else None
        # تنفيذ صريح أسرع وأوضح:
        def _one(d):
            try:
                return self.map_chain.invoke({"input": d.page_content, "context": d.page_content})["text"]
            except KeyError:
                r = self.map_chain.invoke({"input": d.page_content})
                return next(iter(r.values()))
        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(docs) or 1))) as ex:
            mapped = list(ex.map(_one, docs)) if docs else []
        combined = "\n\n".join(mapped)
        try:
            return self.reduce_chain.invoke({"input": combined, "context": combined})
        except Exception:
            return self.reduce_chain.invoke({"input": combined})


class RefineDocumentsChain:
    __slots__ = ("initial_chain", "refine_chain")

    def __init__(self, initial_chain: LLMChain, refine_chain: LLMChain):
        self.initial_chain = initial_chain
        self.refine_chain = refine_chain

    def invoke(self, inputs: dict) -> dict:
        docs: list[Document] = inputs.get("input_documents", [])
        if not docs:
            return {"output_text": ""}
        first = next(iter(self.initial_chain.invoke({"input": docs[0].page_content}).values()))
        for d in docs[1:]:
            try:
                first = next(iter(self.refine_chain.invoke(
                    {"existing_answer": first, "context": d.page_content}).values()))
            except Exception:
                first = next(iter(self.refine_chain.invoke({"input": first + "\n" + d.page_content}).values()))
        return {"output_text": first}


class SequentialChain:
    __slots__ = ("chains", "input_variables", "output_variables")

    def __init__(self, chains: list[LLMChain], input_variables: list[str], output_variables: list[str]):
        self.chains = chains
        self.input_variables = input_variables
        self.output_variables = output_variables

    def invoke(self, inputs: dict) -> dict:
        state = dict(inputs)
        for c in self.chains:
            state.update(c.invoke(state))
        return {k: state.get(k) for k in self.output_variables}


class RetrievalQA:
    """مطابق لـ RetrievalQA / create_stuff_documents_chain + create_retrieval_chain."""

    __slots__ = ("llm", "retriever", "prompt", "return_sources")

    def __init__(self, llm, retriever, prompt=None, return_sources: bool = False):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt
        self.return_sources = return_sources

    @classmethod
    def from_chain_type(cls, llm, retriever, chain_type: str = "stuff", return_source_documents: bool = False,
                        chain_type_kwargs: dict | None = None) -> "RetrievalQA":
        from .prompts import PromptTemplate
        prompt = (chain_type_kwargs or {}).get("prompt") or PromptTemplate.from_template(
            "أجب من السياق التالي:\n{context}\n\nالسؤال: {question}")
        return cls(llm, retriever, prompt, return_sources=return_source_documents)

    def invoke(self, inputs) -> dict:
        q = inputs["query"] if isinstance(inputs, dict) and "query" in inputs else \
            inputs.get("question", inputs.get("input", "")) if isinstance(inputs, dict) else str(inputs)
        docs = self.retriever.invoke(q) if hasattr(self.retriever, "invoke") else self.retriever(q)
        context = "\n\n".join(d.page_content for d in docs)
        tmpl = self.prompt
        if hasattr(tmpl, "format_messages"):
            try:
                msgs = tmpl.format_messages(context=context, question=q, input=q)
            except Exception:
                msgs = tmpl.format_messages(**{"context": context, "question": q})
        else:
            msgs = _render(tmpl, {"context": context, "question": q, "query": q, "input": q})
        out = self.llm.invoke(msgs)
        text = out.content if hasattr(out, "content") else str(out)
        res = {"result": text, "answer": text}
        if self.return_sources:
            res["source_documents"] = docs
        return res


def create_stuff_documents_chain(llm, prompt) -> StuffDocumentsChain:
    return StuffDocumentsChain(llm, prompt)


def create_retrieval_chain(retriever, combine_chain) -> object:
    class _RC:
        def invoke(self, inputs):
            q = inputs.get("input", inputs.get("question", ""))
            docs = retriever.invoke(q)
            ans = combine_chain.invoke({**inputs, "input_documents": docs, "context": "\n\n".join(d.page_content for d in docs)})
            return {"answer": ans.get("output_text", next(iter(ans.values()))), "context": docs}
    return _RC()


__all__ = ["LLMChain", "StuffDocumentsChain", "MapReduceDocumentsChain", "RefineDocumentsChain",
           "SequentialChain", "RetrievalQA", "create_stuff_documents_chain", "create_retrieval_chain"]
