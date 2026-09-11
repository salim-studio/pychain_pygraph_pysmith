"""Tests for pychain / pygraph / pysmith."""
import os
import tempfile

import pychain as pc
from pychain import (AgentExecutor, ChatPromptTemplate, EchoChatModel,
                     FakeListChatModel, HashingEmbeddings, InMemoryVectorStore,
                     LLMChain, PromptTemplate, RecursiveCharacterTextSplitter,
                     StrOutputParser, Tool, tool)
from pygraph import END, START, MemorySaver, StateGraph
from pysmith import Client, evaluate, traceable


def test_prompts():
    p = PromptTemplate.from_template("مرحبا {name}")
    assert p.format(name="سامي") == "مرحبا سامي"
    c = ChatPromptTemplate.from_messages([("system", "أنت مساعد"), ("human", "hi {x}")])
    msgs = c.format_messages(x="there")
    assert msgs[0].content == "أنت مساعد" and msgs[1].content == "hi there"


def test_lcel_pipe():
    prompt = ChatPromptTemplate.from_template("قل: {text}")
    model = FakeListChatModel(responses=["hello"])
    chain = prompt | model | StrOutputParser()
    assert chain.invoke({"text": "x"}) == "hello"
    assert len(chain.batch([{"text": "a"}, {"text": "b"}])) == 2


def test_splitter():
    sp = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
    chunks = sp.split_text("أ " * 100)
    assert len(chunks) > 1 and all(len(c) <= 20 + 8 for c in chunks)
    docs = sp.split_documents([pc.Document("hello world " * 20)])
    assert docs


def test_vectorstore():
    vs = InMemoryVectorStore(embedding=HashingEmbeddings(size=64))
    vs.add_texts(["القط يحب السمك", "السيارة سريعة", "الذكاء الاصطناعي"])
    res = vs.similarity_search("القط والسمك", k=1)
    assert "القط" in res[0].page_content
    r = vs.as_retriever(k=2)
    assert len(r.invoke("سيارة")) == 2


def test_retrieval_qa():
    vs = InMemoryVectorStore.from_texts(["بغداد عاصمة العراق", "باريس عاصمة فرنسا"])
    qa = pc.RetrievalQA.from_chain_type(EchoChatModel(), vs.as_retriever(k=1))
    out = qa.invoke({"query": "ما عاصمة العراق؟"})
    assert "result" in out


def test_llm_chain_and_tools():
    @tool
    def add(a: int, b: int) -> int:
        """جمع رقمين"""
        return a + b
    assert add.invoke({"a": 2, "b": 3}) == "5"
    llm = FakeListChatModel(responses=["Final Answer: 5"])
    ex = AgentExecutor(llm=llm, tools=[add])
    assert "5" in ex.invoke({"input": "كم 2+3؟"})["output"]


def test_memory_cache_parsers():
    m = pc.ConversationBufferMemory()
    m.save_context({"input": "hi"}, {"output": "hello"})
    assert "hi" in m.load_memory_variables()["history"]
    c = pc.InMemoryCache()
    c.update("p", "llm", "v")
    assert c.lookup("p", "llm") == "v"
    assert StrOutputParser().invoke(EchoChatModel().invoke("hi")) == "Echo: hi"
    assert pc.JsonOutputParser().invoke('{"a": 1}') == {"a": 1}


def test_pygraph_linear_and_branch():
    g = StateGraph(dict)
    g.add_node("a", lambda s: {"x": s.get("x", 0) + 1})
    g.add_node("b", lambda s: {"y": s["x"] * 2})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    app = g.compile(checkpointer=MemorySaver())
    assert app.invoke({"x": 1}) == {"x": 2, "y": 4}

    g2 = StateGraph(dict)
    g2.add_node("start", lambda s: {"n": 5})
    g2.add_node("even", lambda s: {"r": "even"})
    g2.add_node("odd", lambda s: {"r": "odd"})
    g2.set_entry_point("start")
    g2.add_conditional_edges("start", lambda s: "even" if s["n"] % 2 == 0 else "odd")
    app2 = g2.compile()
    assert app2.invoke({})["r"] == "odd"
    # checkpointer resume
    tid = {"configurable": {"thread_id": "t1"}}
    app.invoke({"x": 10}, config=tid)
    assert app.get_state(tid)["x"] >= 10


def test_pysmith():
    @traceable(name="demo")
    def f(x: int) -> int:
        return x * 2
    assert f(21) == 42
    with tempfile.TemporaryDirectory() as d:
        c = Client(base_dir=d)
        c.create_dataset("ds")
        c.create_examples("ds", [{"inputs": {"q": "a"}, "outputs": {"a": "a"}}])
        assert len(c.list_examples("ds")) == 1
        res = evaluate(lambda inp: {"a": inp["q"]}, "ds", client=c)
        assert res.summary["n"] == 1
