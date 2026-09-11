# pychain · pygraph · pysmith

بدائل أسرع وخفيفة لـ `langchain` و `langgraph` و `langsmith` — نفس الواجهات، بدون اعتماديات ثقيلة (stdlib فقط + `numpy` اختياري).

| هذه المكتبة | تطابق | السرعة من |
|---|---|---|
| `pychain` | `langchain` / `langchain_core` | `slots` بدون pydantic، `batch` متوازٍ، بحث متجه واحد `matmul`، تقسيم نص بتمريرة واحدة |
| `pygraph` | `langgraph` | حالة `dict` بدون نسخ عميق إلا عند checkpoint، تفرع متوازٍ، مسار سريع للسلاسل الخطية |
| `pysmith` | `langsmith` | تتبع محلي أولاً (JSONL + خيط خلفي غير حاجب)، تقييم متوازٍ، تكلفة ~صفر عند التعطيل |

## التثبيت

```bash
pip install -e .
# أو مباشرة من GitHub
pip install git+https://github.com/salim-studio/pychain_pygraph_pysmith.git
```

## أمثلة

```python
# pychain — سلسلة LCEL
from pychain import ChatPromptTemplate, FakeListChatModel, StrOutputParser

chain = (
    ChatPromptTemplate.from_template("ترجم إلى الفرنسية: {text}")
    | FakeListChatModel(responses=["Bonjour"])
    | StrOutputParser()
)
print(chain.invoke({"text": "صباح الخير"}))
```

```python
# pychain — RAG محلي بالكامل (بدون مفاتيح API)
from pychain import InMemoryVectorStore, RetrievalQA, EchoChatModel

vs = InMemoryVectorStore.from_texts(["بغداد عاصمة العراق", "باريس عاصمة فرنسا"])
qa = RetrievalQA.from_chain_type(EchoChatModel(), vs.as_retriever(k=1))
print(qa.invoke({"query": "ما عاصمة العراق؟"})["result"])
```

```python
# pygraph — رسم حالة
from pygraph import StateGraph, START, END, MemorySaver

g = StateGraph(dict)
g.add_node("a", lambda s: {"x": s.get("x", 0) + 1})
g.add_node("b", lambda s: {"y": s["x"] * 2})
g.add_edge(START, "a")
g.add_edge("a", "b")
g.add_edge("b", END)

app = g.compile(checkpointer=MemorySaver())
print(app.invoke({"x": 1}))  # {'x': 2, 'y': 4}
```

```python
# pysmith — تتبع وتقييم محلي
from pysmith import traceable, Client, evaluate

@traceable(name="qa")
def answer(q: str) -> str:
    return "الإجابة: " + q

c = Client()
c.create_dataset("qa-v1")
c.create_examples("qa-v1", [{"inputs": {"q": "مرحبا"}, "outputs": {"a": "مرحبا"}}])
print(evaluate(lambda inp: {"a": inp["q"]}, "qa-v1", client=c).summary)
```

## ضبط pysmith عبر البيئة

| متغير | الافتراضي | الوصف |
|---|---|---|
| `PYSMITH_TRACING` | `true` | `false` لتعطيل التتبع (تكلفة ~صفر) |
| `PYSMITH_STORE` | `.pysmith_runs.jsonl` | مسار تخزين التشغيلات |
| `PYSMITH_ENDPOINT` | فارغ | إرسال best-effort في الخلفية إن وُجد |

## الاختبارات

```bash
pip install pytest numpy
python -m pytest tests/ -q
```

## التوافق مع الأصل

- `pychain`: `PromptTemplate`، `ChatPromptTemplate`، `Document`، `RecursiveCharacterTextSplitter`، `HashingEmbeddings`/`FakeEmbeddings`/`OpenAIEmbeddings`، `InMemoryVectorStore` (بأسماء `FAISS`/`Chroma`)، `Runnable*` + عامل `|`، `BaseChatModel`/`ChatOpenAI`/`ChatOllama`، `LLMChain`/`RetrievalQA`، `Tool`/`@tool`، `AgentExecutor`، `ConversationBufferMemory`، `Str/Json/PydanticOutputParser`، `InMemoryCache`/`SQLiteCache`.
- `pygraph`: `StateGraph`/`MessageGraph`، `START`/`END`، `Command`/`Send`، `MemorySaver`، `invoke`/`stream`/`batch`/`get_state`/`update_state`.
- `pysmith`: `@traceable`، `Client` (datasets/examples/runs/feedback)، `evaluate`.
