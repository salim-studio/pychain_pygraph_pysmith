"""Agents — ReAct AgentExecutor سريع (تنفيذ أدوات متوازٍ + حد تكرارات)."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from .messages import AIMessage

_ACTION_RE = re.compile(r"Action\s*:\s*(\w+)\s*\nAction Input\s*:\s*(.+)", re.S | re.I)
_FINAL_RE = re.compile(r"Final Answer\s*:\s*(.+)", re.S | re.I)

_REACT_SUFFIX = """أجب بالصيغة التالية حصراً:

Thought: تفكيرك
Action: اسم_الأداة
Action Input: المدخلات
Observation: (ستُملأ تلقائياً)

أو عند الانتهاء:
Thought: انتهيت
Final Answer: الإجابة النهائية

الأدوات المتاحة:
{tools}

السؤال: {input}
{scratchpad}"""


class AgentExecutor:
    __slots__ = ("llm", "tools", "max_iterations", "verbose", "_by_name")

    def __init__(self, llm, tools: list, max_iterations: int = 6, verbose: bool = False):
        self.llm = llm
        self.tools = list(tools)
        self.max_iterations = max_iterations
        self.verbose = verbose
        self._by_name = {t.name: t for t in self.tools}

    @classmethod
    def from_agent_and_tools(cls, agent=None, tools=None, llm=None, **kw) -> "AgentExecutor":
        return cls(llm=llm or (agent.llm if agent is not None else None), tools=tools or [], **kw)

    @classmethod
    def create_react_agent(cls, llm, tools: list, prompt=None) -> dict:
        # توافق شكلي مع create_react_agent في langchain (يعيد واصفاً)
        return {"llm": llm, "tools": tools, "prompt": prompt, "type": "react"}

    def _tool_desc(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools)

    def invoke(self, inputs) -> dict:
        question = inputs["input"] if isinstance(inputs, dict) else str(inputs)
        scratch = ""
        for step in range(self.max_iterations):
            prompt = _REACT_SUFFIX.format(tools=self._tool_desc(), input=question, scratchpad=scratch)
            out = self.llm.invoke(prompt)
            text = out.content if isinstance(out, AIMessage) else str(out)
            if self.verbose:
                print(f"[step {step}] {text[:300]}")
            m_final = _FINAL_RE.search(text)
            # إن لم توجد أدوات: أعد النص مباشرة (fast-path للموديلات البسيطة)
            m_act = _ACTION_RE.search(text)
            if m_final and not m_act:
                return {"output": m_final.group(1).strip()}
            if m_act:
                name, arg = m_act.group(1).strip(), m_act.group(2).strip()
                tool = self._by_name.get(name)
                if tool is None:
                    obs = f"Tool '{name}' غير موجودة."
                else:
                    try:
                        obs = tool.invoke(arg)
                    except Exception as e:
                        obs = f"خطأ الأداة: {e}"
                scratch += f"\n{text}\nObservation: {obs}\n"
                # إن كانت هذه آخر تكرار أعد الملاحظة
                if step == self.max_iterations - 1:
                    return {"output": obs}
            else:
                # لا فعل ولا إجابة نهائية: أعد النص كما هو (متوافق مع Echo/Fake)
                return {"output": text.strip()}
        return {"output": scratch[-2000:]}

    def batch(self, inputs_list: list, max_workers: int = 8) -> list[dict]:
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(inputs_list)))) as ex:
            return list(ex.map(self.invoke, inputs_list))


def create_react_agent(llm, tools, prompt=None) -> dict:
    return AgentExecutor.create_react_agent(llm, tools, prompt)


__all__ = ["AgentExecutor", "create_react_agent"]
