"""Graph — StateGraph / MessageGraph (مطابق لـ langgraph.graph، أسرع تنفيذاً)."""
from __future__ import annotations

from collections.abc import Hashable
from concurrent.futures import ThreadPoolExecutor

START = "__start__"
END = "__end__"


class Command:
    """مطابق لـ langgraph.types.Command: تحديث + قفزة."""
    __slots__ = ("update", "goto")

    def __init__(self, update: dict | None = None, goto=None):
        self.update = update or {}
        self.goto = goto


class Send:
    """مطابق لـ langgraph.types.Send: استدعاء متوازٍ لعقدة بمدخلات مختلفة."""
    __slots__ = ("node", "arg")

    def __init__(self, node: str, arg):
        self.node = node
        self.arg = arg


def _merge_state(state: dict, update: dict, reducers: dict | None = None) -> dict:
    reducers = reducers or {}
    for k, v in (update or {}).items():
        fn = reducers.get(k)
        if fn is not None:
            try:
                state[k] = fn(state.get(k), v)
                continue
            except Exception:
                pass
        # سلوك langgraph الافتراضي: القوائم تُمدَّد عند add، وإلا تُستبدل
        if isinstance(v, list) and isinstance(state.get(k), list) and fn is None:
            # إن كانت العقدة تعيد قائمة كاملة (مثل messages) نمدد — متوافق مع add_messages
            # لكن إن أراد الاستبدال فليستخدم Command(update={k: ...}) مع reducer=None؟ نمدد افتراضياً.
            state[k] = state[k] + v
        else:
            state[k] = v
    return state


def add_messages(left, right):
    if left is None:
        return list(right) if isinstance(right, list) else [right]
    if right is None:
        return left
    r = right if isinstance(right, list) else [right]
    return list(left) + r


class StateGraph:
    """مطابق لـ langgraph.graph.StateGraph."""

    def __init__(self, schema: dict | type | None = None):
        self.schema = schema or dict
        self.nodes: dict[str, object] = {}
        self.edges: dict[str, list[str]] = {}
        self.conditional: dict[str, tuple] = {}  # src -> (fn, mapping)
        self.entry: str | None = None
        self.finish: set[str] = set()
        self.reducers: dict[str, object] = {}
        # استخراج reducers من Annotated إن أمكن (اختياري، لا يعطل)
        try:
            import typing
            hints = typing.get_type_hints(schema, include_extras=True) if isinstance(schema, type) else {}
            for k, h in hints.items():
                meta = getattr(h, "__metadata__", ())
                for m in meta:
                    if callable(m):
                        self.reducers[k] = m
        except Exception:
            pass

    # -- بناء --
    def add_node(self, name: str, fn, **kw) -> "StateGraph":
        self.nodes[name] = fn
        self.edges.setdefault(name, [])
        return self

    def add_edge(self, a: str, b: str) -> "StateGraph":
        if a == START:
            self.entry = b
            return self
        self.edges.setdefault(a, []).append(b)
        return self

    def add_conditional_edges(self, source: str, path, mapping: dict | list | None = None) -> "StateGraph":
        self.conditional[source] = (path, mapping)
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self.entry = name
        return self

    def set_finish_point(self, name: str) -> "StateGraph":
        self.finish.add(name)
        return self

    def add_sequence(self, seq: list) -> "StateGraph":
        prev = None
        for item in seq:
            name = item if isinstance(item, str) else getattr(item, "__name__", str(item))
            if not isinstance(item, str):
                self.add_node(name, item)
            if prev is None:
                if self.entry is None:
                    self.entry = name
            else:
                self.add_edge(prev, name)
            prev = name
        return self

    def compile(self, checkpointer=None, interrupt_before: list[str] | None = None,
                interrupt_after: list[str] | None = None, **kw) -> "CompiledGraph":
        return CompiledGraph(self, checkpointer, interrupt_before or [], interrupt_after or [])


class MessageGraph(StateGraph):
    def __init__(self):
        super().__init__(dict)
        self.reducers["messages"] = add_messages


class CompiledGraph:
    __slots__ = ("builder", "checkpointer", "interrupt_before", "interrupt_after")

    def __init__(self, builder: StateGraph, checkpointer=None, interrupt_before=None, interrupt_after=None):
        self.builder = builder
        self.checkpointer = checkpointer
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []

    def _thread_id(self, config: dict | None) -> str:
        return ((config or {}).get("configurable") or {}).get("thread_id", "default")

    def _initial(self, inputs, tid: str) -> dict:
        if isinstance(inputs, dict):
            state = dict(inputs)
        elif isinstance(inputs, list):
            state = {"messages": list(inputs)}
        else:
            state = {"input": inputs}
        if self.checkpointer is not None:
            saved = self.checkpointer.get(tid)
            if saved:
                # دمج المحفوظ مع الجديد (الجديد يغلب)
                merged = dict(saved)
                merged.update(state)
                # دمج القوائم بذكاء: messages تُمدد
                for k in state:
                    if isinstance(state[k], list) and isinstance(saved.get(k), list) and k == "messages":
                        merged[k] = saved[k] + state[k]
                return merged
        return state

    def _call_node(self, name: str, state: dict):
        fn = self.builder.nodes[name]
        res = fn(state) if not isinstance(fn, type) else fn()(state)
        # Runnable من pychain؟
        if hasattr(res, "invoke") and not isinstance(res, (dict, list, Command, Send)):
            try:
                res = res  # pragma: no cover
            except Exception:
                pass
        return res

    def _next_nodes(self, name: str, state: dict) -> list:
        nxt: list = []
        if name in self.builder.conditional:
            path, mapping = self.builder.conditional[name]
            route = path(state) if callable(path) else path
            # mapping: dict قيمة->عقدة | قائمة Send | None (القيمة نفسها اسم عقدة)
            if mapping is None:
                dests = [route] if isinstance(route, str) else list(route or [])
            elif isinstance(mapping, dict):
                r = mapping.get(route, END)
                dests = [r] if isinstance(r, str) else list(r or [])
            else:
                dests = [route] if isinstance(route, str) else list(route or [])
            # Send objects تمر كما هي
            if isinstance(route, Send):
                return [route]
            if isinstance(route, list) and route and isinstance(route[0], Send):
                return route
            nxt.extend(dests)
        nxt.extend(self.builder.edges.get(name, []))
        # إزالة التكرار مع الحفاظ على الترتيب
        seen, out = set(), []
        for n in nxt:
            if isinstance(n, Send):
                out.append(n)
                continue
            if n not in seen:
                seen.add(n)
                out.append(n)
        return [n for n in out if n != END] if out else []

    def invoke(self, inputs, config: dict | None = None, recursion_limit: int = 25) -> dict:
        return self._run_invoke(inputs, config, recursion_limit)

    def stream(self, inputs, config: dict | None = None, recursion_limit: int = 25, stream_mode: str = "updates"):
        yield from self._run_stream(inputs, config, recursion_limit, stream_mode)

    def batch(self, inputs_list: list, config: dict | None = None, max_workers: int = 8) -> list[dict]:
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(inputs_list)))) as ex:
            return list(ex.map(lambda x: self.invoke(x, config), inputs_list))

    def _core_loop(self, inputs, config, recursion_limit, stream_mode, collect: bool):
        b = self.builder
        tid = self._thread_id(config)
        state = self._initial(inputs, tid)
        if b.entry is None:
            raise ValueError("No entry point — استخدم add_edge(START, 'node') أو set_entry_point")
        queue: list = [b.entry]
        step = 0
        collected = [] if collect else None

        def emit(node, update):
            if stream_mode == "updates":
                return {node: update}
            if stream_mode == "values":
                return dict(state)
            return {node: update}

        while queue and step < recursion_limit:
            step += 1
            # تنفيذ متوازٍ حقيقي عند تفرع fan-out
            if len(queue) > 1:
                with ThreadPoolExecutor(max_workers=min(32, len(queue))) as ex:
                    results = list(ex.map(lambda n: (n, self._dispatch_single(n, state)), queue))
            else:
                results = [(queue[0], self._dispatch_single(queue[0], state))]
            queue = []
            for node, (update, gotos) in results:
                if update:
                    _merge_state(state, update, b.reducers)
                if self.checkpointer is not None:
                    self.checkpointer.put(tid, state, step)
                if collect:
                    collected.append(emit(node, update))
                # المقاطعة بعد العقدة
                if node in self.interrupt_after:
                    queue = []
                    break
                for g in gotos:
                    if isinstance(g, Send):
                        # Send: نفذ العقدة المستهدفة مع state مدمج بالوسيط
                        s2 = dict(state)
                        if isinstance(g.arg, dict):
                            _merge_state(s2, g.arg, b.reducers)
                        else:
                            s2["__send__"] = g.arg
                        upd, gg = self._dispatch_single(g.node, s2)
                        if upd:
                            _merge_state(state, upd, b.reducers)
                        queue.extend([x for x in gg if x != END])
                    elif g != END:
                        queue.append(g)
            # المقاطعة قبل العقد التالية
            if any(q in self.interrupt_before for q in queue):
                break
        return state, collected

    def _run_invoke(self, inputs, config, recursion_limit):
        state, _ = self._core_loop(inputs, config, recursion_limit, "updates", collect=False)
        return state

    def _run_stream(self, inputs, config, recursion_limit, stream_mode="updates"):
        _, collected = self._core_loop(inputs, config, recursion_limit, stream_mode, collect=True)
        yield from collected

    def _dispatch_single(self, node: str, state: dict) -> tuple[dict, list]:
        if node in self.interrupt_before:
            return {}, []
        res = self._call_node(node, state)
        if isinstance(res, Command):
            update = res.update or {}
            goto = res.goto
            if goto is None:
                gotos = self._next_nodes(node, state)
            elif isinstance(goto, str):
                gotos = [goto]
            elif isinstance(goto, Send):
                gotos = [goto]
            else:
                gotos = list(goto)
            return update, gotos
        if isinstance(res, Send):
            return {}, [res]
        if isinstance(res, list) and res and isinstance(res[0], Send):
            return {}, res
        if isinstance(res, dict):
            return res, self._next_nodes(node, {**state, **res})
        if res is None:
            return {}, self._next_nodes(node, state)
        # قيمة مفردة (مثل messages): حاول دمجها بذكاء
        if isinstance(res, (str, tuple)):
            return {"output": res}, self._next_nodes(node, state)
        return {}, self._next_nodes(node, state)

    # -- إدارة الحالة (مطابق لـ langgraph) --
    def get_state(self, config: dict) -> dict:
        tid = self._thread_id(config)
        if self.checkpointer is not None:
            return self.checkpointer.get(tid) or {}
        raise ValueError("No checkpointer attached")

    def update_state(self, config: dict, values: dict):
        tid = self._thread_id(config)
        cur = self.checkpointer.get(tid) if self.checkpointer else {}
        cur = dict(cur or {})
        _merge_state(cur, values, self.builder.reducers)
        if self.checkpointer is not None:
            self.checkpointer.put(tid, cur, 0)


__all__ = ["StateGraph", "MessageGraph", "CompiledGraph", "Command", "Send",
           "START", "END", "add_messages"]
