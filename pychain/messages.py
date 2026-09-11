"""Messages — مطابق لـ langchain_core.messages لكن أسرع (slots + بدون pydantic)."""
from __future__ import annotations


class BaseMessage:
    __slots__ = ("content", "additional_kwargs", "name", "id")
    type: str = "base"

    def __init__(self, content: str = "", additional_kwargs: dict | None = None,
                 name: str | None = None, id: str | None = None):
        self.content = content
        self.additional_kwargs = additional_kwargs or {}
        self.name = name
        self.id = id

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"{type(self).__name__}(content={self.content!r})"

    def __eq__(self, other):
        return (type(self) is type(other) and self.content == other.content
                and self.additional_kwargs == other.additional_kwargs)

    def dict(self) -> dict:
        return {"type": self.type, "content": self.content,
                "additional_kwargs": self.additional_kwargs, "name": self.name}

    def to_dict(self) -> dict:
        return self.dict()


class HumanMessage(BaseMessage):
    __slots__ = ()
    type = "human"


class AIMessage(BaseMessage):
    __slots__ = ("tool_calls",)
    type = "ai"

    def __init__(self, content: str = "", tool_calls: list | None = None, **kw):
        super().__init__(content, **kw)
        self.tool_calls = tool_calls or []
        if tool_calls:
            self.additional_kwargs.setdefault("tool_calls", tool_calls)


class SystemMessage(BaseMessage):
    __slots__ = ()
    type = "system"


class ToolMessage(BaseMessage):
    __slots__ = ("tool_call_id",)
    type = "tool"

    def __init__(self, content: str = "", tool_call_id: str | None = None, **kw):
        super().__init__(content, **kw)
        self.tool_call_id = tool_call_id


class ChatMessage(BaseMessage):
    __slots__ = ("role",)
    type = "chat"

    def __init__(self, content: str = "", role: str = "user", **kw):
        super().__init__(content, **kw)
        self.role = role


def convert_to_messages(data) -> list[BaseMessage]:
    """تحويل مرن: str / dict / tuple / Message -> List[Message] (fast-path)."""
    if isinstance(data, BaseMessage):
        return [data]
    if isinstance(data, str):
        return [HumanMessage(data)]
    if isinstance(data, dict):
        role = data.get("role", data.get("type", "human"))
        content = data.get("content", "")
        r = role.lower()
        if r in ("human", "user"):
            return [HumanMessage(content)]
        if r in ("ai", "assistant"):
            return [AIMessage(content)]
        if r == "system":
            return [SystemMessage(content)]
        if r == "tool":
            return [ToolMessage(content, tool_call_id=data.get("tool_call_id"))]
        return [ChatMessage(content, role=role)]
    if isinstance(data, (list, tuple)):
        out: list[BaseMessage] = []
        for item in data:
            if isinstance(item, BaseMessage):
                out.append(item)
            elif isinstance(item, str):
                out.append(HumanMessage(item))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                role, content = item
                r = str(role).lower()
                if r in ("human", "user"):
                    out.append(HumanMessage(content))
                elif r in ("ai", "assistant"):
                    out.append(AIMessage(content))
                elif r == "system":
                    out.append(SystemMessage(content))
                else:
                    out.append(ChatMessage(content, role=str(role)))
            elif isinstance(item, dict):
                out.extend(convert_to_messages(item))
            else:
                out.append(HumanMessage(str(item)))
        return out
    return [HumanMessage(str(data))]


def get_buffer_string(messages: list[BaseMessage]) -> str:
    parts = []
    for m in messages:
        if isinstance(m, HumanMessage):
            parts.append(f"Human: {m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"AI: {m.content}")
        elif isinstance(m, SystemMessage):
            parts.append(f"System: {m.content}")
        elif isinstance(m, ToolMessage):
            parts.append(f"Tool: {m.content}")
        else:
            parts.append(f"{getattr(m, 'role', m.type)}: {m.content}")
    return "\n".join(parts)


__all__ = ["BaseMessage", "HumanMessage", "AIMessage", "SystemMessage",
           "ToolMessage", "ChatMessage", "convert_to_messages", "get_buffer_string"]
