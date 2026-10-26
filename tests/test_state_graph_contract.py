"""S1 验收测试：State 与 LangGraph 的运行时契约。

这个文件锁定三件容易被误以为理所当然、但一旦不成立整个设计就崩的事：

1. 节点收到的是 `LegalState` 实例（所以 `.get()` 能用）
2. `app.invoke()` 返回的是 **dict**（所以图外不能直接调 dataclass 方法）
3. `add_messages` 真的在合并而不是覆盖

第 2 条尤其重要：如果哪天有人升级 langgraph 后行为变了，
这里会立刻红，而不是等到业务代码里 AttributeError。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from legalqa.state import LegalState, coerce_state


def _build_probe_graph(sink: dict):
    def probe(state):
        sink["type"] = type(state)
        sink["messages"] = list(state.get("messages", []))
        return {
            "messages": [AIMessage(content="reply")],
            "user_query": "探针",
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    g = StateGraph(LegalState)
    g.add_node("probe", probe)
    g.add_edge(START, "probe")
    g.add_edge("probe", END)
    return g.compile()


def test_node_receives_legalstate_instance() -> None:
    sink: dict = {}
    app = _build_probe_graph(sink)
    app.invoke({"messages": [HumanMessage(content="hello")]})

    assert sink["type"] is LegalState, "节点必须收到 LegalState 实例，否则 state.get() 不可用"
    assert [m.content for m in sink["messages"]] == ["hello"]


def test_invoke_returns_dict_not_dataclass() -> None:
    """图输出是 dict —— 边界层必须用 coerce_state 转换。"""
    sink: dict = {}
    app = _build_probe_graph(sink)
    out = app.invoke({"messages": [HumanMessage(content="hello")]})

    assert isinstance(out, dict)
    assert not isinstance(out, LegalState)
    assert not hasattr(out, "with_trace"), "dict 上没有 dataclass 方法，这里不该通过"


def test_coerce_state_roundtrip() -> None:
    sink: dict = {}
    app = _build_probe_graph(sink)
    out = app.invoke({"messages": [HumanMessage(content="hello")]})

    state = coerce_state(out)
    assert isinstance(state, LegalState)
    assert state.user_query == "探针"
    assert state.iteration_count == 1
    assert [m.content for m in state.messages] == ["hello", "reply"]
    # 转换后 dataclass 方法可用
    assert state.with_trace("probe") == ["probe"]


def test_coerce_state_passthrough_and_error() -> None:
    original = LegalState(user_query="x")
    assert coerce_state(original) is original
    try:
        coerce_state(object())
    except TypeError as exc:
        assert "无法转换为 LegalState" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("非 dict / 非 LegalState 应当报错")


def test_add_messages_merges_not_replaces() -> None:
    """锁定 ADR-01 的前提：messages 是合并语义。

    这条测试同时是一个警示 —— 节点返回裁剪后的列表**不会**删除老消息。
    """

    def append_only(state):
        return {"messages": [AIMessage(content="second")]}

    g = StateGraph(LegalState)
    g.add_node("a", append_only)
    g.add_edge(START, "a")
    g.add_edge("a", END)
    app = g.compile()

    out = app.invoke({"messages": [HumanMessage(content="first")]})
    assert [m.content for m in out["messages"]] == ["first", "second"]


def test_add_messages_replaces_same_id_in_place() -> None:
    """OutputGuard 依赖的能力：同 id 是原地替换，不是追加（§6.6）。"""

    def rewrite(state):
        last = state.get("messages", [])[-1]
        return {"messages": [AIMessage(id=last.id, content="已改写")]}

    g = StateGraph(LegalState)
    g.add_node("rw", rewrite)
    g.add_edge(START, "rw")
    g.add_edge("rw", END)
    app = g.compile()

    original = AIMessage(id="fixed-id", content="原始内容")
    out = app.invoke({"messages": [HumanMessage(content="q"), original]})

    assert len(out["messages"]) == 2, "同 id 替换不得新增消息"
    assert out["messages"][-1].content == "已改写"


def test_partial_return_only_updates_returned_fields() -> None:
    """节点只返回增量字典，未返回的字段保持原值（覆盖语义）。"""

    def partial(state):
        return {"trace": ["partial"]}

    g = StateGraph(LegalState)
    g.add_node("p", partial)
    g.add_edge(START, "p")
    g.add_edge("p", END)
    app = g.compile()

    out = app.invoke({"user_query": "保留我", "turn_id": 42})
    assert out["user_query"] == "保留我"
    assert out["turn_id"] == 42
    assert out["trace"] == ["partial"]
