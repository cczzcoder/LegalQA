"""S1 验收测试：State 契约。

对应设计文档 §4.2 / §4.3 与 §11 验收标准：
  - `LegalState()` 零参可构造
  - 所有字段有默认值
  - `messages` 挂了 add_messages reducer
  - 全代码库不存在 `state["key"]` 式读取
"""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path
from typing import get_type_hints

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from legalqa.state import LegalState, append_trace, new_state

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "legalqa"


# ---------------------------------------------------------------------------
# 构造与默认值
# ---------------------------------------------------------------------------
def test_zero_arg_construction() -> None:
    """零参构造必须成功。这是 checkpoint 反序列化缺字段时的最后防线。"""
    state = LegalState()
    assert state.messages == []
    assert state.llm_messages == []
    assert state.iteration_count == 0
    assert state.pending_action is None


def test_every_field_has_default() -> None:
    """所有字段都有默认值（含 default_factory）。"""
    state = LegalState()
    missing = []
    for f in fields(LegalState):
        try:
            getattr(state, f.name)
        except AttributeError:  # pragma: no cover - 理论上不会走到
            missing.append(f.name)
    assert missing == [], f"以下字段缺少默认值: {missing}"


def test_mutable_defaults_are_not_shared() -> None:
    """default_factory 必须生效，否则两个实例会共享同一个 list。"""
    a, b = LegalState(), LegalState()
    a.messages.append(HumanMessage(content="x"))
    a.retrieved_docs.append({"id": 1})
    a.trace.append("a")
    a.tool_call_history.append("read:abc")

    assert b.messages == []
    assert b.retrieved_docs == []
    assert b.trace == []
    assert b.tool_call_history == []


def test_messages_reducer_registered() -> None:
    """messages 必须挂 add_messages，否则历史无法追加（ADR-01）。"""
    hints = get_type_hints(LegalState, include_extras=True)
    annotation = hints["messages"]
    assert getattr(annotation, "__metadata__", None) == (add_messages,)


def test_only_messages_and_trace_have_reducers() -> None:
    """只有 messages（add_messages）和 trace（operator.add）挂 reducer。

    messages 需要合并语义，trace 需要追加语义。其余字段一律覆盖语义，
    这是双轨设计和跨轮重置的前提。
    """
    hints = get_type_hints(LegalState, include_extras=True)
    with_reducer = {
        name for name, ann in hints.items() if getattr(ann, "__metadata__", None)
    }
    assert with_reducer == {"messages", "trace"}


def test_trace_uses_operator_add() -> None:
    import operator

    hints = get_type_hints(LegalState, include_extras=True)
    assert hints["trace"].__metadata__ == (operator.add,)


# ---------------------------------------------------------------------------
# get() 语义
# ---------------------------------------------------------------------------
def test_get_returns_default_for_unknown_key() -> None:
    """拼错 key 时返回默认值，而不是抛 KeyError（BC-19）。"""
    state = LegalState()
    assert state.get("no_such_field", "fallback") == "fallback"
    assert state.get("no_such_field") is None


def test_get_does_not_mask_explicit_none() -> None:
    """字段显式为 None 时返回 None，不回落默认值。

    「字段缺失」与「字段为 None」必须可区分，否则路由层无法正确处理
    rag_score=None 这类脏数据（BC-06）。
    """
    state = LegalState(pending_action=None)
    assert state.get("pending_action", {"a": 1}) is None


def test_get_reads_real_value() -> None:
    state = LegalState(rag_score=0.83)
    assert state.get("rag_score", 0.0) == 0.83


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------
def test_new_state_overrides() -> None:
    state = new_state(user_query="公司拖欠工资", turn_id=7)
    assert state.user_query == "公司拖欠工资"
    assert state.turn_id == 7
    assert state.messages == []


def test_new_state_rejects_unknown_field() -> None:
    """拼错字段名必须立刻报错，不能静默丢弃。"""
    with pytest.raises(TypeError, match="未知的 State 字段"):
        new_state(user_qurey="typo")


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------
def test_with_trace_returns_single_delta() -> None:
    """trace 挂了 operator.add reducer，所以只返回单条增量。

    返回完整列表会让 reducer 把历史重复追加 N 倍。
    """
    state = LegalState(trace=["old"])
    assert state.with_trace("input_guard", "pass") == ["input_guard:pass"]
    assert state.with_trace("agent") == ["agent"]
    assert state.trace == ["old"], "with_trace 不得原地修改"


def test_append_trace_shape() -> None:
    state = LegalState()
    delta = append_trace(state, "context_compress", "kept=5")
    assert delta == {"trace": ["context_compress:kept=5"], "last_node": "context_compress"}


def test_trace_accumulates_across_nodes() -> None:
    """多个节点在同一轮各自追加一条，最终应全部保留。

    早期 trace 没有 reducer，后写的覆盖先写的 —— 实测 7 轮对话只留下
    7 条记录（每轮最后一条），链路信息全丢。
    """
    from langgraph.graph import END, START, StateGraph

    def node_a(state):
        return {"trace": state.with_trace("a")}

    def node_b(state):
        return {"trace": state.with_trace("b")}

    graph = StateGraph(LegalState)
    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)

    out = graph.compile().invoke({"trace": ["pre"]})
    assert out["trace"] == ["pre", "a", "b"]


# ---------------------------------------------------------------------------
# 静态检查：禁止 state["key"]
# ---------------------------------------------------------------------------
def _subscript_state_usages(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id == "state":
                hits.append((node.lineno, ast.unparse(node)))
    return hits


@pytest.mark.parametrize("path", sorted(PACKAGE_DIR.rglob("*.py")), ids=lambda p: p.name)
def test_no_bracket_access_on_state(path: Path) -> None:
    """全代码库禁止 state["key"]（§11 验收标准）。

    用 AST 而非字符串匹配，避免注释和文档字符串里的示例造成误报。
    """
    hits = _subscript_state_usages(path)
    assert hits == [], f"{path.name} 存在下标访问: {hits}"
