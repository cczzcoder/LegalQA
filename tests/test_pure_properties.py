"""S2 属性测试：用 Hypothesis 覆盖手写用例想不全的组合。

对应设计文档 §7.6 防线 3。

固定用例只能覆盖你想到的组合。消息序列的组合空间是爆炸的
（[H, AI(tool), T, AI, H]、[H, AI, H, AI(tool), T, H]……），
漏掉的那一种就是线上的事故。属性测试对**任意输入**验证不变量。
"""

from __future__ import annotations

import math

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from legalqa import config
from legalqa.pure.retrieval import safe_cosine
from legalqa.pure.rewrite import merge_rewrite_result
from legalqa.pure.window import repair_tool_pairs, validate_window

SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


# ---------------------------------------------------------------------------
# 消息序列生成器
# ---------------------------------------------------------------------------
@st.composite
def message_sequences(draw, max_size: int = 12):
    """生成随机消息序列，含残缺的 tool 配对。

    刻意生成「孤儿 Tool」「无结果的 AI」这类畸形结构，
    因为它们正是 trim 之后真实会出现的东西。
    """
    size = draw(st.integers(min_value=0, max_value=max_size))
    kinds = st.sampled_from(["human", "ai", "ai_call", "tool", "system", "empty_ai"])
    picked = [draw(kinds) for _ in range(size)]

    messages: list = []
    for i, kind in enumerate(picked):
        cid = f"c{i}"
        if kind == "human":
            messages.append(HumanMessage(id=f"h{i}", content=f"问题{i}"))
        elif kind == "ai":
            messages.append(AIMessage(id=f"a{i}", content=f"回答{i}"))
        elif kind == "ai_call":
            messages.append(
                AIMessage(
                    id=f"a{i}",
                    content="",
                    tool_calls=[{"id": cid, "name": "read_law", "args": {"q": "x"}}],
                )
            )
        elif kind == "tool":
            messages.append(ToolMessage(id=f"t{i}", content="结果", tool_call_id=cid))
        elif kind == "system":
            messages.append(SystemMessage(content="插入的 system"))
        else:
            messages.append(AIMessage(id=f"e{i}", content="   "))
    return messages


def _orphans(messages) -> list[str]:
    """返回孤儿结构描述，空列表表示配对完整。"""
    problems: list[str] = []
    declared: set[str] = set()
    results: dict[str, int] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                if isinstance(call, dict) and call.get("id"):
                    declared.add(call["id"])
        elif isinstance(message, ToolMessage):
            results[message.tool_call_id] = results.get(message.tool_call_id, 0) + 1

    kept_declared: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            ids = {c.get("id") for c in (message.tool_calls or []) if isinstance(c, dict)}
            ids.discard(None)
            if ids:
                missing = ids - set(results)
                if missing:
                    problems.append(f"AI {message.id} 缺结果 {missing}")
                kept_declared |= ids
    for message in messages:
        if isinstance(message, ToolMessage) and message.tool_call_id not in kept_declared:
            problems.append(f"孤儿 Tool {message.id}")
    return problems


# ---------------------------------------------------------------------------
# 属性 1：修复后不存在孤儿
# ---------------------------------------------------------------------------
@given(message_sequences())
@SETTINGS
def test_repair_leaves_no_orphans(messages) -> None:
    repaired = repair_tool_pairs(messages)
    assert _orphans(repaired) == []


@given(message_sequences())
@SETTINGS
def test_repair_is_idempotent(messages) -> None:
    once = repair_tool_pairs(messages)
    assert repair_tool_pairs(once) == once


@given(message_sequences())
@SETTINGS
def test_repair_never_grows(messages) -> None:
    assert len(repair_tool_pairs(messages)) <= len(messages)


@given(message_sequences())
@SETTINGS
def test_repair_preserves_non_tool_messages(messages) -> None:
    """修复只能删消息，不能改内容、不能重排。"""
    repaired = repair_tool_pairs(messages)
    plain_before = [m.id for m in messages if not isinstance(m, ToolMessage)
                    and not (isinstance(m, AIMessage) and m.tool_calls)]
    plain_after = [m.id for m in repaired if not isinstance(m, ToolMessage)
                   and not (isinstance(m, AIMessage) and m.tool_calls)]
    assert plain_before == plain_after


# ---------------------------------------------------------------------------
# 属性 2：validate_window 永不抛异常
# ---------------------------------------------------------------------------
@given(message_sequences())
@SETTINGS
def test_validate_window_never_raises(messages) -> None:
    """任意畸形输入都必须返回违规列表，而不是炸掉整轮对话。"""
    violations = validate_window(messages)
    assert isinstance(violations, list)
    assert all(isinstance(v, str) for v in violations)


@given(message_sequences(max_size=20))
@SETTINGS
def test_validate_detects_every_empty_window(messages) -> None:
    if not messages:
        assert set(validate_window(messages)) == {"INV-1", "INV-9"}


@given(st.lists(st.text(min_size=0, max_size=30), max_size=5))
@SETTINGS
def test_validate_with_only_humans(texts) -> None:
    window = [SystemMessage(content=config.SYSTEM_PROMPT)] + [
        HumanMessage(id=f"h{i}", content=t or "问") for i, t in enumerate(texts)
    ]
    violations = validate_window(window)
    if len(window) < 2:
        assert "INV-9" in violations
    else:
        assert "INV-1" not in violations


# ---------------------------------------------------------------------------
# 属性 3：余弦永远合法
# ---------------------------------------------------------------------------
@given(
    st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
             min_size=0, max_size=8),
    st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
             min_size=0, max_size=8),
)
@SETTINGS
def test_cosine_always_valid(a, b) -> None:
    score = safe_cosine(a, b)
    assert not math.isnan(score)
    assert not math.isinf(score)
    assert -1.0 - 1e-9 <= score <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 属性 4：改写失败永不返回空串
# ---------------------------------------------------------------------------
@given(st.text(min_size=1, max_size=50), st.text(max_size=300))
@SETTINGS
def test_merge_never_returns_empty(source: str, raw: str) -> None:
    """BC-35：任何路径都必须给出可用 query，否则这一轮直接白丢。"""
    query, _applied, _reason = merge_rewrite_result(source, raw, applied=True)
    assert isinstance(query, str)
    if source.strip():
        assert query.strip() != ""
