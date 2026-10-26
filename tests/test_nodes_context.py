"""S4 测试：上下文压缩节点。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from legalqa import config
from legalqa.nodes import context as ctx
from legalqa.pure.window import validate_window
from legalqa.state import LegalState


def rounds(n: int, chars: int = 20) -> list:
    messages: list = []
    for i in range(n):
        messages.append(HumanMessage(id=f"h{i}", content=f"问题{i}" + "啊" * chars))
        messages.append(AIMessage(id=f"a{i}", content=f"回答{i}" + "呀" * chars))
    return messages


# ---------------------------------------------------------------------------
# 基本行为
# ---------------------------------------------------------------------------
def test_empty_history_returns_system_only() -> None:
    out = ctx.context_compress_node(LegalState())
    assert len(out["llm_messages"]) == 1
    assert isinstance(out["llm_messages"][0], SystemMessage)
    assert out["error"] == "no_body"


def test_short_history_not_trimmed() -> None:
    state = LegalState(messages=rounds(2))
    out = ctx.context_compress_node(state)

    assert out["compressed"] is False
    assert out["dropped_count"] == 0
    assert len(out["llm_messages"]) == 1 + 4
    assert out["llm_messages"][0].content == config.SYSTEM_PROMPT


def test_system_prompt_is_byte_identical() -> None:
    """INV-1：system 必须逐字节等于常量，不是「差不多」。"""
    out = ctx.context_compress_node(LegalState(messages=rounds(8)))
    assert out["llm_messages"][0].content == config.SYSTEM_PROMPT


def test_extra_system_messages_are_stripped() -> None:
    """INV-2：历史里混入的 SystemMessage 不得进入窗口。"""
    messages = [SystemMessage(content="旧的 system"), *rounds(2)]
    out = ctx.context_compress_node(LegalState(messages=messages))

    systems = [m for m in out["llm_messages"] if isinstance(m, SystemMessage)]
    assert len(systems) == 1


# ---------------------------------------------------------------------------
# 长对话裁剪
# ---------------------------------------------------------------------------
def test_long_history_is_trimmed() -> None:
    """T-02：超过 6 轮时裁剪生效，且不破坏 system。"""
    state = LegalState(messages=rounds(10, chars=200))
    out = ctx.context_compress_node(state)

    window = out["llm_messages"]
    assert len(window) <= config.WINDOW_MAX_MESSAGES
    assert len(window) < len(state.messages)
    assert window[0].content == config.SYSTEM_PROMPT
    assert isinstance(window[1], HumanMessage)
    assert out["compressed"] is True
    assert out["dropped_count"] > 0


def test_window_satisfies_all_invariants() -> None:
    out = ctx.context_compress_node(LegalState(messages=rounds(10, chars=200)))
    assert validate_window(out["llm_messages"]) == []


def test_current_question_survives() -> None:
    """INV-7：本轮提问不能被裁掉，否则答非所问。"""
    state = LegalState(messages=rounds(10, chars=200))
    last_human = [m for m in state.messages if isinstance(m, HumanMessage)][-1]
    out = ctx.context_compress_node(state)

    ids = {getattr(m, "id", None) for m in out["llm_messages"]}
    assert last_human.id in ids


def test_round_cap_applies_even_when_tokens_fit() -> None:
    """轮次上限与 token 上限是两条独立约束，短消息也要受轮次限制。"""
    state = LegalState(messages=rounds(20, chars=1))
    out = ctx.context_compress_node(state)
    assert len(out["llm_messages"]) <= config.WINDOW_MAX_MESSAGES


def test_dropped_count_matches_window() -> None:
    state = LegalState(messages=rounds(10, chars=200))
    out = ctx.context_compress_node(state)
    body = [m for m in state.messages if not isinstance(m, SystemMessage)]
    assert out["dropped_count"] == len(body) - (len(out["llm_messages"]) - 1)


# ---------------------------------------------------------------------------
# tool 配对
# ---------------------------------------------------------------------------
def test_orphan_tool_message_is_removed() -> None:
    messages = [
        HumanMessage(id="h1", content="问题"),
        ToolMessage(id="t1", content="孤儿结果", tool_call_id="ghost"),
    ]
    out = ctx.context_compress_node(LegalState(messages=messages))
    assert not any(isinstance(m, ToolMessage) for m in out["llm_messages"])
    assert validate_window(out["llm_messages"]) == []


def test_valid_tool_pair_is_kept() -> None:
    messages = [
        HumanMessage(id="h1", content="问题"),
        AIMessage(
            id="a1",
            content="",
            tool_calls=[{"id": "c1", "name": "read_law", "args": {"query": "x"}}],
        ),
        ToolMessage(id="t1", content="结果", tool_call_id="c1"),
    ]
    out = ctx.context_compress_node(LegalState(messages=messages))
    assert any(isinstance(m, ToolMessage) for m in out["llm_messages"])
    assert validate_window(out["llm_messages"]) == []


# ---------------------------------------------------------------------------
# 降级链
# ---------------------------------------------------------------------------
def test_trim_exception_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """L2：trim 抛异常时退到定长窗口，不整轮失败。"""
    def boom(*args, **kwargs):
        raise RuntimeError("trim down")

    monkeypatch.setattr(ctx, "trim_messages", boom)
    out = ctx.context_compress_node(LegalState(messages=rounds(10, chars=200)))

    assert "trim_failed" in out["error"]
    assert out["llm_messages"][0].content == config.SYSTEM_PROMPT
    assert len(out["llm_messages"]) >= 2
    assert validate_window(out["llm_messages"]) == []


def test_invalid_window_falls_back_to_current_question(monkeypatch: pytest.MonkeyPatch) -> None:
    """L3：自检失败时只保留 system + 当前提问，绝不把畸形窗口喂给模型。"""
    monkeypatch.setattr(ctx, "validate_window", lambda *a, **k: ["INV-4/5"])
    state = LegalState(messages=rounds(6))
    out = ctx.context_compress_node(state)

    assert "invalid_window" in out["error"]
    assert len(out["llm_messages"]) == 2
    assert out["llm_messages"][0].content == config.SYSTEM_PROMPT
    assert isinstance(out["llm_messages"][1], HumanMessage)


def test_no_human_at_all_degrades_to_system_only() -> None:
    """L4：连提问都没有时只给 system，**不注入假的 HumanMessage**。"""
    state = LegalState(messages=[AIMessage(id="a1", content="只有回答")])
    out = ctx.context_compress_node(state)

    assert len(out["llm_messages"]) == 1
    assert "no_human" in (out["error"] or "")


# ---------------------------------------------------------------------------
# 观测字段
# ---------------------------------------------------------------------------
def test_window_tokens_recorded() -> None:
    out = ctx.context_compress_node(LegalState(messages=rounds(3)))
    assert out["window_tokens"] > 0


def test_estimate_tokens_scales_with_length() -> None:
    short = ctx.estimate_tokens([HumanMessage(content="短")])
    long = ctx.estimate_tokens([HumanMessage(content="长" * 1000)])
    assert long > short


def test_estimate_tokens_never_raises_on_weird_content() -> None:
    msg = HumanMessage(content=[{"type": "text", "text": "abc"}])
    assert ctx.estimate_tokens([msg]) >= 0


def test_trace_recorded() -> None:
    out = ctx.context_compress_node(LegalState(messages=rounds(2)))
    assert out["trace"] == ["context_compress"]
    assert out["last_node"] == "context_compress"


def test_does_not_write_messages() -> None:
    """核心约束：压缩只产出视图，绝不回写历史（CM-07）。"""
    out = ctx.context_compress_node(LegalState(messages=rounds(8)))
    assert "messages" not in out
