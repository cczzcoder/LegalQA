"""S6 测试：滚动摘要在压缩节点里的行为。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from legalqa import config
from legalqa.nodes.context import make_context_compress_node
from legalqa.pure.window import validate_window
from legalqa.state import LegalState


class FakeLLM:
    """记录被调用了多少次、最后一次收到的 prompt。"""

    def __init__(self, reply: str = "用户咨询公司拖欠工资，已告知可申请仲裁。", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls = 0
        self.last_prompt = ""

    def invoke(self, messages):
        self.calls += 1
        self.last_prompt = messages[-1].content if messages else ""
        if self.fail:
            raise RuntimeError("summary down")
        return AIMessage(content=self.reply)


def rounds(n: int, chars: int = 60) -> list:
    messages: list = []
    for i in range(n):
        messages.append(HumanMessage(id=f"h{i}", content=f"问题{i}" + "啊" * chars))
        messages.append(AIMessage(id=f"a{i}", content=f"回答{i}" + "呀" * chars))
    return messages


def test_no_llm_means_no_summary() -> None:
    node = make_context_compress_node(None)
    out = node(LegalState(messages=rounds(10)))
    assert out["conversation_summary"] == ""
    assert all(not str(m.content).startswith("【历史对话摘要") for m in out["llm_messages"])


def test_summary_generated_when_backlog_builds() -> None:
    llm = FakeLLM()
    node = make_context_compress_node(llm)
    out = node(LegalState(messages=rounds(12)))

    assert llm.calls == 1
    assert "拖欠工资" in out["conversation_summary"]
    assert out["summarized_count"] > 0


def test_summary_sits_right_after_system() -> None:
    llm = FakeLLM()
    node = make_context_compress_node(llm)
    out = node(LegalState(messages=rounds(12)))

    window = out["llm_messages"]
    assert isinstance(window[0], SystemMessage)
    assert window[0].content == config.SYSTEM_PROMPT
    assert "【历史对话摘要" in str(window[1].content)
    assert isinstance(window[1], HumanMessage)


def test_window_still_valid_with_summary() -> None:
    llm = FakeLLM()
    node = make_context_compress_node(llm)
    out = node(LegalState(messages=rounds(12)))
    assert validate_window(out["llm_messages"]) == []


def test_summary_not_regenerated_without_new_backlog() -> None:
    """同一批被裁消息不该被反复摘要。"""
    llm = FakeLLM()
    node = make_context_compress_node(llm)
    state = LegalState(messages=rounds(12))
    first = node(state)
    assert llm.calls == 1

    # 把第一轮的结果并回 state，再压一次（消息没变）
    merged = LegalState(
        messages=state.messages,
        conversation_summary=first["conversation_summary"],
        summarized_count=first["summarized_count"],
    )
    second = node(merged)
    assert llm.calls == 1, "没有新的被裁消息时不应再调用模型"
    assert second["conversation_summary"] == first["conversation_summary"]


def test_summary_failure_keeps_old_value() -> None:
    """摘要失败不能拖垮主链路，也不能清空已有摘要。"""
    llm = FakeLLM(fail=True)
    node = make_context_compress_node(llm)
    state = LegalState(messages=rounds(12), conversation_summary="旧摘要")
    out = node(state)

    assert out["conversation_summary"] == "旧摘要"
    assert validate_window(out["llm_messages"]) == []


def test_summary_below_threshold_not_triggered() -> None:
    llm = FakeLLM()
    node = make_context_compress_node(llm)
    node(LegalState(messages=rounds(6)))     # 刚好不裁剪
    assert llm.calls == 0


def test_short_history_no_summary() -> None:
    llm = FakeLLM()
    node = make_context_compress_node(llm)
    out = node(LegalState(messages=rounds(2)))
    assert out["conversation_summary"] == ""
    assert llm.calls == 0
