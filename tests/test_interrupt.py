"""S7 测试：用户中途打断后能否接上。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from legalqa.cli import build_interrupted_state
from legalqa.nodes.agent import INTERRUPT_NOTE, last_ai_was_interrupted
from legalqa.state import LegalState


# ---------------------------------------------------------------------------
# 中断状态构造
# ---------------------------------------------------------------------------
def test_interrupted_state_keeps_partial_answer() -> None:
    """半截回答必须入历史。

    不入历史的话，下一轮模型不知道刚才说到哪，会从头再答一遍。
    """
    state = build_interrupted_state(None, "公司拖欠工资", "结论：可以申请仲裁，依据", 0)

    assert len(state.messages) == 2
    assert isinstance(state.messages[0], HumanMessage)
    assert isinstance(state.messages[1], AIMessage)
    assert state.messages[1].content == "结论：可以申请仲裁，依据"
    assert state.messages[1].additional_kwargs["interrupted"] is True


def test_interrupted_state_preserves_history() -> None:
    base = LegalState(
        messages=[
            HumanMessage(id="h0", content="旧问题"),
            AIMessage(id="a0", content="旧回答"),
        ],
        conversation_summary="用户咨询过拖欠工资",
        summarized_count=2,
    )
    state = build_interrupted_state(base, "新问题", "半截", 1)

    assert len(state.messages) == 4
    assert state.messages[0].content == "旧问题"
    assert state.conversation_summary == "用户咨询过拖欠工资"
    assert state.summarized_count == 2


def test_interrupted_state_without_partial_text() -> None:
    """还没开始生成就打断：只留下用户提问，不写空的 AI 消息。"""
    state = build_interrupted_state(None, "问题", "   ", 0)

    assert len(state.messages) == 1
    assert isinstance(state.messages[0], HumanMessage)


def test_interrupted_state_marked_interrupted() -> None:
    """中断用独立字段，不混进 degraded。

    中断是用户主动行为，不是系统故障。混在一起会让「降级率」指标失真。
    """
    state = build_interrupted_state(None, "问题", "半截", 0)
    assert state.interrupted is True
    assert state.degraded is False


def test_interrupt_note_goes_before_user_question() -> None:
    """提示必须插在用户提问**之前**。

    追加到末尾的话，模型收到的最后一句是系统提示，用户问题被挤到中间——
    模型会先回应提示（「好的我立即作答」）而不是回答实际问题。
    """
    import sys

    sys.path.insert(0, "tests")
    from test_graph_e2e import MockChatModel

    from legalqa.nodes.agent import make_agent_node

    llm = MockChatModel(script=[AIMessage(id="a2", content="答")])
    node = make_agent_node(llm)
    interrupted = AIMessage(
        id="a1", content="半截", additional_kwargs={"interrupted": True}
    )
    window = [
        HumanMessage(id="h1", content="原问题"),
        interrupted,
        HumanMessage(id="h2", content="补充：三个月"),
    ]
    node(LegalState(messages=window, llm_messages=window))

    seen = llm.seen_messages[-1]
    assert "中途打断" in str(seen[-2].content), "提示应在倒数第二条"
    assert str(seen[-1].content) == "补充：三个月", "最后一条必须是用户提问"


def test_interrupted_message_id_is_unique() -> None:
    a = build_interrupted_state(None, "q", "x", 1)
    b = build_interrupted_state(None, "q", "x", 2)
    assert a.messages[-1].id != b.messages[-1].id


# ---------------------------------------------------------------------------
# 中断检测
# ---------------------------------------------------------------------------
def test_detect_interrupted_last_ai() -> None:
    messages = [
        HumanMessage(content="q"),
        AIMessage(content="半截", additional_kwargs={"interrupted": True}),
    ]
    assert last_ai_was_interrupted(messages) is True


def test_detect_complete_answer() -> None:
    assert last_ai_was_interrupted([AIMessage(content="完整回答")]) is False


def test_detect_only_looks_at_last_ai() -> None:
    """更早的中断已经处理过了，只看最后一条 AI。"""
    messages = [
        AIMessage(content="更早的半截", additional_kwargs={"interrupted": True}),
        HumanMessage(content="补充"),
        AIMessage(content="完整回答"),
    ]
    assert last_ai_was_interrupted(messages) is False


def test_detect_no_ai_message() -> None:
    assert last_ai_was_interrupted([HumanMessage(content="q")]) is False
    assert last_ai_was_interrupted([]) is False


# ---------------------------------------------------------------------------
# Agent 是否收到提示
# ---------------------------------------------------------------------------
def test_agent_receives_interrupt_note() -> None:
    """被打断时，模型必须被告知「上一条是半截」。"""
    import sys

    sys.path.insert(0, "tests")
    from test_graph_e2e import MockChatModel

    from legalqa.nodes.agent import make_agent_node

    llm = MockChatModel(script=[AIMessage(id="a1", content="继续回答")])
    node = make_agent_node(llm)

    interrupted = AIMessage(
        id="a0", content="结论：可以申请仲裁，依据", additional_kwargs={"interrupted": True}
    )
    state = LegalState(
        messages=[HumanMessage(id="h0", content="公司拖欠工资"), interrupted],
        llm_messages=[HumanMessage(id="h0", content="公司拖欠工资"), interrupted],
    )
    node(state)

    seen = llm.seen_messages[-1]
    assert any(INTERRUPT_NOTE[:20] in str(m.content) for m in seen)


def test_agent_no_note_for_normal_history() -> None:
    import sys

    sys.path.insert(0, "tests")
    from test_graph_e2e import MockChatModel

    from legalqa.nodes.agent import make_agent_node

    llm = MockChatModel(script=[AIMessage(id="a1", content="答")])
    node = make_agent_node(llm)
    state = LegalState(
        messages=[HumanMessage(id="h0", content="q")],
        llm_messages=[HumanMessage(id="h0", content="q")],
    )
    node(state)

    seen = llm.seen_messages[-1]
    assert not any("中途打断" in str(m.content) for m in seen)
