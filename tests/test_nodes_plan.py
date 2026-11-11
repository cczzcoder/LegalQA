"""S9 测试：Plan 节点（确定性问题登记）。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from legalqa.nodes.plan import make_plan_node
from legalqa.state import LegalState


def test_plan_resolves_recent_ordinal() -> None:
    messages = [
        HumanMessage(id="h1", content="第一，A？第二，B？"),
        HumanMessage(id="h2", content="第二个问题怎么算？"),
    ]
    state = LegalState(user_query="第二个问题怎么算？", messages=messages)

    out = make_plan_node()(state)

    assert out["resolved_query"] == "B"
    assert out["ordinal_reason"] == "ordinal_recent"
    assert out["last_node"] == "plan"
    assert out["trace"] == ["plan:ordinal_recent"]


def test_plan_passthrough_without_ordinal() -> None:
    state = LegalState(
        user_query="工伤怎么认定？",
        messages=[HumanMessage(id="h1", content="工伤怎么认定？")],
    )

    out = make_plan_node()(state)

    assert out["resolved_query"] == ""
    assert out["ordinal_reason"] == "not_ordinal"


def test_plan_ignores_current_query_when_scanning_history() -> None:
    """当前提问里的「第二」不能被当成历史编号来源。"""
    state = LegalState(
        user_query="第二个问题怎么算？",
        messages=[HumanMessage(id="h1", content="第二个问题怎么算？")],
    )

    out = make_plan_node()(state)

    assert out["resolved_query"] == ""
    assert out["ordinal_reason"] == "out_of_range"
