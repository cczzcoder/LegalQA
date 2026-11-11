"""S9 测试：Plan 节点（确定性问题登记）。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from legalqa.nodes.plan import make_plan_node
from legalqa.state import LegalState

HARD_QUERY = (
    "在用工身份尚存争议时，居家按指令处理任务受伤的“工作场所”“工作原因”如何认定；"
    "线上碎片化指令下的加班，应以“用人单位安排或追认”还是“劳动者自愿”为界限，"
    "举证责任如何分配？"
)


class _FakeLLM:
    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls = 0

    def invoke(self, messages):  # noqa: ANN001
        self.calls += 1
        if self.error is not None:
            raise self.error
        return type("Response", (), {"content": self.content})()


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


def test_plan_refines_facets_with_llm() -> None:
    llm = _FakeLLM(
        content=(
            '[{"text":"劳动关系确认","query":"怎么确认劳动关系和用工主体责任"},'
            '{"text":"居家工伤","query":"居家办公受伤算不算工伤"},'
            '{"text":"线上加班","query":"线上指令安排加班怎么认定"},'
            '{"text":"举证责任","query":"加班费和工伤的举证责任怎么分配"}]'
        )
    )
    state = LegalState(
        user_query=HARD_QUERY,
        messages=[HumanMessage(id="h1", content=HARD_QUERY)],
    )

    out = make_plan_node(llm)(state)

    assert out["facet_plan_source"] == "llm"
    assert len(out["facets"]) == 4
    assert out["facets"][0]["text"] == "劳动关系确认"
    assert llm.calls == 1


def test_plan_falls_back_when_llm_fails() -> None:
    llm = _FakeLLM(error=RuntimeError("boom"))
    state = LegalState(
        user_query=HARD_QUERY,
        messages=[HumanMessage(id="h1", content=HARD_QUERY)],
    )

    out = make_plan_node(llm)(state)

    assert out["facet_plan_source"] == "deterministic_fallback"
    assert len(out["facets"]) == 3
    assert "居家" in out["facets"][0]["text"]
    assert any("劳动关系" in facet["text"] for facet in out["facets"])
