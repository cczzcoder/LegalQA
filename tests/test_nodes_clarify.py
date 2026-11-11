"""S11 测试：多候选指代澄清节点。"""

from __future__ import annotations

from legalqa.nodes.clarify import clarify_node
from legalqa.state import LegalState


def test_clarify_lists_candidates() -> None:
    state = LegalState(
        turn_id=2,
        clarification_needed=True,
        clarification_candidates=["拖欠工资能否要补偿", "未签合同能否要双倍工资"],
    )

    out = clarify_node(state)

    content = out["messages"][0].content
    assert "哪一个" in content
    assert "1. 拖欠工资能否要补偿" in content
    assert "2. 未签合同能否要双倍工资" in content
    assert "回复编号" in content
    assert out["last_node"] == "clarify"
