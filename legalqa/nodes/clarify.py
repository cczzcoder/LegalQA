"""图节点：多候选指代的澄清。

上一轮有多个问题、本轮只问「如何做？」时，系统不猜，直接把候选列给用户。
用户回复「第二个」后，Plan 的序号解析会接上。
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import AIMessage

from legalqa.state import LegalState

__all__ = ["clarify_node"]


def clarify_node(state: LegalState) -> dict[str, Any]:
    candidates = [
        str(item).strip()
        for item in (state.get("clarification_candidates") or [])
        if str(item).strip()
    ]
    if candidates:
        lines = [
            f"{index}. {text[:80]}"
            for index, text in enumerate(candidates[:5], start=1)
        ]
        content = (
            "你指的是上一轮的哪一个问题？\n"
            + "\n".join(lines)
            + "\n请回复编号（如「第二个」），或直接补充一句完整的问题。"
        )
    else:
        content = "你指的是上一轮的哪一个问题？请补充一句完整的问题，我按新问题处理。"

    message = AIMessage(
        id=f"clarify-{state.get('turn_id', 0)}-{uuid.uuid4().hex[:8]}",
        content=content,
        additional_kwargs={"clarified": True, "candidates": candidates},
    )
    return {
        "messages": [message],
        "trace": state.with_trace("clarify", "ambiguous_elliptical"),
        "last_node": "clarify",
    }
