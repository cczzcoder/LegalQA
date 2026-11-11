"""图节点：确定性问题登记与序号指代解析。

放在 `input_guard` 之后、`context_compress` 之前。它读的是**完整历史**
（最多 20 条，见服务层），不是压缩后的 12 条窗口——这正是为了兜住
「原问题已经滚出窗口，但用户还在说第 N 个问题」的情况。

只做确定性解析，不调 LLM；没有命中序号时原样透传，零额外延迟。
"""

from __future__ import annotations

from typing import Any

from legalqa.pure.questions import resolve_ordinal_from_messages
from legalqa.state import LegalState

__all__ = ["plan_node", "make_plan_node"]


def plan_node(state: LegalState) -> dict[str, Any]:
    query = str(state.get("user_query") or "")
    resolved, reason = resolve_ordinal_from_messages(
        query, state.get("messages") or []
    )
    return {
        "resolved_query": (resolved or "").strip(),
        "ordinal_reason": reason,
        "trace": state.with_trace("plan", reason),
        "last_node": "plan",
    }


def make_plan_node():
    """与其它节点保持同一套闭包工厂风格，便于后续注入配置。"""
    return plan_node
