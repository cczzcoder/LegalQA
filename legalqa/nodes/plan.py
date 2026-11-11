"""图节点：确定性问题登记 + 通用分面检索规划。

放在 `input_guard` 之后、`context_compress` 之前。做两件事：

1. 序号追问：读完整历史把「第 N 个问题」还原成独立问题；
2. 复合问题分面：确定性切分，复杂时让 LLM 细化成 2–4 个方面。

分面结果只用于给 Agent 注入检索清单和做证据闸门，不直接生成回答。
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.messages import HumanMessage

from legalqa.pure.facets import (
    add_precondition_facets,
    build_facet_prompt,
    detect_issue_facets,
    parse_facets,
    should_refine_facets,
    split_query,
)
from legalqa.pure.questions import resolve_ordinal_from_messages
from legalqa.state import LegalState

__all__ = ["plan_node", "make_plan_node"]


def _seed_facets(seed: Sequence[str]) -> list[dict[str, str]]:
    return [{"text": item, "query": item} for item in seed]


def _plan_facets(llm: Any, query: str) -> tuple[list[dict[str, str]], str]:
    # 规则优先：已知法律问题族直接拆成稳定的方面，不调 LLM。
    rule_facets = detect_issue_facets(query)
    if len(rule_facets) >= 2:
        return rule_facets, "rules"

    seed = split_query(query)
    if not should_refine_facets(query, seed):
        return add_precondition_facets(query, _seed_facets(seed)), (
            "deterministic" if seed else "none"
        )

    if llm is None:
        return add_precondition_facets(query, _seed_facets(seed)), "deterministic"

    try:
        response = llm.invoke(
            [HumanMessage(content=build_facet_prompt(query, seed))]
        )
        raw = (
            response.content
            if isinstance(response.content, str)
            else str(response.content or "")
        )
        refined = parse_facets(raw, fallback=[])
        if len(refined) >= 2:
            return add_precondition_facets(query, refined), "llm"
    except Exception:                     # 规划失败绝不能拖垮主链路
        pass
    return add_precondition_facets(query, _seed_facets(seed)), "deterministic_fallback"


def make_plan_node(llm: Any = None):
    """构造节点。传 llm 才会在复杂复合问题上做一次分面细化。"""
    def node(state: LegalState) -> dict[str, Any]:
        query = str(state.get("user_query") or "")
        resolved, reason = resolve_ordinal_from_messages(
            query, state.get("messages") or []
        )
        facets, facet_source = _plan_facets(llm, query)
        return {
            "resolved_query": (resolved or "").strip(),
            "ordinal_reason": reason,
            "facets": facets,
            "facet_plan_source": facet_source,
            "trace": state.with_trace("plan", reason),
            "last_node": "plan",
        }

    return node


# 无 LLM 版本，供测试与不需要分面细化的场景使用
plan_node = make_plan_node()
