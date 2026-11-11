"""通用分面检索规划：把复合问题拆成可分别检索的方面。

不绑定具体领域：按分号/句号/问号做确定性切分，复杂时再让 LLM 细化成
2–4 个方面。每个方面都要求一次检索，回答前检查是否每个方面都有证据。
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from legalqa import config
from legalqa.pure.retrieval import extract_citations, same_law

__all__ = [
    "split_query",
    "should_refine_facets",
    "build_facet_prompt",
    "parse_facets",
    "add_precondition_facets",
    "match_facet",
    "facet_evidence",
    "facet_citation_evidence",
    "effective_tool_budget",
    "build_facet_checklist",
]

_STRONG_SPLIT = re.compile(r"[；;]+")
_SENTENCE_SPLIT = re.compile(r"[。]+")
_QUESTION_SPLIT = re.compile(r"[？?]+")
_REFINE_MARKERS = (
    "另外", "同时", "以及", "并且", "还要", "还需",
    "如何认定", "举证责任", "界限", "标准", "责任主体",
    "劳动关系", "用工身份", "主体资格", "时效", "管辖", "合同效力",
)
_MAX_FACETS = 5
_MIN_FACET_CHARS = 6

# 前提性/程序性争议。它们经常被 LLM 合并进实体问题，导致检索被实体内容淹没。
# 这里保持「问题类型」粒度，不绑定具体法条。
_PRECONDITION_RULES: tuple[tuple[tuple[str, ...], dict[str, str]], ...] = (
    (
        ("用工身份", "劳动关系", "主体资格", "用工主体责任", "责任主体"),
        {
            "text": "用工身份/劳动关系确认与责任主体",
            "query": "确认劳动关系 用工主体责任 仲裁前置 工伤认定",
        },
    ),
    (
        ("诉讼时效", "仲裁时效", "时效期间"),
        {"text": "时效与程序前提", "query": "诉讼时效 仲裁时效 起算 中断"},
    ),
    (
        ("管辖", "哪个法院", "管辖权"),
        {"text": "管辖与程序前提", "query": "管辖 法院 仲裁前置"},
    ),
    (
        ("合同是否有效", "合同效力", "无效", "可撤销"),
        {"text": "合同效力前提", "query": "合同效力 无效 可撤销"},
    ),
)
_OTHER_TOPICS = (
    "居家", "加班", "赔偿", "待遇", "工作场所", "工作原因",
    "工资", "抚养", "侵权", "合同",
)


def _clean(text: str) -> str:
    return re.sub(r"^[\s、，,。；;：:]+|[\s、，,。；;：:]+$", "", str(text or "")).strip()


def split_query(query: str, limit: int = _MAX_FACETS) -> list[str]:
    """确定性切分。返回空列表表示这不是一个可分的复合问题。"""
    text = str(query or "").strip()
    if not text:
        return []

    for pattern in (_STRONG_SPLIT, _SENTENCE_SPLIT):
        parts = [_clean(part) for part in pattern.split(text)]
        parts = [part for part in parts if len(part) >= _MIN_FACET_CHARS]
        if len(parts) >= 2:
            return parts[:limit]

    parts = [_clean(part) for part in _QUESTION_SPLIT.split(text)]
    parts = [part for part in parts if len(part) >= _MIN_FACET_CHARS]
    if len(parts) >= 2:
        return parts[:limit]
    return []


def should_refine_facets(query: str, seed: Sequence[str]) -> bool:
    """是否值得再花一次 LLM 把粗切分细化。

    短的两问句（40 字左右）不细化，避免给现有单问/多问链路增加延迟；
    长复合问题才细化，hard-001 这类嵌套方面会从 2 段拆成 4 个方面。
    """
    text = str(query or "").strip()
    if len(text) < 35:
        return False
    if len(seed) >= 2:
        return True
    if any(marker in text for marker in _REFINE_MARKERS) and any(
        token in text for token in ("？", "?", "；", ";", "，", ",")
    ):
        return True
    # 长句里出现多个逗号，通常也包含多个可独立检索的议题。
    return (text.count("，") + text.count(",")) >= 2


def build_facet_prompt(query: str, seed: Sequence[str]) -> str:
    seeds = json.dumps(list(seed), ensure_ascii=False)
    return (
        "把下面的法律问题拆成 2-5 个可以分别检索的方面。\n"
        "只输出 JSON 数组，不要解释，不要加 Markdown 代码块。\n"
        "每个元素形如 {\"text\": \"方面描述\", \"query\": \"用于检索的独立问题\"}。\n"
        "要求：\n"
        "1. 保留原问题的主体、行为、时间、金额等事实词；\n"
        "2. 不添加原问题和对话里都没有的信息；\n"
        "3. 每个 query 必须能独立检索，不要用「这个」「该问题」等代词；\n"
        "4. 方面之间不要重复；无法拆分时输出单个元素；\n"
        "5. 如果原问题包含前提性/程序性争议（例如劳动关系是否成立、由哪个单位承担"
        "工伤保险责任、是否需要先仲裁或确认），必须单独列为一个方面；\n"
        "6. 每个 query 只聚焦一个议题，不要把「身份争议」和「居家工伤」拼进同一个"
        "检索词，否则检索会被其中一个议题淹没。\n"
        f"原问题：{query}\n"
        f"初步切分（可参考，可合并或拆分）：{seeds}\n"
        "输出："
    )


def parse_facets(
    raw: str | None, limit: int = _MAX_FACETS, fallback: Sequence[str] = ()
) -> list[dict[str, str]]:
    """解析 LLM 输出；失败回退到确定性切分。"""
    text = str(raw or "").strip()
    if text:
        start, end = text.find("["), text.rfind("]")
        if start >= 0 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, list):
                facets: list[dict[str, str]] = []
                seen: set[str] = set()
                for item in payload:
                    if isinstance(item, str):
                        label = query = item.strip()
                    elif isinstance(item, dict):
                        label = str(item.get("text") or item.get("query") or "").strip()
                        query = str(item.get("query") or label).strip()
                    else:
                        continue
                    if not label or not query or label in seen:
                        continue
                    seen.add(label)
                    facets.append({"text": label, "query": query})
                    if len(facets) >= limit:
                        break
                if len(facets) >= 2:
                    return facets

    return [{"text": item, "query": item} for item in list(fallback)[:limit]]


def add_precondition_facets(
    query: str, facets: Sequence[dict[str, Any]], limit: int = _MAX_FACETS
) -> list[dict[str, Any]]:
    """复合问题里单独补出前提性争议，避免被 LLM 合并进实体问题。

    只在已经有至少两个方面时补：单问「怎么确认劳动关系」不需要额外拆。
    """
    result = [dict(item) for item in facets]
    if len(result) < 2:
        return result[:limit]

    text = str(query or "")
    for markers, facet in _PRECONDITION_RULES:
        if not any(marker in text for marker in markers):
            continue
        if any(item.get("text") == facet["text"] for item in result):
            continue
        # 已有方面已经聚焦在这个前提争议上（不含其他实体议题）时不再重复。
        focused = any(
            any(marker in f"{item.get('text', '')} {item.get('query', '')}" for marker in markers)
            and not any(topic in str(item.get("query", "")) for topic in _OTHER_TOPICS)
            for item in result
        )
        if focused:
            continue
        result.append(dict(facet))
        if len(result) >= limit:
            break
    return result[:limit]


def facet_citation_evidence(
    facets: Sequence[dict[str, Any]],
    attempts: Sequence[dict[str, Any]],
    answer: str,
) -> list[dict[str, Any]]:
    """检查每个方面的检索材料是否真的被回答引用。

    比 `facet_evidence` 更严格：检索到了不等于答到了。返回缺失方面，
    `reason` 区分「未检索」和「检索了但未引用依据」。
    """
    cited = extract_citations(answer or "")
    missing: list[dict[str, Any]] = []
    for index, facet in enumerate(facets, start=1):
        relevant = [
            item
            for item in attempts
            if item.get("facet") == index and item.get("hit")
        ]
        docs = [
            doc
            for item in relevant
            for doc in (item.get("docs") or [])
            if isinstance(doc, dict)
        ]
        if not relevant:
            reason = "未检索"
        elif not docs:
            reason = "检索了但未记录材料"
        else:
            used = False
            for law, number in cited:
                for doc in docs:
                    doc_number = doc.get("article_no")
                    if doc_number != number:
                        continue
                    if law is None or same_law(law, str(doc.get("law_name", ""))):
                        used = True
                        break
                if used:
                    break
            if used:
                continue
            reason = "检索了但未引用依据"
        missing.append(
            {
                "index": index,
                "text": str(facet.get("text") or f"第{index}个方面"),
                "reason": reason,
            }
        )
    return missing


def _bigrams(text: str) -> set[str]:
    cleaned = re.sub(r"\s+", "", str(text or ""))
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def match_facet(query: str, facets: Sequence[dict[str, Any]]) -> int | None:
    """模型没传 facet 编号时，用 bigram 相似度把 query 归到最像的方面。

    返回 1-based 编号；相似度太低返回 None（不硬塞给某个方面）。
    """
    if not facets:
        return None
    source = _bigrams(query)
    if not source:
        return None
    best_score = 0.0
    best_index: int | None = None
    for index, facet in enumerate(facets, start=1):
        target = _bigrams(f"{facet.get('text', '')} {facet.get('query', '')}")
        if not target:
            continue
        score = len(source & target) / len(source | target)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index if best_score >= 0.05 else None


def facet_evidence(
    facets: Sequence[dict[str, Any]], attempts: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """返回没有拿到检索结果的方面。"""
    missing: list[dict[str, Any]] = []
    for index, facet in enumerate(facets, start=1):
        relevant = [item for item in attempts if item.get("facet") == index]
        if any(item.get("hit") for item in relevant):
            continue
        missing.append(
            {
                "index": index,
                "text": str(facet.get("text") or f"第{index}个方面"),
                "attempted": bool(relevant),
            }
        )
    return missing


def effective_tool_budget(facets: Sequence[dict[str, Any]]) -> int:
    """复合问题按方面数放宽工具预算，最多 5 次。"""
    count = len(facets or [])
    if not count:
        return config.MAX_TOOL_CALLS
    return min(5, max(config.MAX_TOOL_CALLS, count))


def build_facet_checklist(facets: Sequence[dict[str, Any]]) -> str:
    lines = []
    for index, facet in enumerate(facets, start=1):
        text = str(facet.get("text", ""))[:80]
        query = str(facet.get("query", ""))[:80]
        lines.append(f"{index}. {text}（建议检索词：{query}）")
    return (
        f"【系统提示】本轮包含 {len(facets)} 个需要分别检索的方面：\n"
        + "\n".join(lines)
        + "\n请在同一次回复中为每个方面各发起一次 read_law 调用，"
        "并在 args.facet 中填写方面编号；不要只检索其中最显眼的一个。\n"
        "回答时每个方面控制在 200–300 字：先结论，再列 1–2 条依据"
        "（只写法律名+条号+关键句，不要整段抄录条文；结论涉及金额、倍数、期限时，"
        "必须引用规定该标准的条文，不能只引原则性条款），最后一句建议；"
        "总回答控制在 2000 字以内。"
    )
