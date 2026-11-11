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
    "detect_issue_facets",
    "should_refine_facets",
    "build_facet_prompt",
    "parse_facets",
    "add_precondition_facets",
    "match_facet",
    "facet_evidence",
    "facet_citation_evidence",
    "build_repair_prompt",
    "parse_repair",
    "validate_repair",
    "render_repair_appendix",
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


# 法律问题类型规则：优先于 LLM 拆面，保证同一类问题每次拆出相同的方面。
# 只做「问题族」粒度，不绑定具体法条；未命中的领域再走 LLM 兜底。
_ISSUE_RULES: tuple[tuple[tuple[str, ...], dict[str, str]], ...] = (
    (
        ("用工身份", "劳动关系", "主体资格", "用工主体责任"),
        {
            "text": "用工身份/劳动关系确认与责任主体",
            "query": "确认劳动关系 用工主体责任 仲裁前置",
        },
    ),
    (
        ("居家", "在家办公", "远程办公"),
        {
            "text": "居家办公工伤",
            "query": "居家办公 工作场所 工作原因 工伤认定",
        },
    ),
    (
        ("工伤", "受伤", "事故伤害"),
        {
            "text": "工伤认定与待遇",
            "query": "工伤认定 工作时间 工作场所 工作原因",
        },
    ),
    (
        ("加班", "延长工作时间", "值班"),
        {
            "text": "加班认定",
            "query": "加班 用人单位安排 追认 劳动者自愿",
        },
    ),
    (
        ("举证", "证据", "证明责任"),
        {
            "text": "举证责任",
            "query": "举证责任 用人单位掌握证据 不利后果",
        },
    ),
    (
        ("未签", "没签", "未订立", "没有签订", "双倍工资", "二倍工资"),
        {
            "text": "未签合同二倍工资",
            "query": "未签书面劳动合同 二倍工资",
        },
    ),
    (
        ("辞退", "违法解除", "解除劳动合同", "开除", "解雇"),
        {
            "text": "违法解除劳动合同",
            "query": "违法解除 赔偿金 二倍",
        },
    ),
    (
        ("经济补偿", "补偿金"),
        {
            "text": "经济补偿",
            "query": "经济补偿 计算标准 工作年限",
        },
    ),
    (
        ("转移", "隐匿", "挥霍", "存款"),
        {
            "text": "转移夫妻共同财产",
            "query": "离婚 转移夫妻共同财产 少分 不分",
        },
    ),
    (
        ("分割", "共同财产", "房产"),
        {
            "text": "离婚财产分割",
            "query": "离婚 夫妻共同财产 分割 照顾子女",
        },
    ),
    (
        ("造谣", "诽谤", "名誉", "侮辱"),
        {
            "text": "名誉权侵害",
            "query": "名誉权 诽谤 侮辱 网络传播",
        },
    ),
    (
        ("竞业限制",),
        {
            "text": "竞业限制",
            "query": "竞业限制 经济补偿 违约金",
        },
    ),
    (
        ("抚养权", "抚养", "子女"),
        {
            "text": "子女抚养",
            "query": "子女抚养权 最有利于未成年子女",
        },
    ),
    (
        ("商标", "注册商标"),
        {
            "text": "商标侵权",
            "query": "商标侵权 侵犯注册商标专用权 赔偿",
        },
    ),
    (
        ("高空抛物",),
        {
            "text": "高空抛物",
            "query": "高空抛物 建筑物使用人 物业安全保障",
        },
    ),
    (
        ("社保", "社会保险"),
        {
            "text": "社会保险",
            "query": "社会保险 补缴 工伤保险待遇",
        },
    ),
)


_CORE_SOURCES: dict[str, tuple[tuple[str, int], ...]] = {
    "用工身份/劳动关系确认与责任主体": (
        ("关于确立劳动关系有关事项的通知", 1),
        ("人力资源社会保障部关于执行《工伤保险条例》若干问题的意见（三）", 10),
        ("最高人民法院关于审理工伤保险行政案件若干问题的规定", 3),
    ),
    "居家办公工伤": (
        ("人力资源社会保障部关于执行《工伤保险条例》若干问题的意见（三）", 9),
    ),
    "工伤认定与待遇": (
        ("工伤保险条例", 14),
    ),
    "加班认定": (
        ("中华人民共和国劳动合同法", 31),
        ("中华人民共和国劳动法", 41),
    ),
    "举证责任": (
        ("最高人民法院关于审理劳动争议案件适用法律问题的解释（一）", 42),
        ("中华人民共和国劳动争议调解仲裁法", 6),
        ("工伤保险条例", 19),
    ),
    "未签合同二倍工资": (
        ("中华人民共和国劳动合同法", 82),
    ),
    "违法解除劳动合同": (
        ("中华人民共和国劳动合同法", 87),
    ),
    "经济补偿": (
        ("中华人民共和国劳动合同法", 46),
        ("中华人民共和国劳动合同法", 47),
    ),
    "转移夫妻共同财产": (("中华人民共和国民法典", 1092),),
    "离婚财产分割": (
        ("中华人民共和国民法典", 1087),
        ("中华人民共和国民法典", 1066),
    ),
    "名誉权侵害": (
        ("中华人民共和国民法典", 1024),
    ),
    "竞业限制": (
        ("中华人民共和国劳动合同法", 23),
        ("中华人民共和国劳动合同法", 24),
        ("最高人民法院关于审理劳动争议案件适用法律问题的解释（一）", 36),
        ("最高人民法院关于审理劳动争议案件适用法律问题的解释（一）", 38),
    ),
    "子女抚养": (("中华人民共和国民法典", 1084),),
    "商标侵权": (
        ("中华人民共和国商标法", 72),
        ("中华人民共和国商标法", 77),
    ),
    "高空抛物": (
        ("中华人民共和国民法典", 1254),
        ("中华人民共和国民法典", 1198),
    ),
    "社会保险": (
        ("中华人民共和国劳动法", 72),
        ("工伤保险条例", 62),
        ("中华人民共和国劳动合同法", 38),
    ),
    "时效与程序前提": (
        ("中华人民共和国劳动争议调解仲裁法", 27),
        ("中华人民共和国劳动争议调解仲裁法", 5),
    ),
    "合同效力前提": (
        ("中华人民共和国民法典", 143),
        ("中华人民共和国民法典", 153),
        ("中华人民共和国民法典", 154),
    ),
}

_CORE_QUERIES: dict[str, str] = {
    "用工身份/劳动关系确认与责任主体": "确认劳动关系 三要件 用工主体责任",
    "居家办公工伤": "居家办公 工作场所 工作原因 工伤认定 意见三第九条",
    "工伤认定与待遇": "工伤认定 工作时间 工作场所 工作原因 第十四条",
    "加班认定": "加班 用人单位安排 追认 劳动合同法第三十一条 劳动法第四十一条",
    "举证责任": "加班费 工伤 举证责任 用人单位掌握证据",
    "未签合同二倍工资": "未签书面劳动合同 二倍工资 第八十二条",
    "违法解除劳动合同": "违法解除劳动合同 赔偿金 第八十七条",
    "经济补偿": "经济补偿 第四十六条 第四十七条 工作年限",
    "转移夫妻共同财产": "离婚 转移夫妻共同财产 少分 不分 第一千零九十二条",
    "离婚财产分割": "离婚 夫妻共同财产 分割 第一千零八十七条",
    "名誉权侵害": "名誉权 诽谤 侮辱 第一千零二十四条",
    "竞业限制": "竞业限制 经济补偿 违约金 第二十三条",
    "子女抚养": "子女抚养权 最有利于未成年子女 第一千零八十四条",
    "商标侵权": "商标侵权 侵犯注册商标专用权 赔偿 第七十七条",
    "高空抛物": "高空抛物 建筑物使用人 物业安全保障 第一千二百五十四条",
    "社会保险": "社会保险 补缴 工伤保险待遇 第六十二条",
    "时效与程序前提": "劳动争议 仲裁时效 一年 第二十七条",
    "合同效力前提": "合同效力 无效 可撤销 第一百五十三条",
}


def _with_core(facet: dict[str, Any]) -> dict[str, Any]:
    item = dict(facet)
    core = _CORE_SOURCES.get(str(item.get("text", "")))
    if core:
        item["core"] = [list(pair) for pair in core]
        item["core_query"] = _CORE_QUERIES.get(
            str(item.get("text", "")), item.get("query", "")
        )
    return item


def detect_issue_facets(
    query: str, limit: int = _MAX_FACETS
) -> list[dict[str, Any]]:
    """按法律问题族抽面。命中至少两个才返回，否则交给确定性切分/LLM。"""
    text = str(query or "")
    facets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for markers, facet in _ISSUE_RULES:
        if not any(marker in text for marker in markers):
            continue
        if facet["text"] in seen:
            continue
        seen.add(facet["text"])
        facets.append(_with_core(facet))
        if len(facets) >= limit:
            break
    return facets if len(facets) >= 2 else []


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
        result.append(_with_core(facet))
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
        core = facet.get("core") or []
        if core:
            core_used = any(
                cited_number == number
                and (
                    cited_law is None
                    or same_law(cited_law, str(law))
                )
                for cited_law, cited_number in cited
                for law, number in core
                if isinstance(number, int)
            )
            if core_used:
                continue
            relevant = [
                item
                for item in attempts
                if item.get("facet") == index and item.get("hit")
            ]
            reason = "未检索到核心法源" if not relevant else "未引用核心法源"
            missing.append(
                {
                    "index": index,
                    "text": str(facet.get("text") or f"第{index}个方面"),
                    "reason": reason,
                }
            )
            continue

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


def build_repair_prompt(
    missing: Sequence[dict[str, Any]],
    docs_by_facet: dict[int, Sequence[dict[str, Any]]],
) -> str:
    """构造「只选条文 + 只抄原文」的补引 prompt。"""
    blocks = [
        "你是法律引用核对器。下面每个方面列出本轮检索到的法条。",
        "只从列出的条文中选择能直接支持该方面的条文；不得引用列表外条文。",
        "quote 必须逐字来自给出的正文，不要改写、不要拼接；无法支持就 supported=false。",
        "只输出 JSON，不要解释，不要 Markdown 代码块：",
        '{"facets":[{"index":1,"supported":true,"citations":'
        '[{"law":"法律全称","article_no":87,"quote":"原文片段"}]}]}',
        "每个方面最多 2 条；quote 不短于 10 字。",
    ]
    for item in missing:
        index = int(item["index"])
        blocks.append(f"\n方面 {index}：{item.get('text', '')}")
        docs = docs_by_facet.get(index) or []
        if not docs:
            blocks.append("（本轮未检索到可列出的条文）")
            continue
        for doc in docs[:3]:
            body = " ".join(str(doc.get("text", "")).split())[:500]
            label = doc.get("article_label") or f"第{doc.get('article_no', '?')}条"
            blocks.append(
                f"- 《{doc.get('law_name', '')}》{label}：{body}"
            )
    blocks.append("\n输出：")
    return "\n".join(blocks)


_QUOTE_NOISE = re.compile(
    r"[\s，。；：、！？,.;:!?“”\"'‘’（）()《》【】\[\]]+"
)


def _normalize_quote(text: str) -> str:
    """引文比对忽略空白与标点差异，但不放过任何实词改写。"""
    return _QUOTE_NOISE.sub("", str(text or ""))


def parse_repair(raw: str | None) -> dict[int, dict[str, Any]]:
    """解析补引 JSON。失败返回空 dict。"""
    text = str(raw or "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    items = payload.get("facets") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return {}

    parsed: dict[int, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        citations = item.get("citations") if isinstance(item.get("citations"), list) else []
        parsed[index] = {
            "supported": bool(item.get("supported")),
            "citations": [c for c in citations if isinstance(c, dict)],
        }
    return parsed


def validate_repair(
    payload: dict[int, dict[str, Any]],
    docs_by_facet: dict[int, Sequence[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    """只保留「条文在本方面材料里 + quote 是原文子串」的补引。"""
    accepted: dict[int, list[dict[str, Any]]] = {}
    for index, item in payload.items():
        if not item.get("supported"):
            continue
        docs = list(docs_by_facet.get(index) or [])
        seen: set[tuple[str, int]] = set()
        for citation in item.get("citations") or []:
            law = str(citation.get("law") or "").strip()
            number = citation.get("article_no")
            quote = str(citation.get("quote") or "").strip()
            if not law or not isinstance(number, int) or len(quote) < 10:
                continue
            if (law, number) in seen:
                continue
            for doc in docs:
                if doc.get("article_no") != number:
                    continue
                if not same_law(law, str(doc.get("law_name", ""))):
                    continue
                body = _normalize_quote(str(doc.get("text", "")))
                if _normalize_quote(quote) not in body:
                    continue
                seen.add((law, number))
                accepted.setdefault(index, []).append(
                    {
                        "law_name": str(doc.get("law_name", "")),
                        "article_no": number,
                        "article_label": doc.get("article_label")
                        or f"第{number}条",
                        "quote": quote[:200],
                    }
                )
                break
            if len(accepted.get(index, [])) >= 2:
                break
    return accepted


def render_repair_appendix(
    accepted: dict[int, list[dict[str, Any]]],
    facets: Sequence[dict[str, Any]],
) -> str:
    """渲染「补充依据」附录；没有可接受的补引时返回空串。"""
    if not accepted:
        return ""
    lines = ["【补充依据·来自本轮检索材料】"]
    for index in sorted(accepted):
        label = (
            facets[index - 1].get("text", "")
            if 0 <= index - 1 < len(facets)
            else f"方面 {index}"
        )
        lines.append(f"方面 {index}：{label}")
        for item in accepted[index]:
            lines.append(
                f"- 《{item['law_name']}》{item['article_label']}："
                f"“{item['quote']}”"
            )
    return "\n".join(lines)


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
