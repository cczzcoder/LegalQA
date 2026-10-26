"""图节点：输入护栏与输出护栏。

对应设计文档 §6.1 / §6.6。

两条铁律：

1. **`InputGuard` 只判定，不写 `messages`。** 写入历史只发生在 `Agent` / `SaveState`
   / `OutputGuard`（§4.3）。拦截文案由 `SaveState` 统一落盘，避免两处写、两处文案漂移。
2. **`OutputGuard` 用同 id 替换，不追加。** `add_messages` 对相同 id 执行 update，
   所以原地改写不会增加消息条数；若追加，坏内容会留在历史里，下一轮被当正常
   对话喂回模型，形成自我污染（CM-17）。
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from legalqa import config
from legalqa.pure.retrieval import extract_citations, resolve_law, same_law
from legalqa.pure.window import last_human
from legalqa.state import LegalState

__all__ = [
    "input_guard_node",
    "output_guard_node",
    "make_output_guard_node",
    "extract_text",
    "is_unsafe_promise",
    "reset_turn_fields",
    "DISCLAIMER",
]

DISCLAIMER = "本回答仅供参考，不构成正式法律意见。"

# 输出侧的硬红线：法律咨询不能给胜诉承诺
UNSAFE_OUTPUT_PATTERNS: tuple[str, ...] = (
    "保证胜诉", "包赢", "百分百胜诉", "100%胜诉", "肯定能赢", "一定能赢",
    "保证能赢", "绝对胜诉",
)

# 否定语境豁免：模型说「不能保证胜诉」是**正确**表述，不能当成违规替换掉
_NEGATION_MARKERS: tuple[str, ...] = (
    "不能保证", "无法保证", "不保证", "不存在保证", "无法承诺", "不能承诺",
    "不敢保证", "不能给出保证",
)

# 每轮开始时必须清零的字段。
# 漏掉任何一个都会在多轮对话里出问题：
#   - fallback_saved 残留 → 第二轮降级**不写历史**（用户看到空白）
#   - iteration_count 残留 → 第二轮一进来就超预算，直接强制作答
# 这类 bug 单轮测试永远发现不了，必须由跨轮测试守住。
TURN_SCOPED_RESET: dict[str, Any] = {
    # 循环控制
    "iteration_count": 0,
    "tool_call_history": [],
    "agent_stop_reason": "",
    # 护栏
    "guard_blocked": False,
    "guard_reason": "",
    "output_blocked": False,
    "output_reason": "",
    # 降级
    "fallback_text": "",
    "fallback_saved": False,
    "degraded": False,
    "interrupted": False,
    # 改写
    "rewritten_query": "",
    "rewrite_applied": False,
    "rewrite_reason": "",
    "rewrite_latency_ms": 0,
    # 检索
    "retrieved_docs": [],
    "rag_score": 0.0,
    "rag_hit": False,
    "dense_top1": 0.0,
    "retrieval_channel": "",
    "retrieval_latency_ms": 0,
    # 压缩视图（每轮重算）
    "llm_messages": [],
    "compressed": False,
    "dropped_count": 0,
    "window_tokens": 0,
    # 异常
    "error": "",
}


def reset_turn_fields() -> dict[str, Any]:
    """返回本轮需要重置的字段副本。

    刻意**不重置**：`messages`（历史）、`turn_id`（本轮标识，调用方注入）、
    `trace`（跨轮追踪）、`last_node`。
    """
    return {key: (list(value) if isinstance(value, list) else value)
            for key, value in TURN_SCOPED_RESET.items()}

_SAFE_REPLACEMENT = (
    "关于胜诉可能性的问题，我无法给出保证性结论。案件结果取决于证据、"
    "法律适用和法官裁量，任何承诺都不具备法律效力。\n\n"
    "我可以帮你做的是：梳理请求权基础、列出需要准备的证据、指出对方可能的抗辩。"
    f"\n\n{DISCLAIMER}"
)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def extract_text(message: Any) -> str:
    """从消息里取纯文本。兼容多模态 content（list of blocks）。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p).strip()
    return str(content or "").strip()


def _blocked(reason: str, trace: list[str]) -> dict[str, Any]:
    return {
        **reset_turn_fields(),
        "guard_blocked": True,
        "guard_reason": reason,
        "fallback_text": config.FALLBACK_TEXTS.get(reason, config.FALLBACK_DEFAULT),
        "trace": trace,
        "last_node": "input_guard",
    }


# ---------------------------------------------------------------------------
# 输入护栏
# ---------------------------------------------------------------------------
def input_guard_node(state: LegalState) -> dict[str, Any]:
    """四类硬判定 + 统一提取 `user_query`。

    判定顺序刻意从便宜到昂贵，短路返回。不做领域相关性判定（ADR-04）——
    那是 RAG 阈值的职责，规则判不准「邻居占了我家 30 公分地」是不是法律问题。
    """
    trace = state.with_trace("input_guard")
    messages = state.get("messages") or []

    if not messages:
        return _blocked("empty", trace)

    current = last_human(messages)
    if current is None:
        # 最后一条可能是 ToolMessage —— 没有人类提问就没法答
        return _blocked("no_human", trace)

    text = extract_text(current)
    if not text:
        return _blocked("empty", trace)

    if len(text) > config.MAX_INPUT_CHARS:
        return _blocked("too_long", trace)

    for pattern in config.INJECTION_PATTERNS:
        if pattern.search(text):
            return _blocked("injection", trace)

    return {
        **reset_turn_fields(),
        "user_query": text,
        "trace": trace,
        "last_node": "input_guard",
    }


# ---------------------------------------------------------------------------
# 输出护栏
# ---------------------------------------------------------------------------
def is_unsafe_promise(text: str) -> bool:
    """是否含胜诉承诺。否定语境（「不能保证胜诉」）不算违规。"""
    if any(marker in text for marker in _NEGATION_MARKERS):
        return False
    return any(pattern in text for pattern in UNSAFE_OUTPUT_PATTERNS)


# 回捞条文的展示上限。回答总长受 MAX_ANSWER_CHARS 约束（2500 字），
# 附录不能把正文挤没，所以限条数 + 限单条长度 + 限总长。
_BACKFILL_MAX_ITEMS = 3
_BACKFILL_ITEM_CHARS = 160
_BACKFILL_TOTAL_CHARS = 600


def _render_backfill(docs: Sequence[dict[str, Any]]) -> str:
    """把回捞到的条文渲染成附录。空列表返回空串（不产生任何多余文本）。"""
    if not docs:
        return ""
    from legalqa.tools.read_tool import describe_validity   # 局部导入，避免模块级耦合

    lines: list[str] = []
    used = 0
    for doc in docs[:_BACKFILL_MAX_ITEMS]:
        body = str(doc.get("text", "") or "").strip()
        if len(body) > _BACKFILL_ITEM_CHARS:
            body = body[:_BACKFILL_ITEM_CHARS] + "……"
        label = doc.get("article_label") or f"第{doc.get('article_no', '?')}条"
        line = f"- 《{doc.get('law_name', '')}》{label}（{describe_validity(doc)}）：{body}"
        if used + len(line) > _BACKFILL_TOTAL_CHARS:
            break
        lines.append(line)
        used += len(line)
    if not lines:
        return ""
    return (
        "\n\n【补充核对】回答中引用的以下条文不在本次检索结果里，"
        "已按条号回查法条库确认存在：\n" + "\n".join(lines)
    )


def _looks_like_law(name: str | None) -> bool:
    """粗略判断一个名字是否像法律名，用于把裸条号归到知识库外法律。"""
    return bool(name) and any(
        token in str(name)
        for token in ("法", "条例", "规定", "办法", "解释", "细则", "法典")
    )


def _merge_docs(
    docs: Sequence[dict[str, Any]], extra: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """合并检索材料与回捞条文，按 id 去重，保留原顺序。

    **回捞条目必须打标 `retrieval_source="backfill"`。** 否则下游无法区分
    「检索到的」和「按条号回查到的」：评测的「引用可溯源」会被回捞条目
    抬成「引用在语料库里存在」，不再等于检索覆盖；失败诊断读 `retrieved`
    也会把没召回的法条误认为召回了。
    """
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    def key_of(doc: dict[str, Any]) -> str:
        return str(doc.get("id", "")) or f"{doc.get('law_name')}#{doc.get('article_no')}"

    for doc in docs:
        key = key_of(doc)
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(doc))
    for doc in extra:
        key = key_of(doc)
        if key in seen:
            continue
        seen.add(key)
        merged.append({**dict(doc), "retrieval_source": "backfill"})
    return merged


def make_output_guard_node(index: Any = None):
    """把法条索引注入 OutputGuard，供「引用回捞」使用。

    LangGraph 节点签名只接受 state，索引必须走闭包注入 —— 与
    `make_context_compress_node(llm)` 同一套做法。
    """

    def node(state: LegalState) -> dict[str, Any]:
        return output_guard_node(state, index)

    return node


def output_guard_node(state: LegalState, index: Any = None) -> dict[str, Any]:
    """检查生成内容，必要时**原地改写**最后一条 AIMessage。

    检查顺序：
      0. 悬空 tool_calls → 剥掉（否则会留下空白回答 + 污染下一轮）
      1. 胜诉承诺类表述 → 整条替换为安全文案（不是删掉，删掉会让历史出现空洞）
      2. 引用核实 → 不在本批材料里的条号回查法条库，查不到/已废止才警示
      3. 缺免责声明 → 追加

    `index` 为 None 时（单测、无索引场景）退化为「不在本批材料即未溯源」。
    """
    trace = state.with_trace("output_guard")
    base: dict[str, Any] = {"trace": trace, "last_node": "output_guard"}
    messages = state.get("messages") or []

    if not messages:
        return {**base, "output_blocked": False, "output_reason": "no_message"}

    # 只处理「最终回答」：必须是最后一条、是 AIMessage。
    target = messages[-1]
    if not isinstance(target, AIMessage):
        return {**base, "output_blocked": False, "output_reason": "not_final_answer"}

    # 走到这里说明本轮已经结束，这条消息上的 tool_calls **不可能再被执行**。
    # 留着它有两个后果（E13 实测）：
    #   ① 模型第二轮的这条消息正文为空 → 用户收到 0 字回答，免责声明也无处可加；
    #   ② 历史里出现「悬空 tool_calls」（没有对应 ToolMessage），下一轮喂回模型
    #      会被它当成「上一轮没做完的事」，反复重试同一个检索。
    # 所以剥掉 tool_calls；正文为空时用 unfinished 文案顶上，宁可少说也不给空白。
    dangling = bool(getattr(target, "tool_calls", None))
    if dangling:
        body = extract_text(target)
        if not body:
            body = config.FALLBACK_TEXTS["unfinished"]
        target = target.model_copy(
            update={"content": body, "tool_calls": [], "invalid_tool_calls": []}
        )

    text = extract_text(target)

    if is_unsafe_promise(text):
        safe = target.model_copy(update={"content": _SAFE_REPLACEMENT})
        return {
            **base,
            "messages": [safe],                 # 同 id → 原地替换
            "output_blocked": True,
            "output_reason": "unsafe_promise",
        }

    # ---- 引用溯源 ----
    # 实测 7B 模型会一边看着「第 35/38/47 条」，一边引用记忆里的「第 51/53 条」。
    # 内容可能碰巧正确，但无法用检索材料验证 —— 这正是法律场景不能接受的。
    #
    # **只警示是不够的（E13）。** 100 条评测里有 13~15 条「引对了条号、
    # 只是那条没被 top-6 召回」，旧实现一律打上「无法溯源」——这不是模型的错，
    # 是系统没去核实。法条库就在本地，按 (法律名, 条号) 回查就能确定真假：
    #   查到且现行有效 → 真实条文，把原文附在回答末尾，用户可自行核对
    #   查到但已废止   → 时效问题，必须警示（引用废止条文比不引更危险）
    #   查不到         → 编造，必须警示
    docs = list(state.get("retrieved_docs") or [])
    allowed_nos = {
        doc.get("article_no") for doc in docs if isinstance(doc.get("article_no"), int)
    }
    allowed_laws = {str(doc.get("law_name", "")) for doc in docs if doc.get("law_name")}
    # 「这一条在材料里」必须按 (法律名, 条号) 成对判断，见下面的反例说明
    allowed_pairs = {
        (str(doc.get("law_name", "")), int(doc.get("article_no")))
        for doc in docs
        if isinstance(doc.get("article_no"), int) and doc.get("law_name")
    }
    # 有序的法源列表（按检索相关度），用于「承前省略法律名」时限定解析范围。
    # 用 list 而不是 set：解析结果必须可复现，set 的遍历顺序不稳定。
    batch_laws: list[str] = []
    for doc in docs:
        name = str(doc.get("law_name", "") or "")
        if name and name not in batch_laws:
            batch_laws.append(name)

    unverified: set[int] = set()
    external: set[str] = set()
    repealed: set[str] = set()
    ambiguous: set[int] = set()
    # 本回答内已解析过的条号 → 法律名。模型通常先写「《劳动合同法》第四十六条」，
    # 后文直接写「第46条」；没有这张表，后一次会因为条号跨法重复被判成「不存在」——
    # 刚回捞成功的条文转头被指控编造，这是最不能接受的一类误判。
    resolved: dict[int, str] = {}
    # 用 dict 而不是 list：模型常在回答里把同一条引两次
    # （「依据第四十七条……」「按第四十七条计算……」），
    # 附录里就会出现两条一模一样的条文，既浪费字数又显得系统很蠢。
    backfilled: dict[tuple[str, int], dict[str, Any]] = {}

    # 有本批材料、或者有索引可回查，就做核实。
    # 旧实现在 `allowed_nos` 为空时**整个跳过**——零检索、纯凭记忆作答的回答
    # 反而免检，正是最该查的一种。有索引就不存在「没法查」这个借口。
    if allowed_nos or index is not None:
        # 传 law_names 才能识别裸法名（「依照劳动合同法第八十七条」）。
        # 只认《》会把引文里的交叉引用归到引文所属的法律名下，实测误报过。
        last_law: str | None = None
        for law, number in extract_citations(
            text, index.law_names if index is not None else ()
        ):
            if law:
                last_law = law
            cited_law_in_batch = bool(law) and any(
                same_law(law, known) for known in allowed_laws
            )
            # ① 本批材料里**就是这一条** → 天然可溯源。
            # 必须比对 (法律名, 条号) 这个**对**，不能分开比。
            # 反例（实测 labor-006）：本批有《劳动法》第 38 条、《劳动合同法》第 44 条，
            # 模型引《劳动合同法》第 38 条 ——「38 在条号集合里」成立、
            # 「劳动合同法在法源集合里」也成立，但这一条根本没被检索到。
            if law is None:
                if number in allowed_nos:
                    continue
            elif any(same_law(law, name) and no == number for name, no in allowed_pairs):
                continue

            # ② 回查法条库。认不出法律名时只在条号全局唯一时才认，
            #    否则「第一条」会命中随便哪部法（BC-24 同一个坑）。
            known_law: str | None = None
            law_in_corpus = False
            if law and index is not None:
                # 必须取**唯一/最长**解析：`next(...)` 取首个命中会把
                # 「…婚姻家庭编的解释（一）」解析成「民法典」。
                known_law = resolve_law(law, index.law_names)
                law_in_corpus = known_law is not None

            # ③ 语料库里根本没有这部法律（如《刑法》）→ 超出知识库，不算未溯源
            if index is not None and law and not law_in_corpus:
                external.add(f"《{law}》第{number}条")
                continue

            doc = index.by_article(number, known_law) if index is not None else None

            # ④ 裸条号：沿用同段最近一个显式法律名。
            # 模型常写「《语言法》第十一条……该法第二条」；`extract_citations`
            # 因跨句返回 law=None，直接按条号全局查会因歧义漏掉，或把
            # 知识库外法律（如《刑法》第 133 条）误报成「多部法律中都有」。
            if doc is None and index is not None and law is None and last_law:
                known_last = resolve_law(last_law, index.law_names)
                if known_last is not None:
                    doc = index.by_article(number, known_last)
                elif _looks_like_law(last_law):
                    external.add(f"《{last_law}》第{number}条")
                    continue

            # ⑤ 承前省略法律名 + 条号跨法重复（「第 36 条」在 20 部法里都有）：
            #    `by_article` 因歧义返回 None。按两个层次解析：
            #    a. 本回答前面已经解析过这个条号 → 沿用（模型最常这样写）
            #    b. 本轮材料限定的法源里**唯一**命中 → 认
            #    都不行就是真歧义：不能判「不存在」（它明明存在），
            #    也不能随便挑一部（法律场景挑错法源比不答更糟）。
            if doc is None and index is not None and law is None:
                if number in resolved:
                    doc = index.by_article(number, resolved[number])
                elif batch_laws:
                    hits = [
                        d for d in (index.by_article(number, name) for name in batch_laws)
                        if d is not None
                    ]
                    if len(hits) == 1:
                        doc = hits[0]
                    elif len(hits) > 1:
                        ambiguous.add(number)
                        continue

            if doc is None:
                # 没有索引（单测）时保持旧行为：不在本批就是未溯源
                if index is None and law and allowed_laws and not cited_law_in_batch:
                    external.add(f"《{law}》第{number}条")
                else:
                    unverified.add(number)
                continue

            if doc.get("is_valid", True):
                key = (str(doc.get("law_name", "")), int(doc.get("article_no") or number))
                backfilled.setdefault(key, doc)
                resolved[number] = key[0]
            else:
                label = doc.get("article_label") or f"第{number}条"
                repealed.add(f"《{doc.get('law_name', '')}》{label}")

    if unverified or external or repealed or ambiguous:
        parts = []
        if unverified:
            # 措辞必须与「是否真的查过」一致：有索引是「查不到，可能不存在」，
            # 没索引只是「不在这一批里」。说错话比不说更糟。
            if index is not None:
                parts.append(
                    "以下条号在法条库中查不到，可能不存在，请自行核实："
                    + "、".join(f"第{n}条" for n in sorted(unverified))
                )
            else:
                parts.append(
                    "以下条号未出现在本次检索结果中，无法溯源，请自行核实："
                    + "、".join(f"第{n}条" for n in sorted(unverified))
                )
        if repealed:
            parts.append("以下条文已废止，不能作为现行依据：" + "、".join(sorted(repealed)))
        if ambiguous:
            parts.append(
                "以下条号在多部法律中都有，回答未注明法律名，无法确定具体条文："
                + "、".join(f"第{n}条" for n in sorted(ambiguous))
            )
        if external:
            parts.append(
                "以下引用来自本知识库之外的法律，系统无法校验其内容："
                + "、".join(sorted(external))
            )
        warning = "\n\n【引用提示】" + "；".join(parts)
        content = text + _render_backfill(list(backfilled.values())) + warning
        if DISCLAIMER not in content:
            content += f"\n\n{DISCLAIMER}"
        merged = _merge_docs(docs, list(backfilled.values()))
        return {
            **base,
            "messages": [target.model_copy(update={"content": content})],
            "retrieved_docs": merged,
            "output_blocked": True,
            "output_reason": "unverified_citation",
        }

    if backfilled:
        # 全部引用都核实通过，只是有几条是回捞来的 —— 把原文补进回答，
        # 并记进 retrieved_docs，让「本轮用到的材料」与实际回答一致。
        content = text + _render_backfill(list(backfilled.values()))
        if DISCLAIMER not in content:
            content += f"\n\n{DISCLAIMER}"
        return {
            **base,
            "messages": [target.model_copy(update={"content": content})],
            "retrieved_docs": _merge_docs(docs, list(backfilled.values())),
            "output_blocked": False,
            "output_reason": "citation_backfilled",
        }

    if DISCLAIMER not in text:
        appended = target.model_copy(update={"content": f"{text}\n\n{DISCLAIMER}"})
        return {
            **base,
            "messages": [appended],
            "output_blocked": False,
            "output_reason": "disclaimer_added",
        }

    if dangling:
        # 剥掉 tool_calls 这件事本身就是一次改写，必须写回 state，
        # 否则悬空 tool_calls 会留在历史里污染下一轮。
        return {
            **base,
            "messages": [target],
            "output_blocked": True,
            "output_reason": "dangling_tool_call",
        }

    return {**base, "output_blocked": False, "output_reason": ""}
