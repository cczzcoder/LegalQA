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

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from legalqa import config
from legalqa.pure.retrieval import extract_citations, same_law
from legalqa.pure.window import last_human
from legalqa.state import LegalState

__all__ = [
    "input_guard_node",
    "output_guard_node",
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


def output_guard_node(state: LegalState) -> dict[str, Any]:
    """检查生成内容，必要时**原地改写**最后一条 AIMessage。

    检查顺序：
      1. 胜诉承诺类表述 → 整条替换为安全文案（不是删掉，删掉会让历史出现空洞）
      2. 缺免责声明 → 追加
    """
    trace = state.with_trace("output_guard")
    base: dict[str, Any] = {"trace": trace, "last_node": "output_guard"}
    messages = state.get("messages") or []

    if not messages:
        return {**base, "output_blocked": False, "output_reason": "no_message"}

    # 只处理「最终回答」：必须是最后一条、是 AIMessage、且不带 tool_calls。
    # 循环中途的 AIMessage(tool_calls) 内容为空，给它加免责声明会造出
    # 一条「既要求调工具、又带正文」的畸形消息。
    target = messages[-1]
    if not isinstance(target, AIMessage):
        return {**base, "output_blocked": False, "output_reason": "not_final_answer"}
    if getattr(target, "tool_calls", None):
        return {**base, "output_blocked": False, "output_reason": "tool_call_message"}

    text = extract_text(target)

    if is_unsafe_promise(text):
        safe = target.model_copy(update={"content": _SAFE_REPLACEMENT})
        return {
            **base,
            "messages": [safe],                 # 同 id → 原地替换
            "output_blocked": True,
            "output_reason": "unsafe_promise",
        }

    # ---- 引用溯源：回答里引的条号必须在检索材料里 ----
    # 实测 7B 模型会一边看着「第 35/38/47 条」，一边引用记忆里的「第 51/53 条」。
    # 内容可能碰巧正确，但无法用检索材料验证 —— 这正是法律场景不能接受的。
    # 不整条替换（会丢掉有用内容），只在末尾追加警示并标记。
    docs = state.get("retrieved_docs") or []
    allowed_nos = {
        doc.get("article_no") for doc in docs if isinstance(doc.get("article_no"), int)
    }
    allowed_laws = {str(doc.get("law_name", "")) for doc in docs if doc.get("law_name")}

    unverified: set[int] = set()
    external: set[str] = set()
    if allowed_nos:
        for law, number in extract_citations(text):
            # 引了语料库之外的法律（如《刑法》第 245 条）——不算「未溯源」，
            # 因为本次检索根本不覆盖该法源，提示应区分开。
            if law and allowed_laws and not any(same_law(law, known) for known in allowed_laws):
                external.add(f"《{law}》第{number}条")
                continue
            if number not in allowed_nos:
                unverified.add(number)

    if unverified or external:
        parts = []
        if unverified:
            parts.append(
                "以下条号未出现在本次检索结果中，无法溯源，请自行核实："
                + "、".join(f"第{n}条" for n in sorted(unverified))
            )
        if external:
            parts.append(
                "以下引用来自本知识库之外的法律，系统无法校验其内容："
                + "、".join(sorted(external))
            )
        warning = "\n\n【引用提示】" + "；".join(parts)
        content = text + warning
        if DISCLAIMER not in content:
            content += f"\n\n{DISCLAIMER}"
        return {
            **base,
            "messages": [target.model_copy(update={"content": content})],
            "output_blocked": True,
            "output_reason": "unverified_citation",
        }

    if DISCLAIMER not in text:
        appended = target.model_copy(update={"content": f"{text}\n\n{DISCLAIMER}"})
        return {
            **base,
            "messages": [appended],
            "output_blocked": False,
            "output_reason": "disclaimer_added",
        }

    return {**base, "output_blocked": False, "output_reason": ""}
