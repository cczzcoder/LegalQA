"""图节点：上下文压缩。

对应设计文档 §7。这是全系统最复杂的节点，也是最容易出事的地方。

四道防线在这里落地：
  1. 三明治结构 —— system 物理外置，永不参与裁剪
  2. 双约束 —— 轮次上限 ∩ token 上限
  3. 不信任第三方库 —— trim 之后必须过 `validate_window` 自检
  4. 四级降级 —— 任何一级都不返回空窗口

**只写 `llm_messages`，绝不写 `messages`。** 一旦把裁剪结果写回 messages，
`add_messages` 的合并语义会让裁剪静默失效（ADR-01 / CM-07）。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately

from legalqa import config
from legalqa.pure.summary import (
    build_summary_prompt,
    render_summary_message,
    sanitize_summary,
)
from legalqa.pure.window import (
    last_human,
    repair_tool_pairs,
    restore_current_query,
    validate_window,
)
from legalqa.state import LegalState

__all__ = ["context_compress_node", "make_context_compress_node", "estimate_tokens"]


def estimate_tokens(messages: list[AnyMessage]) -> int:
    """估算窗口 token 数。

    `chars_per_token` 用中文校准值（1.6），不是默认的 4.0 ——
    默认值是按英文设计的，对中文会低估约 2.5 倍（§7.1）。
    """
    if not messages:
        return 0
    try:
        return int(
            count_tokens_approximately(messages, chars_per_token=config.CHARS_PER_TOKEN)
        )
    except Exception:                     # 估算失败不能拖垮压缩
        chars = 0
        for message in messages:
            content = message.content if isinstance(message.content, str) else ""
            chars += len(content)
        return int(chars / max(config.CHARS_PER_TOKEN, 1e-6))


def _system_only() -> list[AnyMessage]:
    return [SystemMessage(content=config.SYSTEM_PROMPT)]


def _build(
    trimmed: list[AnyMessage], current_id: str | None
) -> tuple[list[AnyMessage], list[str]]:
    """拼装窗口并自检。返回 (窗口, 违规列表)。"""
    window: list[AnyMessage] = [*_system_only(), *trimmed]
    violations = validate_window(window, expected_last_human_id=current_id)
    return window, violations


def _update_summary(
    llm: Any,
    existing: str,
    messages: list[AnyMessage],
) -> str:
    """调用 LLM 更新摘要。失败返回空串，由调用方保留旧摘要。"""
    from langchain_core.messages import HumanMessage as _Human

    try:
        prompt = build_summary_prompt(existing, messages)
        response = llm.invoke([_Human(content=prompt)])
        return sanitize_summary(getattr(response, "content", ""))
    except Exception:                    # 摘要失败绝不能拖垮主链路
        return ""


def make_context_compress_node(llm: Any = None):
    """构造压缩节点。传 `llm` 才启用滚动摘要，不传则退化为纯裁剪。"""
    def node(state: LegalState) -> dict[str, Any]:
        return _compress(state, llm)
    return node


def _compress(state: LegalState, llm: Any = None) -> dict[str, Any]:
    history = list(state.get("messages") or [])
    body = [m for m in history if not isinstance(m, SystemMessage)]
    base: dict[str, Any] = {
        "trace": state.with_trace("context_compress"),
        "last_node": "context_compress",
    }

    # ---- 边界：没有任何业务消息 ----
    if not body:
        window = _system_only()
        return {
            **base,
            "llm_messages": window,
            "compressed": False,
            "dropped_count": 0,
            "window_tokens": estimate_tokens(window),
            "error": "no_body",
        }

    current = last_human(body)
    current_id = getattr(current, "id", None) if current is not None else None

    # ---- 约束一：轮次上限（先切片，避免对超长历史做全量估算）----
    capped = (
        body[-config.MAX_BODY_MESSAGES:]
        if len(body) > config.MAX_BODY_MESSAGES
        else body
    )

    # ---- 约束二：token 上限（唯一裁剪点）----
    trimmed: list[AnyMessage]
    try:
        trimmed = list(
            trim_messages(
                capped,
                max_tokens=config.MAX_PROMPT_TOKENS,
                token_counter=estimate_tokens,
                strategy="last",
                include_system=False,      # system 已外置，不参与裁剪
                allow_partial=False,       # 不切断单条消息
                start_on="human",          # 窗口从轮次边界开始
            )
        )
    except Exception as exc:               # L2：裁剪本身失败
        trimmed = list(capped[-config.TRIM_FALLBACK_BODY:])
        base["error"] = f"trim_failed: {type(exc).__name__}"

    # ---- 配对修复 + 当前提问保护 ----
    trimmed = repair_tool_pairs(trimmed)
    trimmed = restore_current_query(trimmed, body)

    # ---- 防线三：自检 ----
    window, violations = _build(trimmed, current_id)

    # ---- L3：自检失败 → 只保留 system + 当前提问 ----
    if violations:
        safe_body = [current] if current is not None else []
        window, again = _build(safe_body, current_id)
        # 拼接而非覆盖：trim_failed 的信息不能因为后续降级而丢失
        prefix = f"{base['error']}|" if base.get("error") else ""
        base["error"] = prefix + "invalid_window:" + ",".join(violations)
        if again:
            base["error"] += "|fallback:" + ",".join(again)

    # ---- L4：连提问都没有 → 只给 system，并标记异常 ----
    # 注意：这里刻意**不注入假的 HumanMessage**。往 prompt 里塞一条
    # 模型会当真的「用户提问」，比空窗口更危险。
    if len(window) < 2:
        window = _system_only()
        base["error"] = (base.get("error") or "") + "|no_human"

    # ---- 滚动摘要：把被裁掉、且还没摘要过的消息压成事实摘要 ----
    summary = str(state.get("conversation_summary") or "")
    summarized_count = int(state.get("summarized_count", 0) or 0)
    kept_ids = {id(m) for m in window}
    dropped_messages = [m for m in body if id(m) not in kept_ids]

    # 只摘要「新增」被裁的部分，避免每次把同样的早期消息重复喂给模型
    pending = dropped_messages[summarized_count:] if summarized_count <= len(dropped_messages) else []
    if llm is not None and len(pending) >= config.SUMMARY_MIN_NEW_MESSAGES:
        updated = _update_summary(llm, summary, pending)
        if updated:
            summary = updated
            summarized_count = len(dropped_messages)

    summary_message = render_summary_message(summary)
    if summary_message is not None:
        # 摘要紧跟在 system 之后：既满足 INV-2（只有一条 system），
        # 又不会让 system 变成动态内容（破坏缓存）。
        window = [window[0], summary_message, *window[1:]]
        base["error"] = base.get("error") or ""

    dropped = max(len(body) - (len(window) - (2 if summary_message else 1)), 0)
    return {
        **base,
        "llm_messages": window,
        "compressed": dropped > 0,
        "dropped_count": dropped,
        "window_tokens": estimate_tokens(window),
        "conversation_summary": summary,
        "summarized_count": summarized_count,
    }


# 无摘要版本，供测试与不需要摘要的场景使用
context_compress_node = make_context_compress_node(None)
