"""图节点：降级文案组装与历史落盘。

对应设计文档 §6.7 / §6.8。

这两个节点是「历史不失忆」的最后保障：
  - `StaticFallback` 只负责选文案，不写历史
  - `SaveState` 是**降级路径上唯一的写入点**，且必须幂等

为什么分成两个节点而不是一个：文案选择是纯逻辑（可单测、可改文案不影响写入），
写入涉及唯一 id 与幂等守卫（出错会污染历史）。混在一起，测试就得同时覆盖两件事。
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import AIMessage

from legalqa import config
from legalqa.state import LegalState

__all__ = ["static_fallback_node", "save_state_node", "build_fallback_text"]


def build_fallback_text(reason: str) -> str:
    """按 guard_reason 选文案。未知 reason 回落默认值，绝不返回空串。"""
    text = config.FALLBACK_TEXTS.get((reason or "").strip(), "")
    return text.strip() or config.FALLBACK_DEFAULT


def static_fallback_node(state: LegalState) -> dict[str, Any]:
    """组装降级文案。

    上游（`InputGuard` / `Agent` / 图外层超时）只需写好 `guard_reason`，
    文案统一在这里选，避免同一句话散落在多个节点里逐渐漂移。
    """
    reason = state.get("guard_reason") or "unknown"
    text = state.get("fallback_text") or ""
    if not text.strip():
        text = build_fallback_text(reason)

    return {
        "fallback_text": text,
        "trace": state.with_trace("static_fallback", reason),
        "last_node": "static_fallback",
    }


def save_state_node(state: LegalState) -> dict[str, Any]:
    """把降级文案落成一条 AIMessage，追加进 `messages`。

    两条硬性要求：

    1. **id 必须唯一**（`fb-{turn_id}-{uuid8}`）。用固定 id 的话，第二次降级会
       原地替换上一条降级消息，表现为「上一轮回答凭空消失」。
    2. **幂等**：`fallback_saved` 已置位就直接返回，避免同一轮写两条。
    """
    base = {
        "trace": state.with_trace("save_state"),
        "last_node": "save_state",
    }
    if state.get("fallback_saved"):
        return {**base, "fallback_saved": True, "degraded": True}

    text = (state.get("fallback_text") or "").strip()
    if not text:
        # BC-08：空文案会让下一轮压缩触发 INV-8，必须回落默认值
        text = config.FALLBACK_DEFAULT

    turn = state.get("turn_id", 0) or 0
    message = AIMessage(
        id=f"{config.DEGRADED_MESSAGE_ID_PREFIX}-{turn}-{uuid.uuid4().hex[:8]}",
        content=text,
        additional_kwargs={
            "degraded": True,
            "reason": state.get("guard_reason") or "unknown",
        },
    )

    return {
        **base,
        "messages": [message],
        "fallback_saved": True,
        "degraded": True,
    }
