"""消息窗口的结构性操作：配对修复、当前提问保护、不变量校验。

对应设计文档 §7.2 / §7.4 / §7.6。

这一层是上下文压缩的安全底线。`trim_messages` 只保证「不切断单条消息」，
它不知道 AI(tool_calls) 与 ToolMessage 是一对，也不知道窗口必须从 human 开始。
所以裁剪之后必须再过一遍这里的函数。

与文档的两处差异（都是 v1.0 循环化之后才暴露的）：

1. 文档里的 `_ensure_last_human_at_tail` 改名并改为 `restore_current_query`。
   原因：循环模式下最后一条消息通常是 `ToolMessage`（刚查完法条），
   强行把 human 挪到末尾会打断 tool 配对。所以只保证「当前提问还在窗口里」，
   不再要求「必须在末尾」。
2. `validate_window` 增加了 `expected_last_human_id` 参数。
   INV-7 的原意是「本轮提问没被裁掉」，只看窗口自身无法判断，必须拿全量历史比对。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from legalqa import config

__all__ = [
    "last_human",
    "insert_before_last_human",
    "repair_tool_pairs",
    "restore_current_query",
    "pairs_ok",
    "validate_window",
    "is_empty_ai",
]


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _msg_id(message: AnyMessage) -> str | None:
    return getattr(message, "id", None)


def _tool_call_ids(message: AnyMessage) -> list[str]:
    """取出 AIMessage 声明的 tool_call_id，过滤掉残缺结构。"""
    calls = getattr(message, "tool_calls", None) or []
    ids: list[str] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        cid = call.get("id")
        if isinstance(cid, str) and cid:
            ids.append(cid)
    return ids


def is_empty_ai(message: AnyMessage) -> bool:
    """空内容的 AIMessage（且没有工具调用）。

    这种消息进了历史，下一轮压缩会把它当正常对话喂回模型，
    还会触发 INV-8。必须在写入口就拦住（BC-16）。
    """
    if not isinstance(message, AIMessage):
        return False
    if _tool_call_ids(message):
        return False
    content = message.content
    if isinstance(content, str):
        return not content.strip()
    return not content


def insert_before_last_human(
    messages: Sequence[AnyMessage], note: str
) -> list[AnyMessage]:
    """把一条系统提示插到**最后一条用户提问之前**。

    为什么不直接追加到末尾：追加后模型收到的最后一句是「系统提示」，
    用户的问题被挤到中间。模型要么先回应提示（「好的，我立即作答」），
    要么把它当成新的用户输入。提示应该出现在问题之前，而不是之后。
    """
    items = list(messages)
    for i in range(len(items) - 1, -1, -1):
        if isinstance(items[i], HumanMessage):
            return [*items[:i], HumanMessage(content=note), *items[i:]]
    return [*items, HumanMessage(content=note)]


def last_human(messages: Sequence[AnyMessage]) -> HumanMessage | None:
    """倒序找最后一条 HumanMessage。找不到返回 None（不要抛异常）。"""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    return None


# ---------------------------------------------------------------------------
# 配对修复
# ---------------------------------------------------------------------------
def repair_tool_pairs(messages: Iterable[AnyMessage]) -> list[AnyMessage]:
    """删除被裁剪破坏的 tool 配对。

    两类孤儿都会被清掉：
      - 找不到发起方的 `ToolMessage`（正向孤儿）
      - 声明了 `tool_calls` 但结果不全的 `AIMessage`（反向孤儿）

    实现用「两遍索引」而不是文档里的正反两次扫描：
    先建立 ai_index / tool_index，再按索引过滤。这样能一次性处理
    「AI 被删 → 它的 Tool 也变孤儿」这种级联情况，而单遍扫描会漏。

    注意：宁可丢掉整对，也不留下半对。Qwen 收到不完整的 tool 结构会直接报错，
    而丢掉一对只是少一次检索结果，模型可以重试。
    """
    msgs = list(messages)

    ai_index: dict[int, list[str]] = {}
    tool_ids: set[str] = set()
    for i, message in enumerate(msgs):
        if isinstance(message, AIMessage):
            ids = _tool_call_ids(message)
            if ids:
                ai_index[i] = ids
        elif isinstance(message, ToolMessage):
            cid = getattr(message, "tool_call_id", None)
            if isinstance(cid, str):
                tool_ids.add(cid)

    # 保留条件：声明的每个 id 都有对应结果
    keep_ai: set[int] = {
        i for i, ids in ai_index.items() if all(cid in tool_ids for cid in ids)
    }

    # 只有被保留的 AI 所声明的 id，其 ToolMessage 才允许留下
    allowed_tool_ids: set[str] = set()
    for i in keep_ai:
        allowed_tool_ids.update(ai_index[i])

    out: list[AnyMessage] = []
    for i, message in enumerate(msgs):
        if i in ai_index:
            if i in keep_ai:
                out.append(message)
            continue
        if isinstance(message, ToolMessage):
            if getattr(message, "tool_call_id", None) in allowed_tool_ids:
                out.append(message)
            continue
        out.append(message)
    return out


# ---------------------------------------------------------------------------
# 当前提问保护
# ---------------------------------------------------------------------------
def restore_current_query(
    trimmed: Sequence[AnyMessage], body: Sequence[AnyMessage]
) -> list[AnyMessage]:
    """确保本轮提问没被裁掉（INV-7）。

    正常情况 `start_on="human"` 已经保证尾部从 human 开始，这个函数是兜底。
    真的丢了（例如单条提问本身超预算，trim 直接返回空），
    就退化成「只保留这一条提问」——宁可丢上下文，也不能答非所问。
    """
    current = last_human(body)
    if current is None:
        return list(trimmed)

    current_id = _msg_id(current)
    if current_id is not None:
        if any(_msg_id(m) == current_id for m in trimmed):
            return list(trimmed)
    elif any(m is current for m in trimmed):
        return list(trimmed)

    return [current]


# ---------------------------------------------------------------------------
# 不变量校验
# ---------------------------------------------------------------------------
def pairs_ok(messages: Sequence[AnyMessage]) -> bool:
    """严格校验 tool 配对完整性（INV-4 / INV-5）。

    与 `repair_tool_pairs` 的宽松判断不同，这里要求**紧邻**：
    AI 声明了 tool_calls，紧跟其后的连续 ToolMessage 必须覆盖全部 id。
    中间隔了别的消息就算不合法。
    """
    i, n = 0, len(messages)
    while i < n:
        message = messages[i]
        ids = _tool_call_ids(message) if isinstance(message, AIMessage) else []
        if ids:
            found: set[str] = set()
            j = i + 1
            while j < n and isinstance(messages[j], ToolMessage):
                cid = getattr(messages[j], "tool_call_id", None)
                if isinstance(cid, str):
                    found.add(cid)
                j += 1
            if not set(ids) <= found:
                return False
            i = j
            continue
        if isinstance(message, ToolMessage):
            return False          # 前面没有 AI，孤儿
        i += 1
    return True


def validate_window(
    window: Sequence[AnyMessage],
    *,
    expected_last_human_id: str | None = None,
) -> list[str]:
    """校验窗口是否满足 INV-1 ~ INV-9，返回违反项列表（空列表 = 通过）。

    失败时调用方**不要抛异常**，按 §7.6 降级到安全窗口并记 trace。
    """
    violations: list[str] = []
    window = list(window)

    if not window:
        return ["INV-1", "INV-9"]

    head = window[0]
    if not isinstance(head, SystemMessage):
        violations.append("INV-1")
    elif head.content != config.SYSTEM_PROMPT:
        violations.append("INV-1")

    if any(isinstance(m, SystemMessage) for m in window[1:]):
        violations.append("INV-2")

    if len(window) < 2:
        violations.append("INV-9")
    elif not isinstance(window[1], HumanMessage):
        violations.append("INV-3")

    if len(window) > config.WINDOW_MAX_MESSAGES:
        violations.append("INV-6")

    if not any(isinstance(m, HumanMessage) for m in window[1:]):
        violations.append("INV-7")
    elif expected_last_human_id is not None:
        if not any(_msg_id(m) == expected_last_human_id for m in window[1:]):
            violations.append("INV-7")

    if any(is_empty_ai(m) for m in window):
        violations.append("INV-8")

    if not pairs_ok(window):
        violations.append("INV-4/5")

    return violations
