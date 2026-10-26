"""滚动摘要：把被裁掉的对话压成事实摘要，解决滑动窗口的失忆。

对应设计文档 §7.5 承认的局限：6 轮之后，用户前面说过的话就真的没了。
`trim_messages` 的定位是「截断」，不是「记忆」。

摘要与裁剪的分工：

| 手段 | 作用 | 代价 |
|---|---|---|
| 裁剪 | 控制 token 预算 | 丢信息 |
| 摘要 | 保留关键事实 | 每 N 轮一次 LLM 调用 |

**摘要只记事实，不记建议。** 模型给的回答已经压缩在「已达成结论」里了，
把整段回答搬进摘要既浪费 token 又会误导下一轮。
"""

from __future__ import annotations

import re
from typing import Sequence

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

__all__ = ["SUMMARY_MAX_CHARS", "build_summary_prompt", "sanitize_summary", "render_summary_message"]

# 300 字对法律咨询不够用：实测摘要塞满后仍被截断，
# 当事人/时间/金额/争议焦点/已达成结论/未决问题 六项塞不下。
SUMMARY_MAX_CHARS = 400

_SUMMARY_INSTRUCTION = (
    "把下面的法律咨询对话压缩成一段事实摘要，不超过 {limit} 字。\n"
    "\n"
    "必须保留：\n"
    "1. 当事人身份与关系（劳动者/用人单位、夫妻、邻里等）\n"
    "2. 关键时间、金额、期限\n"
    "3. 争议焦点\n"
    "4. 已经给出的结论（简要）\n"
    "5. 尚未解决的问题\n"
    "\n"
    "要求：只陈述事实，不评价、不建议、不加解释；用第三人称；直接输出摘要正文。\n"
)

_SUMMARY_PREFIX = "【历史对话摘要（系统生成，非用户输入）】"


def _role(message: AnyMessage) -> str:
    if isinstance(message, HumanMessage):
        return "用户"
    if isinstance(message, AIMessage):
        return "助手"
    return "系统"


def _content_of(message: AnyMessage, limit: int = 400) -> str:
    content = message.content if isinstance(message.content, str) else ""
    content = re.sub(r"\s+", " ", content).strip()
    return content[:limit]


def build_summary_prompt(
    existing_summary: str,
    messages: Sequence[AnyMessage],
    *,
    limit: int = SUMMARY_MAX_CHARS,
) -> str:
    """构造摘要 prompt。

    有旧摘要时做**增量合并**，而不是重头总结——重头总结会随着对话变长
    越来越慢，而且早期事实容易被后面的内容挤掉。
    """
    parts = [_SUMMARY_INSTRUCTION.format(limit=limit)]

    if existing_summary.strip():
        parts.append("已有的历史摘要（需要与新材料合并，不要丢弃其中的事实）：")
        parts.append(existing_summary.strip())
        parts.append("")

    parts.append("新增对话：")
    for message in messages:
        content = _content_of(message)
        if content:
            parts.append(f"{_role(message)}：{content}")
    parts.append("")
    parts.append("输出合并后的完整摘要：")
    return "\n".join(parts)


def sanitize_summary(raw: str | None, *, limit: int = SUMMARY_MAX_CHARS) -> str:
    """清洗摘要输出。

    小模型爱加「摘要：」前缀、爱换行分段。摘要要拼进 prompt，格式必须干净。
    """
    if not raw:
        return ""
    text = str(raw).strip()
    text = re.sub(r"^\s*(摘要|总结|概述)\s*[:：]\s*", "", text)
    text = text.strip().strip('"').strip("「」").strip()
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if len(text) > limit:
        # 在句子边界截断，不要切在「《劳动合同法》第九十四条（2012-12-」这种地方。
        # 摘要里最要紧的就是法条引用，切一半等于把信息毁掉。
        head = text[:limit]
        best = max(head.rfind(sep) for sep in "。；！？")
        text = head[: best + 1] if best >= limit * 0.6 else head.rstrip() + "…"
    return text


def render_summary_message(summary: str) -> HumanMessage | None:
    """把摘要渲染成窗口里的一条消息。

    用 `HumanMessage` 而不是 `SystemMessage`：INV-2 要求窗口里只有 index 0
    那一条 system，再塞一条会破坏不变量；也不能拼进 system，否则 system
    不再是字节恒定的（INV-1 / 缓存安全）。

    前缀明确标注「系统生成，非用户输入」，避免模型把它当成用户说的话。
    """
    text = (summary or "").strip()
    if not text:
        return None
    return HumanMessage(id="conversation-summary", content=f"{_SUMMARY_PREFIX}\n{text}")
