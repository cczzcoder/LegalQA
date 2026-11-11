"""确定性问题登记：从历史用户消息里抽编号问题，并解析「第 N 个」指代。

为什么不用 LLM：序号是确定性的结构信息。原问题滚出 12 条窗口后，
滚动摘要不保证保留「第几个问题」这种编号，让模型猜只会答错；
而「第一/第二/1./2.」本身可以直接解析。

对应本轮修复：lh-003「回到最开始第二个问题」——原问题在窗口外，
模型答成了加班费。这里把指代解析成独立问题，再交给 Agent。
"""

from __future__ import annotations

import re
from typing import Sequence

from langchain_core.messages import HumanMessage

from legalqa import config
from legalqa.pure.retrieval import cn_to_int

__all__ = [
    "extract_numbered_questions",
    "resolve_ordinal",
    "resolve_ordinal_from_messages",
    "is_elliptical_followup",
    "resolve_elliptical_from_messages",
]

_CN_NUM = r"[一二三四五六七八九十两0-9]{1,3}"

# 行首编号：1. / 1、 / 一、 / 一. / （1）
_LINE_MARK = re.compile(
    rf"(?m)^\s*(?:[（(]\s*({_CN_NUM})\s*[）)]\s*|({_CN_NUM})\s*[、.．:：])\s*"
)
# 行内中文编号：第一，/ 第二、 / 第三：
_INLINE_MARK = re.compile(rf"第\s*({_CN_NUM})\s*(?:个)?\s*[、，,：:]\s*")
# 行内中文编号：一、 / 二， / 三：
_INLINE_CN = re.compile(rf"(?:^|[；;\n])\s*({_CN_NUM})\s*[、.．:：]\s*", re.M)
# 行内阿拉伯数字编号：；2、 / 2. / (2)
_INLINE_ARABIC = re.compile(
    rf"(?:^|[；;\n])\s*(?:[（(]\s*(\d{{1,2}})\s*[）)]\s*|(\d{{1,2}})\s*[、.．:：])\s*",
    re.M,
)
_INLINE_ANY_ARABIC = re.compile(
    r"(?<![\d])\s*[（(]?\s*(\d{1,2})\s*[）)]?\s*[、.．:：]\s*(?!\d)"
)
# 括号编号：任意位置的（1）（2）
_PAREN_MARK = re.compile(r"[（(]\s*(\d{1,2})\s*[）)]\s*")

# 「第 N 个问题」。刻意排除「第一个月 / 第一次 / 第二年」这类时间量词。
_ORDINAL = re.compile(
    rf"第\s*({_CN_NUM})\s*(?:"
    rf"个(?!月|年|天|次|周|小时|分钟|元|万)(?:问题|点|件事)?"
    rf"|(?:问题|点|件事)"
    rf")"
)
_FIRST_HINT = re.compile(r"最开始|最初|一开始|开头|最早")
_PREVIOUS_HINT = re.compile(r"上一个|上一条|刚才|前面那个|前一个")
_TRAILING = " \t\r\n？?。；;，,、"


def _clean(segment: str) -> str:
    text = str(segment or "").strip()
    text = re.sub(r"^[\s、，,。；;：:）)】\]]+", "", text)
    text = re.sub(r"[\s？?。；;，,、]+$", "", text)
    return text.strip()


def _split_by(text: str, pattern: re.Pattern[str]) -> list[str]:
    matches = list(pattern.finditer(text))
    if len(matches) < 2:
        return []
    parts: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = _clean(text[start:end])
        if segment:
            parts.append(segment)
    return parts


def extract_numbered_questions(text: str) -> list[str]:
    """把一段用户输入拆成问题列表。

    优先按显式编号切；没有编号时按问号切。返回空列表表示这段输入
    不是一个可枚举的问题列表（例如普通陈述或单个短句）。
    """
    raw = str(text or "").strip()
    if not raw:
        return []

    for pattern in (
        _LINE_MARK,
        _INLINE_MARK,
        _INLINE_CN,
        _INLINE_ARABIC,
        _INLINE_ANY_ARABIC,
        _PAREN_MARK,
    ):
        parts = _split_by(raw, pattern)
        if len(parts) >= 2:
            return parts[:5]

    pieces = [_clean(part) for part in re.split(r"[？?]+", raw)]
    questions = [part for part in pieces if part]
    if len(questions) >= 2:
        return questions[:5]
    if len(questions) == 1 and re.search(r"[？?]", raw):
        return questions[:1]
    return []


def resolve_ordinal(
    query: str, history_texts: Sequence[str]
) -> tuple[str | None, str]:
    """把「第 N 个问题」解析成历史里的具体问题。

    返回 `(解析结果, 原因)`。原因用于 state 观测：
      - ordinal_recent：最近一次含足够多问题的用户消息
      - ordinal_first：最早一次含足够多问题的用户消息（「最开始」）
      - previous：上一个问题（「上一个 / 刚才」）
      - not_ordinal / ambiguous_ordinal / out_of_range / no_history
    """
    text = str(query or "").strip()
    if not text:
        return None, "empty"

    matches = list(_ORDINAL.finditer(text))
    history = [str(item or "") for item in history_texts]

    if not matches:
        if _PREVIOUS_HINT.search(text):
            for item in reversed(history):
                questions = extract_numbered_questions(item)
                if questions:
                    return questions[-1], "previous"
            return None, "no_history"
        return None, "not_ordinal"

    if len(matches) > 1:
        return None, "ambiguous_ordinal"

    number = cn_to_int(matches[0].group(1))
    if not number or number < 1:
        return None, "out_of_range"

    if _FIRST_HINT.search(text):
        for item in history:
            questions = extract_numbered_questions(item)
            if len(questions) >= number:
                return questions[number - 1], "ordinal_first"
        return None, "out_of_range"

    for item in reversed(history):
        questions = extract_numbered_questions(item)
        if len(questions) >= number:
            return questions[number - 1], "ordinal_recent"
    return None, "out_of_range"


def resolve_ordinal_from_messages(
    query: str, messages: Sequence[object]
) -> tuple[str | None, str]:
    """从 LangChain 消息列表里取历史用户输入，再解析序号。

    当前提问通常已经是最后一条 HumanMessage，必须排除，否则
    「第二个问题」会从自己这句话里找编号。
    """
    humans = [
        str(getattr(message, "content", "") or "")
        for message in messages
        if isinstance(message, HumanMessage)
    ]
    if humans and humans[-1].strip() == str(query or "").strip():
        humans = humans[:-1]
    return resolve_ordinal(query, humans)


# 非编号追问的强信号。刻意不把裸「怎么/如何」放进来，否则
# 「怎么申请专利」这类自包含单问会被误判成追问。
_FOLLOWUP_STARTS = (
    "那", "这", "然后", "还有", "继续", "接着", "上面", "前面", "刚才", "具体",
)
_FOLLOWUP_MARKERS = (
    "如何做", "怎么做", "怎么办", "咋办", "然后呢", "接下来",
    "下一步", "具体怎么", "那具体", "这个怎么", "那个怎么",
)


def is_elliptical_followup(query: str) -> bool:
    """是否是省略指代的短追问（「如何做？」「那具体怎么算？」）。"""
    text = str(query or "").strip()
    if not text or len(text) > 24:
        return False
    # 带明确法律实体/术语的短句通常自包含，不覆盖成上一轮问题。
    if any(term in text for term in config.LEGAL_TERMS):
        return False
    if any(marker in text for marker in _FOLLOWUP_MARKERS):
        return True
    if text.startswith(_FOLLOWUP_STARTS):
        return True
    if text.rstrip("？?。！!…~ ").endswith("呢"):
        return True
    return any(token in text for token in config.DEICTIC_PRONOUNS) and len(text) <= 20


def resolve_elliptical_from_messages(
    query: str, messages: Sequence[object]
) -> tuple[str | None, str]:
    """非编号追问：把上一轮用户问题作为确定性兜底指代。"""
    if not is_elliptical_followup(query):
        return None, "not_followup"
    humans = [
        str(getattr(message, "content", "") or "")
        for message in messages
        if isinstance(message, HumanMessage)
    ]
    if humans and humans[-1].strip() == str(query or "").strip():
        humans = humans[:-1]
    if not humans:
        return None, "no_history"
    return humans[-1], "elliptical_previous"
