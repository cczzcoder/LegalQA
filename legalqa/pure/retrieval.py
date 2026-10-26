"""检索侧的纯函数：条号解析、余弦、融合重排。

对应设计文档 §6.4.3 ~ §6.4.5。

一条贯穿全模块的原则（BC-26）：**阈值只吃原始余弦，融合分只用于排序。**
`fuse_rank` 产出的分数是 min-max 归一化后的相对量，top-1 恒为 1.0，
拿它比阈值等于永远不会 fallback。所以这里的函数刻意分开导出，
不提供「一个分数走天下」的便捷接口。
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Sequence

from legalqa import config

__all__ = [
    "cn_to_int",
    "extract_article_no",
    "extract_all_article_nos",
    "extract_citations",
    "match_law_name",
    "law_reference_in",
    "safe_cosine",
    "minmax_normalize",
    "authority_weight",
    "fuse_rank",
]

_CN_DIGITS: dict[str, int] = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS: dict[str, int] = {"十": 10, "百": 100, "千": 1000}

# 同样允许条号内部空格：「第 五 十 八 条」在 OCR/PDF 文本里很常见
_NUM_CHAR = r"[一二三四五六七八九十百千零〇两0-9]"
_ARTICLE_INNER = re.compile(rf"第\s*({_NUM_CHAR}(?:\s*{_NUM_CHAR}){{0,11}})\s*条")


def cn_to_int(text: str) -> int | None:
    """中文数字转整数。失败返回 None，**绝不抛异常**（BC-28）。

    支持到千位（法条号上限约 1500），不支持「万」—— 条号用不到。
    """
    raw = re.sub(r"\s+", "", text or "")
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)

    total = 0
    section = 0
    number = 0
    seen_any = False
    for char in raw:
        if char in _CN_DIGITS:
            number = _CN_DIGITS[char]
            seen_any = True
        elif char in _CN_UNITS:
            unit = _CN_UNITS[char]
            if number == 0:
                number = 1          # 「十六」的十前面没有数字
            section += number * unit
            number = 0
            seen_any = True
        else:
            return None
    if not seen_any:
        return None
    return total + section + number


def extract_article_no(text: str) -> int | None:
    """从文本里抽条号。多处出现时取第一处。

    用户问「民法典第 1165 条」时，条文正文里并不含「民法典」，
    纯语义检索相似度常常只有 0.6 出头，会被 0.7 阈值误杀。
    这条正则 + 一个字典就是为这个高频场景准备的（§6.4.3 通道 B）。
    """
    if not text:
        return None
    match = _ARTICLE_INNER.search(text)
    if not match:
        return None
    return cn_to_int(match.group(1))


def extract_all_article_nos(text: str) -> set[int]:
    """抽取文本里出现的**全部**条号，用于引用溯源校验。

    `extract_article_no` 只取第一处（用于查询解析），这里要全取：
    模型回答里可能引 3~5 个条号，漏检任何一个都会放过未溯源的引用。
    """
    if not text:
        return set()
    found: set[int] = set()
    for match in _ARTICLE_INNER.finditer(text):
        number = cn_to_int(match.group(1))
        if number is not None:
            found.add(number)
    return found


_LAW_CITE = re.compile(r"《([^》]{2,40})》")


def extract_citations(text: str) -> list[tuple[str | None, int]]:
    """抽取「法律名 + 条号」对。

    为什么不能只抽条号：模型会写「《刑法》第二百四十五条」，
    只抽条号会把它当成宪法第 245 条去校验，产生误报。
    法律名取该条号**之前最近**的一个《》标记；没有则返回 None（承前省略）。
    """
    if not text:
        return []
    marks = [(m.start(), m.group(1)) for m in _LAW_CITE.finditer(text)]
    pairs: list[tuple[str | None, int]] = []
    for match in _ARTICLE_INNER.finditer(text):
        pos = match.start()
        law: str | None = None
        for start, name in marks:
            if start < pos:
                law = name
            else:
                break
        number = cn_to_int(match.group(1))
        if number is not None:
            pairs.append((law, number))
    return pairs


_COMMON_PREFIX = "中华人民共和国"

# 提问里「XX法」「XX条例」这类法律名。用来识别「问的是语料外的法律」。
# **贪婪匹配**：非贪婪会把「商标法实施条例」截成「商标法」，
# 导致「商标法实施条例第一条」被解析成「商标法第一条」。
_LAW_NAME_STOP_HEAD = "条款项第的了和与及是在对把被给从这那"

_LAW_NAME_PATTERN = re.compile(
    # {1,18} 而不是 {2,18}：「民法典」是「民」+「法典」，
    # 「宪法」是「宪」+「法」，前缀只需 1 个字符。
    r"([\u4e00-\u9fff]{1,18}(?:法典|条例|细则|办法|解释|规定|法))"
)


def law_reference_in(text: str) -> str | None:
    """抽出提问里提到的法律名（宽松匹配，不要求存在于语料）。

    用途：判断用户问的是不是**语料库之外**的法律。
      - 「民法典第1165条」——语料里没有民法典，不能拿生态环境法典的
        第 1165 条充数（条号恰好唯一，会伪装成精确命中）
      - 「商标法实施条例第一条」——只匹配到「商标法」，返回商标法第一条，
        但用户问的是实施条例

    这两类都必须放弃精确通道，回落到语义通道。
    """
    if not text:
        return None
    # 跳过以条文结构词/虚词开头的候选：
    # 「X法第2条怎么规定的」会匹配出「条怎么规定」，那是句子不是法律名。
    for match in _LAW_NAME_PATTERN.finditer(text):
        candidate = match.group(1)
        if candidate[0] not in _LAW_NAME_STOP_HEAD:
            return candidate
    return None


def match_law_name(text: str, laws: Sequence[str]) -> str | None:
    """从提问里识别法律名。返回命中的**全称**，没有则 None。

    必须识别法律名，否则多法源下「宪法第一条」会被解析成任意一部法律的
    第一条 —— 实测就返回了「劳动法 第一条」，比答错更危险的是它看起来
    像是精确命中（score=1.00）。

    取**最长匹配**：提问「商标法实施条例第十条」同时命中「商标法」和
    「商标法实施条例」，必须选后者。
    """
    if not text:
        return None
    best: tuple[int, str] | None = None
    for law in laws:
        if not law:
            continue
        short = law[len(_COMMON_PREFIX):] if law.startswith(_COMMON_PREFIX) else law
        for candidate in {law, short}:
            if candidate and candidate in text:
                if best is None or len(candidate) > best[0]:
                    best = (len(candidate), law)
    return best[1] if best else None


def same_law(cited: str | None, known: str) -> bool:
    """法律名宽松匹配：「宪法」与「中华人民共和国宪法」视为同一部。"""
    if not cited:
        return True
    a, b = cited.strip(), known.strip()
    if not a or not b:
        return True
    return a in b or b in a


def safe_cosine(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """余弦相似度，零向量返回 0.0 而不是 NaN。

    `HashEmbeddings` 对空串返回全零向量，直接算余弦会得到 NaN，
    而 NaN 参与任何比较都是 False，会让路由静默走错分支（BC-07）。
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    score = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return score


def minmax_normalize(scores: Sequence[float]) -> list[float]:
    """把一组分数压到 [0, 1]。

    注意：**归一化后的 top-1 恒为 1.0**，与绝对相关性无关。
    这正是它不能用来比阈值的原因。
    """
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-9:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def authority_weight(doc_type: str) -> float:
    """效力层级权重。未知类型给默认值，不抛异常。"""
    return config.AUTHORITY_WEIGHTS.get((doc_type or "").strip(), config.AUTHORITY_DEFAULT)


def fuse_rank(candidates: Iterable[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
    """按融合分重排候选，返回新列表（不改原对象）。

    融合分 = 0.75 × 归一化语义分 + 0.25 × 层级权重（§6.4.4）。

    输入每个 dict 需含 `dense_score`；`doc_type` 可选。
    输出的每个 dict 会多一个 `fused_score` 字段，便于观测排序依据。
    """
    items = [dict(item) for item in candidates]
    if not items:
        return []

    dense = []
    for item in items:
        try:
            dense.append(float(item.get("dense_score") or 0.0))
        except (TypeError, ValueError):
            dense.append(0.0)
    normalized = minmax_normalize(dense)

    for item, norm, raw in zip(items, normalized, dense):
        item["dense_score"] = raw
        item["fused_score"] = (
            config.FUSION_DENSE_WEIGHT * norm
            + config.FUSION_AUTHORITY_WEIGHT * authority_weight(item.get("doc_type", ""))
        )

    items.sort(key=lambda x: x["fused_score"], reverse=True)
    if top_k is not None and top_k >= 0:
        return items[:top_k]
    return items
