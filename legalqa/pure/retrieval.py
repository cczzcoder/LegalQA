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
    "int_to_cn",
    "extract_article_no",
    "extract_all_article_nos",
    "extract_citations",
    "match_law_name",
    "law_reference_in",
    "normalize_law",
    "same_law",
    "resolve_law",
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


def int_to_cn(number: int) -> str:
    """整数转中文数字（条号用，1~9999）。超范围或非法输入回退十进制字符串。

    cn_to_int 的反向：引用剥离时需要把「79」写成「第七十九条」去匹配正文。
    """
    if not isinstance(number, int) or not 0 <= number < 10000:
        return str(number)
    if number == 0:
        return "零"
    digits = [int(ch) for ch in str(number)]
    parts: list[str] = []
    zero_pending = False
    for index, digit in enumerate(digits):
        pos = len(digits) - index - 1
        if digit == 0:
            zero_pending = True
            continue
        if zero_pending and parts:
            parts.append("零")
        zero_pending = False
        # 「十五」不是「一十五」；「一百一十」里的「一十」要保留。
        if pos == 1 and digit == 1 and not parts:
            parts.append("十")
        else:
            parts.append("零一二三四五六七八九"[digit])
            parts.append(("", "十", "百", "千")[pos])
    return "".join(parts)


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
_COMMON_PREFIX = "中华人民共和国"
# 法名比较前统一去掉的包装字符：模型写司法解释时常加〈〉，语料里没有。
_LAW_NOISE = str.maketrans("", "", "《》〈〉（）()【】[] \t\u3000")
_SENTENCE_END = "。"
# 法名只归给同一句里其后「紧挨着枚举」的条号（第十四条、第十七条和第三十七条）。
# 中间隔了大段正文再出现另一个条号时不再归因——实测 071：一句里先写
# 《实施条例》第五十二条，隔了 15 个字再写「与第九百二十七条一致」，
# 旧实现把 927 也归到实施条例名下，误报「实施条例第 927 条不存在」。
# 放弃归因的裸条号由 OutputGuard 的「本批材料」兜底解析。
_ENUM_GAP = 6


def normalize_law(name: str | None) -> str:
    """法名归一化：去包装字符、去「中华人民共和国」前缀、去空白。

    只做规范化，不做匹配。匹配规则在 `same_law` / `resolve_law` 里。
    """
    text = str(name or "").translate(_LAW_NOISE).strip()
    if text.startswith(_COMMON_PREFIX):
        text = text[len(_COMMON_PREFIX):]
    return text


def _bare_law_mentions(span: str, laws: Sequence[str]) -> list[tuple[int, str]]:
    """在 span 里找裸法名（不带《》），返回 `(结束位置, 命中的名字)`。

    同一结束位置取更长的名字：「劳动合同法实施条例」不能退化成「劳动合同法」。
    """
    best: dict[int, str] = {}
    for law in laws:
        if not law:
            continue
        candidates = {law}
        if law.startswith(_COMMON_PREFIX):
            candidates.add(law[len(_COMMON_PREFIX):])
        for candidate in candidates:
            if len(candidate) < 2:
                continue
            start = 0
            while True:
                index = span.find(candidate, start)
                if index < 0:
                    break
                end = index + len(candidate)
                current = best.get(end)
                if current is None or len(candidate) > len(current):
                    best[end] = candidate
                start = index + 1
    return list(best.items())


def extract_citations(
    text: str, law_names: Sequence[str] = ()
) -> list[tuple[str | None, int]]:
    """抽取「法律名 + 条号」对。

    为什么不能只抽条号：模型会写「《刑法》第二百四十五条」，
    只抽条号会把它当成宪法第 245 条去校验，产生误报。

    法律名取该条号**之前最近**的一个法名标记；没有则返回 None（承前省略）。
    法名标记有三种，按「结束位置离条号最近」优先：

      1. 《法律名》括号标记
      2. 裸法名（需要传 `law_names`，否则做不到）
      3. 「该法 / 本法」——指代同句内最近一个显式法名

    第 2、3 条是实测补的：模型引用某条原文时，原文里会写
    「依照劳动合同法第八十七条的规定」——只认《》会把 87 归到引文所属的
    《劳动合同法实施条例》名下，于是误报「实施条例第 87 条不存在」。
    """
    if not text:
        return []
    marks = [(m.start(), m.end(), m.group(1)) for m in _LAW_CITE.finditer(text)]
    pairs: list[tuple[str | None, int]] = []
    for match in _ARTICLE_INNER.finditer(text):
        pos = match.start()
        # 法律名与条号之间不能跨句：早期实现把「《通知》……。依据《工伤保险条例》
        # 第十四条」的 14 归到了《通知》名下，溯源时误报「通知第 14 条未检索到」。
        start = text.rfind(_SENTENCE_END, 0, pos) + 1
        sentence_articles = [
            (m.start(), m.end()) for m in _ARTICLE_INNER.finditer(text, start)
        ]

        def _blocked_by_intervening_article(begin: int) -> bool:
            return any(
                begin <= p_start < pos and pos - p_end > _ENUM_GAP
                for p_start, p_end in sentence_articles
            )

        mentions: list[tuple[int, str | None]] = []

        for mark_start, mark_end, name in marks:
            if (
                start <= mark_start < pos
                and mark_end <= pos
                and not _blocked_by_intervening_article(mark_end)
            ):
                mentions.append((mark_end, name))
        for rel_end, candidate in _bare_law_mentions(text[start:pos], law_names):
            end = start + rel_end
            if not _blocked_by_intervening_article(end):
                mentions.append((end, candidate))

        explicit = sorted(mentions, key=lambda item: (item[0], len(item[1] or "")))
        last_law = explicit[-1][1] if explicit else None
        for token in ("该法", "本法"):
            index = text.rfind(token, start, pos)
            if index >= 0:
                mentions.append((index + len(token), last_law))

        law = (
            max(mentions, key=lambda item: (item[0], len(item[1] or "")))[1]
            if mentions
            else None
        )
        number = cn_to_int(match.group(1))
        if number is not None:
            pairs.append((law, number))
    return pairs

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
    """法律名宽松匹配：「宪法」与「中华人民共和国宪法」视为同一部。

    **不能做双向子串**。实测两个方向都会出事：
      - 「解释（一）」含有「民法典」→ 被认成民法典，回捞成民法典第 N 条
      - 「劳动合同法」是「劳动合同法实施条例」的前缀 → 实施条例被当成法
    所以只允许「归一化后相等」或「一方是另一方的后缀」（简称省略前缀），
    且短名至少 2 个字符。
    """
    if not cited or not known:
        return True
    a, b = normalize_law(cited), normalize_law(known)
    if not a or not b:
        return True
    if a == b:
        return True
    # 短名只允许精确匹配：「合同法」是「劳动合同法」的后缀，但不是同一部法。
    # 后缀匹配只用于「婚姻家庭编的解释（一）」这类较长的省略形式。
    if min(len(a), len(b)) < 4:
        return False
    return a.endswith(b) or b.endswith(a)


def resolve_law(cited: str | None, law_names: Sequence[str]) -> str | None:
    """把回答里的法名解析成语料库中的全称。

    优先精确匹配；否则找唯一的后缀匹配。**多义返回 None**——
    法律场景宁可说「无法确定」，也不能随便挑一部（BC-24 同一个坑）。
    """
    if not cited or not law_names:
        return None
    target = normalize_law(cited)
    exact = [name for name in law_names if normalize_law(name) == target]
    if len(exact) == 1:
        return exact[0]
    hits = [name for name in law_names if same_law(cited, name)]
    return hits[0] if len(hits) == 1 else None


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
