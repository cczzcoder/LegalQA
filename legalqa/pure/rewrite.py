"""查询改写的纯逻辑：预判、prompt 构造、输出清洗、保真校验。

对应设计文档 §6.3.2 ~ §6.3.5。

本模块**不调用模型**，只负责「该不该改」和「改完能不能用」。
真正调 LLM 的部分在 `rewrite_query()`，由调用方注入客户端，便于测试。

设计上的一条硬规则（§6.3.3）：预判必须偏向透传。
漏判的代价是走 fallback（安全），误判的代价是检索错、答案错（有害）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from legalqa import config

__all__ = [
    "RewriteDecision",
    "needs_rewrite",
    "build_rewrite_prompt",
    "sanitize_rewrite_output",
    "extract_entities",
    "entities_preserved",
    "similarity_ok",
    "finalize_rewrite",
]

_PREFIX_NOISE = re.compile(r"^\s*(改写|输出|结果|独立问题|改写后)\s*[:：]\s*")
_QUOTE_WRAP = re.compile(r"^[\"'“”‘’「」『』]+|[\"'“”‘’「」『』]+$")


@dataclass(frozen=True)
class RewriteDecision:
    """预判结果。`reason` 会写进 state.rewrite_reason，用于统计改写率。"""

    needed: bool
    reason: str


# ---------------------------------------------------------------------------
# 预判
# ---------------------------------------------------------------------------
def _first_pronoun_pos(text: str) -> int:
    """返回第一个代词出现的下标，没有则 -1。"""
    best = -1
    for token in (*config.DEICTIC_PRONOUNS, *config.PERSONAL_PRONOUNS):
        pos = text.find(token)
        if pos >= 0 and (best < 0 or pos < best):
            best = pos
    return best


def needs_rewrite(query: str) -> RewriteDecision:
    """三层预判。判定顺序不可调换：L3 豁免优先于 L1/L2。

    顺序写反的后果：一条 40 字的完整问句里出现「该公司」会被送去改写，
    而小模型改写长句最容易跑偏。
    """
    text = (query or "").strip()
    if not text:
        return RewriteDecision(False, "empty")

    # ---- L3 豁免：显式条号，必然自包含 ----
    if config.ARTICLE_NO_PATTERN.search(text):
        return RewriteDecision(False, "exempt_article_no")

    # ---- L3 豁免：长句 + 代词出现在句中（有句内先行词） ----
    # 这条是对文档规则的细化：文档只写「≥30 字且无代词」，
    # 但「公司说我这个月绩效不达标要辞退我，这样合法吗」同样自包含。
    pronoun_pos = _first_pronoun_pos(text)
    if len(text) >= config.REWRITE_LONG_CHARS and pronoun_pos >= 8:
        return RewriteDecision(False, "exempt_long")

    # 去掉句末标点再判「呢」，否则「工伤呢？」永远匹配不到
    stripped = text.rstrip("？?。！!…~ ")

    # ---- L1 强信号 ----
    if text.startswith(config.REWRITE_STRONG_PREFIX):
        return RewriteDecision(True, "strong_prefix")
    # 只认「呢」，不认「吗」：「吗」是完整疑问句的标准语气词，
    # 「公司拖欠工资三个月了，我能要赔偿吗」是自包含的，不该被改写。
    if stripped.endswith("呢") and len(stripped) < config.REWRITE_TAIL_SHORT:
        return RewriteDecision(True, "tail_question")

    # ---- L2 弱信号 ----
    # 放在 too_short 之前：同样要改写，「这个合同有效吗」判 deictic
    # 比判 too_short 更有诊断价值（说明是指代问题，不是信息量问题）。
    if any(token in text for token in config.DEICTIC_PRONOUNS):
        return RewriteDecision(True, "deictic")
    if any(token in text for token in config.PERSONAL_PRONOUNS):
        return RewriteDecision(True, "personal")

    # ---- 兜底：太短且没有明显信号 ----
    if len(stripped) < config.REWRITE_MIN_CHARS:
        return RewriteDecision(True, "too_short")

    return RewriteDecision(False, "passthrough")


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------
_REWRITE_INSTRUCTION = (
    "把「最新问题」改写成一个不依赖上下文的独立问题。\n"
    "\n"
    "要求：\n"
    "1. 只输出改写后的问句，不要解释，不要加前缀\n"
    "2. 保留原文的主体、行为、金额、时间\n"
    "3. 不添加原文和对话里都没有的信息\n"
    "4. 无法补全时，原样输出\n"
)

_FEW_SHOT = (
    ("公司没签劳动合同，能要双倍工资吗", "那能要多少？",
     "未签劳动合同的双倍工资最多可以主张多少个月？"),
    ("我在工地摔伤了，包工头不认", "工伤呢？",
     "在工地摔伤但包工头不承认劳动关系，能否认定工伤？"),
    ("", "劳动合同到期不续签有补偿吗", "劳动合同到期不续签有补偿吗"),
)


def _recent_history(messages: Sequence[AnyMessage]) -> list[str]:
    """只取最近 N 轮，更早的与指代消解无关，还白占 token。"""
    limit = config.REWRITE_HISTORY_TURNS * 2
    recent = list(messages)[-limit:] if messages else []
    lines: list[str] = []
    for message in recent:
        content = message.content if isinstance(message.content, str) else ""
        content = content.strip()
        if not content:
            continue
        role = "用户" if isinstance(message, HumanMessage) else "助手"
        lines.append(f"{role}：{content[:200]}")
    return lines


def build_rewrite_prompt(query: str, messages: Sequence[AnyMessage]) -> str:
    """构造改写 prompt。

    三个 few-shot 里第三个是「原样输出」——明确告诉模型「不改」也是合法输出，
    显著降低 1.5B 模型的乱改倾向。
    """
    parts: list[str] = [_REWRITE_INSTRUCTION]
    for i, (hist, question, answer) in enumerate(_FEW_SHOT, start=1):
        parts.append(f"示例{i}：")
        if hist:
            parts.append(f"历史：用户问「{hist}」")
        parts.append(f"最新问题：{question}")
        parts.append(f"输出：{answer}")
        parts.append("")

    history_lines = _recent_history(messages)
    parts.append("对话历史：")
    parts.extend(history_lines or ["（无）"])
    parts.append("")
    parts.append(f"最新问题：{query.strip()}")
    parts.append("输出：")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 输出清洗
# ---------------------------------------------------------------------------
def sanitize_rewrite_output(raw: str | None) -> str | None:
    """剥前缀、去引号、只取第一行、长度校验。

    任何一步不通过返回 None，调用方据此回退原文。
    """
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None

    # 只取第一行（小模型爱输出「改写：xxx\n解释：yyy」）
    text = text.splitlines()[0].strip()
    text = _PREFIX_NOISE.sub("", text)
    text = _QUOTE_WRAP.sub("", text).strip()

    if not (config.REWRITE_OUTPUT_MIN <= len(text) <= config.REWRITE_OUTPUT_MAX):
        return None
    return text


# ---------------------------------------------------------------------------
# 实体保真
# ---------------------------------------------------------------------------
def extract_entities(text: str) -> set[str]:
    """抽取必须保真的实体：数字+单位、裸数字、法律术语。

    不做语义理解，纯正则 + 词表。宁可粗糙也不能漏 ——
    小模型把「3 年」改成「5 年」是最危险的失败（§6.3.5）。
    """
    if not text:
        return set()
    found: set[str] = set()
    for pattern in config.ENTITY_PATTERNS:
        found.update(m.group(0).strip() for m in pattern.finditer(text))
    for term in config.LEGAL_TERMS:
        if term in text:
            found.add(term)
    return {item for item in found if item}


def entities_preserved(source: str, rewritten: str) -> bool:
    """原文实体必须全部出现在改写结果里（子集校验）。

    只要求「包含」，不要求「不多」—— 补全省略成分本来就会引入新词。
    """
    required = extract_entities(source)
    if not required:
        return True
    target = rewritten or ""
    return all(entity in target for entity in required)


def _bigrams(text: str) -> set[str]:
    cleaned = re.sub(r"\s+", "", text or "")
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def similarity_ok(source: str, rewritten: str) -> bool:
    """语义漂移粗检：字符 bigram 的 Jaccard 相似度。

    **只用一个判据，不用长度膨胀率。** 原设计想用「相似度低 且 膨胀超 3 倍」
    双条件，实测发现膨胀率是个坏信号：合理补全本来就会显著变长
    （「工伤呢」→ 24 字，膨胀 6 倍），拿它卡必然误杀。

    所以规则简化为：
      - 空输入直接判否（空向量算余弦会得 NaN）
      - 源问句 < 10 字直接放行（太短，Jaccard 本身不可靠；
        漂移风险由实体保真 + 200 字上限兜住）
      - 其余要求 Jaccard ≥ REWRITE_MIN_SIMILARITY
    """
    src = (source or "").strip()
    dst = (rewritten or "").strip()
    if not src or not dst:
        return False
    if len(src) < 10:
        return True

    a, b = _bigrams(src), _bigrams(dst)
    if not a or not b:
        return False
    jaccard = len(a & b) / len(a | b)
    return jaccard >= config.REWRITE_MIN_SIMILARITY


def finalize_rewrite(source: str, raw: str | None) -> tuple[str | None, str]:
    """把模型原始输出变成可用查询。

    返回 `(可用改写, 原因)`；`None` 表示必须回退原文。
    原因取值与 state.rewrite_reason 对齐。
    """
    cleaned = sanitize_rewrite_output(raw)
    if cleaned is None:
        return None, "invalid_output"
    if cleaned == source.strip():
        return cleaned, "rewritten"          # 模型选择原样输出，也算合法
    if not entities_preserved(source, cleaned):
        return None, "invalid_output"
    if not similarity_ok(source, cleaned):
        return None, "invalid_output"
    return cleaned, "rewritten"


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------
def merge_rewrite_result(
    source: str, raw: str | None, *, applied: bool
) -> tuple[str, bool, str]:
    """产出 `(rewritten_query, rewrite_applied, rewrite_reason)` 三元组。

    **任何失败路径都返回可用的 query，绝不返回空串**（BC-35）：
    空串会让 ToolNode 直接判 low_score，等于白丢一轮。
    """
    if not applied:
        return source, False, "passthrough"
    cleaned, reason = finalize_rewrite(source, raw)
    if cleaned is None:
        return source, False, reason
    return cleaned, True, reason
