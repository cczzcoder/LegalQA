"""评测检查项：把「答案好不好」拆成可自动判定的事实。

对应设计文档 §14。

**为什么先做规则检查，而不是直接上 LLM 打分。**

LLM-as-judge 能评「答案是否准确」，但成本高、结果不稳定（同一个答案两次打分可能不同），
而且它评不了最关键的一条——**引用是不是编的**。

引用幻觉是法律问答的头号风险，而它完全可以用规则判定：把回答里的每个
「《法律》第 X 条」拿去索引里查，查不到就是编的。这是确定性的、零成本的、
可以每轮都跑的。

所以分两层：
  1. 规则检查（本模块）—— 每次改动都能跑，秒级反馈
  2. LLM judge（可选）—— 阶段性评估，看答案质量
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from legalqa.nodes.guards import DISCLAIMER
from legalqa.pure.retrieval import extract_citations, same_law

__all__ = [
    "CheckResult",
    "check_disclaimer",
    "check_length",
    "check_traceability",
    "check_hallucination",
    "check_refusal",
    "check_keywords",
    "check_expected_citation",
]

# 答案长度合理区间。太短说明敷衍，太长说明没抓住重点。
MIN_ANSWER_CHARS = 80
MAX_ANSWER_CHARS = 2500

# 表达「没找到依据」的说法。无关问题必须出现其中之一。
_REFUSAL_MARKERS = (
    "现有资料不足", "未涵盖", "未找到", "没有检索到", "未检索到",
    "无法给出", "无法回答", "暂无法", "不在本知识库", "超出",
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseReport:
    case_id: str
    query: str
    checks: list[CheckResult] = field(default_factory=list)
    latency_ms: int = 0
    iterations: int = 0
    channel: str = ""
    score: float = 0.0

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


# ---------------------------------------------------------------------------
# 单项检查
# ---------------------------------------------------------------------------
def check_disclaimer(text: str) -> CheckResult:
    ok = DISCLAIMER in (text or "")
    return CheckResult("免责声明", ok, "" if ok else "回答缺少免责声明")


def check_length(text: str) -> CheckResult:
    length = len((text or "").strip())
    if length < MIN_ANSWER_CHARS:
        return CheckResult("长度合理", False, f"仅 {length} 字，可能敷衍")
    if length > MAX_ANSWER_CHARS:
        return CheckResult("长度合理", False, f"{length} 字，可能啰嗦")
    return CheckResult("长度合理", True, f"{length} 字")


def check_traceability(text: str, docs: Sequence[dict[str, Any]]) -> CheckResult:
    """引用的条号是否都能在本次检索材料里找到。"""
    allowed_nos = {
        d.get("article_no") for d in docs if isinstance(d.get("article_no"), int)
    }
    allowed_laws = {str(d.get("law_name", "")) for d in docs if d.get("law_name")}
    if not allowed_nos:
        return CheckResult("引用可溯源", True, "无检索材料，跳过")

    # 按条号去重：同一部法律在回答里可能写成「劳动合同法」「中华人民共和国劳动合同法」
    # 两种形式，按字符串去重会重复列出。
    unverified: set[int] = set()
    for law, number in extract_citations(text):
        if law and allowed_laws and not any(same_law(law, k) for k in allowed_laws):
            continue                       # 语料外法律，不归此项管
        if number not in allowed_nos:
            unverified.add(number)
    if unverified:
        shown = "、".join(f"第{n}条" for n in sorted(unverified)[:5])
        return CheckResult("引用可溯源", False, f"未溯源: {shown}")
    return CheckResult("引用可溯源", True)


def check_hallucination(text: str, index: Any) -> CheckResult:
    """引用的条号是否真实存在于语料。

    区分两种情况：
      - 引用了语料里的法律，但条号不存在（「劳动法第 999 条」）→ **幻觉**
      - 引用了语料外的法律（《刑法》第 245 条）→ 不算幻觉，只是超出知识库

    这是法律问答最该守住的一条：编造条号比答不出来恶劣得多。
    """
    if index is None or len(index) == 0:
        return CheckResult("无幻觉条号", True, "无索引，跳过")

    law_names = index.law_names
    fake: dict[int, str] = {}
    for law, number in extract_citations(text):
        if not law:
            continue
        matched = next((n for n in law_names if same_law(law, n)), None)
        if matched is None:
            continue                       # 语料外法律，不算幻觉
        if index.by_article(number, matched) is None:
            fake[number] = f"《{matched}》第{number}条"

    if fake:
        shown = "、".join(fake[n] for n in sorted(fake)[:5])
        return CheckResult("无幻觉条号", False, f"不存在的条号: {shown}")
    return CheckResult("无幻觉条号", True)


def check_refusal(text: str, should_answer: bool) -> CheckResult:
    """拒答是否恰当。

    无关问题该拒答（或明确说没找到），相关问题不该一上来就拒答。
    """
    said_no = any(marker in (text or "") for marker in _REFUSAL_MARKERS)
    if should_answer and said_no and len((text or "").strip()) < MIN_ANSWER_CHARS:
        return CheckResult("拒答恰当", False, "该回答却拒答了")
    if not should_answer and not said_no:
        return CheckResult("拒答恰当", False, "无关问题未明确说明缺少依据")
    return CheckResult("拒答恰当", True)


def check_keywords(text: str, keywords: Sequence[str]) -> CheckResult:
    """关键要素是否说到点子上。

    `keywords` 是**同义选项**，命中任意一个即通过。比如问拖欠工资，
    「仲裁」「投诉」「劳动监察」提到哪个都算答到点子上了；
    要求全部命中会把同义表达判成失败。

    法律回答的价值不只在引用正确——问拖欠工资却一个维权途径都没提，
    即使法条引对了也是残缺的。
    """
    if not keywords:
        return CheckResult("关键要素", True, "未指定")
    body = text or ""
    if any(k in body for k in keywords):
        return CheckResult("关键要素", True)
    return CheckResult("关键要素", False, "均未提及: " + "、".join(keywords))


def check_expected_citation(
    text: str, expected_law: str | None, expected_article_no: int | None
) -> CheckResult:
    """答案有没有引用「正确的那条法条」。

    这个指标和「引用可溯源」互补：
      - 可溯源 = 引的条号在检索材料里（测检索覆盖）
      - 命中期望 = 引的条号是对的（测整条链路）

    端到端指标之所以对检索改善不敏感，就是因为缺这一项——
    「引用可溯源」只要求「引的是材料里的」，不要求「引的是对的」。
    """
    if not expected_law or expected_article_no is None:
        return CheckResult("命中期望法条", True, "未指定")

    for cited_law, number in extract_citations(text):
        if number != expected_article_no:
            continue
        # 没写法律名时放宽（模型常承前省略：「依据第四十七条」）
        if cited_law is None or same_law(cited_law, expected_law):
            return CheckResult("命中期望法条", True)
    short = str(expected_law).replace("中华人民共和国", "")
    return CheckResult("命中期望法条", False, f"未引用《{short}》第{expected_article_no}条")


def extract_answer(messages: Sequence[Any]) -> str:
    """取最后一条 AI 消息的正文。"""
    from langchain_core.messages import AIMessage

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return str(message.content)
    return ""
