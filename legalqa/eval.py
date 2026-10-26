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
from legalqa.pure.retrieval import extract_citations, resolve_law, same_law

__all__ = [
    "CheckResult",
    "check_disclaimer",
    "check_length",
    "check_traceability",
    "check_hallucination",
    "check_refusal",
    "check_keywords",
    "check_expected_citation",
    "extract_answer",
    "strip_backfill_appendix",
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
    # 留证据。没有这两项，任何一次「为什么失败」都得重跑 6 分钟评测才能看，
    # 而重跑的结果和上一次不完全一样（并发 + 网络），根本没法逐条比对。
    answer: str = ""
    # 只放**工具真正检索到的**材料。回捞条目单独放 backfilled，
    # 这样诊断时能区分「检索到了」和「没检索到但回查确认存在」。
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    backfilled: list[dict[str, Any]] = field(default_factory=list)
    # 检索覆盖：引用是否落在**原始检索材料**里。不参与通过判定，
    # 只用来回答「检索层到底漏了多少」——这是和「引用是否可核验」不同的指标。
    retrieval_coverage: CheckResult | None = None

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
    """引用的条号是否都能在给定材料里找到。

    `docs` 可以是「检索 + 回捞核验」的合并材料（通过判定用），也可以是
    纯原始检索材料（`retrieval_coverage` 用）。两者语义不同，调用方明确传入。

    **必须按 (法律名, 条号) 成对判断。** 分开比会放过一类真实错误：
    材料里有《劳动法》第 38 条、《劳动合同法》第 44 条，回答引《劳动合同法》
    第 38 条——「38 在条号集合里」和「劳动合同法在法源集合里」都成立，
    但这一条压根没检索到。实测 labor-006 就是这么漏过去的。
    """
    allowed_pairs = {
        (str(d.get("law_name", "")), int(d.get("article_no")))
        for d in docs
        if isinstance(d.get("article_no"), int) and d.get("law_name")
    }
    if not allowed_pairs:
        return CheckResult("引用可溯源", True, "无检索材料，跳过")
    allowed_nos = {no for _, no in allowed_pairs}
    allowed_laws = {name for name, _ in allowed_pairs}

    # 按条号去重：同一部法律在回答里可能写成「劳动合同法」「中华人民共和国劳动合同法」
    # 两种形式，按字符串去重会重复列出。
    unverified: set[int] = set()
    # 传 allowed_laws 才能识别裸法名（「依照劳动合同法第八十七条」）。
    # sorted 保证解析结果可复现（set 的遍历顺序不稳定）。
    for law, number in extract_citations(text, sorted(allowed_laws)):
        if law and not any(same_law(law, k) for k in allowed_laws):
            continue                       # 本批材料之外的法律，不归此项管
        if law is None:
            # 承前省略法律名：只看条号，放宽（否则「依据第四十七条」会大面积误报）
            if number not in allowed_nos:
                unverified.add(number)
            continue
        if not any(same_law(law, name) and no == number for name, no in allowed_pairs):
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
    for law, number in extract_citations(text, law_names):
        if not law:
            continue
        # 用唯一解析，不能 `next(...)`：嵌套解释名会被首个命中的「民法典」吃掉。
        matched = resolve_law(law, law_names)
        if matched is None:
            continue                       # 语料外法律（或多义），不算幻觉
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


# 引用回捞附录的起始标记（与 guards._render_backfill 对应）
_BACKFILL_MARK = "【补充核对】"
_WARNING_MARK = "【引用提示】"


def strip_backfill_appendix(text: str) -> str:
    """去掉「【补充核对】」附录，只保留模型自己写的正文。

    **附录里是法条原文，不能当作模型的引用行为。** 实测 labor-006：
    回捞的《劳动合同法》第四十六条原文里写着「（一）劳动者依照本法第三十八条
    规定解除劳动合同的……」——评测把这几处交叉引用当成了模型引的条号，
    于是判「第36/38/40/41 条未溯源」。系统因为忠实引用原文而被扣分。

    更危险的是反方向：`check_expected_citation` 可能在附录原文里碰巧撞上
    期望条号，让一条**没引对**的回答被判成通过。两个方向都必须切掉附录。

    保留附录之后的【引用提示】和免责声明——那是系统的输出，不是法条原文。
    """
    if not text:
        return ""
    start = text.find(_BACKFILL_MARK)
    if start < 0:
        return text
    warn = text.find(_WARNING_MARK, start)
    if warn >= 0:
        return text[:start] + text[warn:]
    disclaimer = text.find(DISCLAIMER, start)
    return text[:start] + (text[disclaimer:] if disclaimer >= 0 else "")
