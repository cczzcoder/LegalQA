"""阶段 0 唯一的工具：法条检索。

对应设计文档 §6.4。工具在循环里的角色与 v0.4 的「节点」不同：
**调不调、调几次由模型决定**，本模块只负责「给定 query，返回带判断依据的结果」。

返回给模型的不是裸数据，而是一段带结论的文本 —— 因为检索分数低时，
模型需要知道「该换关键词重试」还是「该如实说没找到」。没有这句话，
1.5B 会拿 0.62 的结果硬答。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from langchain_core.messages import AnyMessage

from legalqa import config
from legalqa.corpus.index import Embedder, VectorIndex
from legalqa.pure.retrieval import (
    extract_article_no,
    fuse_rank,
    law_reference_in,
    match_law_name,
    same_law,
)
from legalqa.pure.rewrite import merge_rewrite_result, needs_rewrite

__all__ = ["ReadTool", "ToolResult", "format_context", "READ_TOOL_SPEC"]

Rewriter = Callable[[str, Sequence[AnyMessage]], str | None]

# 供 llm.bind_tools 使用。描述要写清「什么时候该调」，1.5B 不会自己推断。
READ_TOOL_SPEC: dict[str, Any] = {
    "name": "read_law",
    "description": (
        "检索中国法律法规条文。当用户的问题涉及具体法律依据、权利义务、"
        "责任认定、赔偿标准时调用。可以多次调用：如果第一次结果相关度偏低，"
        "换用更具体的法律术语再试一次。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索用的法律问题描述，应包含主体、行为和争议点，不要用代词",
            }
        },
        "required": ["query"],
    },
}


@dataclass
class ToolResult:
    """一次检索的完整结果。字段与 State 的 E 组对应。"""

    docs: list[dict[str, Any]] = field(default_factory=list)
    score: float = 0.0            # 仅语义通道原始余弦；精确通道恒为 1.0
    dense_top1: float = 0.0
    channel: str = "none"         # exact / dense / none
    latency_ms: int = 0
    query: str = ""               # 实际用于检索的串（可能是改写结果）
    rewrite_applied: bool = False
    rewrite_reason: str = ""
    message: str = ""             # 给模型看的文本
    error: str = ""

    @property
    def hit(self) -> bool:
        return bool(self.docs)


# ---------------------------------------------------------------------------
# 上下文渲染
# ---------------------------------------------------------------------------
def describe_validity(doc: dict[str, Any]) -> str:
    """法条的时效说明。

    法律有时效性，引用时必须让模型和用户都能看到「这是哪个版本」。
    优先级：施行日期 > 版本日期 > 仅「现行有效」。
    """
    status = "现行有效" if doc.get("is_valid", True) else "已废止"
    effective = str(doc.get("effective_date") or "").strip()
    version = str(doc.get("version_date") or "").strip()
    if effective:
        return f"{status}，{effective} 施行"
    if version:
        return f"{status}，{version} 版本"
    return status


def format_context(result: ToolResult, *, threshold: float | None = None) -> str:
    """把检索结果渲染成模型可读的文本。

    三要素缺一不可（§6.4.6）：序号、法律名+条号、有效性。
    少任何一个，小模型就有编造空间。
    """
    limit = config.RAG_THRESHOLD if threshold is None else threshold
    head = f"【检索结果】相关度 {result.score:.2f}（可信阈值 {limit:.2f}）"

    if not result.docs:
        return (
            f"{head}\n"
            "未检索到相关法条。\n"
            "【提示】请如实告知用户没有找到直接依据，并请用户补充时间、主体、"
            "争议金额等要素；不要凭印象作答。"
        )

    blocks = [head]
    for i, doc in enumerate(result.docs, start=1):
        law = doc.get("law_name", "")
        label = doc.get("article_label") or f"第{doc.get('article_no', '?')}条"
        blocks.append(f"【法条 {i}】《{law}》{label}（{describe_validity(doc)}）\n{doc.get('text', '')}")

    blocks.append(
        "【引用约束】只能引用上述列出的条文；若上述条文不足以回答，"
        "请明确说明「现有资料不足」，不得引用未列出的法条，不得虚构条号或条文内容。\n"
        "引用时须注明该条文的版本（括号里的日期），例如「《民法典》第 1087 条"
        "（2020-05-28 版本）」。法律有时效性，读者需要知道依据的是哪一版。"
    )

    if result.score < limit:
        blocks.append(
            "【提示】相关度偏低，可能是关键词不匹配。建议换用更具体的法律术语重试一次，"
            "或如实告知用户未找到依据。"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
class ReadTool:
    """只读检索工具。无副作用，可重复调用。"""

    name = "read_law"

    def __init__(
        self,
        index: VectorIndex,
        embedder: Embedder,
        rewriter: Rewriter | None = None,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._rewriter = rewriter

    # -- 内部 ---------------------------------------------------------
    def _resolve_query(
        self, query: str, history: Sequence[AnyMessage] | None
    ) -> tuple[str, bool, str]:
        """改写预判 + 执行。任何失败都回退原文（BC-35）。"""
        text = (query or "").strip()
        decision = needs_rewrite(text)
        if not decision.needed or self._rewriter is None:
            return text, False, "passthrough" if self._rewriter else "no_rewriter"
        if not history:
            return text, False, "no_history"
        try:
            raw = self._rewriter(text, history)
        except Exception:                      # 改写永远不能阻塞主链路
            return text, False, "timeout"
        return merge_rewrite_result(text, raw, applied=True)

    def run(
        self,
        query: str,
        history: Sequence[AnyMessage] | None = None,
    ) -> ToolResult:
        started = time.perf_counter()
        result = ToolResult()

        try:
            text, applied, reason = self._resolve_query(query, history)
            result.query = text
            result.rewrite_applied = applied
            result.rewrite_reason = reason

            if not text:
                result.error = "empty_query"
                result.message = format_context(result)
                return result

            # ---- 通道 B：条号精确命中，短路 ----
            article_no = extract_article_no(text)
            if article_no is not None:
                # 先识别法律名：「宪法第一条」和「劳动法第一条」是不同条文。
                # 认不出法律名且该条号在多部法律里都存在时，by_article 返回 None，
                # 自动回落到语义通道，不会给出「看起来精确」的错误答案。
                law_name = match_law_name(text, self._index.law_names)
                law_ref = law_reference_in(text)

                # 提问提到的法律与语料里匹配到的不是同一部 → 放弃精确通道。
                # 「民法典第1165条」不能拿生态环境法典第1165条充数；
                # 「商标法实施条例第一条」不能退化成「商标法第一条」。
                if law_ref and not (law_name and same_law(law_ref, law_name)):
                    doc = None
                else:
                    doc = self._index.by_article(article_no, law_name)

                if doc is not None:
                    result.docs = [doc]
                    result.score = 1.0
                    result.dense_top1 = 0.0
                    result.channel = "exact"
                    result.message = format_context(result)
                    return result

            # ---- 通道 A：语义召回 + 融合重排 ----
            vector = self._embedder.embed_query(text)
            hits = self._index.search(vector, k=config.RAG_RECALL_K)
            if not hits:
                result.error = "no_candidates"
                result.message = format_context(result)
                return result

            result.dense_top1 = hits[0].score
            result.score = hits[0].score          # 阈值只吃这个值（BC-26）
            candidates = [
                {**hit.doc, "dense_score": hit.score} for hit in hits
            ]
            ranked = fuse_rank(candidates, top_k=config.RAG_TOP_K)
            result.docs = ranked
            result.channel = "dense"
            result.message = format_context(result)
            return result

        except Exception as exc:                  # 工具异常不穿透，转成可读文本
            result.error = f"{type(exc).__name__}: {exc}"
            result.channel = "none"
            result.docs = []
            result.score = 0.0
            result.message = (
                f"【检索失败】{result.error}\n"
                "【提示】检索服务异常，请如实告知用户暂时无法查阅法条依据，"
                "并建议稍后重试。不要凭印象作答。"
            )
            return result
        finally:
            result.latency_ms = int((time.perf_counter() - started) * 1000)
