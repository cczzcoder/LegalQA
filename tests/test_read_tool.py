"""S3 测试：检索工具。"""

from __future__ import annotations

import numpy as np
import pytest
from langchain_core.messages import HumanMessage

from legalqa import config
from legalqa.corpus.index import VectorIndex
from legalqa.tools.read_tool import READ_TOOL_SPEC, ReadTool, ToolResult, format_context

DOCS = [
    {"id": "x-0001", "law_name": "X法", "article_no": 1, "article_label": "第一条",
     "doc_type": "法律", "text": "第一条正文", "is_valid": True, "effective_date": "2020-01-01"},
    {"id": "x-0002", "law_name": "X法", "article_no": 2, "article_label": "第二条",
     "doc_type": "法律", "text": "第二条正文", "is_valid": True, "effective_date": "2020-01-01"},
    {"id": "x-0003", "law_name": "X法", "article_no": 3, "article_label": "第三条",
     "doc_type": "地方性法规", "text": "第三条正文", "is_valid": True, "effective_date": ""},
]

# 三个正交向量，方便构造确定的相关度
VECTORS = {
    "第一条正文": [1.0, 0.0, 0.0],
    "第二条正文": [0.0, 1.0, 0.0],
    "第三条正文": [0.0, 0.0, 1.0],
}


class FakeEmbedder:
    def __init__(self, query_vector: list[float] | None = None, fail: bool = False) -> None:
        self.query_vector = query_vector or [1.0, 0.0, 0.0]
        self.fail = fail
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [VECTORS.get(t, [0.0, 0.0, 0.0]) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding down")
        return self.query_vector


def make_tool(embedder: FakeEmbedder | None = None, rewriter=None) -> ReadTool:
    emb = embedder or FakeEmbedder()
    index = VectorIndex.build(DOCS, emb)
    return ReadTool(index, emb, rewriter=rewriter)


# ---------------------------------------------------------------------------
# 通道 B：条号精确
# ---------------------------------------------------------------------------
def test_exact_channel_short_circuits() -> None:
    tool = make_tool()
    result = tool.run("X法第2条怎么规定的")

    assert result.channel == "exact"
    assert result.score == 1.0
    assert len(result.docs) == 1
    assert result.docs[0]["article_no"] == 2


def test_exact_channel_skips_embedding() -> None:
    """精确命中时不应该再调向量化 —— 省一次推理。"""
    emb = FakeEmbedder()
    tool = make_tool(emb)
    tool.run("第1条")
    assert emb.calls == 0


def test_exact_miss_falls_back_to_dense() -> None:
    """条号不存在时回落语义通道，不报错（BC-24）。"""
    tool = make_tool()
    result = tool.run("第9999条")
    assert result.channel == "dense"
    assert result.error == ""


# ---------------------------------------------------------------------------
# 通道 A：语义
# ---------------------------------------------------------------------------
def test_dense_channel_returns_top_k() -> None:
    tool = make_tool()
    result = tool.run("某条正文")
    assert result.channel == "dense"
    # 语料只有 3 条，top_k=4 时取实际条数
    assert len(result.docs) == min(config.RAG_TOP_K, len(DOCS))
    assert result.docs[0]["article_no"] == 1


def test_score_is_raw_cosine_not_fused() -> None:
    """BC-26：state 里的 score 必须是原始余弦，不能是归一化后的融合分。"""
    tool = make_tool(FakeEmbedder([1.0, 0.0, 0.0]))
    result = tool.run("某条正文")

    assert result.score == pytest.approx(1.0, abs=1e-5)
    assert result.dense_top1 == pytest.approx(1.0, abs=1e-5)
    # 融合分不在 docs[0] 之外暴露，且 score 不等于 1.0 之外被改写
    assert "fused_score" in result.docs[0]


def test_low_score_keeps_raw_value() -> None:
    """相关度低时也不能把 score 抬到阈值以上。"""
    # 与三个正交向量都只有 0.577 的余弦，低于当前阈值
    tool = make_tool(FakeEmbedder([1.0, 1.0, 1.0]))
    result = tool.run("某条正文")
    assert result.score == pytest.approx(0.5773, abs=1e-3)
    assert result.score < config.RAG_THRESHOLD


# ---------------------------------------------------------------------------
# 异常与边界
# ---------------------------------------------------------------------------
def test_empty_query() -> None:
    tool = make_tool()
    result = tool.run("   ")
    assert result.docs == []
    assert result.error == "empty_query"
    assert "未检索到" in result.message


def test_embedder_failure_is_contained() -> None:
    """工具异常不穿透，转成模型可读文本。"""
    tool = make_tool(FakeEmbedder(fail=True))
    result = tool.run("某个法律问题")

    assert result.error.startswith("RuntimeError")
    assert result.docs == []
    assert result.score == 0.0
    assert "检索失败" in result.message
    assert "不要凭印象作答" in result.message


def test_latency_recorded() -> None:
    tool = make_tool()
    assert tool.run("某条正文").latency_ms >= 0


# ---------------------------------------------------------------------------
# 改写集成
# ---------------------------------------------------------------------------
def test_rewriter_not_called_when_not_needed() -> None:
    calls: list[str] = []

    def rewriter(query, history):
        calls.append(query)
        return "改写过的问题"

    tool = make_tool(rewriter=rewriter)
    result = tool.run("劳动合同到期不续签有补偿吗", [HumanMessage(content="上文")])

    assert calls == []
    assert result.rewrite_applied is False
    assert result.rewrite_reason == "passthrough"


def test_rewriter_called_when_needed() -> None:
    def rewriter(query, history):
        return "公司未签劳动合同的双倍工资可以主张多少个月"

    tool = make_tool(rewriter=rewriter)
    result = tool.run("那能要多少？", [HumanMessage(content="公司没签劳动合同")])

    assert result.rewrite_applied is True
    assert result.rewrite_reason == "rewritten"
    assert "双倍工资" in result.query


def test_rewriter_without_history_is_skipped() -> None:
    tool = make_tool(rewriter=lambda q, h: "不该被调用")
    result = tool.run("那能要多少？", None)
    assert result.rewrite_applied is False
    assert result.rewrite_reason == "no_history"


def test_rewriter_exception_falls_back() -> None:
    def boom(query, history):
        raise TimeoutError("slow")

    tool = make_tool(rewriter=boom)
    result = tool.run("那能要多少？", [HumanMessage(content="上文")])

    assert result.rewrite_applied is False
    assert result.rewrite_reason == "timeout"
    assert result.query == "那能要多少？"


def test_rewriter_tampering_falls_back() -> None:
    """改写篡改数字 → 回退原文（BC-33）。"""
    tool = make_tool(rewriter=lambda q, h: "工作 5 年可以主张多少经济补偿")
    result = tool.run("工作 3 年可以主张多少经济补偿呢", [HumanMessage(content="上文")])

    assert result.rewrite_applied is False
    assert result.rewrite_reason == "invalid_output"
    assert "3 年" in result.query


# ---------------------------------------------------------------------------
# 上下文渲染
# ---------------------------------------------------------------------------
def test_format_context_contains_three_elements() -> None:
    result = ToolResult(
        docs=[DOCS[0]], score=0.9, channel="dense",
    )
    text = format_context(result)

    assert "【法条 1】" in text
    assert "《X法》" in text
    assert "第一条" in text
    assert "现行有效" in text
    assert "2020-01-01 施行" in text
    assert "【引用约束】" in text
    assert "不得虚构条号" in text


def test_format_context_low_score_adds_hint() -> None:
    result = ToolResult(docs=[DOCS[0]], score=0.5, channel="dense")
    text = format_context(result)
    assert "相关度偏低" in text
    assert "建议换用更具体的法律术语重试" in text


def test_format_context_high_score_has_no_hint() -> None:
    result = ToolResult(docs=[DOCS[0]], score=0.95, channel="dense")
    assert "相关度偏低" not in format_context(result)


def test_format_context_empty() -> None:
    text = format_context(ToolResult())
    assert "未检索到相关法条" in text
    assert "不要凭印象作答" in text


def test_format_context_marks_invalid_doc() -> None:
    doc = {**DOCS[0], "is_valid": False, "effective_date": ""}
    text = format_context(ToolResult(docs=[doc], score=0.9))
    assert "已废止" in text


def test_tool_spec_is_bindable() -> None:
    """工具描述必须能直接喂给 bind_tools。"""
    assert READ_TOOL_SPEC["name"] == "read_law"
    assert "query" in READ_TOOL_SPEC["parameters"]["properties"]
    assert READ_TOOL_SPEC["parameters"]["required"] == ["query"]
    assert "多次调用" in READ_TOOL_SPEC["description"]


# ---------------------------------------------------------------------------
# 多法源：精确通道不能串号
# ---------------------------------------------------------------------------
def test_exact_channel_rejects_out_of_corpus_law() -> None:
    """提问提到的法律不在语料里时，不能拿别的法律同条号充数。

    实测：「民法典第1165条」返回了生态环境法典第一千一百六十五条，
    score 还是 1.00 —— 看起来像精确命中，实际答错了法。
    """
    tool = make_tool()
    result = tool.run("民法典第1165条怎么规定的")

    assert result.channel == "dense", "语料外法律必须放弃精确通道"
    assert all(d["law_name"] != "民法典" for d in result.docs)


def test_exact_channel_requires_law_match() -> None:
    """「XX法实施条例第一条」不能退化成「XX法第一条」。"""
    tool = make_tool()
    result = tool.run("X法实施条例第一条")

    assert result.channel == "dense"


def test_exact_channel_still_works_with_law_name() -> None:
    tool = make_tool()
    result = tool.run("X法第2条怎么规定的")
    assert result.channel == "exact"
    assert result.docs[0]["article_no"] == 2


# ---------------------------------------------------------------------------
# 时效性
# ---------------------------------------------------------------------------
def test_describe_validity_prefers_effective_date() -> None:
    from legalqa.tools.read_tool import describe_validity

    doc = {"is_valid": True, "effective_date": "2021-01-01", "version_date": "2020-05-28"}
    assert describe_validity(doc) == "现行有效，2021-01-01 施行"


def test_describe_validity_falls_back_to_version_date() -> None:
    from legalqa.tools.read_tool import describe_validity

    doc = {"is_valid": True, "effective_date": "", "version_date": "2020-05-28"}
    assert describe_validity(doc) == "现行有效，2020-05-28 版本"


def test_describe_validity_without_dates() -> None:
    from legalqa.tools.read_tool import describe_validity

    assert describe_validity({"is_valid": True}) == "现行有效"


def test_describe_validity_repealed() -> None:
    from legalqa.tools.read_tool import describe_validity

    assert describe_validity({"is_valid": False, "version_date": "1999-01-01"}) == "已废止，1999-01-01 版本"


def test_format_context_shows_version() -> None:
    """没有施行日期时回落到版本日期。"""
    doc = {**DOCS[0], "effective_date": "", "version_date": "2020-05-28"}
    text = format_context(ToolResult(docs=[doc], score=0.9))
    assert "2020-05-28 版本" in text


def test_format_context_prefers_effective_over_version() -> None:
    """有施行日期时优先显示它——施行日期比「文件版本日期」对用户更有意义。"""
    doc = {**DOCS[0], "effective_date": "2021-01-01", "version_date": "2020-05-28"}
    text = format_context(ToolResult(docs=[doc], score=0.9))
    assert "2021-01-01 施行" in text
    # 法条行只显示施行日期，不重复显示版本日期
    article_line = next(line for line in text.splitlines() if line.startswith("【法条 1】"))
    assert "2020-05-28" not in article_line


# ---------------------------------------------------------------------------
# 用户原问题补召回
# ---------------------------------------------------------------------------
def _docs(prefix: str, n: int) -> list[dict]:
    return [
        {**DOCS[i % len(DOCS)], "id": f"{prefix}-{i}"}
        for i in range(1, n + 1)
    ]


def test_supplement_adds_at_most_two_new_docs() -> None:
    from legalqa.tools.read_tool import supplement_with_user_query

    primary = ToolResult(docs=_docs("p", 6), score=0.8, channel="dense", query="model")
    secondary = ToolResult(
        docs=[_docs("p", 2)[0], *_docs("s", 4)],
        score=0.7,
        channel="dense",
        query="user",
    )
    out = supplement_with_user_query(primary, secondary)

    assert [d["id"] for d in out.docs] == [f"p-{i}" for i in range(1, 7)] + ["s-1", "s-2"]
    assert out.channel == "dense+user"
    assert out.score == 0.8
    assert out.query == "model"


def test_supplement_returns_secondary_when_primary_empty() -> None:
    from legalqa.tools.read_tool import supplement_with_user_query

    primary = ToolResult(error="empty_query")
    secondary = ToolResult(docs=DOCS[:2], score=0.7, channel="dense")
    assert supplement_with_user_query(primary, secondary) is secondary


def test_supplement_skips_exact_primary() -> None:
    from legalqa.tools.read_tool import supplement_with_user_query

    primary = ToolResult(docs=DOCS[:1], score=1.0, channel="exact")
    secondary = ToolResult(docs=DOCS[1:], score=0.7, channel="dense")
    assert supplement_with_user_query(primary, secondary) is primary


def test_tool_node_supplements_user_query() -> None:
    """ToolNode 会用用户原问题再检索一次，但只补模型没覆盖的条文。"""
    from langchain_core.messages import AIMessage

    from legalqa.nodes.agent import make_tool_node
    from legalqa.state import LegalState

    class FakeTool:
        name = "read_law"

        def __init__(self) -> None:
            self.queries: list[str] = []

        def run(self, query: str, history=None) -> ToolResult:
            self.queries.append(query)
            no = 1 if query == "模型查询" else 2
            doc = {
                "id": f"x-{no}", "law_name": "X法", "article_no": no,
                "article_label": f"第{no}条", "text": f"第{no}条正文",
                "is_valid": True, "effective_date": "2020-01-01",
            }
            score = 0.8 if query == "模型查询" else 0.9
            result = ToolResult(docs=[doc], score=score, channel="dense", query=query)
            result.message = format_context(result)
            return result

    tool = FakeTool()
    state = LegalState(
        messages=[
            AIMessage(
                id="a1",
                content="",
                tool_calls=[{"id": "c1", "name": "read_law", "args": {"query": "模型查询"}}],
            )
        ],
        user_query="用户原问题",
    )
    out = make_tool_node(tool, supplement_user_query=True)(state)

    assert tool.queries == ["模型查询", "用户原问题"]
    content = out["messages"][0].content
    assert "第1条正文" in content
    assert "第2条正文" in content
    assert {d["article_no"] for d in out["retrieved_docs"]} == {1, 2}


def test_tool_node_skips_supplement_when_user_query_is_worse() -> None:
    """用户原问题得分没有明显更高时不补，避免稀释模型自己的 query。"""
    from langchain_core.messages import AIMessage

    from legalqa.nodes.agent import make_tool_node
    from legalqa.state import LegalState

    class FakeTool:
        name = "read_law"

        def __init__(self) -> None:
            self.queries: list[str] = []

        def run(self, query: str, history=None) -> ToolResult:
            self.queries.append(query)
            score = 0.8 if query == "模型查询" else 0.7
            doc = {
                "id": query, "law_name": "X法", "article_no": 1,
                "article_label": "第一条", "text": "正文",
                "is_valid": True, "effective_date": "2020-01-01",
            }
            result = ToolResult(docs=[doc], score=score, channel="dense", query=query)
            result.message = format_context(result)
            return result

    tool = FakeTool()
    state = LegalState(
        messages=[
            AIMessage(
                id="a1",
                content="",
                tool_calls=[{"id": "c1", "name": "read_law", "args": {"query": "模型查询"}}],
            )
        ],
        user_query="用户原问题",
    )
    out = make_tool_node(tool, supplement_user_query=True)(state)

    assert tool.queries == ["模型查询", "用户原问题"]
    assert len(out["retrieved_docs"]) == 1
