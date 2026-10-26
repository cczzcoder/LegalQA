"""S8 测试：评测检查项。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from legalqa.corpus.index import VectorIndex
from legalqa.eval import (
    CheckResult,
    check_disclaimer,
    check_hallucination,
    check_keywords,
    check_length,
    check_refusal,
    check_traceability,
    extract_answer,
)
from legalqa.nodes.guards import DISCLAIMER


class FakeEmbedder:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        seed = sum(ord(c) for c in text) or 1
        return [float((seed + i) % 7 + 1) for i in range(self.dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make_index() -> VectorIndex:
    docs = [
        {"id": "labor-0050", "law_name": "中华人民共和国劳动法", "article_no": 50,
         "article_label": "第五十条", "doc_type": "法律", "text": "工资应当按月支付", "is_valid": True},
        {"id": "contract-0087", "law_name": "中华人民共和国劳动合同法", "article_no": 87,
         "article_label": "第八十七条", "doc_type": "法律", "text": "违法解除赔偿", "is_valid": True},
    ]
    return VectorIndex.build(docs, FakeEmbedder())


# ---------------------------------------------------------------------------
# 免责声明 / 长度
# ---------------------------------------------------------------------------
def test_disclaimer_present() -> None:
    assert check_disclaimer(f"结论如下。{DISCLAIMER}").passed


def test_disclaimer_missing() -> None:
    result = check_disclaimer("结论如下。")
    assert not result.passed
    assert "免责声明" in result.detail


def test_length_too_short() -> None:
    assert not check_length("太短").passed


def test_length_too_long() -> None:
    assert not check_length("字" * 3000).passed


def test_length_ok() -> None:
    assert check_length("字" * 200).passed


# ---------------------------------------------------------------------------
# 引用可溯源
# ---------------------------------------------------------------------------
def test_traceability_all_found() -> None:
    docs = [{"law_name": "中华人民共和国劳动法", "article_no": 50}]
    assert check_traceability("依据《劳动法》第五十条。", docs).passed


def test_traceability_reports_missing() -> None:
    docs = [{"law_name": "中华人民共和国劳动法", "article_no": 50}]
    result = check_traceability("依据《劳动法》第五十条和第八十七条。", docs)
    assert not result.passed
    assert "第87条" in result.detail


def test_traceability_ignores_external_law() -> None:
    """语料外法律不归「可溯源」这项管（由引用提示负责）。"""
    docs = [{"law_name": "中华人民共和国劳动法", "article_no": 50}]
    result = check_traceability("另见《刑法》第二百四十五条。", docs)
    assert result.passed


def test_traceability_skips_without_docs() -> None:
    assert check_traceability("任意内容", []).passed


def test_traceability_dedupes_by_article_no() -> None:
    """同一部法律两种写法不该重复列出。"""
    docs = [{"law_name": "中华人民共和国劳动合同法", "article_no": 87}]
    result = check_traceability(
        "《劳动合同法》第47条、《中华人民共和国劳动合同法》第四十七条。", docs
    )
    assert not result.passed
    assert result.detail.count("47") == 1


# ---------------------------------------------------------------------------
# 幻觉条号
# ---------------------------------------------------------------------------
def test_hallucination_detects_nonexistent_article() -> None:
    """引用了语料里的法律，但条号不存在 → 幻觉。

    这是法律问答最该守住的一条：编造条号比答不出来恶劣得多。
    """
    index = make_index()
    result = check_hallucination("依据《劳动法》第九百九十九条。", index)
    assert not result.passed
    assert "999" in result.detail


def test_hallucination_accepts_real_article() -> None:
    index = make_index()
    assert check_hallucination("依据《劳动法》第五十条。", index).passed


def test_hallucination_ignores_external_law() -> None:
    """《刑法》不在语料里，引用它不算幻觉，只是超出知识库。"""
    index = make_index()
    assert check_hallucination("另见《刑法》第二百四十五条。", index).passed


def test_hallucination_skips_without_index() -> None:
    assert check_hallucination("依据《劳动法》第九百九十九条。", None).passed


# ---------------------------------------------------------------------------
# 拒答
# ---------------------------------------------------------------------------
def test_refusal_expected_and_given() -> None:
    assert check_refusal("现有资料不足，无法回答。", should_answer=False).passed


def test_refusal_expected_but_not_given() -> None:
    result = check_refusal("这个问题涉及个人所得税，税率为 3%。", should_answer=False)
    assert not result.passed


def test_refusal_unexpected_and_short() -> None:
    """该回答却一上来就拒答。"""
    result = check_refusal("现有资料不足。", should_answer=True)
    assert not result.passed


def test_refusal_answer_with_caveat_is_ok() -> None:
    """回答里带「未涵盖某部分」是正常的，不该判失败。"""
    answer = "结论：可以主张赔偿。" + "详细说明。" * 30 + "现有资料未涵盖地方细则。"
    assert check_refusal(answer, should_answer=True).passed


# ---------------------------------------------------------------------------
# 关键要素
# ---------------------------------------------------------------------------
def test_keywords_any_match() -> None:
    """同义选项命中任意一个即通过。"""
    assert check_keywords("可以申请劳动仲裁。", ["仲裁", "投诉"]).passed


def test_keywords_none_match() -> None:
    result = check_keywords("可以向法院起诉。", ["仲裁", "投诉"])
    assert not result.passed
    assert "仲裁" in result.detail


def test_keywords_empty_list() -> None:
    assert check_keywords("任意内容", []).passed


# ---------------------------------------------------------------------------
# 提取回答
# ---------------------------------------------------------------------------
def test_extract_answer_takes_last_ai() -> None:
    messages = [
        HumanMessage(content="问"),
        AIMessage(content="旧回答"),
        HumanMessage(content="再问"),
        AIMessage(content="新回答"),
    ]
    assert extract_answer(messages) == "新回答"


def test_extract_answer_without_ai() -> None:
    assert extract_answer([HumanMessage(content="问")]) == ""
