"""S8 测试：评测检查项。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from legalqa.corpus.index import VectorIndex
from legalqa.eval import (
    CheckResult,
    check_coverage,
    check_disclaimer,
    check_hallucination,
    check_keywords,
    check_length,
    check_refusal,
    check_required_citations,
    check_traceability,
    extract_answer,
    strip_system_notes,
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


def test_traceability_skips_bare_number_after_external_law() -> None:
    """《刑法》第 133 条后的裸「第 133 条」仍属于外部法律，不该判未溯源。"""
    docs = [{"law_name": "中华人民共和国劳动法", "article_no": 50}]
    text = "醉驾的刑事责任规定在《中华人民共和国刑法》中，第一百三十三条之一。\n建议查第一百三十三条之一。"
    assert check_traceability(text, docs).passed


def test_traceability_bare_number_after_internal_law_still_checked() -> None:
    """最近法是本批法律时，裸条号仍要核实，不能一律放过。"""
    docs = [{"law_name": "中华人民共和国劳动法", "article_no": 50}]
    text = "依据《中华人民共和国劳动法》第五十条。\n另见第六十条。"
    result = check_traceability(text, docs)
    assert not result.passed
    assert "第60条" in result.detail


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


def test_hallucination_handles_bare_law_inside_quoted_article() -> None:
    """引文里的「依照劳动合同法第八十七条」不是实施条例第 87 条。"""
    docs = [
        {"id": "lf-87", "law_name": "中华人民共和国劳动合同法", "article_no": 87,
         "article_label": "第八十七条", "doc_type": "法律", "text": "违法解除赔偿", "is_valid": True},
        {"id": "lfr-25", "law_name": "中华人民共和国劳动合同法实施条例", "article_no": 25,
         "article_label": "第二十五条", "doc_type": "行政法规", "text": "赔偿金与经济补偿", "is_valid": True},
    ]
    index = VectorIndex.build(docs, FakeEmbedder())
    text = (
        "《劳动合同法实施条例》第二十五条：“用人单位违反劳动合同法的规定解除或者终止"
        "劳动合同，依照劳动合同法第八十七条的规定支付了赔偿金的，不再支付经济补偿。”"
    )
    assert check_hallucination(text, index).passed
    # 真的引用了实施条例第 87 条时，仍然要判幻觉
    assert not check_hallucination("《劳动合同法实施条例》第八十七条。", index).passed


def test_expected_citation_matches_nested_interpretation() -> None:
    """〈〉与否不能影响「是否引用了期望法条」的判断。"""
    from legalqa.eval import check_expected_citation

    text = "《最高人民法院关于适用〈中华人民共和国民法典〉婚姻家庭编的解释（一）》第二十三条"
    expected = "最高人民法院关于适用中华人民共和国民法典婚姻家庭编的解释（一）"
    assert check_expected_citation(text, expected, 23).passed


def test_expected_citation_accepts_alternative_article() -> None:
    """同一结论的并列条文都算命中，不能只认合成金标那一条。"""
    from legalqa.eval import check_expected_citation

    text = "依据《劳动合同法》第八十二条，应支付二倍工资。"
    alternatives = [{"law": "中华人民共和国劳动合同法", "article_no": 82}]

    assert not check_expected_citation(
        text, "中华人民共和国劳动合同法实施条例", 6
    ).passed
    assert check_expected_citation(
        text, "中华人民共和国劳动合同法实施条例", 6, alternatives
    ).passed
    # 只给备选、不给主答案也能判
    assert check_expected_citation(text, None, None, alternatives).passed
    # 备选条号不对仍然失败
    assert not check_expected_citation(
        text, None, None, [{"law": "中华人民共和国劳动合同法", "article_no": 87}]
    ).passed


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


def test_traceability_requires_law_and_number_pair() -> None:
    """同一批材料里有《甲法》第 38 条，回答引《乙法》第 38 条 → 不算可溯源。"""
    from legalqa.eval import check_traceability

    # 甲法在批内（提供了第 38 条），乙法也在批内（提供了第 44 条）；
    # 回答引的乙法第 38 条却根本没检索到 —— 只有成对判断才能发现。
    docs = [{"law_name": "甲法", "article_no": 38},
            {"law_name": "乙法", "article_no": 44}]
    result = check_traceability("依据《乙法》第三十八条。", docs)
    assert result.passed is False
    assert "38" in result.detail


def test_traceability_accepts_matching_pair() -> None:
    from legalqa.eval import check_traceability

    docs = [{"law_name": "中华人民共和国劳动合同法", "article_no": 38}]
    assert check_traceability("依据《劳动合同法》第三十八条。", docs).passed is True


def test_traceability_ignores_cross_reference_inside_quote() -> None:
    """引号内是法条原文，里面的「本法第四十七条」不是模型引的条号。"""
    from legalqa.eval import check_traceability

    docs = [{"law_name": "中华人民共和国劳动合同法", "article_no": 87}]
    text = (
        "依据《劳动合同法》第八十七条：“用人单位违反本法规定解除或者终止劳动合同的，"
        "应当依照本法第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金。”"
    )
    assert check_traceability(text, docs).passed


def test_strip_backfill_appendix_removes_quoted_statute() -> None:
    """附录里是法条原文，其中的交叉引用不算模型的引用行为。"""
    from legalqa.eval import strip_backfill_appendix

    text = (
        "依据《劳动合同法》第四十六条，单位应支付补偿。\n\n"
        "本回答仅供参考，不构成正式法律意见。\n\n"
        "【补充核对】回答中引用的以下条文不在本次检索结果里：\n"
        "- 《中华人民共和国劳动合同法》第四十六条（现行有效）：有下列情形之一的……"
        "（一）劳动者依照本法第三十八条规定解除劳动合同的；（二）用人单位依照本法"
        "第三十六条规定……\n\n"
        "本回答仅供参考，不构成正式法律意见。"
    )
    body = strip_backfill_appendix(text)
    assert "第三十八条" not in body
    assert "第三十六条" not in body
    assert "第四十六条" in body          # 模型自己写的那句要保留
    assert "本回答仅供参考" in body       # 免责声明不能被切掉


def test_strip_backfill_appendix_keeps_warning() -> None:
    """附录之后的【引用提示】是系统输出，必须保留。"""
    from legalqa.eval import strip_backfill_appendix

    text = (
        "正文。\n\n【补充核对】……原文里的第三十八条……\n\n"
        "【引用提示】以下条号在法条库中查不到：第999条\n\n本回答仅供参考，不构成正式法律意见。"
    )
    body = strip_backfill_appendix(text)
    assert "第三十八条" not in body
    assert "第999条" in body


def test_strip_backfill_appendix_passthrough() -> None:
    from legalqa.eval import strip_backfill_appendix

    assert strip_backfill_appendix("普通回答") == "普通回答"
    assert strip_backfill_appendix("") == ""


def test_refusal_recognizes_out_of_scope_wording() -> None:
    """074 类：明确说明「不属于法律范畴、无需检索」应算合格的拒答说明。"""
    from legalqa.eval import check_refusal

    text = (
        "该问题不属于法律适用范畴，因此无需调用 read_law 工具检索法条。\n"
        "结论：就业方向可参考教育部门公开信息。"
    )
    assert check_refusal(text, False).passed
    assert not check_refusal(text, True).passed


def test_refusal_recognizes_alternative_wording() -> None:
    """off-006：模型说「未直接规定 / 不足以完整回答」也算合格拒答。"""
    from legalqa.eval import check_refusal

    text = (
        "现有检索结果中未直接规定“如何申请专利”的程序性要求。所列条文仅涉及"
        "专利申请权的归属，不涉及申请步骤、形式要件或行政程序。因此，当前材料"
        "不足以完整回答“怎么申请专利”，需要补充检索。"
    )
    assert check_refusal(text, False).passed

    text2 = (
        "现有检索结果未直接列明股权转让所涉具体税种，仅原则性规定纳税义务，"
        "但未明确税种、税率及征管规则。当前材料未包含相关税收实体法条文，"
        "无法从所提供材料中提取具体计税依据。"
    )
    assert check_refusal(text2, False).passed

    from legalqa import config

    assert check_refusal(config.FALLBACK_TEXTS["unfinished"], False).passed

    text3 = (
        "根据现有检索结果，未直接查到关于醉驾量刑的具体刑法条文。"
        "上述《刑法》条文未在本次检索结果中出现，说明当前工具返回材料不完整。"
    )
    assert check_refusal(text3, False).passed


# ---------------------------------------------------------------------------
# 问题覆盖（多问 / 复杂问）
# ---------------------------------------------------------------------------
def test_coverage_all_present() -> None:
    result = check_coverage(
        "先讲经济补偿，再讲二倍工资。",
        [
            {"text": "经济补偿", "must_mention": ["经济补偿", "补偿金"]},
            {"text": "双倍工资", "must_mention": ["二倍工资", "双倍工资"]},
        ],
    )
    assert result.passed
    assert result.covered == 2
    assert result.ratio == 1.0


def test_coverage_reports_missing_labels() -> None:
    result = check_coverage(
        "只说了经济补偿。",
        [
            {"text": "经济补偿", "must_mention": ["经济补偿"]},
            {"text": "双倍工资", "must_mention": ["二倍工资", "双倍工资"]},
        ],
    )
    assert not result.passed
    assert result.covered == 1
    assert result.ratio == 0.5
    assert result.missing == ["双倍工资"]
    assert "双倍工资" in result.detail


def test_coverage_without_questions_passes() -> None:
    result = check_coverage("任意内容", [])
    assert result.passed
    assert result.total == 0
    assert result.detail == "未指定"


def test_coverage_without_keywords_is_treated_as_covered() -> None:
    """评测规则没给关键词时宁可漏报，也不能冤枉一条完整回答。"""
    assert check_coverage("任意内容", [{"text": "无关键词"}]).passed


def test_required_citations_all_groups_hit() -> None:
    groups = [
        [["A法", 1], ["B法", 2]],
        [["C法", 3]],
    ]
    result = check_required_citations("依据《A法》第一条和《C法》第三条。", groups)
    assert result.passed
    assert "2/2" in result.detail


def test_required_citations_reports_missing_group() -> None:
    groups = [
        [["A法", 1], ["B法", 2]],
        [["C法", 3]],
    ]
    result = check_required_citations("依据《A法》第一条。", groups)
    assert not result.passed
    assert "1/2" in result.detail
    assert "C法" in result.detail


# ---------------------------------------------------------------------------
# 系统提示剥离（引用提示不应被当成模型幻觉）
# ---------------------------------------------------------------------------
def test_strip_system_notes_removes_warning_keeps_disclaimer() -> None:
    text = (
        "正文。\n\n【引用提示】以下条号在法条库中不存在：第89条\n\n"
        "本回答仅供参考，不构成正式法律意见。"
    )
    body = strip_system_notes(text)
    assert "【引用提示】" not in body
    assert "第89条" not in body
    assert "本回答仅供参考" in body


def test_strip_system_notes_removes_backfill_and_warning() -> None:
    text = (
        "正文。\n\n【补充核对】……原文里的第三十八条……\n\n"
        "【引用提示】以下条号不存在：第89条\n\n"
        "本回答仅供参考，不构成正式法律意见。"
    )
    body = strip_system_notes(text)
    assert "第三十八条" not in body
    assert "第89条" not in body
    assert "本回答仅供参考" in body


def test_strip_system_notes_passthrough() -> None:
    assert strip_system_notes("普通回答") == "普通回答"
    assert strip_system_notes("") == ""
