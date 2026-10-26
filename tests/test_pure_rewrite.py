"""S2 测试：改写纯逻辑。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from legalqa.pure.rewrite import (
    build_rewrite_prompt,
    entities_preserved,
    extract_entities,
    finalize_rewrite,
    merge_rewrite_result,
    needs_rewrite,
    sanitize_rewrite_output,
    similarity_ok,
)


# ---------------------------------------------------------------------------
# 预判
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "query,needed,reason",
    [
        ("", False, "empty"),
        ("民法典第1165条怎么规定的", False, "exempt_article_no"),
        ("民法典第一千一百六十五条", False, "exempt_article_no"),
        ("劳动合同到期后公司不给我续签，还说这个岗位取消了，我能要补偿吗", False, "exempt_long"),
        ("还有呢", True, "strong_prefix"),
        ("那要是公司不承认呢", True, "strong_prefix"),
        ("怎么办", True, "too_short"),
        ("工伤呢？", True, "tail_question"),
        ("这个合同有效吗", True, "deictic"),
        ("他能要多少", True, "personal"),
        ("劳动合同到期不续签有补偿吗", False, "passthrough"),
        ("公司拖欠工资三个月了，我能要赔偿吗", False, "passthrough"),
    ],
)
def test_needs_rewrite(query: str, needed: bool, reason: str) -> None:
    decision = needs_rewrite(query)
    assert decision.needed is needed, f"{query!r} -> {decision}"
    assert decision.reason == reason


def test_exemption_beats_strong_signal() -> None:
    """L3 豁免必须优先于 L1：含条号的长句即使以「如果」开头也不改写。"""
    query = "如果违反民法典第1165条的规定造成了损害，责任应该怎么划分才合理"
    decision = needs_rewrite(query)
    assert decision.needed is False
    assert decision.reason == "exempt_article_no"


def test_question_particle_does_not_trigger_rewrite() -> None:
    """「吗」结尾的完整问句不该被改写 —— 这是最容易误判的一类。"""
    assert needs_rewrite("公司拖欠工资三个月了，我能要赔偿吗").needed is False


def test_needs_rewrite_never_raises() -> None:
    for weird in ["   ", "\n\n", "。。。", "第条", "第零条"]:
        assert isinstance(needs_rewrite(weird).needed, bool)


# ---------------------------------------------------------------------------
# 输出清洗
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ('改写："工伤能否认定？"', "工伤能否认定？"),
        ("输出：在工地摔伤能否认定工伤", "在工地摔伤能否认定工伤"),
        ("  「未签合同能要双倍工资吗」  ", "未签合同能要双倍工资吗"),
        ("第一行是一句完整的问题\n第二行解释", "第一行是一句完整的问题"),
    ],
)
def test_sanitize_cleans_noise(raw: str, expected: str) -> None:
    assert sanitize_rewrite_output(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "短", "x" * 500])
def test_sanitize_rejects_bad_length(raw) -> None:
    assert sanitize_rewrite_output(raw) is None


# ---------------------------------------------------------------------------
# 实体保真
# ---------------------------------------------------------------------------
def test_extract_entities_numbers_and_terms() -> None:
    found = extract_entities("工作 3 年，月薪 8000 元，主张经济补偿")
    assert "3 年" in found or "3" in found
    assert "8000 元" in found or "8000" in found
    assert "经济补偿" in found


def test_entities_preserved_accepts_faithful_rewrite() -> None:
    source = "工作 3 年，能要多少经济补偿"
    rewritten = "在公司工作 3 年的员工，解除劳动合同时可以主张多少经济补偿？"
    assert entities_preserved(source, rewritten) is True


def test_entities_preserved_rejects_number_tampering() -> None:
    """最危险的失败：3 年改成 5 年。检索照样命中，答案是错的。"""
    source = "工作 3 年，能要多少经济补偿"
    rewritten = "在公司工作 5 年的员工，可以主张多少经济补偿？"
    assert entities_preserved(source, rewritten) is False


def test_entities_preserved_rejects_term_swap() -> None:
    source = "劳动合同到期不续签有补偿吗"
    rewritten = "劳务合同到期不续签有补偿吗"
    assert entities_preserved(source, rewritten) is False


def test_entities_preserved_without_entities() -> None:
    assert entities_preserved("怎么办", "遇到纠纷应该怎么办？") is True


# ---------------------------------------------------------------------------
# 相似度
# ---------------------------------------------------------------------------
def test_similarity_accepts_reasonable_expansion() -> None:
    source = "工伤呢？"
    rewritten = "在工地摔伤但包工头不承认劳动关系，能否认定工伤？"
    assert similarity_ok(source, rewritten) is True


def test_similarity_rejects_complete_drift() -> None:
    """长问句被整段改写成别的主题 → 判跑偏。"""
    source = "公司拖欠工资三个月了我能要赔偿吗"
    rewritten = "房屋租赁合同到期后押金不退还可以起诉吗需要准备哪些材料呢"
    assert similarity_ok(source, rewritten) is False


def test_similarity_handles_empty() -> None:
    assert similarity_ok("", "abc") is False
    assert similarity_ok("abc", "") is False


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------
def test_finalize_returns_none_on_tamper() -> None:
    cleaned, reason = finalize_rewrite("工作 3 年能要多少补偿", "工作 5 年能要多少补偿")
    assert cleaned is None
    assert reason == "invalid_output"


def test_finalize_accepts_identity_output() -> None:
    cleaned, reason = finalize_rewrite("劳动合同到期不续签有补偿吗", "劳动合同到期不续签有补偿吗")
    assert cleaned == "劳动合同到期不续签有补偿吗"
    assert reason == "rewritten"


def test_merge_passthrough() -> None:
    assert merge_rewrite_result("原始问题", "随便什么", applied=False) == (
        "原始问题", False, "passthrough",
    )


def test_merge_falls_back_to_source_on_failure() -> None:
    """任何失败都必须回退原文，**绝不返回空串**（BC-35）。"""
    for raw in [None, "", "   ", "短", "x" * 500]:
        query, applied, reason = merge_rewrite_result("原始问题", raw, applied=True)
        assert query == "原始问题"
        assert applied is False
        assert reason == "invalid_output"


def test_merge_never_returns_empty_query() -> None:
    for raw in [None, "", "\n", "？？"]:
        query, _, _ = merge_rewrite_result("有内容的问题", raw, applied=True)
        assert query.strip() != ""


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
def test_prompt_contains_fewshot_and_history() -> None:
    messages = [
        HumanMessage(content="公司没签劳动合同"),
        AIMessage(content="可以主张双倍工资"),
    ]
    prompt = build_rewrite_prompt("那能要多少？", messages)
    assert "示例3" in prompt                 # 原样输出的示范必须存在
    assert "用户：公司没签劳动合同" in prompt
    assert "最新问题：那能要多少？" in prompt
    assert prompt.rstrip().endswith("输出：")


def test_prompt_without_history() -> None:
    prompt = build_rewrite_prompt("工伤呢", [])
    assert "（无）" in prompt


def test_prompt_truncates_long_history() -> None:
    messages = [HumanMessage(content="很长" * 500), AIMessage(content="回答" * 500)]
    prompt = build_rewrite_prompt("还有呢", messages)
    assert len(prompt) < 3000
