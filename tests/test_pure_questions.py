"""S9 测试：确定性问题登记与序号指代解析。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from legalqa.pure.questions import (
    extract_numbered_questions,
    is_elliptical_followup,
    resolve_elliptical_from_messages,
    resolve_ordinal,
    resolve_ordinal_from_messages,
)


# ---------------------------------------------------------------------------
# 问题抽取
# ---------------------------------------------------------------------------
def test_extract_inline_chinese_numbered() -> None:
    text = (
        "我想问三件事：第一，拖欠工资能不能辞职要补偿？"
        "第二，没签合同能不能要二倍工资？第三，周末加班费怎么算？"
    )
    questions = extract_numbered_questions(text)
    assert len(questions) == 3
    assert "二倍工资" in questions[1]


def test_extract_question_marks() -> None:
    questions = extract_numbered_questions("离婚时房子怎么分？孩子抚养权归谁？")
    assert questions == ["离婚时房子怎么分", "孩子抚养权归谁"]


def test_extract_line_arabic() -> None:
    questions = extract_numbered_questions("1. 拖欠工资\n2. 双倍工资")
    assert len(questions) == 2
    assert "双倍工资" in questions[1]


def test_extract_inline_chinese_semicolon() -> None:
    questions = extract_numbered_questions("一、拖欠工资；二、双倍工资")
    assert len(questions) == 2
    assert "双倍工资" in questions[1]


def test_extract_parenthesized_numbers() -> None:
    questions = extract_numbered_questions("（1）拖欠工资（2）双倍工资")
    assert len(questions) == 2
    assert "双倍工资" in questions[1]


def test_extract_single_statement_returns_empty() -> None:
    assert extract_numbered_questions("公司拖欠我三个月工资") == []


# ---------------------------------------------------------------------------
# 序号解析
# ---------------------------------------------------------------------------
def test_resolve_recent_ordinal() -> None:
    history = ["我想问三件事：第一，A？第二，B？第三，C？"]
    resolved, reason = resolve_ordinal("第二个问题具体怎么算？", history)
    assert reason == "ordinal_recent"
    assert resolved == "B"


def test_resolve_first_ordinal_across_long_history() -> None:
    history = ["离婚时房子怎么分？孩子抚养权归谁？"] + ["公司拖欠工资怎么办？"] * 7
    resolved, reason = resolve_ordinal("回到最开始第二个问题，那个归谁？", history)
    assert reason == "ordinal_first"
    assert resolved == "孩子抚养权归谁"


def test_resolve_out_of_range() -> None:
    resolved, reason = resolve_ordinal("第五个问题呢？", ["A？B？"])
    assert resolved is None
    assert reason == "out_of_range"


def test_resolve_not_ordinal() -> None:
    assert resolve_ordinal("那具体怎么算？", ["A？B？"]) == (None, "not_ordinal")


def test_resolve_ignores_time_quantifier() -> None:
    """「第一个月」不是「第一个问题」，不能误判。"""
    assert resolve_ordinal("第一个月工资怎么算？", ["A？B？"]) == (None, "not_ordinal")


def test_resolve_previous_question() -> None:
    resolved, reason = resolve_ordinal("上一个问题呢？", ["A？B？"])
    assert reason == "previous"
    assert resolved == "B"


def test_resolve_ambiguous_ordinals() -> None:
    resolved, reason = resolve_ordinal("第一个和第二个问题分别怎么算？", ["A？B？"])
    assert resolved is None
    assert reason == "ambiguous_ordinal"


def test_resolve_from_messages_excludes_current_query() -> None:
    messages = [
        HumanMessage(id="h1", content="A？B？"),
        HumanMessage(id="h2", content="第二个问题呢？"),
    ]
    resolved, reason = resolve_ordinal_from_messages("第二个问题呢？", messages)
    assert reason == "ordinal_recent"
    assert resolved == "B"


def test_elliptical_followup_markers() -> None:
    assert is_elliptical_followup("如何做？")
    assert is_elliptical_followup("那具体怎么算？")
    assert is_elliptical_followup("然后呢")
    assert is_elliptical_followup("刚才那个怎么办")


def test_self_contained_short_question_is_not_elliptical() -> None:
    assert not is_elliptical_followup("工伤怎么认定？")
    assert not is_elliptical_followup("怎么申请专利？")


def test_resolve_elliptical_uses_previous_question() -> None:
    messages = [
        HumanMessage(id="h1", content="公司拖欠工资，我想辞职还能要补偿吗？"),
        HumanMessage(id="h2", content="如何做？"),
    ]
    resolved, reason = resolve_elliptical_from_messages("如何做？", messages)
    assert reason == "elliptical_previous"
    assert "拖欠工资" in (resolved or "")
