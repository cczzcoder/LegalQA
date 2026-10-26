"""S6 测试：滚动摘要。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from legalqa.pure.summary import (
    SUMMARY_MAX_CHARS,
    build_summary_prompt,
    render_summary_message,
    sanitize_summary,
)


def test_build_prompt_without_existing() -> None:
    messages = [HumanMessage(content="公司拖欠工资"), AIMessage(content="可以申请仲裁")]
    prompt = build_summary_prompt("", messages)
    assert "用户：公司拖欠工资" in prompt
    assert "助手：可以申请仲裁" in prompt
    assert "已有的历史摘要" not in prompt


def test_build_prompt_with_existing_merges() -> None:
    prompt = build_summary_prompt("此前用户已工作三年", [HumanMessage(content="那赔偿呢")])
    assert "此前用户已工作三年" in prompt
    assert "合并" in prompt


def test_build_prompt_truncates_long_content() -> None:
    prompt = build_summary_prompt("", [HumanMessage(content="很长" * 500)])
    assert len(prompt) < 2000


def test_build_prompt_skips_empty_messages() -> None:
    prompt = build_summary_prompt("", [HumanMessage(content="   "), AIMessage(content="")])
    assert "用户：" not in prompt


def test_sanitize_strips_prefix_and_newlines() -> None:
    raw = "摘要：用户于2020年入职\n某公司，月薪8000元。"
    assert sanitize_summary(raw) == "用户于2020年入职 某公司，月薪8000元。"


def test_sanitize_strips_quotes() -> None:
    assert sanitize_summary("「用户已工作三年」") == "用户已工作三年"


def test_sanitize_truncates() -> None:
    result = sanitize_summary("字" * 500)
    assert len(result) <= SUMMARY_MAX_CHARS + 1
    assert result.endswith("…")


def test_sanitize_handles_empty() -> None:
    assert sanitize_summary("") == ""
    assert sanitize_summary(None) == ""


def test_render_summary_message_is_human() -> None:
    """摘要必须是 HumanMessage，不能是 SystemMessage。

    INV-2 要求窗口里只有 index 0 一条 system；拼进 system 又会破坏
    system 的字节恒定（缓存安全）。
    """
    message = render_summary_message("用户已工作三年")
    assert isinstance(message, HumanMessage)
    assert not isinstance(message, SystemMessage)
    assert "系统生成" in message.content
    assert "用户已工作三年" in message.content
    assert message.id == "conversation-summary"


def test_render_summary_message_empty() -> None:
    assert render_summary_message("") is None
    assert render_summary_message("   ") is None


def test_sanitize_cuts_at_sentence_boundary() -> None:
    """截断不能切在法条引用中间。

    实测摘要被截成「依据为《劳动合同法》第九十四条（2012-12-…」，
    最关键的法条引用被毁掉了。
    """
    text = "用户咨询工伤。" + "填充内容。" * 60 + "依据为《劳动合同法》第九十四条（2012-12-28 版本）。"
    result = sanitize_summary(text)
    assert result.endswith("。"), f"应在句号处结束，实际结尾: {result[-20:]!r}"
    assert not result.endswith("…")


def test_sanitize_hard_cut_when_no_boundary() -> None:
    """找不到合适句号时退化为硬截断 + 省略号。"""
    result = sanitize_summary("字" * 500)
    assert result.endswith("…")
