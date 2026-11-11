"""S12 测试：库外权威法源白名单。"""

from __future__ import annotations

from legalqa.external import ExternalEntry, ExternalWhitelist, load_whitelist


def _whitelist() -> ExternalWhitelist:
    return ExternalWhitelist(
        [
            ExternalEntry(
                law_name="中华人民共和国刑法",
                article_no=246,
                article_label="第二百四十六条",
                text=(
                    "以暴力或者其他方法公然侮辱他人或者捏造事实诽谤他人，"
                    "情节严重的，处三年以下有期徒刑、拘役、管制或者剥夺政治权利。"
                ),
                source="测试白名单",
                source_url="https://example.test",
                version_date="2023-12-29",
            )
        ]
    )


def test_verify_existing_article() -> None:
    status, entry = _whitelist().verify("刑法", 246)
    assert status == "verified_exists"
    assert entry is not None and entry.law_name == "中华人民共和国刑法"


def test_verify_quote_match() -> None:
    status, entry = _whitelist().verify(
        "中华人民共和国刑法", 246, "捏造事实诽谤他人"
    )
    assert status == "verified_quote"
    assert entry is not None


def test_verify_quote_mismatch() -> None:
    status, entry = _whitelist().verify("刑法", 246, "可以随便造谣")
    assert status == "quote_mismatch"
    assert entry is not None


def test_verify_quote_ignores_punctuation() -> None:
    status, _entry = _whitelist().verify(
        "刑法", 246, "捏造事实诽谤他人，情节严重的，处三年以下有期徒刑。"
    )
    assert status == "verified_quote"


def test_verify_missing_article() -> None:
    status, entry = _whitelist().verify("刑法", 245)
    assert status == "not_found"
    assert entry is None


def test_seed_whitelist_loads() -> None:
    whitelist = load_whitelist()
    assert len(whitelist) >= 1
    status, entry = whitelist.verify("刑法", 246)
    assert status in {"verified_exists", "verified_quote"}
    assert entry is not None
    assert entry.source_url
