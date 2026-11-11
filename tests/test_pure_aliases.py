"""语言文字域确定性别名扩展。"""

from __future__ import annotations

from legalqa.pure.aliases import expand_query


def test_expand_language_query() -> None:
    query = "孩子在海外学中文，老师主要教方言和繁体字"
    expanded, applied = expand_query(query)
    assert applied == ("国际中文教育 普通话 规范汉字",)
    assert expanded.startswith(query)
    assert "国际中文教育" in expanded


def test_expand_unrelated_query_unchanged() -> None:
    query = "公司拖欠工资不发怎么办"
    assert expand_query(query) == (query, ())


def test_expand_is_idempotent() -> None:
    once, _ = expand_query("招牌上用了点方言字")
    twice, applied = expand_query(once)
    assert applied == ()
    assert twice == once


def test_expand_empty_query() -> None:
    assert expand_query("  ") == ("", ())
