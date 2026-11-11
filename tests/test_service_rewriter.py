"""查询改写开关：默认关闭，LEGALQA_REWRITE=1 时开启。"""

from __future__ import annotations

from legalqa.service import _build_rewriter


def test_rewriter_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("LEGALQA_REWRITE", raising=False)
    assert _build_rewriter(object()) is None


def test_rewriter_disabled_when_flag_is_zero(monkeypatch) -> None:
    monkeypatch.setenv("LEGALQA_REWRITE", "0")
    assert _build_rewriter(object()) is None


def test_rewriter_enabled_by_flag(monkeypatch) -> None:
    monkeypatch.setenv("LEGALQA_REWRITE", "1")
    assert callable(_build_rewriter(object()))
