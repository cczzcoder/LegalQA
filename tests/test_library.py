"""法条目录与评测报告数据层的单元测试（不启动 FastAPI / 模型）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legalqa.library import EvalStore, LawCatalog

DOCS = [
    {
        "id": "labor-1",
        "law_name": "中华人民共和国劳动合同法",
        "doc_type": "法律",
        "article_no": 10,
        "article_label": "第十条",
        "chapter": "第二章",
        "section": "",
        "text": "建立劳动关系，应当订立书面劳动合同。",
        "is_valid": True,
        "version_date": "2012-12-28",
    },
    {
        "id": "labor-2",
        "law_name": "中华人民共和国劳动合同法",
        "doc_type": "法律",
        "article_no": 87,
        "article_label": "第八十七条",
        "chapter": "第七章",
        "section": "",
        "text": "用人单位违法解除劳动合同的，应当支付赔偿金。",
        "is_valid": True,
        "version_date": "2012-12-28",
    },
    {
        "id": "civil-1",
        "law_name": "中华人民共和国民法典",
        "doc_type": "法律",
        "article_no": 87,
        "article_label": "第八十七条",
        "chapter": "第三章",
        "section": "",
        "text": "为公益目的成立的非营利法人，不得向出资人分配利润。",
        "is_valid": True,
        "version_date": "2020-05-28",
    },
    {
        "id": "civil-2",
        "law_name": "中华人民共和国民法典",
        "doc_type": "法律",
        "article_no": 1165,
        "article_label": "第一千一百六十五条",
        "chapter": "第七章",
        "section": "",
        "text": "行为人因过错侵害他人民事权益造成损害的，应当承担侵权责任。",
        "is_valid": True,
        "version_date": "2020-05-28",
    },
]


@pytest.fixture()
def catalog(tmp_path: Path) -> LawCatalog:
    path = tmp_path / "docs.json"
    path.write_text(json.dumps(DOCS, ensure_ascii=False), encoding="utf-8")
    return LawCatalog(path)


def test_catalog_stats_and_laws(catalog: LawCatalog) -> None:
    stats = catalog.stats()
    assert stats["total_laws"] == 2
    assert stats["total_articles"] == 4
    assert stats["valid_articles"] == 4
    assert {item["name"]: item["article_count"] for item in catalog.laws()} == {
        "中华人民共和国劳动合同法": 2,
        "中华人民共和国民法典": 2,
    }

    labor = next(item for item in catalog.laws() if item["name"].endswith("劳动合同法"))
    assert labor["latest_version"] == "2012-12-28"
    assert labor["versions"] == ["2012-12-28"]


def test_articles_full_text_search(catalog: LawCatalog) -> None:
    result = catalog.articles(q="赔偿金")
    assert result["total"] == 1
    assert result["items"][0]["id"] == "labor-2"


def test_articles_filter_by_law(catalog: LawCatalog) -> None:
    result = catalog.articles(law="民法典", limit=1)
    assert result["total"] == 2
    assert len(result["items"]) == 1
    assert result["items"][0]["law_name"] == "中华人民共和国民法典"

    result = catalog.articles(law="民法典", offset=1, limit=1)
    assert result["items"][0]["article_no"] == 1165


def test_article_number_query_scoped_by_law(catalog: LawCatalog) -> None:
    result = catalog.articles(q="第八十七条")
    assert result["total"] == 2

    result = catalog.articles(law="中华人民共和国民法典", q="第八十七条")
    assert result["total"] == 1
    assert result["items"][0]["id"] == "civil-1"

    result = catalog.articles(q="劳动合同法第八十七条")
    assert result["total"] == 1
    assert result["items"][0]["id"] == "labor-2"


def test_unknown_law_returns_empty(catalog: LawCatalog) -> None:
    assert catalog.articles(law="中华人民共和国刑法")["total"] == 0


def test_law_shorthand_does_not_match_implementation_regulation(tmp_path: Path) -> None:
    docs = [
        {**DOCS[0], "id": "law-1", "law_name": "中华人民共和国劳动合同法"},
        {
            **DOCS[0],
            "id": "rule-1",
            "law_name": "中华人民共和国劳动合同法实施条例",
            "article_no": 1,
            "article_label": "第一条",
        },
    ]
    path = tmp_path / "docs.json"
    path.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    catalog = LawCatalog(path)

    assert catalog.articles(law="劳动合同法")["total"] == 1
    assert catalog.articles(law="中华人民共和国劳动合同法实施条例")["total"] == 1


CASES = [
    {
        "id": "c1",
        "query": "试用期多久",
        "passed": True,
        "latency_ms": 1000,
        "iterations": 2,
        "channel": "dense",
        "score": 0.8,
        "answer": "ok",
        "retrieved": [],
        "checks": [
            {"name": "长度合理", "passed": True, "detail": "120 字"},
            {"name": "引用可溯源", "passed": True, "detail": ""},
        ],
        "retrieval_coverage": {"passed": True, "detail": ""},
    },
    {
        "id": "c2",
        "query": "违法解除怎么赔",
        "passed": False,
        "latency_ms": 3000,
        "iterations": 3,
        "channel": "exact",
        "score": 0.7,
        "answer": "答案",
        "retrieved": [],
        "checks": [
            {"name": "长度合理", "passed": True, "detail": "100 字"},
            {"name": "引用可溯源", "passed": False, "detail": "第八十八条未检索到"},
        ],
        "retrieval_coverage": {"passed": False, "detail": "top-6 未覆盖"},
    },
]


@pytest.fixture()
def store(tmp_path: Path) -> EvalStore:
    (tmp_path / "qwen37_fact_100.json").write_text(
        json.dumps(CASES, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
    return EvalStore(tmp_path)


def test_eval_store_summary(store: EvalStore) -> None:
    runs = store.list_runs()
    assert len(runs) == 1
    run = runs[0]
    assert run["id"] == "qwen37_fact_100"
    assert run["pinned"] is True
    assert run["case_count"] == 2
    assert run["passed"] == 1
    assert run["pass_rate"] == 0.5
    assert run["avg_latency_ms"] == 2000
    assert run["channel_counts"] == {"dense": 1, "exact": 1}

    checks = {item["name"]: item for item in run["checks"]}
    assert checks["引用可溯源"]["failed"] == 1
    assert checks["检索覆盖"]["failed"] == 1
    assert checks["长度合理"]["fail_rate"] == 0.0


def test_eval_store_detail_and_case(store: EvalStore) -> None:
    detail = store.detail("qwen37_fact_100")
    assert detail is not None
    assert [item["id"] for item in detail["failures"]] == ["c2"]
    assert {item["name"] for item in detail["failures"][0]["failed_checks"]} == {
        "引用可溯源",
        "检索覆盖",
    }

    case = store.case("qwen37_fact_100", "c2")
    assert case is not None
    assert case["answer"] == "答案"

    assert store.detail("missing") is None
    assert store.case("qwen37_fact_100", "missing") is None
