"""S3 测试：语料解析与向量索引。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from legalqa.corpus.index import VectorIndex
from legalqa.corpus.ingest import Article, parse_articles, read_docx_paragraphs, write_jsonl, load_jsonl

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus" / "constitution.jsonl"


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def test_parse_skips_toc_and_tracks_chapter() -> None:
    paragraphs = [
        "中华人民共和国宪法",
        "目　　录",
        "第一章　总　　纲",          # 目录里的
        "第二章　公民的基本权利和义务",
        "第一章　总　　纲",          # 正文开始
        "第一条　中华人民共和国是社会主义国家。",
        "社会主义制度是根本制度。",   # 续段（款）
        "第二条　一切权力属于人民。",
        "第二章　公民的基本权利和义务",
        "第三十三条　凡具有国籍的人都是公民。",
    ]
    articles = parse_articles(paragraphs, law_name="中华人民共和国宪法")

    assert [a.article_no for a in articles] == [1, 2, 33]
    assert articles[0].text == "中华人民共和国是社会主义国家。\n社会主义制度是根本制度。"
    assert articles[0].chapter == "第一章总纲"
    assert articles[2].chapter == "第二章公民的基本权利和义务"


def test_parse_handles_chinese_numerals() -> None:
    paragraphs = ["第一章　总纲", "第一百四十三条　首都是北京。"]
    articles = parse_articles(paragraphs, law_name="中华人民共和国宪法")
    assert len(articles) == 1
    assert articles[0].article_no == 143
    assert articles[0].article_label == "第一百四十三条"


def test_parse_keeps_paragraph_breaks_for_clauses() -> None:
    paragraphs = ["第一章　总纲", "第一条　第一款。", "第二款。", "第三款。"]
    articles = parse_articles(paragraphs, law_name="中华人民共和国宪法")
    assert articles[0].text.count("\n") == 2


def test_parse_stops_at_tail_explanation() -> None:
    """法条尾巴上的「关于《……》的说明」不是条文，不能被吞进最后一条。

    实测国家通用语言文字法第 32 条因此吞掉整篇修订说明（12 字 → 946 字）。
    """
    paragraphs = [
        "第一章　总纲",
        "第一条　正文。",
        "第三十二条　本法自年月日起施行。",
        "关于《中华人民共和国X法（修订草案）》的说明———",
        "一、修法的必要性。",
    ]
    articles = parse_articles(paragraphs, law_name="X法")

    assert [a.article_no for a in articles] == [1, 32]
    assert articles[-1].text == "本法自年月日起施行。"
    assert "说明" not in articles[-1].text


def test_parse_stops_at_split_tail_explanation() -> None:
    """PDF 把「关于《……」和「……的说明」折成两行时也要截止。"""
    paragraphs = [
        "第一条　正文。",
        "第三十二条　本法自年月日起施行。",
        "关于 《中华人民共和国X法",
        "（修订草案）》 的 说 明",
        "一、修法的必要性。",
    ]
    articles = parse_articles(paragraphs, law_name="X法")

    assert [a.article_no for a in articles] == [1, 32]
    assert articles[-1].text == "本法自年月日起施行。"


def test_parse_drops_preamble() -> None:
    """第一个「第X条」之前的内容（序言）整体丢弃。"""
    paragraphs = ["第一章　总纲", "序言段落一。", "序言段落二。", "第一条　正文。"]
    articles = parse_articles(paragraphs, law_name="中华人民共和国宪法")
    assert len(articles) == 1
    assert articles[0].text == "正文。"


def test_parse_empty_input() -> None:
    assert parse_articles([], law_name="x") == []
    assert parse_articles(["没有条文的段落"], law_name="x") == []


def test_parse_section_tracking() -> None:
    paragraphs = [
        "第三章　国家机构",
        "第一节　全国人民代表大会",
        "第五十七条　全国人大是最高国家权力机关。",
    ]
    articles = parse_articles(paragraphs, law_name="中华人民共和国宪法")
    assert articles[0].chapter == "第三章国家机构"
    assert articles[0].section == "第一节全国人民代表大会"


# ---------------------------------------------------------------------------
# 真实语料（宪法）
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CORPUS.exists(), reason="未构建语料，先跑 scripts/build_corpus.py")
def test_real_corpus_is_complete() -> None:
    docs = load_jsonl(CORPUS)
    numbers = [d["article_no"] for d in docs]

    assert len(docs) == 143, "宪法共 143 条"
    assert numbers == list(range(1, 144)), "条号必须 1..143 连续无缺无重"
    assert all(d["law_name"] == "中华人民共和国宪法" for d in docs)
    assert all(d["text"].strip() for d in docs)
    assert all(d["id"].startswith("constitution-") for d in docs)


@pytest.mark.skipif(not CORPUS.exists(), reason="未构建语料")
def test_real_corpus_spot_check() -> None:
    docs = {d["article_no"]: d for d in load_jsonl(CORPUS)}
    assert "一切权力属于人民" in docs[2]["text"]
    assert docs[143]["text"].strip() == "中华人民共和国首都是北京。"
    assert docs[1]["chapter"] == "第一章总纲"


def test_docx_reader_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_docx_paragraphs(tmp_path / "nope.docx")


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------
def test_jsonl_roundtrip(tmp_path: Path) -> None:
    articles = [
        Article(
            id="constitution-0001",
            law_name="中华人民共和国宪法",
            doc_type="宪法",
            article_no=1,
            article_label="第一条",
            chapter="第一章总纲",
            section="",
            text="正文",
        )
    ]
    target = tmp_path / "corpus.jsonl"
    assert write_jsonl(articles, target) == 1
    loaded = load_jsonl(target)
    assert loaded[0]["article_no"] == 1
    assert loaded[0]["text"] == "正文"


# ---------------------------------------------------------------------------
# 索引
# ---------------------------------------------------------------------------
class FixedEmbedder:
    """按关键词返回固定向量，让排序结果可控。"""

    def __init__(self, mapping: dict[str, list[float]], default: list[float]) -> None:
        self.mapping = mapping
        self.default = default

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping.get(t, self.default) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.mapping.get(text, self.default)


def make_docs() -> list[dict]:
    return [
        {"id": "a", "law_name": "X法", "article_no": 1, "article_label": "第一条",
         "doc_type": "法律", "text": "AAA", "is_valid": True},
        {"id": "b", "law_name": "X法", "article_no": 2, "article_label": "第二条",
         "doc_type": "法律", "text": "BBB", "is_valid": True},
        {"id": "c", "law_name": "Y法", "article_no": 2, "article_label": "第二条",
         "doc_type": "地方性法规", "text": "CCC", "is_valid": True},
    ]


def make_index() -> VectorIndex:
    docs = make_docs()
    emb = FixedEmbedder(
        {"AAA": [1.0, 0.0, 0.0], "BBB": [0.0, 1.0, 0.0], "CCC": [0.0, 0.0, 1.0]},
        [0.0, 0.0, 0.0],
    )
    return VectorIndex.build(docs, emb)


def test_search_orders_by_cosine() -> None:
    index = make_index()
    hits = index.search([0.9, 0.1, 0.0], k=3)
    assert hits[0].doc["id"] == "a"
    assert hits[0].score == pytest.approx(0.9938, abs=1e-3)


def test_search_respects_k() -> None:
    index = make_index()
    assert len(index.search([1.0, 1.0, 1.0], k=2)) == 2
    assert len(index.search([1.0, 1.0, 1.0], k=99)) == 3


def test_search_zero_vector_returns_empty() -> None:
    index = make_index()
    assert index.search([0.0, 0.0, 0.0]) == []


def test_search_dimension_mismatch_raises() -> None:
    index = make_index()
    with pytest.raises(ValueError, match="维度"):
        index.search([1.0, 0.0])


def test_by_article_respects_law_name() -> None:
    """条号通道必须带法律名，否则多法源会串号。"""
    index = make_index()
    assert index.by_article(2, "X法")["id"] == "b"
    assert index.by_article(2, "Y法")["id"] == "c"
    assert index.by_article(999) is None


def test_by_article_ambiguous_returns_none() -> None:
    """不带法律名且条号在多部法律里都存在时，必须返回 None。

    实测教训：「宪法第一条」返回了「劳动法第一条」，而且 score=1.00，
    看起来像精确命中。歧义时宁可回落语义通道，也不能随便挑一个。
    """
    index = make_index()
    assert index.by_article(2) is None, "X法 与 Y法 都有第 2 条，属于歧义"


def test_by_article_unique_without_law_name() -> None:
    """条号唯一时可以不传法律名。"""
    index = make_index()
    assert index.by_article(1)["id"] == "a"


def test_law_names_property() -> None:
    index = make_index()
    assert index.law_names == ["X法", "Y法"]


def test_index_save_load(tmp_path: Path) -> None:
    index = make_index()
    index.save(tmp_path / "idx")
    loaded = VectorIndex.load(tmp_path / "idx")
    assert len(loaded) == 3
    assert loaded.by_article(1, "X法")["id"] == "a"


def test_empty_index() -> None:
    index = VectorIndex([], np.zeros((0, 4), dtype=np.float32))
    assert len(index) == 0
    assert index.search([1.0, 0.0, 0.0, 0.0]) == []
    assert index.by_article(1) is None


def test_docs_vectors_length_mismatch() -> None:
    with pytest.raises(ValueError, match="数量不一致"):
        VectorIndex(make_docs(), np.zeros((2, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# 回归：解析器的两个已修缺陷
# ---------------------------------------------------------------------------
def test_parse_preserves_internal_spaces() -> None:
    """条文内部的空格必须保留。

    早期实现用归一化后的文本提取正文，把「工作 3 年」压成「工作3年」，
    法条文本保真度受损，实体保真校验也会跟着失准。
    """
    paragraphs = ["第一章　总纲", "第一条　劳动者工作 3 年，月薪 8000 元。"]
    articles = parse_articles(paragraphs, law_name="X法")
    assert articles[0].text == "劳动者工作 3 年，月薪 8000 元。"
    assert "3 年" in articles[0].text


def test_body_start_with_multi_chapter_toc() -> None:
    """目录列了多个章时，「第二个第一章」会指到目录里的第二章。

    必须靠「目 录」标记定位，不能靠数第一章出现的次数。
    """
    from legalqa.corpus.ingest import _find_body_start

    paragraphs = [
        "目　　录",
        "第一章　总纲",
        "第二章　公民权利",
        "第三章　国家机构",
        "第一章　总纲",
        "第一条　正文",
    ]
    start = _find_body_start(paragraphs)
    # 目录之后即正文区（目录项不含条文，解析时会被自然丢弃）
    assert start <= 4
    articles = parse_articles(paragraphs, law_name="X法")
    assert [a.article_no for a in articles] == [1]
    assert articles[0].chapter == "第一章总纲"


# ---------------------------------------------------------------------------
# PDF 与目录导入
# ---------------------------------------------------------------------------
def test_scanned_pdf_is_rejected(tmp_path: Path) -> None:
    """无文本层的 PDF 必须明确报错，不能静默产出 0 条。

    实测一份 29 页宪法扫描件：每页只有一张图、无字体对象，
    extract_text() 返回空串。如果不过滤，调用方会以为「这份法律没有条文」。
    """
    from pypdf import PdfWriter

    from legalqa.corpus.ingest import ScannedPdfError, read_pdf_paragraphs

    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)

    with pytest.raises(ScannedPdfError, match="扫描件"):
        read_pdf_paragraphs(path)


def test_read_any_dispatch(tmp_path: Path) -> None:
    from legalqa.corpus.ingest import read_any

    txt = tmp_path / "a.txt"
    txt.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持的格式"):
        read_any(txt)


def test_page_noise_filtered_from_pdf_lines() -> None:
    """页码、页眉行必须滤掉，否则会被当成条文续段拼进正文。"""
    from legalqa.corpus.ingest import _PAGE_NOISE

    noise = ["12", "- 3 -", "— 7 —", "第 5 页", "第 2 页 / 共 10 页", "共 8 页"]
    for line in noise:
        assert any(p.match(line) for p in _PAGE_NOISE), f"{line} 应被过滤"

    content = ["第一条　正文内容", "中华人民共和国民法典", "（2020年5月28日通过）"]
    for line in content:
        assert not any(p.match(line) for p in _PAGE_NOISE), f"{line} 不该被过滤"


def test_slug_strips_common_prefix() -> None:
    from legalqa.corpus.ingest import _slug

    assert _slug("中华人民共和国宪法") == "constitution"
    assert _slug("中华人民共和国民法典") == "民法典"
    assert _slug("中华人民共和国劳动合同法") == "劳动合同法"


def test_article_report_flags_missing_numbers() -> None:
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "ingest_folder", _P(__file__).resolve().parent.parent / "scripts" / "ingest_folder.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    articles = [
        Article(id="x-0001", law_name="X", doc_type="法律", article_no=1,
                article_label="第一条", chapter="", section="", text="a"),
        Article(id="x-0003", law_name="X", doc_type="法律", article_no=3,
                article_label="第三条", chapter="", section="", text="c"),
    ]
    report = module.article_report(articles)
    assert "缺 1 号" in report
    assert "2" in report


def test_infer_doc_type() -> None:
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "ingest_folder2", _P(__file__).resolve().parent.parent / "scripts" / "ingest_folder.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    assert module.infer_doc_type("中华人民共和国宪法") == "宪法"
    assert module.infer_doc_type("最高人民法院关于适用民法典的解释") == "司法解释"
    assert module.infer_doc_type("中华人民共和国道路交通安全法实施条例") == "行政法规"
    assert module.infer_doc_type("中华人民共和国民法典") == "法律"


# ---------------------------------------------------------------------------
# PDF 提取的两个真实陷阱
# ---------------------------------------------------------------------------
def test_article_number_with_internal_spaces() -> None:
    """PDF 常把「第二十七条」拆成「第二十七 条」。

    不允许条号内空格会静默漏条：实测民用航空法 PDF 因此漏了 16 条，
    而且不报任何错。
    """
    paragraphs = [
        "第一章　总则",
        "第二十七 条　这是正文。",
        "第五十 八 条　这也是正文。",
    ]
    articles = parse_articles(paragraphs, law_name="X")
    assert [a.article_no for a in articles] == [27, 58]


def test_backward_reference_is_not_a_new_article() -> None:
    """正文里的回溯引用折行到行首时，不能当成新条文。

    「违反本法第十六条至第十九条……规定的」折行后以「第十六条」开头，
    早期实现会造出正文只有「至」一个字的假条。
    """
    paragraphs = [
        "第一章　总则",
        "第十六条　第一款内容比较长，足够构成一条真条文。",
        "第十七条　第二款内容同样足够长，是真正的条文正文。",
        "第十六条至第十九条、第二十一条规定的行为，依照前款规定处罚。",
    ]
    articles = parse_articles(paragraphs, law_name="X")
    assert [a.article_no for a in articles] == [16, 17]
    assert "至第十九条" in articles[-1].text, "回溯引用应归到上一条"


def test_tiny_body_is_treated_as_continuation() -> None:
    """条号递增但正文极短（「第十七条、」）也是折行残留。"""
    paragraphs = [
        "第一章　总则",
        "第十六条　一条足够长的正文内容。",
        "第十七条、",
    ]
    articles = parse_articles(paragraphs, law_name="X")
    assert [a.article_no for a in articles] == [16]


def test_law_name_normalization() -> None:
    """文件名里的日期、版本说明、书名号都要剥掉。"""
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "ingest_folder3", _P(__file__).resolve().parent.parent / "scripts" / "ingest_folder.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    assert module.law_name_from_path(_P("中华人民共和国劳动法_20181229.docx")) == "中华人民共和国劳动法"
    assert module.law_name_from_path(_P("中华人民共和国宪法（2018年修正文本）_20180311.pdf")) == "中华人民共和国宪法"
    assert module.law_name_from_path(_P("《中华人民共和国民法典》.docx")) == "中华人民共和国民法典"


# ---------------------------------------------------------------------------
# 增量索引
# ---------------------------------------------------------------------------
class CountingEmbedder:
    """记录向量化了多少条，用于验证增量是否真的跳过未变化条文。"""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.embedded = 0

    def _vec(self, text: str) -> list[float]:
        seed = sum(ord(c) for c in text) or 1
        return [float((seed + i) % 7 + 1) for i in range(self.dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def _docs(n: int, prefix: str = "x") -> list[dict]:
    return [
        {"id": f"{prefix}-{i:04d}", "law_name": "X法", "article_no": i,
         "article_label": f"第{i}条", "doc_type": "法律", "text": f"条文{i}", "is_valid": True}
        for i in range(1, n + 1)
    ]


def test_incremental_reuses_unchanged() -> None:
    """新增一条时，只有这一条需要向量化。"""
    emb = CountingEmbedder()
    base = _docs(5)
    index, stats = VectorIndex.build_incremental(base, emb, None)
    assert stats.embedded == 5

    emb2 = CountingEmbedder()
    new_doc = {"id": "y-0001", "law_name": "Y法", "article_no": 1,
               "article_label": "第一条", "doc_type": "法律", "text": "新条文", "is_valid": True}
    extended = [*base, new_doc]              # 加一条新的
    new_index, stats2 = VectorIndex.build_incremental(extended, emb2, index)

    assert stats2.embedded == 1, "只有新增那条需要向量化"
    assert stats2.reused == 5
    assert len(new_index) == 6


def test_incremental_reembeds_changed_text() -> None:
    """id 相同但正文变了（法律修订）必须重新向量化。"""
    emb = CountingEmbedder()
    docs = _docs(3)
    index, _ = VectorIndex.build_incremental(docs, emb, None)

    emb2 = CountingEmbedder()
    revised = [dict(d) for d in docs]
    revised[1]["text"] = "修订后的条文2"
    _new, stats = VectorIndex.build_incremental(revised, emb2, index)

    assert stats.embedded == 1
    assert stats.reused == 2


def test_incremental_reports_removed() -> None:
    emb = CountingEmbedder()
    index, _ = VectorIndex.build_incremental(_docs(5), emb, None)

    emb2 = CountingEmbedder()
    _new, stats = VectorIndex.build_incremental(_docs(3), emb2, index)
    assert stats.removed == 2
    assert stats.reused == 3
    assert stats.embedded == 0


def test_incremental_without_previous_is_full() -> None:
    emb = CountingEmbedder()
    _index, stats = VectorIndex.build_incremental(_docs(4), emb, None)
    assert stats.embedded == 4
    assert stats.reused == 0


def test_incremental_preserves_search_quality() -> None:
    """增量索引的向量必须与全量构建一致，否则检索会漂移。"""
    emb = CountingEmbedder()
    docs = _docs(6)
    full, _ = VectorIndex.build_incremental(docs, emb, None)

    emb2 = CountingEmbedder()
    partial, _ = VectorIndex.build_incremental(docs[:3], emb2, None)
    incremental, _stats = VectorIndex.build_incremental(docs, CountingEmbedder(), partial)

    query = emb.embed_query("条文2")
    assert [h.doc["id"] for h in full.search(query, k=3)] == [
        h.doc["id"] for h in incremental.search(query, k=3)
    ]


# ---------------------------------------------------------------------------
# 时效性：失效条文不参与检索
# ---------------------------------------------------------------------------
def test_search_excludes_repealed_articles() -> None:
    """已废止条文必须被屏蔽。

    引用一条已废止的旧条文比答不出来更危险 —— 用户会当真。
    """
    docs = _docs(3)
    docs[0]["is_valid"] = False          # 第 1 条废止
    emb = CountingEmbedder()
    index = VectorIndex.build(docs, emb)

    hits = index.search(emb.embed_query("条文1"), k=3)
    assert all(h.doc["id"] != docs[0]["id"] for h in hits)
    assert len(hits) == 2

    # 允许显式包含失效条文（查历史版本时）
    all_hits = index.search(emb.embed_query("条文1"), k=3, only_valid=False)
    assert len(all_hits) == 3


def test_search_all_repealed_returns_empty() -> None:
    docs = _docs(2)
    for d in docs:
        d["is_valid"] = False
    index = VectorIndex.build(docs, CountingEmbedder())
    assert index.search([1.0, 1.0, 1.0, 1.0], k=2) == []


def test_version_date_parsed_from_filename() -> None:
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "ingest_folder4", _P(__file__).resolve().parent.parent / "scripts" / "ingest_folder.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    assert module.parse_version_date(_P("中华人民共和国民法典_20200528.docx")) == "2020-05-28"
    assert module.parse_version_date(_P("中华人民共和国宪法.docx")) == ""
    assert module.parse_version_date(_P("x_99999999.docx")) == ""
