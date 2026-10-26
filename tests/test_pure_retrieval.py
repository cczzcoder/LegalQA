"""S2 测试：检索纯函数。"""

from __future__ import annotations

import math

import pytest

from legalqa import config
from legalqa.pure.retrieval import (
    authority_weight,
    cn_to_int,
    extract_article_no,
    fuse_rank,
    minmax_normalize,
    safe_cosine,
)


# ---------------------------------------------------------------------------
# 中文数字
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("1165", 1165),
        ("一", 1),
        ("十", 10),
        ("十六", 16),
        ("二十", 20),
        ("一百零五", 105),
        ("一千一百六十五", 1165),
        ("九百九十九", 999),
        ("两", 2),
        ("〇", 0),
    ],
)
def test_cn_to_int(text: str, expected: int) -> None:
    assert cn_to_int(text) == expected


@pytest.mark.parametrize("text", ["", "  ", "abc", "第", "一条龙", "十百千", "3.5"])
def test_cn_to_int_returns_none_on_garbage(text: str) -> None:
    """脏输入返回 None，绝不抛异常（BC-28）。"""
    result = cn_to_int(text)
    assert result is None or isinstance(result, int)


# ---------------------------------------------------------------------------
# 条号抽取
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("民法典第1165条怎么规定的", 1165),
        ("《中华人民共和国民法典》第一千一百六十五条", 1165),
        ("第 3 条", 3),
        ("劳动合同法第 87 条和民法典第 1165 条", 87),   # 取第一处
        ("没有条号的一句话", None),
        ("第条", None),
    ],
)
def test_extract_article_no(text: str, expected: int | None) -> None:
    assert extract_article_no(text) == expected


def test_extract_article_no_never_raises() -> None:
    for text in ["", "第" * 50, "第999999999999999条", "第〇条"]:
        extract_article_no(text)


# ---------------------------------------------------------------------------
# 余弦
# ---------------------------------------------------------------------------
def test_safe_cosine_basic() -> None:
    assert safe_cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert safe_cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert safe_cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_safe_cosine_zero_vector_is_zero_not_nan() -> None:
    """空向量必须返回 0.0 —— NaN 会让路由静默走错分支（BC-07）。"""
    score = safe_cosine([0.0, 0.0], [1.0, 1.0])
    assert score == 0.0
    assert not math.isnan(score)


def test_safe_cosine_dimension_mismatch() -> None:
    assert safe_cosine([1.0, 2.0], [1.0]) == 0.0
    assert safe_cosine([], []) == 0.0


# ---------------------------------------------------------------------------
# 归一化与融合
# ---------------------------------------------------------------------------
def test_minmax_normalize() -> None:
    assert minmax_normalize([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
    assert minmax_normalize([5.0, 5.0]) == [0.0, 0.0]
    assert minmax_normalize([]) == []


def test_normalized_top1_is_always_one() -> None:
    """锁定 BC-26 的前提：归一化后 top-1 恒为 1.0，与绝对相关性无关。

    这就是「融合分不能拿来比阈值」的数学原因。
    """
    for scores in ([0.31, 0.30, 0.29], [0.91, 0.12, 0.05]):
        assert max(minmax_normalize(scores)) == pytest.approx(1.0)


def test_authority_weight() -> None:
    assert authority_weight("法律") == 1.0
    assert authority_weight("司法解释") == 0.95
    assert authority_weight("未知类型") == config.AUTHORITY_DEFAULT
    assert authority_weight("") == config.AUTHORITY_DEFAULT


def test_fuse_rank_prefers_authority_on_tie() -> None:
    candidates = [
        {"id": "a", "dense_score": 0.80, "doc_type": "地方性法规"},
        {"id": "b", "dense_score": 0.80, "doc_type": "法律"},
    ]
    ranked = fuse_rank(candidates)
    assert ranked[0]["id"] == "b"


def test_fuse_rank_does_not_mutate_input() -> None:
    original = [{"id": "a", "dense_score": 0.5, "doc_type": "法律"}]
    snapshot = dict(original[0])
    fuse_rank(original)
    assert original[0] == snapshot
    assert "fused_score" not in original[0]


def test_fuse_rank_top_k() -> None:
    candidates = [
        {"id": str(i), "dense_score": 0.1 * i, "doc_type": "法律"} for i in range(1, 6)
    ]
    assert len(fuse_rank(candidates, top_k=2)) == 2
    assert fuse_rank(candidates, top_k=0) == []
    assert len(fuse_rank(candidates)) == 5


def test_fuse_rank_handles_dirty_scores() -> None:
    """dense_score 为 None / 字符串 / 缺失时不得崩溃。"""
    candidates = [
        {"id": "a", "dense_score": None},
        {"id": "b", "dense_score": "bad"},
        {"id": "c"},
    ]
    ranked = fuse_rank(candidates)
    assert len(ranked) == 3
    assert all("fused_score" in item for item in ranked)


def test_fuse_rank_empty() -> None:
    assert fuse_rank([]) == []


def test_extract_all_article_nos() -> None:
    from legalqa.pure.retrieval import extract_all_article_nos

    text = "依据《宪法》第三十五条、第五十一条和第九十九条，以及第 1165 条。"
    assert extract_all_article_nos(text) == {35, 51, 99, 1165}
    assert extract_all_article_nos("没有条号") == set()
    assert extract_all_article_nos("") == set()


def test_extract_citations_pairs_law_with_article() -> None:
    from legalqa.pure.retrieval import extract_citations

    text = "《中华人民共和国宪法》第三十五条与第五十一条；另见《刑法》第二百四十五条。"
    pairs = extract_citations(text)
    assert ("中华人民共和国宪法", 35) in pairs
    assert ("中华人民共和国宪法", 51) in pairs
    assert ("刑法", 245) in pairs


def test_extract_citations_without_law_name() -> None:
    from legalqa.pure.retrieval import extract_citations

    assert extract_citations("依据第五十一条") == [(None, 51)]
    assert extract_citations("") == []


def test_match_law_name() -> None:
    from legalqa.pure.retrieval import match_law_name

    laws = ["中华人民共和国宪法", "中华人民共和国劳动法", "中华人民共和国商标法",
            "中华人民共和国商标法实施条例"]
    assert match_law_name("宪法第一条怎么规定", laws) == "中华人民共和国宪法"
    assert match_law_name("劳动法第五十条", laws) == "中华人民共和国劳动法"
    # 最长匹配：「商标法实施条例」不能退化成「商标法」
    assert match_law_name("商标法实施条例第十条", laws) == "中华人民共和国商标法实施条例"
    assert match_law_name("商标法第三十二条", laws) == "中华人民共和国商标法"
    assert match_law_name("第一条", laws) is None
    assert match_law_name("", laws) is None


def test_law_reference_in() -> None:
    from legalqa.pure.retrieval import law_reference_in

    assert law_reference_in("民法典第1165条") == "民法典"
    assert law_reference_in("宪法第一条") == "宪法"
    # 贪婪匹配：「商标法实施条例」不能截断成「商标法」
    assert law_reference_in("商标法实施条例第一条") == "商标法实施条例"
    assert law_reference_in("第一条") is None
    # 句子里的「条怎么规定」不是法律名
    assert law_reference_in("X法第2条怎么规定的") is None
    assert law_reference_in("依据第1165条的规定") is None


def test_extract_citations_does_not_cross_sentence() -> None:
    """法律名与条号之间跨句时，不归属。

    实测模型写「《关于确立劳动关系有关事项的通知》第一条规定……。
    依据《工伤保险条例》第十四条」，早期实现把第十四条归到《通知》名下，
    溯源时误报「通知第 14 条未检索到」。
    """
    from legalqa.pure.retrieval import extract_citations

    text = "《关于确立劳动关系有关事项的通知》第一条规定了认定标准。依据《工伤保险条例》第十四条应当认定工伤。"
    pairs = extract_citations(text)
    assert ("关于确立劳动关系有关事项的通知", 1) in pairs
    assert ("工伤保险条例", 14) in pairs
    assert ("关于确立劳动关系有关事项的通知", 14) not in pairs


def test_extract_citations_same_sentence_multiple_articles() -> None:
    """同一句内多个条号都归属同一个法律名。"""
    from legalqa.pure.retrieval import extract_citations

    text = "《工伤保险条例》第十四条、第十七条和第三十七条均有规定。"
    pairs = extract_citations(text)
    assert ("工伤保险条例", 14) in pairs
    assert ("工伤保险条例", 17) in pairs
    assert ("工伤保险条例", 37) in pairs


# ---------------------------------------------------------------------------
# 法名解析：嵌套解释 / 实施条例 / 裸法名（E14 回归）
# ---------------------------------------------------------------------------
def test_same_law_normalizes_nested_interpretation() -> None:
    """模型写〈〉、语料不带〈〉，必须视为同一部。"""
    from legalqa.pure.retrieval import same_law

    cited = "最高人民法院关于适用〈中华人民共和国民法典〉婚姻家庭编的解释（一）"
    known = "最高人民法院关于适用中华人民共和国民法典婚姻家庭编的解释（一）"
    assert same_law(cited, known)
    assert same_law(known, cited)


def test_same_law_does_not_match_base_law_inside_interpretation() -> None:
    """「解释（一）」里含「民法典」，但两者不是同一部法。

    旧实现用双向子串，`same_law(解释（一）, 民法典)` 为 True，
    于是解释第十条被回查成民法典第十条，附录给出无关条文。
    """
    from legalqa.pure.retrieval import same_law

    cited = "最高人民法院关于适用〈中华人民共和国民法典〉婚姻家庭编的解释（一）"
    assert not same_law(cited, "中华人民共和国民法典")
    assert not same_law("中华人民共和国民法典", cited)


def test_same_law_rejects_implementation_regulation_as_law() -> None:
    """「劳动合同法」是「劳动合同法实施条例」的前缀，不能混为一谈。"""
    from legalqa.pure.retrieval import same_law

    assert not same_law("中华人民共和国劳动合同法实施条例", "中华人民共和国劳动合同法")
    assert not same_law("劳动合同法", "中华人民共和国劳动合同法实施条例")


def test_same_law_allows_suffix_abbreviation() -> None:
    """允许省略前缀的简称，但不允许任意子串。"""
    from legalqa.pure.retrieval import same_law

    full = "最高人民法院关于适用中华人民共和国民法典婚姻家庭编的解释（一）"
    assert same_law("婚姻家庭编的解释（一）", full)


def test_same_law_does_not_match_short_suffix_as_law() -> None:
    """「合同法」是「劳动合同法」的后缀，但两者不是同一部法。"""
    from legalqa.pure.retrieval import same_law

    assert not same_law("中华人民共和国合同法", "中华人民共和国劳动合同法")
    assert not same_law("合同法", "劳动合同法")


def test_resolve_law_returns_unique_and_refuses_ambiguous() -> None:
    from legalqa.pure.retrieval import resolve_law

    laws = [
        "中华人民共和国劳动合同法",
        "中华人民共和国劳动合同法实施条例",
        "中华人民共和国商标法实施条例",
    ]
    assert resolve_law("劳动合同法", laws) == "中华人民共和国劳动合同法"
    assert resolve_law("劳动合同法实施条例", laws) == "中华人民共和国劳动合同法实施条例"
    assert resolve_law("实施条例", laws) is None      # 多义：不能随便挑
    assert resolve_law("刑法", laws) is None          # 语料外


def test_extract_citations_prefers_bare_law_inside_quoted_article() -> None:
    """引文正文里的「劳动合同法第八十七条」不能被归到引文所属的实施条例。"""
    from legalqa.pure.retrieval import extract_citations

    laws = [
        "中华人民共和国劳动合同法",
        "中华人民共和国劳动合同法实施条例",
    ]
    text = (
        "《劳动合同法实施条例》第二十五条：“用人单位违反劳动合同法的规定解除或者终止"
        "劳动合同，依照劳动合同法第八十七条的规定支付了赔偿金的，不再支付经济补偿。”"
    )
    pairs = extract_citations(text, laws)
    assert ("劳动合同法实施条例", 25) in pairs
    assert ("劳动合同法", 87) in pairs
    assert ("劳动合同法实施条例", 87) not in pairs


def test_extract_citations_resolves_this_law() -> None:
    """「该法第八十七条」指代同句最近一个显式法名。"""
    from legalqa.pure.retrieval import extract_citations

    text = "依据《中华人民共和国劳动合同法》第四十六条，后续按该法第八十七条处理。"
    pairs = extract_citations(text)
    assert ("中华人民共和国劳动合同法", 46) in pairs
    assert ("中华人民共和国劳动合同法", 87) in pairs


def test_extract_citations_keeps_outer_interpretation_name() -> None:
    """外层解释名比括号内的「民法典」结束得更晚，必须选外层。"""
    from legalqa.pure.retrieval import extract_citations, same_law

    laws = [
        "中华人民共和国民法典",
        "最高人民法院关于适用中华人民共和国民法典婚姻家庭编的解释（二）",
    ]
    text = "《最高人民法院关于适用〈中华人民共和国民法典〉婚姻家庭编的解释（二）》第十条"
    pairs = extract_citations(text, laws)
    assert len(pairs) == 1
    law, number = pairs[0]
    assert number == 10
    assert same_law(law, laws[1])
    assert not same_law(law, laws[0])
