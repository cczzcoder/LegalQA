"""S10 测试：通用分面检索规划与证据闸门。"""

from __future__ import annotations

from legalqa import config
from legalqa.pure.facets import (
    detect_issue_facets,
    effective_tool_budget,
    facet_citation_evidence,
    facet_evidence,
    match_facet,
    parse_facets,
    parse_repair,
    render_repair_appendix,
    should_refine_facets,
    split_query,
    validate_repair,
)


HARD_QUERY = (
    "在用工身份尚存争议时，居家按指令处理任务受伤的“工作场所”“工作原因”如何认定；"
    "线上碎片化指令下的加班，应以“用人单位安排或追认”还是“劳动者自愿”为界限，"
    "举证责任如何分配？"
)


def test_split_compound_by_semicolon() -> None:
    parts = split_query(HARD_QUERY)
    assert len(parts) == 2
    assert "居家" in parts[0]
    assert "加班" in parts[1]


def test_split_short_single_returns_empty() -> None:
    assert split_query("工伤怎么认定？") == []


def test_detect_issue_facets_hard_query() -> None:
    facets = detect_issue_facets(HARD_QUERY)
    assert len(facets) == 5
    texts = " ".join(facet["text"] for facet in facets)
    assert "用工身份" in texts
    assert "居家" in texts
    assert "加班" in texts
    assert "举证" in texts


def test_detect_issue_facets_marriage_query() -> None:
    facets = detect_issue_facets(
        "离婚时对方转移存款，还在网上造谣我，我能怎么分割财产？"
    )
    texts = " ".join(facet["text"] for facet in facets)
    assert "转移" in texts
    assert "财产分割" in texts
    assert "名誉权" in texts


def test_detect_issue_facets_single_issue_returns_empty() -> None:
    assert detect_issue_facets("工伤怎么认定？") == []


def test_should_refine_long_compound() -> None:
    assert should_refine_facets(HARD_QUERY, split_query(HARD_QUERY))


def test_should_not_refine_short_two_questions() -> None:
    query = "公司拖欠工资能要补偿吗？没签合同能要双倍工资吗？"
    assert not should_refine_facets(query, split_query(query))


def test_parse_facets_json() -> None:
    raw = (
        "```json\n"
        '[{"text": "劳动关系确认", "query": "怎么确认劳动关系和用工主体责任"},'
        ' {"text": "居家工伤", "query": "居家办公受伤算不算工伤"}]'
        "\n```"
    )
    facets = parse_facets(raw)
    assert len(facets) == 2
    assert facets[0]["text"] == "劳动关系确认"
    assert "居家" in facets[1]["query"]


def test_parse_facets_falls_back() -> None:
    facets = parse_facets("不是 JSON", fallback=["方面A", "方面B"])
    assert facets == [
        {"text": "方面A", "query": "方面A"},
        {"text": "方面B", "query": "方面B"},
    ]


def test_match_facet_by_overlap() -> None:
    facets = [
        {"text": "劳动关系确认与用工主体责任", "query": "确认劳动关系 用工主体责任"},
        {"text": "居家办公工伤认定", "query": "居家办公 工作原因 工伤"},
    ]
    assert match_facet("怎么确认劳动关系和用工主体责任", facets) == 1
    assert match_facet("居家办公期间受伤算工伤吗", facets) == 2


def test_facet_evidence_reports_missing() -> None:
    facets = [{"text": "方面A"}, {"text": "方面B"}]
    attempts = [
        {"facet": 1, "hit": True},
        {"facet": 2, "hit": False},
    ]
    missing = facet_evidence(facets, attempts)
    assert [item["index"] for item in missing] == [2]
    assert missing[0]["attempted"] is True


def test_facet_citation_evidence_detects_uncited_material() -> None:
    facets = [{"text": "方面A"}, {"text": "方面B"}]
    attempts = [
        {
            "facet": 1,
            "hit": True,
            "docs": [{"law_name": "X法", "article_no": 1}],
        },
        {
            "facet": 2,
            "hit": True,
            "docs": [{"law_name": "X法", "article_no": 2}],
        },
    ]
    missing = facet_citation_evidence(facets, attempts, "依据《X法》第一条。")
    assert [item["index"] for item in missing] == [2]
    assert missing[0]["reason"] == "检索了但未引用依据"


def test_facet_citation_evidence_detects_unsearched_facet() -> None:
    facets = [{"text": "方面A"}]
    missing = facet_citation_evidence(facets, [], "没有引用。")
    assert missing[0]["reason"] == "未检索"


def test_facet_citation_evidence_requires_core_source() -> None:
    facets = [
        {
            "text": "身份",
            "core": [["关于确立劳动关系有关事项的通知", 1]],
        }
    ]
    attempts = [
        {
            "facet": 1,
            "hit": True,
            "docs": [
                {
                    "law_name": "中华人民共和国劳动争议调解仲裁法",
                    "article_no": 6,
                }
            ],
        }
    ]
    missing = facet_citation_evidence(
        facets, attempts, "依据《劳动争议调解仲裁法》第六条。"
    )
    assert missing[0]["reason"] == "未引用核心法源"

    assert facet_citation_evidence(
        facets,
        attempts,
        "依据《关于确立劳动关系有关事项的通知》第一条。",
    ) == []


def test_effective_tool_budget() -> None:
    assert effective_tool_budget([]) == config.MAX_TOOL_CALLS
    assert effective_tool_budget([{}, {}]) == 2
    assert effective_tool_budget([{}] * 4) == 4


def _repair_docs() -> dict[int, list[dict]]:
    return {
        1: [
            {
                "law_name": "中华人民共和国劳动合同法",
                "article_no": 87,
                "article_label": "第八十七条",
                "text": (
                    "用人单位违反本法规定解除或者终止劳动合同的，"
                    "应当依照本法第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金。"
                ),
            }
        ]
    }


def test_parse_repair_json() -> None:
    raw = (
        '{"facets":[{"index":1,"supported":true,"citations":'
        '[{"law":"劳动合同法","article_no":87,"quote":"二倍向劳动者支付赔偿金"}]}]}'
    )
    parsed = parse_repair(raw)
    assert parsed[1]["supported"] is True
    assert parsed[1]["citations"][0]["article_no"] == 87


def test_validate_repair_accepts_verbatim_quote() -> None:
    payload = parse_repair(
        '{"facets":[{"index":1,"supported":true,"citations":'
        '[{"law":"劳动合同法","article_no":87,'
        '"quote":"应当依照本法第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金"}]}]}'
    )
    accepted = validate_repair(payload, _repair_docs())
    assert accepted[1][0]["article_no"] == 87
    assert "经济补偿标准" in accepted[1][0]["quote"]


def test_validate_repair_rejects_wrong_quote() -> None:
    payload = parse_repair(
        '{"facets":[{"index":1,"supported":true,"citations":'
        '[{"law":"劳动合同法","article_no":87,"quote":"用人单位应当支付三倍赔偿金"}]}]}'
    )
    assert validate_repair(payload, _repair_docs()) == {}


def test_validate_repair_rejects_article_outside_material() -> None:
    payload = parse_repair(
        '{"facets":[{"index":1,"supported":true,"citations":'
        '[{"law":"劳动合同法","article_no":47,"quote":"经济补偿按工作年限计算"}]}]}'
    )
    assert validate_repair(payload, _repair_docs()) == {}


def test_render_repair_appendix() -> None:
    accepted = {
        1: [
            {
                "law_name": "中华人民共和国劳动合同法",
                "article_no": 87,
                "article_label": "第八十七条",
                "quote": "二倍向劳动者支付赔偿金",
            }
        ]
    }
    text = render_repair_appendix(accepted, [{"text": "违法解除赔偿"}])
    assert "【补充依据" in text
    assert "方面 1：违法解除赔偿" in text
    assert "第八十七条" in text
