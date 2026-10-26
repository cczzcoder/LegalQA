"""S2 测试：窗口结构操作（配对修复 / 当前提问保护 / 不变量校验）。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from legalqa import config
from legalqa.pure.window import (
    is_empty_ai,
    last_human,
    pairs_ok,
    repair_tool_pairs,
    restore_current_query,
    validate_window,
)


def ai_call(cid: str, text: str = "") -> AIMessage:
    return AIMessage(
        id=f"ai-{cid}",
        content=text,
        tool_calls=[{"id": cid, "name": "read_law", "args": {"q": "x"}}],
    )


def tool(cid: str, text: str = "结果") -> ToolMessage:
    return ToolMessage(id=f"tool-{cid}", content=text, tool_call_id=cid)


# ---------------------------------------------------------------------------
# repair_tool_pairs
# ---------------------------------------------------------------------------
def test_repair_keeps_valid_pair() -> None:
    msgs = [HumanMessage(id="h1", content="q"), ai_call("c1"), tool("c1")]
    assert repair_tool_pairs(msgs) == msgs


def test_repair_drops_orphan_tool() -> None:
    """只有 ToolMessage、没有发起方 → 丢弃（正向孤儿）。"""
    msgs = [HumanMessage(id="h1", content="q"), tool("c1")]
    out = repair_tool_pairs(msgs)
    assert len(out) == 1
    assert isinstance(out[0], HumanMessage)


def test_repair_drops_ai_without_result() -> None:
    """AI 声明了 tool_calls 但没有结果 → 整条丢弃（反向孤儿）。"""
    msgs = [HumanMessage(id="h1", content="q"), ai_call("c1")]
    out = repair_tool_pairs(msgs)
    assert len(out) == 1
    assert isinstance(out[0], HumanMessage)


def test_repair_drops_partial_result_set() -> None:
    """两个调用只回了一个结果 → 整对丢弃，不留半对。"""
    two_calls = AIMessage(
        id="ai-multi",
        content="",
        tool_calls=[
            {"id": "c1", "name": "read_law", "args": {}},
            {"id": "c2", "name": "read_law", "args": {}},
        ],
    )
    msgs = [HumanMessage(id="h1", content="q"), two_calls, tool("c1")]
    out = repair_tool_pairs(msgs)
    assert [type(m).__name__ for m in out] == ["HumanMessage"]


def test_repair_cascade_removes_tools_of_dropped_ai() -> None:
    """级联：AI 被删后，它本来合法的 Tool 也变成孤儿，必须一起删。"""
    msgs = [
        HumanMessage(id="h1", content="q"),
        ai_call("c1"),          # 无结果，会被删
        tool("c2"),             # 孤儿
        tool("c3"),             # 孤儿
    ]
    out = repair_tool_pairs(msgs)
    assert [type(m).__name__ for m in out] == ["HumanMessage"]


def test_repair_is_idempotent() -> None:
    msgs = [HumanMessage(id="h1", content="q"), ai_call("c1"), tool("c1")]
    once = repair_tool_pairs(msgs)
    assert repair_tool_pairs(once) == once


def test_repair_handles_empty_and_plain() -> None:
    assert repair_tool_pairs([]) == []
    plain = [HumanMessage(content="a"), AIMessage(content="b")]
    assert repair_tool_pairs(plain) == plain


def test_repair_ignores_malformed_tool_calls() -> None:
    """tool_calls 里混入非 dict / 无 id 的脏数据时不得崩溃。

    正常构造会被 LangChain 校验拦下，所以这里用 model_construct 模拟
    「从 checkpoint 反序列化出来的脏数据」——这正是 BC-19 描述的场景。
    """
    dirty = AIMessage.model_construct(
        id="ai-d",
        content="",
        type="ai",
        additional_kwargs={},
        response_metadata={},
        tool_calls=[{"name": "x"}, "oops"],   # 缺 id、混入字符串
        invalid_tool_calls=[],
    )
    msgs = [HumanMessage(content="q"), dirty, tool("c1")]
    out = repair_tool_pairs(msgs)
    assert isinstance(out[0], HumanMessage)
    assert not any(isinstance(m, ToolMessage) for m in out)


# ---------------------------------------------------------------------------
# pairs_ok
# ---------------------------------------------------------------------------
def test_pairs_ok_valid() -> None:
    assert pairs_ok([HumanMessage(content="q"), ai_call("c1"), tool("c1")])


def test_pairs_ok_rejects_orphan_tool() -> None:
    assert not pairs_ok([HumanMessage(content="q"), tool("c1")])


def test_pairs_ok_rejects_separated_pair() -> None:
    """AI 与 Tool 之间插了别的消息 → 不合法（严格要求紧邻）。"""
    msgs = [
        HumanMessage(content="q"),
        ai_call("c1"),
        HumanMessage(content="打断"),
        tool("c1"),
    ]
    assert not pairs_ok(msgs)


# ---------------------------------------------------------------------------
# restore_current_query
# ---------------------------------------------------------------------------
def test_restore_keeps_window_when_query_present() -> None:
    body = [HumanMessage(id="h1", content="q1"), AIMessage(id="a1", content="r1"),
            HumanMessage(id="h2", content="q2")]
    trimmed = body[1:]
    assert restore_current_query(trimmed, body) == trimmed


def test_restore_falls_back_to_query_only() -> None:
    """当前提问被裁掉时，退化为「只保留这条提问」，而不是答非所问。"""
    body = [HumanMessage(id="h1", content="旧问题"), AIMessage(id="a1", content="旧回答"),
            HumanMessage(id="h2", content="当前问题")]
    out = restore_current_query([], body)
    assert len(out) == 1
    assert out[0].id == "h2"


def test_restore_handles_no_human() -> None:
    body = [AIMessage(id="a1", content="x")]
    assert restore_current_query([], body) == []


# ---------------------------------------------------------------------------
# validate_window
# ---------------------------------------------------------------------------
def good_window() -> list:
    return [
        SystemMessage(content=config.SYSTEM_PROMPT),
        HumanMessage(id="h1", content="公司拖欠工资怎么办"),
        AIMessage(id="a1", content="可以先协商"),
        HumanMessage(id="h2", content="那能要多少"),
    ]


def test_validate_accepts_good_window() -> None:
    assert validate_window(good_window()) == []


def test_validate_empty_window() -> None:
    assert set(validate_window([])) == {"INV-1", "INV-9"}


def test_validate_inv1_wrong_system() -> None:
    window = good_window()
    window[0] = SystemMessage(content="你是一个通用助手")
    assert "INV-1" in validate_window(window)


def test_validate_inv1_missing_system() -> None:
    assert "INV-1" in validate_window(good_window()[1:])


def test_validate_inv2_extra_system() -> None:
    window = good_window()
    window.insert(2, SystemMessage(content="插入的"))
    assert "INV-2" in validate_window(window)


def test_validate_inv3_first_body_not_human() -> None:
    window = [SystemMessage(content=config.SYSTEM_PROMPT), AIMessage(content="我先说")]
    assert "INV-3" in validate_window(window)


def test_validate_inv6_too_many() -> None:
    window = [SystemMessage(content=config.SYSTEM_PROMPT)]
    for i in range(config.WINDOW_MAX_MESSAGES):
        window.append(HumanMessage(id=f"h{i}", content="q"))
    assert "INV-6" in validate_window(window)


def test_validate_inv7_query_dropped() -> None:
    window = [SystemMessage(content=config.SYSTEM_PROMPT), AIMessage(content="x")]
    assert "INV-7" in validate_window(window)


def test_validate_inv7_expected_id_missing() -> None:
    assert "INV-7" in validate_window(good_window(), expected_last_human_id="h999")
    assert "INV-7" not in validate_window(good_window(), expected_last_human_id="h2")


def test_validate_inv8_empty_ai() -> None:
    window = good_window()
    window.append(AIMessage(id="empty", content="   "))
    assert "INV-8" in validate_window(window)


def test_validate_inv4_orphan_tool() -> None:
    window = good_window()
    window.append(tool("c-missing"))
    assert "INV-4/5" in validate_window(window)


def test_validate_inv9_too_short() -> None:
    assert "INV-9" in validate_window([SystemMessage(content=config.SYSTEM_PROMPT)])


# ---------------------------------------------------------------------------
# 其他
# ---------------------------------------------------------------------------
def test_last_human() -> None:
    msgs = [HumanMessage(id="h1", content="a"), AIMessage(content="b"),
            HumanMessage(id="h2", content="c")]
    found = last_human(msgs)
    assert found is not None and found.id == "h2"
    assert last_human([AIMessage(content="x")]) is None


def test_is_empty_ai() -> None:
    assert is_empty_ai(AIMessage(content=""))
    assert is_empty_ai(AIMessage(content="  \n "))
    assert not is_empty_ai(AIMessage(content="有内容"))
    assert not is_empty_ai(ai_call("c1"))          # 有 tool_calls 不算空
    assert not is_empty_ai(HumanMessage(content=""))


def test_insert_before_last_human() -> None:
    from legalqa.pure.window import insert_before_last_human

    messages = [
        HumanMessage(id="h1", content="第一问"),
        AIMessage(id="a1", content="答"),
        HumanMessage(id="h2", content="第二问"),
    ]
    result = insert_before_last_human(messages, "【提示】")
    assert [m.content for m in result] == ["第一问", "答", "【提示】", "第二问"]
    assert isinstance(result[-1], HumanMessage)


def test_insert_before_last_human_without_human() -> None:
    from legalqa.pure.window import insert_before_last_human

    result = insert_before_last_human([AIMessage(content="只有回答")], "【提示】")
    assert [m.content for m in result] == ["只有回答", "【提示】"]
