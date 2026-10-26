"""图组装与路由函数。

对应设计文档 §3.2 / §3.5 / §3.7。

两档模式共用同一批节点实现，只有 `add_edge` 的连法不同：
  - `loop`（默认）：带环 Agent Loop，模型决定调不调工具
  - `pipeline`：固定流水线，模型不绑工具，检索由 `read_once` 前置完成
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from legalqa import config
from legalqa.nodes.agent import make_agent_node, make_tool_node
from legalqa.nodes.context import make_context_compress_node
from legalqa.nodes.fallback import save_state_node, static_fallback_node
from legalqa.nodes.guards import input_guard_node, make_output_guard_node
from legalqa.state import LegalState
from legalqa.tools.read_tool import READ_TOOL_SPEC, ReadTool

__all__ = [
    "build_graph",
    "route_after_guard",
    "route_after_agent",
    "LOOP",
    "PIPELINE",
]

LOOP = "loop"
PIPELINE = "pipeline"


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
def route_after_guard(state: LegalState) -> str:
    """入口分流。必须返回字面量，隐式 None 会触发 InvalidUpdateError。"""
    return "fallback" if state.get("guard_blocked") else "agent"


def route_after_agent(state: LegalState) -> str:
    """循环出口判定。**判定顺序不可调换。**

    1. 出错 → fallback（必须排第一，否则带着 error 继续跑）
    2. 预算耗尽 → end（必须排在 tool_calls 判定之前，BC-44）
    3. 无 tool_calls → end
    4. 工具调用次数超限 → **answer**（不是 end）
    5. 其余 → tools

    第 4 步返回 `answer` 而不是 `end`，是 E13 修掉的一个真实事故：
    模型想继续检索但次数已用尽时，直接 `end` 会让**最后一条消息停留在
    「带 tool_calls、正文为空」的 AIMessage 上**。OutputGuard 见到 tool_calls
    会跳过处理（它只管最终回答），于是用户收到 0 字回答，免责声明也没处加。
    正确做法是把模型送回 Agent 再生成一次 —— 那一轮工具已解绑（见
    `make_agent_node` 的 `tool_budget_exhausted`），它只能输出正文。
    `iteration >= MAX_ITER` 会兜住这一跳，不会无限循环。
    """
    if state.get("error"):
        return "fallback"

    if int(state.get("iteration_count", 0) or 0) >= config.MAX_ITER:
        return "end"

    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if last is None:
        return "end"
    if not getattr(last, "tool_calls", None):
        return "end"

    if len(state.get("tool_call_history") or []) >= config.MAX_TOOL_CALLS:
        return "answer"
    return "tools"


# ---------------------------------------------------------------------------
# pipeline 专用：前置检索
# ---------------------------------------------------------------------------
def make_read_once_node(read_tool: ReadTool):
    """pipeline 模式的单次检索节点。

    检索结果以 HumanMessage 追加。这是降级模式的已知妥协：正常 loop 模式下
    结果走 ToolMessage（更符合模型训练分布），这里没有 tool_calls 可挂，
    只能塞进对话。用醒目前缀让它与真实用户输入区分开。
    """

    def read_once_node(state: LegalState) -> dict[str, Any]:
        query = state.get("user_query") or ""
        result = read_tool.run(query, state.get("messages") or [])
        message = HumanMessage(
            id=f"ctx-{state.get('turn_id', 0)}",
            content=f"【系统检索结果，非用户输入】\n{result.message}",
        )
        return {
            "messages": [message],
            "retrieved_docs": result.docs,
            "rag_score": result.score,
            "dense_top1": result.dense_top1,
            "rag_hit": result.hit,
            "retrieval_channel": result.channel,
            "retrieval_latency_ms": result.latency_ms,
            "rewritten_query": result.query,
            "rewrite_applied": result.rewrite_applied,
            "rewrite_reason": result.rewrite_reason,
            "trace": state.with_trace("read_once"),
            "last_node": "read_once",
        }

    return read_once_node


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------
def build_graph(
    *,
    llm: Any,
    read_tool: ReadTool,
    mode: str = LOOP,
    index: Any = None,
) -> Any:
    if mode not in (LOOP, PIPELINE):
        raise ValueError(f"未知模式 {mode!r}，可选：{LOOP} / {PIPELINE}")

    # 索引用于 OutputGuard 的「引用回捞」：回答里引的条号不在本批材料时，
    # 按 (法律名, 条号) 回查法条库，区分「只是没召回」和「编造」。
    # 不传就退化为旧行为（不在材料即未溯源）。
    if index is None:
        index = getattr(read_tool, "index", None)

    graph = StateGraph(LegalState)
    graph.add_node("input_guard", input_guard_node)
    # 传 llm 才启用滚动摘要：窗口之外的历史压成事实摘要，避免 6 轮后失忆
    graph.add_node("context_compress", make_context_compress_node(llm))
    graph.add_node("output_guard", make_output_guard_node(index))
    graph.add_node("static_fallback", static_fallback_node)
    graph.add_node("save_state", save_state_node)

    graph.add_edge(START, "input_guard")
    graph.add_edge("output_guard", END)
    graph.add_edge("static_fallback", "save_state")
    graph.add_edge("save_state", END)

    if mode == LOOP:
        graph.add_node("agent", make_agent_node(llm, [READ_TOOL_SPEC]))
        graph.add_node("tool_node", make_tool_node(read_tool))
        graph.add_conditional_edges(
            "input_guard",
            route_after_guard,
            {"agent": "context_compress", "fallback": "static_fallback"},
        )
        graph.add_edge("context_compress", "agent")
        graph.add_conditional_edges(
            "agent",
            route_after_agent,
            {
                "tools": "tool_node",
                "end": "output_guard",
                # 次数用尽但模型还想调工具 → 回 compress 再走一次 agent，
                # 那一轮已解绑工具，只能出正文（E13）
                "answer": "context_compress",
                "fallback": "static_fallback",
            },
        )
        graph.add_edge("tool_node", "context_compress")   # 回边，构成循环
    else:
        # pipeline：模型不绑工具，检索前置一次。
        #
        # **顺序必须是 read_once → context_compress → agent，不能反过来。**
        # `Agent` 只读 `llm_messages`，而 `llm_messages` 是 `ContextCompress`
        # 从 `messages` 算出来的。如果 read_once 排在 compress 之后，
        # 它追加的检索结果永远进不了窗口 —— 模型看不到任何法条，
        # 整条 pipeline 退化成「无检索裸生成」，而且不报错。
        graph.add_node("agent", make_agent_node(llm, []))
        graph.add_node("read_once", make_read_once_node(read_tool))
        graph.add_conditional_edges(
            "input_guard",
            route_after_guard,
            {"agent": "read_once", "fallback": "static_fallback"},
        )
        graph.add_edge("read_once", "context_compress")
        graph.add_edge("context_compress", "agent")
        graph.add_conditional_edges(
            "agent",
            route_after_agent,
            {
                "end": "output_guard",
                "fallback": "static_fallback",
                "tools": "output_guard",
                # pipeline 不绑工具，理论上产不出 tool_calls，`answer` 不可达；
                # 仍显式映射，避免某个模型硬编 tool_calls 时抛 KeyError。
                "answer": "context_compress",
            },
        )

    return graph.compile()
