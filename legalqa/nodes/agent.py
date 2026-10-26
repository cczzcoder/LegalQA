"""图节点：Agent（循环核心）与 ToolNode（工具执行器）。

对应设计文档 §3.3 / §6.4 / §6.5。

循环的终止**绝不依赖模型自觉**。1.5B 会反复调同一个工具、反复改写同一个
query，永远不会「自己停下来」。所以：

  - 超预算时 `bind_tools([])` —— 模型**物理上**无法产生合法 tool_calls
  - 即使它硬编了 tool_calls 结构，第 5 步会手动剥离
  - 路由函数再兜一层（`iteration_count >= MAX_ITER` 一律 "end"）

三重保险，任何一重单独生效即可终止。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Sequence

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from legalqa import config
from legalqa.pure.window import insert_before_last_human
from legalqa.state import LegalState
from legalqa.tools.read_tool import READ_TOOL_SPEC, ReadTool, ToolResult

__all__ = [
    "make_agent_node",
    "make_tool_node",
    "call_signature",
    "last_ai_was_interrupted",
    "INTERRUPT_NOTE",
]

# 用户中途打断时追加给模型的提示。
# 半截回答已经进了历史，但如果不说清「它是被打断的」，模型会把它当成完整结论，
# 继续往下答时要么重复、要么基于一个未完成的论断推理。
INTERRUPT_NOTE = (
    "【系统提示】上一轮回答被用户中途打断，只输出到一半（那条消息标记了 interrupted）。"
    "用户的新输入可能是对它的补充、修正，或者一个全新问题：\n"
    "- 若是补充或修正：结合被打断的内容一起回答，把上一轮没说完的补完整；\n"
    "- 若是全新问题：直接作答，不要重复上一轮已经说过的内容。"
)

_FALLBACK_REPLY = (
    "我暂时无法组织出可靠回答。可以请您把问题拆得更具体一些吗？"
    "比如说明事情经过、涉及哪一方、您希望达成什么结果。"
)


# ---------------------------------------------------------------------------
# 工具调用签名（用于去重）
# ---------------------------------------------------------------------------
def call_signature(name: str, args: Any) -> str:
    """生成调用指纹。相同 (工具, 参数) 视为重复调用。

    用 sha256 而不是内置 hash：后者带随机盐，跨进程不稳定，
    重启后去重账本会全部失效。
    """
    payload = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{name}:{digest}"


def _last_message(messages: Sequence[AnyMessage]) -> AnyMessage | None:
    return messages[-1] if messages else None


def last_ai_was_interrupted(messages: Sequence[AnyMessage]) -> bool:
    """窗口里最后一条 AI 消息是否被用户打断。

    只看最后一条 AI：更早的中断已经处理过了，再提示会干扰模型。
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return bool((message.additional_kwargs or {}).get("interrupted"))
    return False


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
def make_agent_node(llm: Any, tool_specs: Sequence[dict] | None = None) -> Callable[[LegalState], dict]:
    """构造 Agent 节点。LLM 通过闭包注入，便于测试替换。"""
    specs = list(tool_specs if tool_specs is not None else [READ_TOOL_SPEC])

    def agent_node(state: LegalState) -> dict[str, Any]:
        iteration = int(state.get("iteration_count", 0) or 0) + 1
        # MAX_ITER 是「最多几次 LLM 调用」。第 MAX_ITER 次不再给工具，
        # 强制它输出最终答案 —— 这样「最多 2 轮检索 + 1 轮作答」，
        # 与 route_after_agent 的 `iteration >= MAX_ITER -> end` 完全一致。
        # 两处判据必须同步，否则会出现「预算已耗尽但仍放行到 ToolNode」（BC-44）。
        over_budget = iteration >= config.MAX_ITER
        messages = list(state.get("llm_messages") or [])
        trace = state.with_trace("agent", f"iter={iteration}")

        if not messages:
            return {
                "iteration_count": iteration,
                "agent_stop_reason": "error",
                "error": "empty_window",
                "guard_reason": "agent_error",
                "trace": trace,
                "last_node": "agent",
            }

        # 上一轮被用户打断：把半截回答标记出来，避免模型当它是完整结论
        if last_ai_was_interrupted(messages):
            messages = insert_before_last_human(messages, INTERRUPT_NOTE)

        # 超预算时不绑工具，并明确告知「不能再检索」。
        # 否则模型会输出「再检索一下工伤认定程序……」这种半句话 —— 它想调工具
        # 但工具已经没了，于是把意图写进正文，用户看到一条没头没尾的回答。
        if over_budget:
            messages = insert_before_last_human(
                messages,
                "【系统提示】检索次数已用尽，不能再调用任何工具。"
                "请立即基于以上已有材料给出最终回答；"
                "材料不足的部分直接说明「现有资料未涵盖」，不要描述你打算做什么。",
            )

        try:
            target = llm
            if not over_budget and specs:
                try:
                    target = llm.bind_tools(specs)
                except (AttributeError, NotImplementedError):
                    target = llm          # 模型不支持工具绑定，退化为纯生成
            response = target.invoke(messages)
        except Exception as exc:
            return {
                "iteration_count": iteration,
                "agent_stop_reason": "error",
                "error": f"llm_failed: {type(exc).__name__}",
                "guard_reason": "agent_error",
                "trace": trace,
                "last_node": "agent",
            }

        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(getattr(response, "content", "") or ""))

        tool_calls = list(getattr(response, "tool_calls", None) or [])

        # 超预算还硬编 tool_calls → 手动剥离（终止保证第 2 重）
        if over_budget and tool_calls:
            response = response.model_copy(
                update={"tool_calls": [], "invalid_tool_calls": []}
            )
            tool_calls = []

        content = response.content if isinstance(response.content, str) else ""
        stop_reason = "tool_call" if tool_calls else "no_tool_call"

        # 空回答不入历史（INV-8），改写兜底文案
        if not content.strip() and not tool_calls:
            response = response.model_copy(update={"content": _FALLBACK_REPLY})
            stop_reason = "empty_output"

        if over_budget:
            stop_reason = "budget_exhausted"

        return {
            "messages": [response],
            "iteration_count": iteration,
            "agent_stop_reason": stop_reason,
            "degraded": stop_reason == "empty_output" or state.get("degraded", False),
            "trace": trace,
            "last_node": "agent",
        }

    return agent_node


# ---------------------------------------------------------------------------
# ToolNode
# ---------------------------------------------------------------------------
def make_tool_node(read_tool: ReadTool) -> Callable[[LegalState], dict]:
    """构造工具执行节点。"""

    def tool_node(state: LegalState) -> dict[str, Any]:
        trace = state.with_trace("tool_node")
        base: dict[str, Any] = {"trace": trace, "last_node": "tool_node"}

        messages = list(state.get("messages") or [])
        last = _last_message(messages)
        calls = list(getattr(last, "tool_calls", None) or []) if isinstance(last, AIMessage) else []

        if not calls:
            return {**base, "error": "no_tool_calls"}

        history = list(state.get("tool_call_history") or [])
        seen = set(history)
        new_results: list[ToolMessage] = []
        executed: list[str] = []
        best: ToolResult | None = None
        # **累积本轮所有检索结果**，不能只留分数最高的那一次。
        # 实测：模型检索两次（一次查认定条件、一次查待遇标准），只留 best 的话
        # 另一次的法律名不在 retrieved_docs 里，OutputGuard 的引用溯源会把
        # 《工伤保险条例》误判成「知识库之外的法律」。
        # 把 state 里已有的结果也带上：`retrieved_docs` 没有 reducer，
        # 第二次 tool_node 返回时会**覆盖**第一次的结果。实测模型检索两次
        # （认定条件 + 待遇标准），最后 state 里只剩后一次的 4 条，
        # OutputGuard 的引用溯源于是把前一次查到的条文判成「未溯源」。
        accumulated: list[dict] = list(state.get("retrieved_docs") or [])

        for call in calls:
            name = str(call.get("name", ""))
            args = call.get("args") or {}
            call_id = str(call.get("id") or f"call-{len(new_results)}")
            signature = call_signature(name, args)

            # 去重：相同调用不再执行（终止保证第 4 重）
            if signature in seen:
                new_results.append(
                    ToolMessage(
                        # id 必须带序号：模型可能在同一轮复用同一个 call id，
                        # 只按 call_id 生成会让两条 ToolMessage 撞 id，
                        # add_messages 合并后少一条 → 配对完整性被破坏。
                        id=f"tm-{call_id}-{len(new_results)}",
                        content="该查询本轮已执行过，结果见上文，请勿重复调用。",
                        tool_call_id=call_id,
                    )
                )
                continue

            seen.add(signature)
            executed.append(signature)

            if name != read_tool.name:
                new_results.append(
                    ToolMessage(
                        id=f"tm-{call_id}-{len(new_results)}",
                        content=f"未知工具：{name}。当前只支持 {read_tool.name}。",
                        tool_call_id=call_id,
                    )
                )
                continue

            query = str(args.get("query", "") or "")
            try:
                result = read_tool.run(query, messages)
            except Exception as exc:      # 双保险：工具内部已兜底，这里再兜一层
                result = ToolResult(error=f"{type(exc).__name__}: {exc}")
                result.message = f"【检索失败】{result.error}\n请如实告知用户暂时无法查阅法条。"

            new_results.append(
                ToolMessage(
                    id=f"tm-{call_id}-{len(new_results)}",
                    content=result.message,
                    tool_call_id=call_id,
                )
            )
            accumulated.extend(result.docs)
            if best is None or result.score > best.score:
                best = result

        payload: dict[str, Any] = {
            **base,
            "messages": new_results,
            "tool_call_history": [*history, *executed],
        }
        if best is not None:
            # 按 id 去重，保留首次出现的顺序（已按融合分排序）
            merged: list[dict] = []
            seen_ids: set[str] = set()
            for doc in accumulated:
                key = str(doc.get("id", ""))
                if key and key not in seen_ids:
                    seen_ids.add(key)
                    merged.append(doc)
            payload.update(
                {
                    "retrieved_docs": merged,
                    "rag_score": best.score,
                    "dense_top1": best.dense_top1,
                    "rag_hit": best.hit,
                    "retrieval_channel": best.channel,
                    "retrieval_latency_ms": best.latency_ms,
                    "rewritten_query": best.query,
                    "rewrite_applied": best.rewrite_applied,
                    "rewrite_reason": best.rewrite_reason,
                }
            )
        return payload

    return tool_node
