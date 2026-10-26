"""图状态定义。

对应设计文档 §4：State 契约。

三条硬性约束（违反任意一条都会被 Code Review 打回）：

1. **所有字段必须有默认值** —— 保证 `LegalState()` 零参可构造，
   测试不必拼全字段，checkpoint 反序列化缺字段时也不会炸（BC-19）。
2. **`messages` 用 `add_messages` reducer，其余字段一律覆盖语义** —— 见 ADR-01。
3. **节点读取必须走 `state.get(key, default)`** —— 本类显式提供 `get()`，
   让「KeyError 在物理上不可能发生」，而不是靠人记住别写下标读取。

关于第 3 点的实现说明：
    LangGraph 对 dataclass schema 会构造实例后传给节点，节点拿到的是 `LegalState`
    对象而非 dict。所以这里补了一个 `.get()` 方法，既满足「统一用 .get()」的
    编码规范，又不会因为拼错 key 抛异常（拼错只会拿到默认值，配合测试可发现）。
    故意 **不实现** `__getitem__`：下标访问会直接抛 TypeError，
    这样静态检查（grep 或 AST）能一眼发现违规写法。
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field, fields
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from legalqa import config

_MISSING = object()


@dataclass
class LegalState:
    """全图共享状态。字段分组与设计文档 §4.2 一一对应。"""

    # ------------------------------------------------------------------
    # A. 对话历史 —— source of truth，只增不减
    # ------------------------------------------------------------------
    # 唯一带 reducer 的字段：节点返回的 messages 会被「合并」而非覆盖。
    # 注意 add_messages 是 merge 语义，被裁掉的老消息不会消失（ADR-01）。
    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)

    # ------------------------------------------------------------------
    # B. 压缩视图 —— 每轮由 ContextCompress 重算，不持久化语义
    # ------------------------------------------------------------------
    llm_messages: list[AnyMessage] = field(default_factory=list)
    compressed: bool = False
    dropped_count: int = 0
    # 滚动摘要：窗口之外的历史压成的事实摘要，拼在 system 之后。
    # 滑动窗口的固有缺陷是「6 轮之后就真的忘了」，摘要用来兜住这个洞。
    conversation_summary: str = ""
    # 已经被摘要覆盖的消息条数（按 body 顺序计）。用来避免重复摘要同一批消息。
    summarized_count: int = 0

    # ------------------------------------------------------------------
    # C. 本轮输入
    # ------------------------------------------------------------------
    user_query: str = ""
    rewritten_query: str = ""
    rewrite_applied: bool = False
    rewrite_reason: str = ""
    rewrite_latency_ms: int = 0
    turn_id: int = 0

    # ------------------------------------------------------------------
    # D. 护栏
    # ------------------------------------------------------------------
    guard_blocked: bool = False
    guard_reason: str = ""
    output_blocked: bool = False
    output_reason: str = ""

    # ------------------------------------------------------------------
    # E. RAG
    # ------------------------------------------------------------------
    retrieved_docs: list[dict] = field(default_factory=list)
    rag_score: float = 0.0          # 仅语义通道 top-1 原始余弦（BC-26）
    rag_hit: bool = False
    dense_top1: float = 0.0
    retrieval_channel: str = ""     # exact / dense / none
    retrieval_latency_ms: int = 0

    # ------------------------------------------------------------------
    # F. 降级与兜底
    # ------------------------------------------------------------------
    fallback_text: str = ""
    fallback_saved: bool = False
    # 系统降级（兜底回答、空输出兜底）
    degraded: bool = False
    # 用户主动中断。与 degraded 分开统计：中断是用户行为，不是系统故障，
    # 混在一起会让「降级率」这个指标失真。
    interrupted: bool = False

    # ------------------------------------------------------------------
    # G. 可观测
    # ------------------------------------------------------------------
    # trace 必须挂 reducer。多个节点在同一轮都会返回 trace 增量，
    # 没有 reducer 时后写的会覆盖先写的 —— 实测 7 轮对话只留下 7 条记录
    # （每轮最后一条），前面 input_guard / agent / tool_node 全丢了，
    # 排查问题时等于没有链路。
    trace: Annotated[list[str], operator.add] = field(default_factory=list)
    last_node: str = ""
    error: str = ""

    # ------------------------------------------------------------------
    # H. 循环控制（v1.0 新增，见 §3.3）
    # ------------------------------------------------------------------
    iteration_count: int = 0
    tool_call_history: list[str] = field(default_factory=list)
    agent_stop_reason: str = ""
    window_tokens: int = 0

    # ------------------------------------------------------------------
    # I. 阶段 1 预留（只定义，不接线）
    # ------------------------------------------------------------------
    pending_action: dict | None = None
    requires_approval: bool = False
    approval_granted: bool = False
    tool_permissions: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # 读取接口
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """安全读取。字段不存在时返回 default，绝不抛 KeyError。

        注意：字段存在但值为 None 时返回 None（不回落 default）。
        这样「字段缺失」与「字段显式为 None」不会混淆，路由层需自行处理 None。
        """
        value = getattr(self, key, _MISSING)
        return default if value is _MISSING else value

    def with_trace(self, node: str, note: str = "") -> list[str]:
        """返回**单条** trace 增量，由 `operator.add` reducer 追加。

        注意不要返回完整列表：trace 有 reducer，返回 `[旧..., 新]` 会让
        历史被重复追加 N 倍。
        """
        entry = f"{node}:{note}" if note else node
        return [entry]

    def to_dict(self) -> dict[str, Any]:
        """转为纯 dict，用于日志与 checkpoint 快照。

        messages 保留原对象（不序列化），由调用方决定是否 dump。
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------
def new_state(**overrides: Any) -> LegalState:
    """构造 State 并覆盖指定字段。未知字段直接报错，避免拼写错误被静默吞掉。"""
    known = {f.name for f in fields(LegalState)}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(f"未知的 State 字段: {sorted(unknown)}")
    return LegalState(**overrides)


def append_trace(state: LegalState, node: str, note: str = "") -> dict[str, Any]:
    """节点返回的通用增量：trace + last_node。"""
    return {"trace": [f"{node}:{note}" if note else node], "last_node": node}


def coerce_state(obj: Any) -> LegalState:
    """把图输出转成 LegalState。

    **重要契约（已用测试锁定）：LangGraph 的节点收到 `LegalState` 实例，
    但 `app.invoke()` 返回的是普通 dict。** 这个不对称是真实存在的坑：
    在图外对 `result` 调 `with_trace()` 会 AttributeError。

    边界层（CLI / API / 测试）统一用本函数转换，不要在业务代码里散落
    `LegalState(**result)`。
    """
    if isinstance(obj, LegalState):
        return obj
    if isinstance(obj, dict):
        # LangGraph 可能在输出里混入内部 key（如 __interrupt__），
        # 直接展开会触发 new_state 的未知字段校验，这里先过滤掉。
        known = {f.name for f in fields(LegalState)}
        payload = {k: v for k, v in obj.items() if k in known}
        return new_state(**payload)
    raise TypeError(f"无法转换为 LegalState: {type(obj).__name__}")
