"""Web 服务层：把 LangGraph 会话封装成 SSE 事件流。

CLI 的打印逻辑不在这里复用；本模块只负责「一次提问 → 结构化事件」，
前端不需要了解 LangGraph、tool_calls 或 State 字段。
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Sequence

from langchain_core.messages import AIMessage, HumanMessage

from legalqa.cli import ROOT, build_embedder, build_llm, final_answer, load_index
from legalqa.graph import LOOP, build_graph
from legalqa.state import coerce_state
from legalqa.tools.read_tool import ReadTool

__all__ = ["ChatEvent", "LegalQAService", "get_service", "set_service", "reload_service"]


@dataclass
class ChatEvent:
    type: str
    text: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_sse(self) -> str:
        import json

        return f"data: {json.dumps(asdict(self), ensure_ascii=False)}\n\n"


class LegalQAService:
    """进程内单例服务：索引/LLM/图只初始化一次。"""

    def __init__(self, *, mode: str = LOOP) -> None:
        from legalqa.env import load_dotenv

        load_dotenv()
        self.embedder = build_embedder("auto")
        self.index = load_index(
            ROOT / "data/index/all",
            ROOT / "data/corpus/all.jsonl",
            self.embedder,
        )
        self.llm = build_llm(
            argparse.Namespace(dry_run=False, provider=None, model=None)
        )
        # 与当前评测基线一致：不做 LLM 重排、不做无条件补召回
        self._mode = mode
        self.read_tool = ReadTool(self.index, self.embedder)
        self.app = build_graph(llm=self.llm, read_tool=self.read_tool, mode=mode)

    def reload(self) -> None:
        """文献入库后重载索引：语料变了就自动重建，并换掉工具与图里的旧引用。"""
        self.index = load_index(
            ROOT / "data/index/all",
            ROOT / "data/corpus/all.jsonl",
            self.embedder,
        )
        self.read_tool = ReadTool(self.index, self.embedder)
        self.app = build_graph(llm=self.llm, read_tool=self.read_tool, mode=self._mode)

    # ------------------------------------------------------------------
    def _messages(self, query: str, history: Sequence[dict[str, Any]], turn_id: int):
        messages = []
        for index, item in enumerate(list(history)[-20:]):
            role = str(item.get("role", ""))
            content = str(item.get("content", "") or "")
            if not content.strip():
                continue
            if role == "user":
                messages.append(HumanMessage(content=content, id=f"h-{turn_id}-{index}"))
            elif role == "assistant":
                messages.append(AIMessage(content=content, id=f"a-{turn_id}-{index}"))
        messages.append(HumanMessage(content=query, id=f"h-{turn_id}-{len(messages)}"))
        return messages

    @staticmethod
    def _citations(state: Any) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        seen: set[tuple[str, Any]] = set()
        for doc in state.retrieved_docs or []:
            law = str(doc.get("law_name", ""))
            number = doc.get("article_no")
            key = (law, number)
            if key in seen:
                continue
            seen.add(key)
            label = doc.get("article_label") or (
                f"第{number}条" if number is not None else ""
            )
            citations.append(
                {
                    "law_name": law,
                    "article_no": number,
                    "article_label": label,
                    "version": doc.get("version_date")
                    or doc.get("effective_date")
                    or "",
                    "effective_date": doc.get("effective_date") or "",
                    "valid": bool(doc.get("is_valid", True)),
                    "text": str(doc.get("text", "")),
                    "source": doc.get("retrieval_source") or "tool",
                    "doc_type": doc.get("doc_type") or "",
                }
            )
            if len(citations) >= 12:
                break
        return citations

    # ------------------------------------------------------------------
    def stream(
        self,
        query: str,
        history: Sequence[dict[str, Any]] | None = None,
        turn_id: int = 0,
    ) -> Iterator[ChatEvent]:
        started = time.perf_counter()
        payload = {
            "messages": self._messages(query, history or [], turn_id),
            "turn_id": turn_id,
        }
        yield ChatEvent("status", meta={"stage": "thinking"})

        partial: list[str] = []
        final_state: dict[str, Any] | None = None
        try:
            for mode, chunk in self.app.stream(
                payload, stream_mode=["messages", "values"]
            ):
                if mode == "messages":
                    message, metadata = chunk
                    node = metadata.get("langgraph_node") if isinstance(metadata, dict) else None
                    if node == "tool_node":
                        # 上一段 agent 输出只是「我来查一下」这类中间产物
                        partial.clear()
                        yield ChatEvent("reset")
                        yield ChatEvent("status", meta={"stage": "retrieving"})
                    elif node == "agent":
                        content = message.content if isinstance(message.content, str) else ""
                        if content:
                            partial.append(content)
                            yield ChatEvent("delta", text=content)
                elif mode == "values":
                    final_state = chunk
        except Exception as exc:  # noqa: BLE001 - 前端要拿到可读错误
            yield ChatEvent(
                "error",
                error=f"{type(exc).__name__}: {exc}",
                meta={"stage": "error"},
            )
            return

        if final_state is None:
            yield ChatEvent("error", error="empty_state")
            return

        state = coerce_state(final_state)
        answer = final_answer(state)
        if answer != "".join(partial):
            # OutputGuard 可能追加免责声明/回捞附录/引用提示，统一以最终状态为准
            yield ChatEvent("replace", text=answer)

        yield ChatEvent("citations", citations=self._citations(state))
        yield ChatEvent(
            "meta",
            meta={
                "degraded": bool(state.degraded),
                "guard_blocked": bool(state.guard_blocked),
                "guard_reason": state.guard_reason,
                "output_blocked": bool(state.output_blocked),
                "output_reason": state.output_reason,
                "iterations": int(state.iteration_count),
                "channel": state.retrieval_channel,
                "score": round(float(state.rag_score or 0.0), 4),
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "error": state.error,
            },
        )
        yield ChatEvent("done")


_service: LegalQAService | None = None
_service_lock = threading.Lock()


def get_service() -> LegalQAService:
    global _service
    if _service is not None:
        return _service
    with _service_lock:
        if _service is None:
            _service = LegalQAService()
    return _service


def set_service(service: LegalQAService | None) -> None:
    """测试/多环境注入用。"""
    global _service
    with _service_lock:
        _service = service


def reload_service() -> bool:
    """服务已初始化时重载索引（文献入库后调用）；未初始化返回 False。"""
    global _service
    with _service_lock:
        running = _service
    if running is None:
        return False
    running.reload()
    return True
