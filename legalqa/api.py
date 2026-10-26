"""FastAPI 入口：/api/chat 用 SSE 返回流式事件，/ 可选托管前端构建产物。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from legalqa.service import get_service

__all__ = ["app", "ChatRequest", "HistoryItem"]

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "frontend" / "dist"


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    history: list[HistoryItem] = Field(default_factory=list, max_length=40)
    turn_id: int = 0


app = FastAPI(title="LegalMind API", version="0.1.0")

# 本地开发时前端跑在 5173；生产同源由 FastAPI 托管 dist
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "LegalMind"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        service = get_service()
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - build_llm 缺 key 会抛 SystemExit
        raise HTTPException(
            status_code=503,
            detail=f"服务初始化失败：{type(exc).__name__}: {exc}",
        ) from exc

    def events():
        for event in service.stream(
            req.query,
            [item.model_dump() for item in req.history],
            req.turn_id,
        ):
            yield event.to_sse()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# 构建后的前端（npm run build）由 FastAPI 直接托管；开发时用 Vite 代理
if DIST.exists():
    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="frontend")
