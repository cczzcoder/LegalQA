"""FastAPI 入口：登录、限流、会话落库、SSE 聊天，以及前端静态托管。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from legalqa.service import get_service
from legalqa.store import (
    COOKIE_NAME,
    TOKEN_TTL_SECONDS,
    Database,
    RateLimiter,
)

__all__ = ["app", "create_app", "AuthRequest", "ChatQuery", "SessionCreate"]

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "frontend" / "dist"

COOKIE_SECURE = os.environ.get("LEGALMIND_COOKIE_SECURE", "0") == "1"
CHAT_PER_MINUTE = int(os.environ.get("LEGALMIND_CHAT_PER_MINUTE", "10"))
CHAT_PER_HOUR = int(os.environ.get("LEGALMIND_CHAT_PER_HOUR", "60"))
LOGIN_PER_MINUTE = int(os.environ.get("LEGALMIND_LOGIN_PER_MINUTE", "8"))
REGISTER_PER_MINUTE = int(os.environ.get("LEGALMIND_REGISTER_PER_MINUTE", "5"))


class AuthRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=1, max_length=128)


class SessionCreate(BaseModel):
    title: str = Field(default="新对话", max_length=60)


class ChatQuery(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _rate_limit(request: Request, key: str, limit: int, window: float) -> None:
    retry = request.app.state.limiter.retry_after(key, limit, window)
    if retry:
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试",
            headers={"Retry-After": str(retry)},
        )


def _set_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=TOKEN_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=COOKIE_SECURE,
        path="/",
    )


def current_user(request: Request) -> dict[str, Any]:
    token = request.cookies.get(COOKIE_NAME)
    user = request.app.state.db.user_by_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return user


def create_app(db_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="LegalMind API", version="0.2.0")
    app.state.db = Database(db_path or ROOT / "data/app.db")
    app.state.limiter = RateLimiter()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "LegalMind"}

    # -- 登录 / 注册 ---------------------------------------------------
    @app.post("/api/auth/register")
    def register(
        payload: AuthRequest, request: Request, response: Response
    ) -> dict[str, Any]:
        _rate_limit(request, f"register:{_client_ip(request)}", REGISTER_PER_MINUTE, 60)
        try:
            user = request.app.state.db.create_user(payload.username, payload.password)
        except ValueError as exc:
            status = 409 if "已存在" in str(exc) else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        _set_cookie(response, request.app.state.db.create_token(user["id"]))
        return user

    @app.post("/api/auth/login")
    def login(
        payload: AuthRequest, request: Request, response: Response
    ) -> dict[str, Any]:
        _rate_limit(request, f"login:ip:{_client_ip(request)}", LOGIN_PER_MINUTE, 60)
        _rate_limit(
            request,
            f"login:user:{payload.username.strip().lower()}",
            max(LOGIN_PER_MINUTE // 2, 3),
            60,
        )
        user = request.app.state.db.verify_user(payload.username, payload.password)
        if user is None:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        _set_cookie(response, request.app.state.db.create_token(user["id"]))
        return user

    @app.post("/api/auth/logout")
    def logout(request: Request, response: Response) -> dict[str, bool]:
        request.app.state.db.delete_token(request.cookies.get(COOKIE_NAME))
        response.delete_cookie(COOKIE_NAME, path="/")
        return {"ok": True}

    @app.get("/api/auth/me")
    def me(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
        return user

    # -- 会话 -----------------------------------------------------------
    @app.get("/api/sessions")
    def list_sessions(
        request: Request,
        user: dict[str, Any] = Depends(current_user),
    ) -> list[dict[str, Any]]:
        return request.app.state.db.list_sessions(user["id"])

    @app.post("/api/sessions")
    def new_session(
        payload: SessionCreate,
        request: Request,
        user: dict[str, Any] = Depends(current_user),
    ) -> dict[str, Any]:
        return request.app.state.db.create_session(user["id"], payload.title)

    @app.get("/api/sessions/{session_id}")
    def session_detail(
        session_id: str,
        request: Request,
        user: dict[str, Any] = Depends(current_user),
    ) -> dict[str, Any]:
        session = request.app.state.db.get_session(user["id"], session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        session["messages"] = request.app.state.db.get_messages(session_id)
        return session

    @app.delete("/api/sessions/{session_id}")
    def remove_session(
        session_id: str,
        request: Request,
        user: dict[str, Any] = Depends(current_user),
    ) -> dict[str, bool]:
        if not request.app.state.db.delete_session(user["id"], session_id):
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"ok": True}

    # -- 聊天（SSE） -----------------------------------------------------
    @app.post("/api/sessions/{session_id}/chat")
    def chat(
        session_id: str,
        payload: ChatQuery,
        request: Request,
        user: dict[str, Any] = Depends(current_user),
    ):
        db = request.app.state.db
        session = db.get_session(user["id"], session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")

        _rate_limit(
            request, f"chat:{user['id']}:minute", CHAT_PER_MINUTE, 60
        )
        _rate_limit(request, f"chat:{user['id']}:hour", CHAT_PER_HOUR, 3600)

        try:
            service = get_service()
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail=f"服务初始化失败：{type(exc).__name__}: {exc}",
            ) from exc

        history = [
            {"role": item["role"], "content": item["content"]}
            for item in db.get_messages(session_id, limit=20)
        ]
        db.add_message(session_id, "user", payload.query)
        turn_id = max(db.count_user_messages(session_id) - 1, 0)
        if session["title"] == "新对话":
            db.touch_session(
                session_id, title=payload.query[:24].strip() or "新对话", user_id=user["id"]
            )

        def events():
            parts: list[str] = []
            citations: list[dict[str, Any]] = []
            meta: dict[str, Any] = {}
            try:
                for event in service.stream(payload.query, history, turn_id):
                    if event.type == "delta":
                        parts.append(event.text)
                    elif event.type == "reset":
                        parts.clear()
                    elif event.type == "replace":
                        parts = [event.text]
                    elif event.type == "citations":
                        citations = event.citations
                    elif event.type == "meta":
                        meta = event.meta
                    yield event.to_sse()
            finally:
                answer = "".join(parts).strip()
                if not answer:
                    answer = "本轮生成失败，请稍后重试。"
                    meta = {**meta, "degraded": True, "error": meta.get("error") or "empty_stream"}
                db.add_message(session_id, "assistant", answer, citations, meta)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    if DIST.exists():
        app.mount("/", StaticFiles(directory=str(DIST), html=True), name="frontend")

    return app


app = create_app()
