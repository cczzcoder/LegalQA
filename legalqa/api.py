"""FastAPI 入口：登录、限流、会话落库、SSE 聊天、法条库/评测、管理端与前端静态托管。"""

from __future__ import annotations

import json
import os
import secrets
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from legalqa.importer import MAX_UPLOAD_BYTES, SUPPORTED_SUFFIXES, Importer
from legalqa.env import load_dotenv
from legalqa.library import EvalStore, LawCatalog
from legalqa.service import get_service, reload_service
from legalqa.store import (
    COOKIE_NAME,
    TOKEN_TTL_SECONDS,
    Database,
    RateLimiter,
)

__all__ = ["app", "create_app", "AuthRequest", "ChatQuery", "SessionCreate"]

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "frontend" / "dist"

COOKIE_SECURE = os.environ.get("LEGALQA_COOKIE_SECURE", "0") == "1"
CSRF_COOKIE_NAME = "lm_csrf"
CSRF_HEADER_NAME = "X-CSRF-Token"
CHAT_PER_MINUTE = int(os.environ.get("LEGALQA_CHAT_PER_MINUTE", "10"))
CHAT_PER_HOUR = int(os.environ.get("LEGALQA_CHAT_PER_HOUR", "60"))
LOGIN_PER_MINUTE = int(os.environ.get("LEGALQA_LOGIN_PER_MINUTE", "8"))
REGISTER_PER_MINUTE = int(os.environ.get("LEGALQA_REGISTER_PER_MINUTE", "5"))


class AuthRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=1, max_length=128)


class SessionCreate(BaseModel):
    title: str = Field(default="新对话", max_length=60)


class ChatQuery(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class FeedbackIn(BaseModel):
    session_id: str
    message_id: str
    category: str = Field(max_length=20)
    comment: str = Field(default="", max_length=1000)


class FeedbackStatusIn(BaseModel):
    status: str = Field(max_length=20)


class ImportApplyIn(BaseModel):
    replace: bool = True


FEEDBACK_CATEGORIES = {"引用错误", "答案错误", "条号不存在", "其他"}
FEEDBACK_STATUSES = {"open", "done", "ignored"}


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


def _set_csrf_cookie(response: Response, token: str) -> None:
    # 不 httpOnly：双提交模式下前端要读它并放回 X-CSRF-Token 头。
    response.set_cookie(
        CSRF_COOKIE_NAME,
        token,
        max_age=TOKEN_TTL_SECONDS,
        httponly=False,
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


def current_admin(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可访问")
    return user


def create_app(
    db_path: str | Path | None = None,
    docs_path: str | Path | None = None,
    eval_dir: str | Path | None = None,
    importer: Importer | None = None,
) -> FastAPI:
    load_dotenv()
    app = FastAPI(title="LegalQA API", version="0.2.0")
    app.state.db = Database(db_path or ROOT / "data/app.db")
    app.state.limiter = RateLimiter()
    app.state.catalog = LawCatalog(docs_path)
    app.state.evals = EvalStore(eval_dir)
    app.state.importer = importer or Importer()

    @app.middleware("http")
    async def csrf_guard(request: Request, call_next):
        # Cookie 双提交：不安全方法必须带 X-CSRF-Token 且与 lm_csrf cookie 一致。
        if request.url.path.startswith("/api/") and request.method in {
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
        }:
            cookie_token = request.cookies.get(CSRF_COOKIE_NAME, "")
            header_token = request.headers.get(CSRF_HEADER_NAME, "")
            if (
                not cookie_token
                or not header_token
                or not secrets.compare_digest(cookie_token, header_token)
            ):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF 校验失败，请刷新页面后重试"},
                )
        return await call_next(request)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
        )
        # Element Plus 运行时注入 <style>，style-src 必须放开 unsafe-inline。
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; font-src 'self' data:; connect-src 'self'",
        )
        return response

    admin_username = os.environ.get("LEGALQA_ADMIN_USERNAME", "").strip()
    if admin_username:
        promoted = app.state.db.set_admin(admin_username)
        if promoted:
            app.state.db.add_audit(
                "admin_promote", username=admin_username, detail="环境变量提升为管理员"
            )

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
        return {"status": "ok", "service": "LegalQA"}

    # -- 法条库浏览（公开只读） -----------------------------------------
    @app.get("/api/laws")
    def list_laws(request: Request) -> dict[str, Any]:
        catalog: LawCatalog = request.app.state.catalog
        try:
            return {"stats": catalog.stats(), "laws": catalog.laws()}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/laws/articles")
    def list_articles(
        request: Request,
        law: str | None = None,
        q: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        catalog: LawCatalog = request.app.state.catalog
        try:
            return catalog.articles(law=law, q=q, offset=offset, limit=limit)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    # -- 评测看板（公开只读，读 data/eval 历史结果） ---------------------
    @app.get("/api/eval/runs")
    def list_eval_runs(request: Request) -> dict[str, Any]:
        return {"runs": request.app.state.evals.list_runs()}

    @app.get("/api/eval/runs/{run_id}")
    def eval_run_detail(run_id: str, request: Request) -> dict[str, Any]:
        detail = request.app.state.evals.detail(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="评测结果不存在")
        return detail

    @app.get("/api/eval/runs/{run_id}/cases/{case_id}")
    def eval_case_detail(run_id: str, case_id: str, request: Request) -> dict[str, Any]:
        case = request.app.state.evals.case(run_id, case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="评测用例不存在")
        return case

    @app.get("/api/eval/compare")
    def eval_compare(
        left: str,
        right: str,
        request: Request,
    ) -> dict[str, Any]:
        store = request.app.state.evals
        a = store.detail(left)
        b = store.detail(right)
        if a is None or b is None:
            raise HTTPException(status_code=404, detail="评测结果不存在")

        def _summary(item: dict[str, Any]) -> dict[str, Any]:
            return {
                "id": item["id"],
                "label": item["label"],
                "case_count": item["case_count"],
                "passed": item["passed"],
                "pass_rate": item["pass_rate"],
                "avg_latency_ms": item["avg_latency_ms"],
                "failures": [f["id"] for f in item["failures"]],
            }

        checks_a = {c["name"]: c for c in a["checks"]}
        checks_b = {c["name"]: c for c in b["checks"]}
        names = list(dict.fromkeys([*checks_a.keys(), *checks_b.keys()]))
        rows = []
        for name in names:
            ca, cb = checks_a.get(name), checks_b.get(name)
            rows.append(
                {
                    "name": name,
                    "left": (
                        {"failed": ca["failed"], "total": ca["total"], "fail_rate": ca["fail_rate"]}
                        if ca
                        else None
                    ),
                    "right": (
                        {"failed": cb["failed"], "total": cb["total"], "fail_rate": cb["fail_rate"]}
                        if cb
                        else None
                    ),
                }
            )
        failures_a = {f["id"] for f in a["failures"]}
        failures_b = {f["id"] for f in b["failures"]}
        return {
            "left": _summary(a),
            "right": _summary(b),
            "checks": rows,
            "common_failures": sorted(failures_a & failures_b),
        }

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
        csrf_token = secrets.token_urlsafe(32)
        _set_csrf_cookie(response, csrf_token)
        request.app.state.db.add_audit(
            "register", user["id"], user["username"], "注册成功"
        )
        return {**user, "csrf_token": csrf_token}

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
            request.app.state.db.add_audit(
                "login_failed", username=payload.username.strip(), detail="用户名或密码错误"
            )
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        _set_cookie(response, request.app.state.db.create_token(user["id"]))
        csrf_token = secrets.token_urlsafe(32)
        _set_csrf_cookie(response, csrf_token)
        request.app.state.db.add_audit(
            "login", user["id"], user["username"], "登录成功"
        )
        return {**user, "csrf_token": csrf_token}

    @app.post("/api/auth/logout")
    def logout(request: Request, response: Response) -> dict[str, bool]:
        db = request.app.state.db
        user = db.user_by_token(request.cookies.get(COOKIE_NAME))
        if user:
            db.add_audit("logout", user["id"], user["username"], "退出登录")
        db.delete_token(request.cookies.get(COOKIE_NAME))
        response.delete_cookie(COOKIE_NAME, path="/")
        return {"ok": True}

    @app.get("/api/auth/me")
    def me(request: Request):
        token = request.cookies.get(COOKIE_NAME)
        user = request.app.state.db.user_by_token(token)
        # 无论登录与否都确保 CSRF cookie 存在：登录表单提交、已登录用户的
        # 首次访问（旧会话 cookie 没有配对 token）都依赖它。
        response = JSONResponse(
            status_code=200 if user else 401,
            content=user or {"detail": "请先登录"},
        )
        if not request.cookies.get(CSRF_COOKIE_NAME):
            _set_csrf_cookie(response, secrets.token_urlsafe(32))
        return response

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

    # -- 用户反馈 -------------------------------------------------------
    @app.post("/api/feedback")
    def submit_feedback(
        payload: FeedbackIn,
        request: Request,
        user: dict[str, Any] = Depends(current_user),
    ) -> dict[str, Any]:
        if payload.category not in FEEDBACK_CATEGORIES:
            raise HTTPException(status_code=400, detail="反馈分类不合法")
        _rate_limit(request, f"feedback:{user['id']}:hour", 20, 3600)
        db = request.app.state.db
        if db.get_session(user["id"], payload.session_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        context = db.get_message_context(payload.session_id, payload.message_id)
        if context is None:
            raise HTTPException(status_code=404, detail="消息不存在")
        return db.add_feedback(
            user["id"],
            payload.session_id,
            payload.message_id,
            payload.category,
            payload.comment.strip(),
            context["question"],
            context["answer"],
            context["citations"],
        )

    # -- 管理端 ----------------------------------------------------------
    @app.get("/api/admin/overview")
    def admin_overview(
        request: Request,
        user: dict[str, Any] = Depends(current_admin),
    ) -> dict[str, Any]:
        db = request.app.state.db
        catalog: LawCatalog = request.app.state.catalog
        importer: Importer = request.app.state.importer

        warnings: list[dict[str, Any]] = []
        corpus: dict[str, Any] = {"total_laws": 0, "total_articles": 0}
        try:
            stats = catalog.stats()
            corpus["total_laws"] = stats["total_laws"]
            corpus["total_articles"] = stats["total_articles"]
            today_year = date.today().year
            for law in catalog.laws():
                version = law.get("latest_version", "")
                if not version:
                    warnings.append(
                        {"name": law["name"], "version": "", "years": None, "reason": "无版本日期"}
                    )
                    continue
                years = max(today_year - int(version[:4]), 0)
                if years >= 5:
                    warnings.append(
                        {
                            "name": law["name"],
                            "version": version,
                            "years": years,
                            "reason": f"版本距今 {years} 年",
                        }
                    )
        except FileNotFoundError as exc:
            corpus["error"] = str(exc)

        warnings.sort(key=lambda item: (item["years"] is None, -(item["years"] or 0)))
        return {
            "feedback_open": db.count_feedback_open(),
            "imports_pending": db.count_imports_pending(),
            "index": importer.index_lag(),
            "corpus": corpus,
            "version_warnings": warnings,
            "recent_audit": db.list_audit(10),
        }

    @app.get("/api/admin/audit")
    def list_audit_admin(
        request: Request,
        limit: int = 100,
        user: dict[str, Any] = Depends(current_admin),
    ) -> dict[str, Any]:
        return {"items": request.app.state.db.list_audit(limit)}

    @app.get("/api/admin/feedback")
    def list_feedback_admin(
        request: Request,
        status: str | None = None,
        offset: int = 0,
        limit: int = 50,
        user: dict[str, Any] = Depends(current_admin),
    ) -> dict[str, Any]:
        return request.app.state.db.list_feedback(
            status or "", min(max(limit, 1), 100), max(offset, 0)
        )

    @app.patch("/api/admin/feedback/{feedback_id}")
    def patch_feedback(
        feedback_id: str,
        payload: FeedbackStatusIn,
        request: Request,
        user: dict[str, Any] = Depends(current_admin),
    ) -> dict[str, bool]:
        if payload.status not in FEEDBACK_STATUSES:
            raise HTTPException(status_code=400, detail="状态不合法")
        if not request.app.state.db.update_feedback_status(feedback_id, payload.status):
            raise HTTPException(status_code=404, detail="反馈不存在")
        request.app.state.db.add_audit(
            "feedback_status",
            user["id"],
            user["username"],
            f"反馈 {feedback_id} → {payload.status}",
        )
        return {"ok": True}

    @app.get("/api/admin/imports")
    def list_imports_admin(
        request: Request,
        user: dict[str, Any] = Depends(current_admin),
    ) -> dict[str, Any]:
        return {"items": request.app.state.db.list_imports()}

    @app.post("/api/admin/imports")
    async def upload_import(
        request: Request,
        file: UploadFile = File(...),
        user: dict[str, Any] = Depends(current_admin),
    ) -> dict[str, Any]:
        _rate_limit(request, f"import:{user['id']}:minute", 5, 60)
        filename = Path(file.filename or "").name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"只支持 {' / '.join(sorted(SUPPORTED_SUFFIXES))} 格式",
            )
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="文件超过 20MB 上限")
        if not content:
            raise HTTPException(status_code=400, detail="文件为空")
        importer: Importer = request.app.state.importer
        import_id, stored = importer.save_upload(filename, content)
        record = request.app.state.db.create_import(filename, str(stored), len(content))
        request.app.state.db.add_audit(
            "import_upload",
            user["id"],
            user["username"],
            f"{filename}（{len(content)} 字节）→ {import_id}",
        )
        return record

    @app.delete("/api/admin/imports/{import_id}")
    def delete_import(
        import_id: str,
        request: Request,
        user: dict[str, Any] = Depends(current_admin),
    ) -> dict[str, bool]:
        db = request.app.state.db
        record = db.get_import(import_id)
        if record is None:
            raise HTTPException(status_code=404, detail="导入记录不存在")
        if record["status"] == "applied":
            raise HTTPException(status_code=400, detail="已入库的导入不能删除")
        request.app.state.importer.remove_upload(record["stored_path"])
        db.delete_import(import_id)
        db.add_audit(
            "import_delete",
            user["id"],
            user["username"],
            f"{record['filename']}（{record['status']}）",
        )
        return {"ok": True}

    @app.post("/api/admin/imports/{import_id}/analyze")
    def analyze_import(
        import_id: str,
        request: Request,
        user: dict[str, Any] = Depends(current_admin),
    ) -> dict[str, Any]:
        _rate_limit(request, f"analyze:{user['id']}:minute", 10, 60)
        db = request.app.state.db
        record = db.get_import(import_id)
        if record is None:
            raise HTTPException(status_code=404, detail="导入记录不存在")
        if record["status"] == "applied":
            raise HTTPException(status_code=400, detail="该文件已入库")
        report = request.app.state.importer.analyze(record["stored_path"])
        db.update_import(
            import_id,
            status="analyzed" if report.get("ok") else "failed",
            law_name=report.get("law_name", ""),
            version_date=report.get("version_date", ""),
            article_count=report.get("article_count", 0),
            report_json=json.dumps(report, ensure_ascii=False),
            error=report.get("error", ""),
        )
        db.add_audit(
            "import_analyze",
            user["id"],
            user["username"],
            f"{record['filename']} → {report.get('suggestion', report.get('error', ''))}",
        )
        return report

    @app.post("/api/admin/imports/{import_id}/apply")
    def apply_import(
        import_id: str,
        payload: ImportApplyIn,
        request: Request,
        user: dict[str, Any] = Depends(current_admin),
    ) -> dict[str, Any]:
        _rate_limit(request, f"apply:{user['id']}:hour", 5, 3600)
        db = request.app.state.db
        record = db.get_import(import_id)
        if record is None:
            raise HTTPException(status_code=404, detail="导入记录不存在")
        if record["status"] != "analyzed":
            raise HTTPException(status_code=400, detail="请先完成分析再入库")
        if not record.get("report", {}).get("ok"):
            raise HTTPException(status_code=400, detail="分析未通过，无法入库")
        result = request.app.state.importer.apply(record, replace=payload.replace)
        if result.get("ok"):
            report = dict(record.get("report") or {})
            report["apply_result"] = result
            db.update_import(
                import_id,
                status="applied",
                applied_at=datetime.now().isoformat(timespec="seconds"),
                report_json=json.dumps(report, ensure_ascii=False),
            )
            request.app.state.catalog.reload()
            reload_service()
            db.add_audit(
                "import_apply",
                user["id"],
                user["username"],
                f"{record['filename']}：+{result.get('added', 0)} 条"
                f"（替换 {result.get('removed', 0)}）· {result.get('index_status', '')}",
            )
        return result

    if DIST.exists():
        app.mount("/", StaticFiles(directory=str(DIST), html=True), name="frontend")

    return app


app = create_app()
