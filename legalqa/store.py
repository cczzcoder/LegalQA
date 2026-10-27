"""SQLite 持久化：用户、登录令牌、会话、消息；以及进程内滑动窗口限流。

单进程 MVP 用 SQLite 足够；多 worker/多机部署时把限流换成 Redis、
把数据库换成 Postgres 即可，API 层不需要改。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "COOKIE_NAME",
    "TOKEN_TTL_SECONDS",
    "Database",
    "RateLimiter",
    "hash_token",
    "verify_password",
]

COOKIE_NAME = "legalmind_token"
TOKEN_TTL_SECONDS = 7 * 24 * 3600
USERNAME_RE = re.compile(r"^[A-Za-z0-9_\u4e00-\u9fa5]{3,32}$")
PBKDF2_ITERATIONS = 200_000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    ).hex()


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    return hmac.compare_digest(_hash_password(password, salt), hash_hex)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class RateLimiter:
    """进程内滑动窗口限流。key 例如 chat:{user_id}:minute / login:{ip}。"""

    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def retry_after(self, key: str, limit: int, window: float) -> int:
        now = time.monotonic()
        cutoff = now - window
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if t > cutoff]
            if len(hits) >= max(limit, 1):
                self._hits[key] = hits
                return max(1, int(window - (now - hits[0])) + 1)
            hits.append(now)
            self._hits[key] = hits
            if len(self._hits) > 10_000:
                self._hits = {
                    k: [t for t in v if t > cutoff]
                    for k, v in self._hits.items()
                    if any(t > cutoff for t in v)
                }
            return 0


class RedisRateLimiter:
    """Redis 滑动窗口限流（ZSET），多 worker 共享计数。

    语义与进程内 RateLimiter 一致：窗口内命中数达上限即拒绝并给出
    retry-after（取最老命中的剩余时间）。
    """

    def __init__(self, url: str) -> None:
        import redis

        self._client = redis.Redis.from_url(url, decode_responses=True)

    def retry_after(self, key: str, limit: int, window: float) -> int:
        now = time.time()
        self._client.zremrangebyscore(key, 0, now - window)
        count = self._client.zcard(key)
        if count >= max(limit, 1):
            oldest = self._client.zrange(key, 0, 0, withscores=True)
            if oldest:
                return max(1, int(window - (now - float(oldest[0][1]))) + 1)
            return int(window)
        member = f"{now:.6f}:{uuid.uuid4().hex[:8]}"
        self._client.zadd(key, {member: now})
        self._client.expire(key, int(window) + 1)
        return 0


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    salt TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS auth_tokens (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    expires_at REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations_json TEXT NOT NULL DEFAULT '[]',
    meta_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    category TEXT NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL DEFAULT '',
    answer TEXT NOT NULL DEFAULT '',
    citations_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS imports (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'uploaded',
    law_name TEXT NOT NULL DEFAULT '',
    version_date TEXT NOT NULL DEFAULT '',
    article_count INTEGER NOT NULL DEFAULT 0,
    report_json TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT '',
    username TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user
    ON chat_sessions(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_tokens_user ON auth_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_status
    ON feedback(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_imports_status
    ON imports(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_created
    ON audit_log(created_at DESC);
"""

# Postgres 方言差异：自增主键与 float8。
_SCHEMA_PG = (
    _SCHEMA.replace(
        "id INTEGER PRIMARY KEY AUTOINCREMENT,",
        "id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,",
    ).replace("expires_at REAL NOT NULL", "expires_at DOUBLE PRECISION NOT NULL")
)


class _PGAdapter:
    """psycopg 连接包装：把 sqlite 风格 SQL（? 占位）透明翻译成 %s。

    本模块所有方法共用同一套 SQL 文本，Postgres 只需换占位符与建表方言。
    """

    def __init__(self, conn) -> None:
        self._conn = conn

    def execute(self, sql: str, params=()):
        cursor = self._conn.execute(sql.replace("?", "%s"), params)
        self.rowcount = cursor.rowcount
        return cursor


class Database:
    """SQLite（默认，传 path）或 Postgres（传 postgresql:// 的 url）存储层。"""

    def __init__(
        self, path: str | Path | None = None, url: str | None = None
    ) -> None:
        if url:
            import psycopg

            self.pg = True
            self._pg_conn = psycopg.connect(
                url, row_factory=psycopg.rows.dict_row, autocommit=True
            )
            self.path: Path | None = None
        else:
            if path is None:
                raise ValueError("SQLite 模式必须提供 path")
            self.pg = False
            self._pg_conn = None
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def _conn(self):
        if self.pg:
            yield _PGAdapter(self._pg_conn)
            return
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        if self.pg:
            with self._conn() as conn:
                for statement in (s.strip() for s in _SCHEMA_PG.split(";") if s.strip()):
                    conn.execute(statement)
        else:
            with self._conn() as conn:
                conn.executescript(_SCHEMA)
        # 旧库迁移：role 列是后加的，CREATE TABLE IF NOT EXISTS 不会补列。
        with self._conn() as conn:
            if self.pg:
                rows = conn.execute(
                    "SELECT column_name AS name FROM information_schema.columns "
                    "WHERE table_name = 'users'"
                ).fetchall()
            else:
                rows = conn.execute("PRAGMA table_info(users)").fetchall()
            columns = {row["name"] for row in rows}
            if "role" not in columns:
                conn.execute(
                    "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'"
                )

    # -- 用户 / 登录 ---------------------------------------------------
    def create_user(self, username: str, password: str) -> dict[str, Any]:
        username = username.strip()
        if not USERNAME_RE.fullmatch(username):
            raise ValueError("用户名需为 3-32 位中文、字母、数字或下划线")
        if len(password) < 8:
            raise ValueError("密码至少 8 位")
        salt = secrets.token_bytes(16)
        user_id = uuid.uuid4().hex
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO users(id, username, salt, password_hash, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (user_id, username, salt.hex(), _hash_password(password, salt), _now_iso()),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("用户名已存在") from exc
        return {"id": user_id, "username": username, "role": "user"}

    def verify_user(self, username: str, password: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, username, role, salt, password_hash FROM users "
                "WHERE username = ?",
                (username.strip(),),
            ).fetchone()
        if row is None or not verify_password(
            password, row["salt"], row["password_hash"]
        ):
            return None
        return {"id": row["id"], "username": row["username"], "role": row["role"]}

    def set_admin(self, username: str) -> bool:
        """把指定用户提升为管理员；返回是否真正发生了提升（已是 admin 返回 False）。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT role FROM users WHERE username = ?", (username.strip(),)
            ).fetchone()
            if row is None or row["role"] == "admin":
                return False
            conn.execute(
                "UPDATE users SET role = 'admin' WHERE username = ?",
                (username.strip(),),
            )
            return True

    def create_token(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO auth_tokens(token_hash, user_id, expires_at, created_at) "
                "VALUES (?, ?, ?, ?)",
                (hash_token(token), user_id, time.time() + TOKEN_TTL_SECONDS, _now_iso()),
            )
        return token

    def user_by_token(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None
        token_hash = hash_token(token)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT u.id, u.username, u.role, t.expires_at FROM auth_tokens t "
                "JOIN users u ON u.id = t.user_id WHERE t.token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] <= time.time():
                conn.execute("DELETE FROM auth_tokens WHERE token_hash = ?", (token_hash,))
                return None
            return {
                "id": row["id"],
                "username": row["username"],
                "role": row["role"],
            }

    def delete_token(self, token: str | None) -> None:
        if not token:
            return
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM auth_tokens WHERE token_hash = ?", (hash_token(token),)
            )

    # -- 会话 -----------------------------------------------------------
    def create_session(self, user_id: str, title: str = "新对话") -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        now = _now_iso()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO chat_sessions(id, user_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, title, now, now),
            )
        return {"id": session_id, "title": title, "updated_at": now}

    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT s.id, s.title, s.updated_at, "
                "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count "
                "FROM chat_sessions s WHERE s.user_id = ? "
                "ORDER BY s.updated_at DESC LIMIT 50",
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_session(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM chat_sessions "
                "WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def delete_session(self, user_id: str, session_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            )
            return cur.rowcount > 0

    def touch_session(
        self, session_id: str, title: str | None = None, *, user_id: str | None = None
    ) -> None:
        now = _now_iso()
        with self._conn() as conn:
            if title:
                if user_id:
                    conn.execute(
                        "UPDATE chat_sessions SET title = ?, updated_at = ? "
                        "WHERE id = ? AND user_id = ?",
                        (title, now, session_id, user_id),
                    )
                else:
                    conn.execute(
                        "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                        (title, now, session_id),
                    )
            else:
                conn.execute(
                    "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                    (now, session_id),
                )

    # -- 消息 -----------------------------------------------------------
    def get_messages(
        self, session_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        with self._conn() as conn:
            if limit:
                rows = conn.execute(
                    "SELECT * FROM (SELECT * FROM messages WHERE session_id = ? "
                    "ORDER BY created_at DESC LIMIT ?) ORDER BY created_at ASC",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                    (session_id,),
                ).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["citations"] = json.loads(item.pop("citations_json") or "[]")
            except json.JSONDecodeError:
                item["citations"] = []
            try:
                item["meta"] = json.loads(item.pop("meta_json") or "{}")
            except json.JSONDecodeError:
                item["meta"] = {}
            messages.append(item)
        return messages

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        message_id = uuid.uuid4().hex
        now = _now_iso()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages(id, session_id, role, content, citations_json, "
                "meta_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    session_id,
                    role,
                    content,
                    json.dumps(citations or [], ensure_ascii=False),
                    json.dumps(meta or {}, ensure_ascii=False),
                    now,
                ),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
        return {"id": message_id, "created_at": now}

    def count_user_messages(self, session_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE session_id = ? AND role = 'user'",
                (session_id,),
            ).fetchone()
        return int(row["n"]) if row else 0

    # -- 反馈 -----------------------------------------------------------
    def get_message_context(
        self, session_id: str, message_id: str
    ) -> dict[str, Any] | None:
        """反馈用的消息快照：回答本身 + 它之前最近一条用户提问。"""
        with self._conn() as conn:
            message = conn.execute(
                "SELECT id, role, content, citations_json, created_at FROM messages "
                "WHERE id = ? AND session_id = ? AND role = 'assistant'",
                (message_id, session_id),
            ).fetchone()
            if message is None:
                return None
            question = conn.execute(
                "SELECT content FROM messages WHERE session_id = ? AND role = 'user' "
                "AND created_at <= ? ORDER BY created_at DESC LIMIT 1",
                (session_id, message["created_at"]),
            ).fetchone()
        try:
            citations = json.loads(message["citations_json"] or "[]")
        except json.JSONDecodeError:
            citations = []
        return {
            "message_id": message_id,
            "question": question["content"] if question else "",
            "answer": message["content"],
            "citations": citations,
        }

    def add_feedback(
        self,
        user_id: str,
        session_id: str,
        message_id: str,
        category: str,
        comment: str,
        question: str,
        answer: str,
        citations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        feedback_id = uuid.uuid4().hex
        now = _now_iso()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO feedback(id, user_id, session_id, message_id, category, "
                "comment, question, answer, citations_json, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)",
                (
                    feedback_id,
                    user_id,
                    session_id,
                    message_id,
                    category,
                    comment,
                    question,
                    answer,
                    json.dumps(citations or [], ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return {"id": feedback_id, "status": "open", "created_at": now}

    def list_feedback(
        self, status: str = "", limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        where = "WHERE status = ?" if status else ""
        params: list[Any] = [status] if status else []
        with self._conn() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) AS n FROM feedback {where}", params
            ).fetchone()["n"]
            rows = conn.execute(
                f"SELECT * FROM feedback {where} "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            try:
                item["citations"] = json.loads(item.pop("citations_json") or "[]")
            except json.JSONDecodeError:
                item["citations"] = []
            items.append(item)
        return {"total": int(total), "items": items}

    def update_feedback_status(self, feedback_id: str, status: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE feedback SET status = ?, updated_at = ? WHERE id = ?",
                (status, _now_iso(), feedback_id),
            )
            return cur.rowcount > 0

    def count_feedback_open(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM feedback WHERE status = 'open'"
            ).fetchone()
        return int(row["n"]) if row else 0

    # -- 文献导入 --------------------------------------------------------
    def create_import(
        self, filename: str, stored_path: str, size_bytes: int
    ) -> dict[str, Any]:
        import_id = uuid.uuid4().hex
        now = _now_iso()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO imports(id, filename, stored_path, size_bytes, status, "
                "created_at) VALUES (?, ?, ?, ?, 'uploaded', ?)",
                (import_id, filename, stored_path, size_bytes, now),
            )
        return {
            "id": import_id,
            "filename": filename,
            "size_bytes": size_bytes,
            "status": "uploaded",
            "created_at": now,
        }

    def get_import(self, import_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM imports WHERE id = ?", (import_id,)
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        try:
            item["report"] = json.loads(item.pop("report_json") or "{}")
        except json.JSONDecodeError:
            item["report"] = {}
        return item

    def list_imports(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM imports ORDER BY created_at DESC LIMIT 200"
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            try:
                item["report"] = json.loads(item.pop("report_json") or "{}")
            except json.JSONDecodeError:
                item["report"] = {}
            items.append(item)
        return items

    def update_import(self, import_id: str, **fields: Any) -> dict[str, Any] | None:
        allowed = {
            "status",
            "law_name",
            "version_date",
            "article_count",
            "report_json",
            "error",
            "applied_at",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return self.get_import(import_id)
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._conn() as conn:
            cur = conn.execute(
                f"UPDATE imports SET {assignments} WHERE id = ?",
                [*updates.values(), import_id],
            )
            if cur.rowcount == 0:
                return None
        return self.get_import(import_id)

    def count_imports_pending(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM imports WHERE status IN ('uploaded', 'analyzed')"
            ).fetchone()
        return int(row["n"]) if row else 0

    def delete_import(self, import_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM imports WHERE id = ?", (import_id,))
            return cur.rowcount > 0

    # -- 审计日志 --------------------------------------------------------
    def add_audit(
        self,
        action: str,
        user_id: str = "",
        username: str = "",
        detail: str = "",
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO audit_log(user_id, username, action, detail, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, username, action, detail[:500], _now_iso()),
            )

    def list_audit(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
                (min(max(limit, 1), 500),),
            ).fetchall()
        return [dict(row) for row in rows]
