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


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
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
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
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
                CREATE INDEX IF NOT EXISTS idx_sessions_user
                    ON chat_sessions(user_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_tokens_user ON auth_tokens(user_id);
                """
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
        return {"id": user_id, "username": username}

    def verify_user(self, username: str, password: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, username, salt, password_hash FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
        if row is None or not verify_password(
            password, row["salt"], row["password_hash"]
        ):
            return None
        return {"id": row["id"], "username": row["username"]}

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
                "SELECT u.id, u.username, t.expires_at FROM auth_tokens t "
                "JOIN users u ON u.id = t.user_id WHERE t.token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] <= time.time():
                conn.execute("DELETE FROM auth_tokens WHERE token_hash = ?", (token_hash,))
                return None
            return {"id": row["id"], "username": row["username"]}

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
