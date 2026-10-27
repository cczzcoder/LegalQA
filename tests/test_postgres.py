"""Postgres / Redis 集成测试（需显式配置测试实例，未配置时自动跳过）。

本地跑法（复用本机已有镜像起独立容器，端口 5433/6380，勿动原项目数据）：
    docker run -d --name legalqa-pg-test \
        -e POSTGRES_USER=legalqa -e POSTGRES_PASSWORD=legalqa -e POSTGRES_DB=legalqa \
        -p 127.0.0.1:5433:5432 pgvector/pgvector:pg17
    docker run -d --name legalqa-redis-test -p 127.0.0.1:6380:6379 redis:7-alpine
    LEGALQA_TEST_DATABASE_URL=postgresql://legalqa:legalqa@127.0.0.1:5433/legalqa \\
    LEGALQA_TEST_REDIS_URL=redis://127.0.0.1:6380/0 pytest tests/test_postgres.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest

from legalqa.store import Database, RedisRateLimiter

PG_URL = os.environ.get("LEGALQA_TEST_DATABASE_URL", "")
REDIS_URL = os.environ.get("LEGALQA_TEST_REDIS_URL", "")

pytestmark = pytest.mark.skipif(
    not PG_URL or not REDIS_URL, reason="未配置测试用 Postgres/Redis"
)


@pytest.fixture(scope="module")
def db() -> Database:
    return Database(url=PG_URL)


def test_postgres_users_sessions_messages(db: Database) -> None:
    username = f"t{uuid.uuid4().hex[:8]}"
    user = db.create_user(username, "password123")
    assert user["role"] == "user"

    token = db.create_token(user["id"])
    me = db.user_by_token(token)
    assert me is not None
    assert me["username"] == username
    assert me["role"] == "user"

    session = db.create_session(user["id"])
    db.add_message(session["id"], "user", "拖欠工资怎么办")
    db.add_message(session["id"], "assistant", "结论：可以主张。")
    messages = db.get_messages(session["id"])
    assert [item["role"] for item in messages] == ["user", "assistant"]

    db.add_feedback(
        user["id"],
        session["id"],
        messages[1]["id"],
        "引用错误",
        "",
        "拖欠工资怎么办",
        "结论：可以主张。",
        [],
    )
    assert db.list_feedback()["total"] >= 1

    db.add_audit("login", user["id"], username, "登录成功")
    assert any(item["action"] == "login" for item in db.list_audit(10))

    assert db.delete_session(user["id"], session["id"]) is True


def test_postgres_admin_promotion(db: Database) -> None:
    username = f"b{uuid.uuid4().hex[:8]}"
    db.create_user(username, "password123")
    assert db.set_admin(username) is True
    assert db.verify_user(username, "password123")["role"] == "admin"


def test_redis_rate_limiter_sliding_window() -> None:
    limiter = RedisRateLimiter(REDIS_URL)
    key = f"legalqa-test:{uuid.uuid4().hex}"
    try:
        for _ in range(3):
            assert limiter.retry_after(key, 3, 60) == 0
        assert limiter.retry_after(key, 3, 60) > 0
    finally:
        limiter._client.delete(key)
