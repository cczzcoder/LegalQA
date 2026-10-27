"""SQLite 存储与限流测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from legalqa.store import Database, RateLimiter


def test_user_token_session_message_roundtrip(tmp_path: Path) -> None:
    db = Database(tmp_path / "app.db")
    user = db.create_user("测试用户", "password123")
    assert db.verify_user("测试用户", "password123")["id"] == user["id"]
    assert db.verify_user("测试用户", "wrong") is None

    token = db.create_token(user["id"])
    assert db.user_by_token(token)["username"] == "测试用户"
    assert db.user_by_token("bad-token") is None
    db.delete_token(token)
    assert db.user_by_token(token) is None

    session = db.create_session(user["id"])
    assert db.get_session(user["id"], session["id"])["id"] == session["id"]
    assert db.get_session("other-user", session["id"]) is None

    db.add_message(session["id"], "user", "没签合同能要双倍工资吗")
    db.add_message(
        session["id"],
        "assistant",
        "可以主张。",
        citations=[{"law_name": "劳动合同法", "article_no": 82}],
        meta={"degraded": False},
    )
    messages = db.get_messages(session["id"])
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["citations"][0]["article_no"] == 82
    assert messages[1]["meta"]["degraded"] is False
    assert db.count_user_messages(session["id"]) == 1

    db.touch_session(session["id"], title="未签合同")
    assert db.list_sessions(user["id"])[0]["title"] == "未签合同"
    assert db.list_sessions(user["id"])[0]["message_count"] == 2

    assert db.delete_session(user["id"], session["id"]) is True
    assert db.get_session(user["id"], session["id"]) is None


def test_user_validation(tmp_path: Path) -> None:
    db = Database(tmp_path / "app.db")
    with pytest.raises(ValueError, match="用户名"):
        db.create_user("a", "password123")
    with pytest.raises(ValueError, match="密码"):
        db.create_user("validuser", "short")
    db.create_user("validuser", "password123")
    with pytest.raises(ValueError, match="已存在"):
        db.create_user("validuser", "password123")


def test_rate_limiter_sliding_window() -> None:
    limiter = RateLimiter()
    assert limiter.retry_after("chat:u1", 2, 60) == 0
    assert limiter.retry_after("chat:u1", 2, 60) == 0
    assert limiter.retry_after("chat:u1", 2, 60) > 0
    assert limiter.retry_after("chat:u2", 2, 60) == 0


def test_set_password_revokes_tokens_and_verification(tmp_path: Path) -> None:
    db = Database(tmp_path / "pw.db")
    user = db.create_user("tester2", "password123", "t@b.com")
    token = db.create_token(user["id"])
    assert db.user_by_token(token) is not None

    db.create_verification(user["id"], "password_reset", "123456", 600)
    assert db.consume_verification(user["id"], "password_reset", "000000") is False
    assert db.consume_verification(user["id"], "password_reset", "123456") is True
    assert db.consume_verification(user["id"], "password_reset", "123456") is False

    db.set_password(user["id"], "newpassword9")
    assert db.user_by_token(token) is None
    assert db.verify_user("tester2", "password123") is None
    assert db.verify_user("tester2", "newpassword9") is not None
