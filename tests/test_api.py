"""FastAPI 接口测试：认证、会话落库、SSE；不启动真实模型/索引。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import legalqa.api as api
from legalqa.service import ChatEvent


class FakeService:
    def stream(self, query: str, history, turn_id: int = 0):
        assert query
        yield ChatEvent("status", meta={"stage": "thinking"})
        yield ChatEvent("delta", text="结论：可以主张。")
        yield ChatEvent(
            "citations",
            citations=[
                {
                    "law_name": "中华人民共和国劳动合同法",
                    "article_no": 82,
                    "article_label": "第八十二条",
                    "valid": True,
                    "version": "2012-12-28",
                    "text": "用人单位自用工之日起超过一个月不满一年未签订书面劳动合同的……",
                }
            ],
        )
        yield ChatEvent("meta", meta={"degraded": False, "channel": "dense"})
        yield ChatEvent("done")


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api, "get_service", lambda: FakeService())
    app = api.create_app(tmp_path / "test.db")
    return TestClient(app)


def _register(client: TestClient) -> dict:
    response = client.post(
        "/api/auth/register",
        json={"username": "tester", "password": "password123"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_health(client: TestClient) -> None:
    assert client.get("/api/health").json()["status"] == "ok"


def test_auth_register_login_me_logout(client: TestClient) -> None:
    assert client.get("/api/auth/me").status_code == 401
    user = _register(client)
    assert user["username"] == "tester"
    assert client.get("/api/auth/me").json()["username"] == "tester"

    assert client.post("/api/auth/logout").json()["ok"] is True
    assert client.get("/api/auth/me").status_code == 401

    login = client.post(
        "/api/auth/login",
        json={"username": "tester", "password": "password123"},
    )
    assert login.status_code == 200
    assert client.get("/api/auth/me").json()["username"] == "tester"

    bad = client.post(
        "/api/auth/login",
        json={"username": "tester", "password": "wrong-password"},
    )
    assert bad.status_code == 401


def test_sessions_and_chat_persist(client: TestClient) -> None:
    _register(client)
    created = client.post("/api/sessions", json={})
    assert created.status_code == 200, created.text
    session_id = created.json()["id"]

    chat = client.post(
        f"/api/sessions/{session_id}/chat",
        json={"query": "没签劳动合同能要双倍工资吗"},
    )
    assert chat.status_code == 200
    assert chat.headers["content-type"].startswith("text/event-stream")
    assert '"type": "done"' in chat.text
    assert "第八十二条" in chat.text

    detail = client.get(f"/api/sessions/{session_id}").json()
    assert detail["title"].startswith("没签劳动合同")
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][0]["content"] == "没签劳动合同能要双倍工资吗"
    assert "结论" in detail["messages"][1]["content"]
    assert detail["messages"][1]["citations"][0]["article_no"] == 82

    listing = client.get("/api/sessions").json()
    assert listing[0]["id"] == session_id
    assert listing[0]["message_count"] == 2

    assert client.delete(f"/api/sessions/{session_id}").json()["ok"] is True
    assert client.get(f"/api/sessions/{session_id}").status_code == 404


def test_chat_requires_login(client: TestClient) -> None:
    response = client.post(
        "/api/sessions/nope/chat", json={"query": "测试"}
    )
    assert response.status_code == 401
