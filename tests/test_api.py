"""FastAPI 接口测试：认证、会话落库、SSE；不启动真实模型/索引。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import legalqa.api as api
from legalqa.service import ChatEvent

DOCS = [
    {
        "id": "labor-1",
        "law_name": "中华人民共和国劳动合同法",
        "doc_type": "法律",
        "article_no": 10,
        "article_label": "第十条",
        "chapter": "第二章",
        "section": "",
        "text": "建立劳动关系，应当订立书面劳动合同。",
        "is_valid": True,
        "version_date": "2012-12-28",
    },
    {
        "id": "civil-1",
        "law_name": "中华人民共和国民法典",
        "doc_type": "法律",
        "article_no": 1165,
        "article_label": "第一千一百六十五条",
        "chapter": "第七章",
        "section": "",
        "text": "行为人因过错侵害他人民事权益造成损害的，应当承担侵权责任。",
        "is_valid": True,
        "version_date": "2020-05-28",
    },
]

EVAL_CASES = [
    {
        "id": "c1",
        "query": "试用期多久",
        "passed": True,
        "latency_ms": 1000,
        "iterations": 2,
        "channel": "dense",
        "score": 0.8,
        "answer": "回答一",
        "retrieved": [],
        "checks": [{"name": "引用可溯源", "passed": True, "detail": ""}],
        "retrieval_coverage": {"passed": True, "detail": ""},
    },
    {
        "id": "c2",
        "query": "违法解除怎么赔",
        "passed": False,
        "latency_ms": 3000,
        "iterations": 3,
        "channel": "exact",
        "score": 0.7,
        "answer": "回答二",
        "retrieved": [],
        "checks": [{"name": "引用可溯源", "passed": False, "detail": "未检索到"}],
        "retrieval_coverage": {"passed": False, "detail": "top-6 未覆盖"},
    },
]


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
    docs = tmp_path / "docs.json"
    docs.write_text(json.dumps(DOCS, ensure_ascii=False), encoding="utf-8")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "qwen37_fact_100.json").write_text(
        json.dumps(EVAL_CASES, ensure_ascii=False), encoding="utf-8"
    )
    app = api.create_app(tmp_path / "test.db", docs_path=docs, eval_dir=eval_dir)
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


def test_law_library(client: TestClient) -> None:
    catalog = client.get("/api/laws").json()
    assert catalog["stats"]["total_laws"] == 2
    assert catalog["stats"]["total_articles"] == 2

    result = client.get(
        "/api/laws/articles", params={"law": "劳动合同法", "q": "书面劳动合同"}
    ).json()
    assert result["total"] == 1
    assert result["items"][0]["article_label"] == "第十条"

    empty = client.get(
        "/api/laws/articles", params={"law": "中华人民共和国刑法"}
    ).json()
    assert empty["total"] == 0


def test_eval_dashboard(client: TestClient) -> None:
    runs = client.get("/api/eval/runs").json()["runs"]
    assert runs[0]["id"] == "qwen37_fact_100"
    assert runs[0]["pass_rate"] == 0.5
    assert runs[0]["pinned"] is True

    detail = client.get("/api/eval/runs/qwen37_fact_100").json()
    assert detail["passed"] == 1
    assert [item["id"] for item in detail["failures"]] == ["c2"]

    case = client.get("/api/eval/runs/qwen37_fact_100/cases/c2").json()
    assert case["answer"] == "回答二"

    assert client.get("/api/eval/runs/missing").status_code == 404
    assert client.get("/api/eval/runs/qwen37_fact_100/cases/missing").status_code == 404
