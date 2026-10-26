"""FastAPI 接口测试：不启动真实模型/索引。"""

from __future__ import annotations

from fastapi.testclient import TestClient

import legalqa.api as api
from legalqa.service import ChatEvent


class FakeService:
    def stream(self, query: str, history, turn_id: int = 0):
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


def test_health() -> None:
    client = TestClient(api.app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_rejects_empty_query() -> None:
    client = TestClient(api.app)
    response = client.post("/api/chat", json={"query": "", "history": []})
    assert response.status_code == 422


def test_chat_streams_sse(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_service", lambda: FakeService())
    client = TestClient(api.app)
    response = client.post(
        "/api/chat",
        json={
            "query": "没签劳动合同能要双倍工资吗",
            "history": [{"role": "user", "content": "我在一家公司干了十个月"}],
            "turn_id": 1,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert '"type": "delta"' in body
    assert "第八十二条" in body
    assert '"type": "done"' in body
