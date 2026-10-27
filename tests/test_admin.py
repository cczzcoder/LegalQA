"""管理端测试：角色、反馈、文献导入（分析/入库/增量索引）。

不调真实 embedding：入库测试注入 HashEmbeddings。
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import legalqa.api as api
import legalqa.importer as importer
from legalqa.importer import Importer
from legalqa.llm_factory import HashEmbeddings
from legalqa.service import ChatEvent
from legalqa.store import Database
from tests.csrf_client import CsrfClient


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
DOCX_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def build_docx(path: Path, paragraphs: list[str]) -> None:
    body = "".join(
        f"<w:p><w:r><w:t>{para}</w:t></w:r></w:p>" for para in paragraphs
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{DOCX_NS}"><w:body>{body}</w:body></w:document>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)


CIVIL_PARAS = [
    "中华人民共和国民法典",
    "第一编 总则",
    "第一条 为了保护民事主体的合法权益，调整民事关系，制定本法。",
    "第二条 民法调整平等主体的自然人、法人和非法人组织之间的财产关系。",
    "第三条 民事主体的人身权利受法律保护。",
]

LABOR_ROW = {
    "id": "劳动合同法-0001",
    "law_name": "中华人民共和国劳动合同法",
    "doc_type": "法律",
    "article_no": 1,
    "article_label": "第一条",
    "chapter": "第一章",
    "section": "",
    "text": "为了完善劳动合同制度，制定本法。",
    "is_valid": True,
    "effective_date": "",
    "version_date": "2012-12-28",
    "source": "test.docx",
}


def make_importer(tmp_path: Path) -> Importer:
    corpus = tmp_path / "corpus" / "all.jsonl"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text(
        json.dumps(LABOR_ROW, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return Importer(
        corpus_path=corpus,
        index_dir=tmp_path / "index",
        uploads_dir=tmp_path / "uploads",
        backups_dir=tmp_path / "backups",
    )


class FakeService:
    def stream(self, query: str, history, turn_id: int = 0):
        assert query
        yield ChatEvent("delta", text="结论：可以主张。")
        yield ChatEvent("done")


# ---------------------------------------------------------------------------
# store：角色与反馈
# ---------------------------------------------------------------------------
def test_role_migration_and_admin_promotion(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    user = db.create_user("boss", "password123")
    assert user["role"] == "user"

    legacy = tmp_path / "legacy.db"
    import sqlite3

    conn = sqlite3.connect(legacy)
    conn.execute(
        "CREATE TABLE users (id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, "
        "salt TEXT NOT NULL, password_hash TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()

    migrated = Database(legacy)
    with migrated._conn() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
    assert "role" in columns

    assert db.set_admin("boss") is True
    assert db.verify_user("boss", "password123")["role"] == "admin"


def test_feedback_roundtrip(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    user = db.create_user("tester", "password123")
    session = db.create_session(user["id"])
    question = db.add_message(session["id"], "user", "拖欠工资怎么办")
    answer = db.add_message(
        session["id"],
        "assistant",
        "结论：可以主张。",
        citations=[{"law_name": "中华人民共和国劳动法", "article_no": 50}],
    )

    context = db.get_message_context(session["id"], answer["id"])
    assert context is not None
    assert context["question"] == "拖欠工资怎么办"
    assert context["citations"][0]["article_no"] == 50
    assert db.get_message_context(session["id"], question["id"]) is None

    feedback = db.add_feedback(
        user["id"],
        session["id"],
        answer["id"],
        "引用错误",
        "条号不对",
        context["question"],
        context["answer"],
        context["citations"],
    )
    listing = db.list_feedback()
    assert listing["total"] == 1
    assert listing["items"][0]["status"] == "open"
    assert db.count_feedback_open() == 1

    assert db.update_feedback_status(feedback["id"], "done") is True
    assert db.count_feedback_open() == 0
    assert db.list_feedback(status="done")["total"] == 1


# ---------------------------------------------------------------------------
# importer：分析与入库
# ---------------------------------------------------------------------------
def test_analyze_new_law(tmp_path: Path) -> None:
    importer = make_importer(tmp_path)
    path = tmp_path / "中华人民共和国民法典_20200528.docx"
    build_docx(path, CIVIL_PARAS)

    report = importer.analyze(path)
    assert report["ok"] is True
    assert report["law_name"] == "中华人民共和国民法典"
    assert report["version_date"] == "2020-05-28"
    assert report["article_count"] == 3
    assert report["suggestion"].startswith("新法律")


def test_analyze_duplicate_and_old_version(tmp_path: Path) -> None:
    importer = make_importer(tmp_path)
    civil_rows = []
    for number, text in (
        (1, "为了保护民事主体的合法权益，调整民事关系，制定本法。"),
        (2, "民法调整平等主体的自然人、法人和非法人组织之间的财产关系。"),
        (3, "民事主体的人身权利受法律保护。"),
    ):
        civil_rows.append(
            {
                **LABOR_ROW,
                "id": f"民法典-{number:04d}",
                "law_name": "中华人民共和国民法典",
                "article_no": number,
                "article_label": f"第{number}条",
                "text": text,
                "version_date": "2020-05-28",
            }
        )
    with importer.corpus_path.open("a", encoding="utf-8") as handle:
        for row in civil_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    duplicate = tmp_path / "中华人民共和国民法典_20200528.docx"
    build_docx(duplicate, CIVIL_PARAS)
    report = importer.analyze(duplicate)
    assert report["suggestion"].startswith("重复")
    assert report["existing"]["overlap_rate"] == 1.0

    older = tmp_path / "中华人民共和国民法典_20170315.docx"
    build_docx(older, CIVIL_PARAS)
    report = importer.analyze(older)
    assert report["timeliness"]["corpus_has_newer"] is True
    assert "更新版本" in report["suggestion"]

    no_version = tmp_path / "中华人民共和国民法典.docx"
    build_docx(no_version, CIVIL_PARAS)
    report = importer.analyze(no_version)
    assert report["timeliness"]["version_missing"] is True
    assert "疑似重复" in report["suggestion"]


def test_apply_import_incremental_index(tmp_path: Path) -> None:
    importer = make_importer(tmp_path)
    path = tmp_path / "中华人民共和国民法典_20200528.docx"
    build_docx(path, CIVIL_PARAS)
    report = importer.analyze(path)
    assert report["ok"] is True

    record = {
        "id": "import-1",
        "stored_path": str(path),
        "filename": path.name,
        "report": report,
        "created_at": "2026-10-27T10:00:00",
    }
    result = importer.apply(record, replace=True, embedder=HashEmbeddings())
    assert result["ok"] is True
    assert result["added"] == 3
    assert result["corpus_total"] == 4
    assert result["index_lagged"] is False
    assert "增量完成" in result["index_status"]

    lag = importer.index_lag()
    assert lag["lagged"] is False
    assert lag["corpus_count"] == 4
    assert lag["index_count"] == 4

    # 再次入库同一部法（替换）：总量不变，旧条文被移除
    again = importer.apply(record, replace=True, embedder=HashEmbeddings())
    assert again["ok"] is True
    assert again["removed"] == 3
    assert again["corpus_total"] == 4

    # 不替换：两个版本并存
    both = importer.apply(record, replace=False, embedder=HashEmbeddings())
    assert both["corpus_total"] == 7
    assert importer.index_lag()["lagged"] is False


# ---------------------------------------------------------------------------
# API：反馈 + 管理端
# ---------------------------------------------------------------------------
@pytest.fixture()
def admin_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api, "get_service", lambda: FakeService())
    monkeypatch.setattr(importer, "_embedder", lambda: HashEmbeddings())
    monkeypatch.setenv("LEGALQA_ADMIN_USERNAME", "boss")

    runtime = make_importer(tmp_path / "runtime")
    # 管理员必须在应用启动前存在，才能被环境变量提升。
    db = Database(tmp_path / "test.db")
    db.create_user("boss", "password123")
    db.set_admin("boss")

    app = api.create_app(
        tmp_path / "test.db",
        docs_path=runtime.corpus_path,
        eval_dir=tmp_path / "eval",
        importer=runtime,
    )
    (tmp_path / "eval").mkdir()
    return CsrfClient(app)


def _login(client: TestClient, username: str, password: str = "password123") -> None:
    login = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    if login.status_code != 200:
        registered = client.post(
            "/api/auth/register", json={"username": username, "password": password}
        )
        assert registered.status_code == 200, registered.text
        assert registered.json()["role"] == "user"


def test_admin_guard_and_overview(admin_client: TestClient) -> None:
    _login(admin_client, "tester")
    assert admin_client.get("/api/admin/overview").status_code == 403
    admin_client.post("/api/auth/logout")

    _login(admin_client, "boss")
    overview = admin_client.get("/api/admin/overview").json()
    assert overview["feedback_open"] == 0
    assert overview["imports_pending"] == 0
    assert overview["index"]["lagged"] is True  # 测试语料是空文件，无索引
    assert admin_client.get("/api/auth/me").json()["role"] == "admin"


def test_feedback_submit_and_admin_flow(admin_client: TestClient) -> None:
    _login(admin_client, "tester")
    session = admin_client.post("/api/sessions", json={}).json()
    chat = admin_client.post(
        f"/api/sessions/{session['id']}/chat", json={"query": "拖欠工资怎么办"}
    )
    assert chat.status_code == 200
    messages = admin_client.get(f"/api/sessions/{session['id']}").json()["messages"]
    answer = next(item for item in messages if item["role"] == "assistant")

    response = admin_client.post(
        "/api/feedback",
        json={
            "session_id": session["id"],
            "message_id": answer["id"],
            "category": "引用错误",
            "comment": "条号不对",
        },
    )
    assert response.status_code == 200, response.text
    admin_client.post("/api/auth/logout")

    _login(admin_client, "boss")
    listing = admin_client.get("/api/admin/feedback").json()
    assert listing["total"] == 1
    item = listing["items"][0]
    assert item["category"] == "引用错误"
    assert item["question"] == "拖欠工资怎么办"
    assert "结论" in item["answer"]

    patched = admin_client.patch(
        f"/api/admin/feedback/{item['id']}", json={"status": "done"}
    )
    assert patched.status_code == 200
    assert admin_client.get("/api/admin/feedback", params={"status": "open"}).json()[
        "total"
    ] == 0


def test_import_upload_analyze_apply(admin_client: TestClient, tmp_path: Path) -> None:
    _login(admin_client, "boss")

    docx = tmp_path / "中华人民共和国民法典_20200528.docx"
    build_docx(docx, CIVIL_PARAS)
    with docx.open("rb") as handle:
        uploaded = admin_client.post(
            "/api/admin/imports",
            files={"file": ("中华人民共和国民法典_20200528.docx", handle, "application/octet-stream")},
        )
    assert uploaded.status_code == 200, uploaded.text
    record = uploaded.json()
    assert record["status"] == "uploaded"

    analyzed = admin_client.post(f"/api/admin/imports/{record['id']}/analyze").json()
    assert analyzed["ok"] is True
    assert analyzed["suggestion"].startswith("新法律")

    applied = admin_client.post(
        f"/api/admin/imports/{record['id']}/apply", json={"replace": True}
    ).json()
    assert applied["ok"] is True
    assert applied["added"] == 3
    assert applied["index_lagged"] is False

    listing = admin_client.get("/api/admin/imports").json()["items"]
    assert listing[0]["status"] == "applied"

    assert admin_client.delete(f"/api/admin/imports/{record['id']}").status_code == 400

    articles = admin_client.get(
        "/api/laws/articles", params={"law": "中华人民共和国民法典"}
    ).json()
    assert articles["total"] == 3

    overview = admin_client.get("/api/admin/overview").json()
    assert overview["index"]["lagged"] is False


def test_import_can_be_deleted_before_apply(admin_client: TestClient, tmp_path: Path) -> None:
    _login(admin_client, "boss")
    docx = tmp_path / "中华人民共和国民法典_20200528.docx"
    build_docx(docx, CIVIL_PARAS)
    with docx.open("rb") as handle:
        uploaded = admin_client.post(
            "/api/admin/imports",
            files={"file": ("中华人民共和国民法典_20200528.docx", handle, "application/octet-stream")},
        )
    record = uploaded.json()
    assert admin_client.delete(f"/api/admin/imports/{record['id']}").json()["ok"] is True
    assert admin_client.get("/api/admin/imports").json()["items"] == []
    assert admin_client.get("/api/admin/overview").json()["imports_pending"] == 0


def test_import_rejects_non_admin_and_bad_suffix(admin_client: TestClient) -> None:
    _login(admin_client, "tester")
    response = admin_client.post(
        "/api/admin/imports",
        files={"file": ("evil.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 403
    admin_client.post("/api/auth/logout")

    _login(admin_client, "boss")
    response = admin_client.post(
        "/api/admin/imports",
        files={"file": ("evil.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "格式" in response.json()["detail"]


def test_audit_log_admin_only(admin_client: TestClient) -> None:
    _login(admin_client, "tester")
    assert admin_client.get("/api/admin/audit").status_code == 403
    admin_client.post("/api/auth/logout")

    _login(admin_client, "boss")
    items = admin_client.get("/api/admin/audit").json()["items"]
    actions = {item["action"] for item in items}
    assert "login" in actions
    assert "register" in actions
    assert any(item["action"] == "login" and item["username"] == "boss" for item in items)

    overview = admin_client.get("/api/admin/overview").json()
    assert isinstance(overview["recent_audit"], list)
