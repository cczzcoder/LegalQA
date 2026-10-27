"""文献导入：上传落盘、解析、查重/时效分析、确认入库与增量索引。

原则：
- 分析（analyze）只读文件，绝不改语料/索引；确认（apply）才落库。
- 入库前必须备份语料；索引增量只向量化新增/变更条文（复用 VectorIndex.build_incremental）。
- embedding 失败不阻断语料入库，但标记索引落后——检索服务下次加载会自动全量重建。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from legalqa.corpus.ingest import (
    ScannedPdfError,
    load_jsonl,
    parse_articles,
    read_any,
)
from legalqa.corpus.index import VectorIndex
from legalqa.pure.retrieval import normalize_law

__all__ = ["Importer", "SUPPORTED_SUFFIXES", "MAX_UPLOAD_BYTES", "law_name_from_path", "parse_version_date"]

ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_SUFFIXES = {".docx", ".docm", ".pdf"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_FILENAME_CHARS = 120

_DATE_SUFFIX = re.compile(r"[_\-\s]*(\d{8})$")
_VERSION_SUFFIX = re.compile(r"[（(][^）)]{0,20}(修正|修订|年版|文本)[^）)]{0,10}[）)]")

_DOC_TYPE_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("宪法",), "宪法"),
    (("解释", "批复", "答复"), "司法解释"),
    (("条例",), "行政法规"),
    (("办法", "细则", "规则"), "部门规章"),
)


def law_name_from_path(path: Path | str) -> str:
    """文件名 → 法律全称。与 scripts/ingest_folder.py 同规则：
    剥书名号、8 位日期后缀、版本说明后缀。"""
    stem = Path(path).stem.replace("《", "").replace("》", "")
    stem = _DATE_SUFFIX.sub("", stem)
    stem = _VERSION_SUFFIX.sub("", stem)
    stem = re.sub(r"\s*[（(]\d+[）)]\s*$", "", stem)
    stem = re.sub(r"\s*-\s*副本\s*$", "", stem)
    return stem.strip()


def parse_version_date(path: Path | str) -> str:
    match = _DATE_SUFFIX.search(Path(path).stem)
    if not match:
        return ""
    raw = match.group(1)
    try:
        year, month, day = int(raw[:4]), int(raw[4:6]), int(raw[6:])
        if not (1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31):
            return ""
        return f"{year:04d}-{month:02d}-{day:02d}"
    except ValueError:
        return ""


def doc_type_for(law_name: str) -> str:
    for markers, doc_type in _DOC_TYPE_RULES:
        if any(marker in law_name for marker in markers):
            return doc_type
    return "法律"


def _clean_filename(filename: str) -> str:
    name = Path(filename or "").name
    return re.sub(r"[^\w\u4e00-\u9fff.\-]", "_", name)[:MAX_FILENAME_CHARS]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _embedder():
    from legalqa.env import load_dotenv
    from legalqa.llm_factory import get_embeddings

    load_dotenv()
    return get_embeddings()


class Importer:
    def __init__(
        self,
        corpus_path: str | Path | None = None,
        index_dir: str | Path | None = None,
        uploads_dir: str | Path | None = None,
        backups_dir: str | Path | None = None,
    ) -> None:
        self.corpus_path = Path(corpus_path or ROOT / "data/corpus/all.jsonl")
        self.index_dir = Path(index_dir or ROOT / "data/index/all")
        self.uploads_dir = Path(uploads_dir or ROOT / "data/imports")
        self.backups_dir = Path(backups_dir or ROOT / "data/backups")

    # -- 上传 ----------------------------------------------------------
    def save_upload(self, filename: str, content: bytes) -> tuple[str, Path]:
        clean = _clean_filename(filename) or "document"
        import_id = uuid.uuid4().hex
        folder = self.uploads_dir / import_id
        folder.mkdir(parents=True, exist_ok=True)
        stored = folder / clean
        stored.write_bytes(content)
        return import_id, stored

    def remove_upload(self, stored_path: str | Path) -> None:
        shutil.rmtree(Path(stored_path).parent, ignore_errors=True)

    # -- 分析 ----------------------------------------------------------
    def analyze(self, stored_path: str | Path) -> dict[str, Any]:
        path = Path(stored_path)
        filename = path.name
        law_name = law_name_from_path(path)
        version_date = parse_version_date(path)
        base: dict[str, Any] = {
            "ok": False,
            "error": "",
            "law_name": law_name,
            "version_date": version_date,
            "doc_type": doc_type_for(law_name),
            "article_count": 0,
            "missing_numbers": [],
            "duplicated_numbers": [],
            "existing": {"found": False, "versions": [], "article_count": 0, "overlap_rate": 0.0},
            "timeliness": {"version_missing": not version_date, "years_since_version": None, "corpus_has_newer": False},
            "suggestion": "",
        }

        try:
            paragraphs = read_any(path)
            articles = parse_articles(
                paragraphs,
                law_name=law_name,
                doc_type=base["doc_type"],
                version_date=version_date,
                source=filename,
            )
        except ScannedPdfError as exc:
            base["error"] = f"扫描版 PDF 无法解析：{exc}"
            base["suggestion"] = "请提供文字版 PDF 或 docx"
            return base
        except (OSError, ValueError, KeyError) as exc:
            base["error"] = f"{type(exc).__name__}: {exc}"
            base["suggestion"] = "文件解析失败，请检查格式"
            return base

        if not articles:
            base["error"] = "未解析出任何条文（未找到「第X条」结构）"
            base["suggestion"] = "文件可能不是法律条文文本"
            return base

        numbers = sorted(article.article_no for article in articles)
        base["article_count"] = len(articles)
        if numbers[-1] > len(numbers):
            base["missing_numbers"] = [
                n for n in range(1, numbers[-1] + 1) if n not in set(numbers)
            ]
        base["duplicated_numbers"] = sorted(
            {n for n in numbers if numbers.count(n) > 1}
        )

        if version_date:
            years = date.today().year - int(version_date[:4])
            base["timeliness"]["years_since_version"] = max(years, 0)

        corpus_rows = load_jsonl(self.corpus_path) if self.corpus_path.exists() else []
        same_law = [
            row for row in corpus_rows
            if normalize_law(str(row.get("law_name", ""))) == normalize_law(law_name)
        ]
        if same_law:
            existing = base["existing"]
            existing["found"] = True
            existing["article_count"] = len(same_law)
            existing["versions"] = sorted(
                {str(row.get("version_date", "")) for row in same_law}
            )
            known = {
                (int(row.get("article_no", 0)), re.sub(r"\s+", "", str(row.get("text", ""))))
                for row in same_law
            }
            own = {
                (article.article_no, re.sub(r"\s+", "", article.text))
                for article in articles
            }
            existing["overlap_rate"] = round(
                len(known & own) / len(own), 4
            ) if own else 0.0

            corpus_versions = [v for v in existing["versions"] if v]
            if version_date and version_date in corpus_versions:
                base["suggestion"] = "重复：语料库已有同名同版本，建议跳过"
            elif corpus_versions and version_date and version_date < max(corpus_versions):
                base["timeliness"]["corpus_has_newer"] = True
                base["suggestion"] = "警告：语料库已有更新版本，导入会倒退版本，建议跳过"
            elif not version_date and existing["overlap_rate"] >= 0.9:
                base["suggestion"] = "疑似重复：条文高度重合但无版本信息，建议跳过"
            elif not version_date:
                base["suggestion"] = "版本日期缺失，建议核对后确认（默认为替换现有同名条文）"
            else:
                base["suggestion"] = f"同名新版本：语料库现有 {', '.join(corpus_versions) or '无日期'}，建议替换"
        else:
            if version_date:
                base["suggestion"] = "新法律：可直接入库"
            else:
                base["suggestion"] = "新法律，但文件名缺少版本日期，建议补日期后确认"

        base["ok"] = True
        return base

    # -- 入库 ----------------------------------------------------------
    def apply(
        self,
        record: dict[str, Any],
        replace: bool = True,
        embedder: Any = None,
    ) -> dict[str, Any]:
        report = record.get("report") or {}
        law_name = report.get("law_name") or law_name_from_path(record["stored_path"])
        version_date = report.get("version_date") or parse_version_date(record["stored_path"])

        paragraphs = read_any(record["stored_path"])
        articles = parse_articles(
            paragraphs,
            law_name=law_name,
            doc_type=report.get("doc_type") or doc_type_for(law_name),
            version_date=version_date,
            source=record.get("filename", ""),
        )
        if not articles:
            return {"ok": False, "error": "未解析出条文，无法入库"}

        corpus_rows = load_jsonl(self.corpus_path) if self.corpus_path.exists() else []
        normalized = normalize_law(law_name)
        if replace:
            kept = [
                row for row in corpus_rows
                if normalize_law(str(row.get("law_name", ""))) != normalized
            ]
        else:
            kept = list(corpus_rows)
        removed = len(corpus_rows) - len(kept)
        new_rows = [article.to_dict() for article in articles]
        updated = kept + new_rows

        if self.corpus_path.exists():
            self.backups_dir.mkdir(parents=True, exist_ok=True)
            stamp = re.sub(r"[:.]", "-", record.get("created_at", "")) or uuid.uuid4().hex[:8]
            backup = self.backups_dir / f"all-{stamp}.jsonl"
            if not backup.exists():
                shutil.copy2(self.corpus_path, backup)
        _write_rows(self.corpus_path, updated)

        index_status = ""
        index_lagged = False
        try:
            embedder = embedder or _embedder()
            previous = (
                VectorIndex.load(self.index_dir)
                if (self.index_dir / "vectors.npy").exists()
                else None
            )
            index, result = VectorIndex.build_incremental(updated, embedder, previous)
            meta = dict(previous.meta) if previous is not None else {}
            if not meta.get("embedder_model"):
                meta["embedder_provider"] = os.environ.get("LEGALQA_EMBEDDING_PROVIDER", "")
                meta["embedder_model"] = os.environ.get("LEGALQA_EMBEDDING_MODEL", "")
                meta["embedder_dimension"] = index.dim
            index.save(self.index_dir, meta=meta)
            index_status = (
                f"索引增量完成：新向量化 {result.embedded} 条，复用 {result.reused} 条"
            )
        except Exception as exc:  # noqa: BLE001 —— embedding 失败不阻断语料入库
            index_lagged = True
            index_status = f"索引重建失败（{type(exc).__name__}: {exc}）；服务重启后自动全量重建"

        return {
            "ok": True,
            "removed": removed,
            "added": len(new_rows),
            "corpus_total": len(updated),
            "index_status": index_status,
            "index_lagged": index_lagged,
        }

    # -- 一致性 --------------------------------------------------------
    def index_lag(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "corpus_count": 0,
            "index_count": 0,
            "lagged": False,
            "model": "",
            "reason": "",
        }
        if not self.corpus_path.exists():
            info.update(lagged=True, reason="语料文件不存在")
            return info
        try:
            rows = load_jsonl(self.corpus_path)
            info["corpus_count"] = len(rows)
        except (OSError, json.JSONDecodeError) as exc:
            info.update(lagged=True, reason=f"语料读取失败：{exc}")
            return info
        if not (self.index_dir / "vectors.npy").exists():
            info.update(lagged=True, reason="索引文件不存在")
            return info
        try:
            index = VectorIndex.load(self.index_dir)
            info["index_count"] = len(index)
            info["model"] = str(index.meta.get("embedder_model", ""))
            if len(index) != len(rows) or index.fingerprint() != VectorIndex.fingerprint_docs(rows):
                info.update(lagged=True, reason="索引与语料不一致（条数或内容指纹不同）")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            info.update(lagged=True, reason=f"索引读取失败：{exc}")
        return info
