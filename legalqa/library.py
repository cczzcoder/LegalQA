"""法条库浏览与评测看板的只读数据层。

两块数据都不走模型：

- 法条目录直接读索引的 docs（缺失时回落到 data/corpus/all.jsonl），
  起服务后第一次访问才加载，避免 CLI/测试为它付启动成本。
- 评测看板读 data/eval 下的历史 JSON，不重跑评测，避免看一次看板就烧一次 API。

这个模块只做「读文件 + 聚合 + 打分」，不依赖 FastAPI，方便单测。
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from legalqa.pure.retrieval import (
    cn_to_int,
    extract_article_no,
    match_law_name,
    same_law,
)

__all__ = ["LawCatalog", "EvalStore", "default_docs_path", "default_eval_dir"]

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval"

_CHECK_ORDER = ("免责声明", "长度合理", "引用可溯源", "无幻觉条号", "命中期望法条", "拒答恰当", "检索覆盖")

# 关键节点在列表里置顶。note 只写实验结论，不写分数——分数由文件实时算，
# 写死在文案里迟早和文件对不上。
_RUN_META: dict[str, tuple[str, str]] = {
    "qwen37_fact_100": ("当前基线 · 100 条", "E19 工具 query 保留事实词（采纳）"),
    "qwen37_barefix_34": ("当前回归 · 34 条", "E20 裸引条号按最近法名解析（采纳）"),
    "qwen37_fact_10": ("硬失败子集 · 10 条", "E19 逐案复盘的子集"),
    "qwen37_100_a": ("qwen3.7 embedding · 100 条 A", "E17 换 embedding 后首轮"),
    "qwen37_100_b": ("qwen3.7 embedding · 100 条 B", "E17 换 embedding 后复跑"),
    "qwen37_34": ("qwen3.7 embedding · 34 条", "E17 回归集"),
    "goldfix_100_a": ("金标审计后 · 100 条 A", "E15 修金标与语料尾部"),
    "goldfix_100_b": ("金标审计后 · 100 条 B", "E15 复跑"),
    "s100_final": ("E13 修复后基线 · 100 条", "循环终止、引用回捞、度量口径"),
    "s100_base": ("E13 之前 · 100 条", "修复前基线"),
    "qwen37_supp_100": ("补召回 · 100 条", "E18 未采纳"),
    "qwen37_prompt_100": ("引用选择 prompt · 100 条", "E18 未采纳"),
    "qwen37_gatedsupp_100": ("有门槛补召回 · 100 条", "E22 未采纳"),
    "qwen37plus_10": ("qwen3.7-plus · 10 条", "E21 未采纳"),
}
_PINNED = ("qwen37_fact_100", "qwen37_barefix_34", "qwen37_fact_10")

_STOP_CHARS = set("的了吗呢吧啊呀么怎如何什么哪些可以是否请问我你他她它您这那有和与或被把对在")
_NON_WORD = re.compile(r"[^\u4e00-\u9fffa-zA-Z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]+")
_ALNUM = re.compile(r"[a-zA-Z0-9]{2,}")


def default_docs_path() -> Path:
    env = os.environ.get("LEGALMIND_LAW_DOCS")
    if env:
        return Path(env)
    indexed = ROOT / "data" / "index" / "qwen37" / "docs.json"
    if indexed.exists():
        return indexed
    return ROOT / "data" / "corpus" / "all.jsonl"


def default_eval_dir() -> Path:
    return Path(os.environ.get("LEGALMIND_EVAL_DIR") or EVAL_DIR)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path.name} 不是法条列表")
    return data


def _tokens(text: str) -> list[str]:
    """中文按 bigram、英文数字按词切。去重后保留顺序。"""
    cleaned = "".join(ch for ch in text if ch not in _STOP_CHARS)
    tokens: list[str] = []
    for part in _NON_WORD.split(cleaned):
        if not part:
            continue
        for run in _CJK.findall(part):
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        tokens.extend(_ALNUM.findall(part))
    return list(dict.fromkeys(tokens))


def _score(doc: dict[str, Any], query: str, article_no: int | None) -> float:
    if article_no is not None and int(doc.get("article_no") or 0) == article_no:
        return 1000.0
    haystack = " ".join(
        str(doc.get(key) or "")
        for key in ("law_name", "article_label", "chapter", "section", "text")
    )
    if query in haystack:
        return 200.0 + min(len(query), 20) / 10
    tokens = _tokens(query)
    if not tokens:
        return 0.0
    coverage = sum(1 for token in tokens if token in haystack) / len(tokens)
    return round(coverage * 100, 2) if coverage >= 0.5 else 0.0


def _parse_article_query(query: str) -> int | None:
    found = extract_article_no(query)
    if found is not None:
        return found
    if query.isdigit() and len(query) <= 4:
        return cn_to_int(query)
    return None


class LawCatalog:
    def __init__(self, docs_path: str | Path | None = None) -> None:
        self.docs_path = Path(docs_path) if docs_path else default_docs_path()
        self._docs: list[dict[str, Any]] | None = None
        self._laws: list[dict[str, Any]] | None = None

    def _load(self) -> list[dict[str, Any]]:
        if self._docs is None:
            if not self.docs_path.exists():
                raise FileNotFoundError(f"法条数据不存在：{self.docs_path}")
            rows = _load_rows(self.docs_path)
            self._docs = [row for row in rows if row.get("law_name") and row.get("article_no")]
        return self._docs

    def reload(self) -> None:
        """文献入库后调用：清掉缓存，下次访问重新读文件。"""
        self._docs = None
        self._laws = None

    def laws(self) -> list[dict[str, Any]]:
        if self._laws is None:
            grouped: dict[str, dict[str, Any]] = {}
            for doc in self._load():
                name = str(doc["law_name"])
                item = grouped.setdefault(
                    name,
                    {
                        "name": name,
                        "doc_type": "",
                        "article_count": 0,
                        "valid_count": 0,
                        "versions": set(),
                    },
                )
                item["article_count"] += 1
                if doc.get("is_valid", True):
                    item["valid_count"] += 1
                if doc.get("doc_type"):
                    item["doc_type"] = doc["doc_type"]
                if doc.get("version_date"):
                    item["versions"].add(str(doc["version_date"]))
            laws = []
            for item in grouped.values():
                versions = sorted(item.pop("versions"))
                item["versions"] = versions
                item["latest_version"] = versions[-1] if versions else ""
                laws.append(item)
            laws.sort(key=lambda item: (-item["article_count"], item["name"]))
            self._laws = laws
        return self._laws

    def _filter_law(self, docs: list[dict[str, Any]], law: str) -> list[dict[str, Any]]:
        """目录里的全称走精确匹配；简称先解析成唯一全称，再回落到宽松匹配。"""
        names = [item["name"] for item in self.laws()]
        matched = law if law in names else match_law_name(law, names)
        if matched:
            return [doc for doc in docs if str(doc["law_name"]) == matched]
        return [doc for doc in docs if same_law(law, str(doc["law_name"]))]

    def stats(self) -> dict[str, Any]:
        docs = self._load()
        doc_types = Counter(str(doc.get("doc_type") or "未分类") for doc in docs)
        return {
            "total_laws": len(self.laws()),
            "total_articles": len(docs),
            "valid_articles": sum(1 for doc in docs if doc.get("is_valid", True)),
            "doc_types": [
                {"name": name, "count": count}
                for name, count in doc_types.most_common()
            ],
        }

    def articles(
        self,
        law: str | None = None,
        q: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        offset = max(0, offset)
        limit = min(max(1, limit), 100)
        docs = self._load()

        if law:
            docs = self._filter_law(docs, law)

        query = (q or "").strip()
        if query:
            # 「劳动合同法第八十七条」只按条号搜会在每部法律里各命中一条，
            # 用户一旦带了法名，就先把范围收敛到那部法。
            if not law:
                inferred = match_law_name(query, [item["name"] for item in self.laws()])
                if inferred:
                    docs = [doc for doc in docs if str(doc["law_name"]) == inferred]
            article_no = _parse_article_query(query)
            exact = []
            if article_no is not None:
                exact = [
                    doc
                    for doc in docs
                    if int(doc.get("article_no") or 0) == article_no
                ]
            if exact:
                docs = sorted(
                    exact,
                    key=lambda doc: (str(doc["law_name"]), int(doc["article_no"])),
                )
            else:
                scored = []
                for doc in docs:
                    score = _score(doc, query, article_no)
                    if score > 0:
                        scored.append((score, doc))
                scored.sort(
                    key=lambda pair: (
                        -pair[0],
                        str(pair[1]["law_name"]),
                        int(pair[1]["article_no"]),
                    )
                )
                docs = [doc for _, doc in scored]
        else:
            # 浏览模式按「法律条目数」排序，和左侧目录一致：首屏先看到民法典，
            # 而不是按法名拼音排出来的劳动争议调解仲裁法。
            law_order = {law["name"]: index for index, law in enumerate(self.laws())}
            docs = sorted(
                docs,
                key=lambda doc: (
                    law_order.get(str(doc["law_name"]), len(law_order)),
                    int(doc["article_no"]),
                ),
            )

        total = len(docs)
        page = docs[offset : offset + limit]
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "law": law or "",
            "query": query,
            "items": [self._brief(doc) for doc in page],
        }

    @staticmethod
    def _brief(doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": doc.get("id", ""),
            "law_name": doc.get("law_name", ""),
            "doc_type": doc.get("doc_type", ""),
            "article_no": doc.get("article_no", 0),
            "article_label": doc.get("article_label", ""),
            "chapter": doc.get("chapter", ""),
            "section": doc.get("section", ""),
            "text": doc.get("text", ""),
            "is_valid": bool(doc.get("is_valid", True)),
            "version_date": doc.get("version_date", ""),
        }


def _percentile(sorted_values: list[int], ratio: float) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * ratio))))
    return sorted_values[index]


def _check_summary(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}

    def bump(name: str, passed: bool) -> None:
        item = buckets.setdefault(name, {"name": name, "total": 0, "failed": 0})
        item["total"] += 1
        if not passed:
            item["failed"] += 1

    for case in cases:
        for check in case.get("checks") or []:
            if isinstance(check, dict) and check.get("name"):
                bump(str(check["name"]), bool(check.get("passed")))
        coverage = case.get("retrieval_coverage")
        if isinstance(coverage, dict):
            bump("检索覆盖", bool(coverage.get("passed")))

    order = {name: index for index, name in enumerate(_CHECK_ORDER)}
    checks = sorted(
        buckets.values(),
        key=lambda item: (order.get(item["name"], len(order)), item["name"]),
    )
    for item in checks:
        item["fail_rate"] = round(item["failed"] / item["total"], 4) if item["total"] else 0.0
    return checks


def _analyze(cases: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = sorted(
        int(case["latency_ms"])
        for case in cases
        if isinstance(case.get("latency_ms"), (int, float)) and case["latency_ms"] > 0
    )
    passed = sum(1 for case in cases if case.get("passed"))
    return {
        "case_count": len(cases),
        "passed": passed,
        "pass_rate": round(passed / len(cases), 4) if cases else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
        "p90_latency_ms": _percentile(latencies, 0.9),
        "channel_counts": dict(Counter(str(case.get("channel") or "未知") for case in cases)),
        "checks": _check_summary(cases),
    }


def _failure(case: dict[str, Any]) -> dict[str, Any]:
    failed = [
        {"name": check.get("name", ""), "detail": check.get("detail", "")}
        for check in case.get("checks") or []
        if isinstance(check, dict) and not check.get("passed")
    ]
    coverage = case.get("retrieval_coverage")
    if isinstance(coverage, dict) and not coverage.get("passed"):
        failed.append({"name": "检索覆盖", "detail": coverage.get("detail", "")})
    return {
        "id": case.get("id", ""),
        "query": case.get("query", ""),
        "latency_ms": case.get("latency_ms", 0),
        "iterations": case.get("iterations", 0),
        "score": case.get("score", 0),
        "channel": case.get("channel", ""),
        "failed_checks": failed,
    }


class EvalStore:
    def __init__(self, eval_dir: str | Path | None = None) -> None:
        self.eval_dir = Path(eval_dir) if eval_dir else default_eval_dir()
        self._cache: dict[str, tuple[float, int, list[dict[str, Any]]]] = {}

    def _cases(self, run_id: str) -> list[dict[str, Any]] | None:
        path = self.eval_dir / f"{run_id}.json"
        if not path.is_file() or path.parent.resolve() != self.eval_dir.resolve():
            return None
        stat = path.stat()
        cached = self._cache.get(run_id)
        if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            return cached[2]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, list) or not data:
            return None
        cases = [item for item in data if isinstance(item, dict) and "passed" in item]
        if not cases:
            return None
        self._cache[run_id] = (stat.st_mtime, stat.st_size, cases)
        return cases

    def _paths(self) -> dict[str, Path]:
        if not self.eval_dir.is_dir():
            return {}
        return {path.stem: path for path in self.eval_dir.glob("*.json") if path.is_file()}

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for run_id, path in self._paths().items():
            cases = self._cases(run_id)
            if not cases:
                continue
            summary = _analyze(cases)
            label, note = _RUN_META.get(run_id, (run_id, ""))
            summary.update(
                {
                    "id": run_id,
                    "label": label,
                    "note": note,
                    "pinned": run_id in _PINNED,
                    "updated_at": datetime.fromtimestamp(
                        path.stat().st_mtime, timezone.utc
                    ).isoformat(timespec="seconds"),
                }
            )
            runs.append(summary)

        pinned_order = {run_id: index for index, run_id in enumerate(_PINNED)}
        # 非置顶项按更新时间倒序；置顶项保持 _PINNED 顺序。
        pinned = [run for run in runs if run["pinned"]]
        pinned.sort(key=lambda run: pinned_order[run["id"]])
        rest = sorted(
            (run for run in runs if not run["pinned"]),
            key=lambda run: run["updated_at"],
            reverse=True,
        )
        return pinned + rest

    def detail(self, run_id: str) -> dict[str, Any] | None:
        cases = self._cases(run_id)
        if not cases:
            return None
        label, note = _RUN_META.get(run_id, (run_id, ""))
        detail = _analyze(cases)
        detail.update(
            {
                "id": run_id,
                "label": label,
                "note": note,
                "failures": [_failure(case) for case in cases if not case.get("passed")],
            }
        )
        return detail

    def case(self, run_id: str, case_id: str) -> dict[str, Any] | None:
        cases = self._cases(run_id)
        if not cases:
            return None
        return next((case for case in cases if case.get("id") == case_id), None)
