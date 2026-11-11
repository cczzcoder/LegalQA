"""库外权威法源白名单：只做原文核对，不自动联网抓取。

白名单条目必须来自官方/权威发布渠道，人工复核后写入
`data/external/whitelist.jsonl`。系统对库外引用的策略是：
  - 白名单命中且 quote 与原文一致 → 记为「库外已核验」；
  - 白名单未收录 / quote 不一致 → 保持「库外未校验」，绝不当成已核实。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from legalqa.pure.retrieval import normalize_law, resolve_law

__all__ = ["ExternalEntry", "ExternalWhitelist", "load_whitelist"]

DEFAULT_WHITELIST_PATH = (
    Path(__file__).resolve().parent.parent / "data/external/whitelist.jsonl"
)

_PUNCT_NOISE = re.compile(
    r"[\s，。；：、！？,.;:!?“”\"'‘’（）()《》【】\[\]]+"
)


def _normalize_text(text: str) -> str:
    """引文比对忽略空白与标点差异，但不放过任何实词改写。"""
    return _PUNCT_NOISE.sub("", str(text or ""))


@dataclass(frozen=True)
class ExternalEntry:
    law_name: str
    article_no: int
    article_label: str
    text: str
    source: str = ""
    source_url: str = ""
    version_date: str = ""
    effective_date: str = ""
    doc_type: str = "法律"

    def to_doc(self) -> dict:
        return {
            "id": f"external:{self.law_name}:{self.article_no}",
            "law_name": self.law_name,
            "article_no": self.article_no,
            "article_label": self.article_label,
            "text": self.text,
            "doc_type": self.doc_type,
            "is_valid": True,
            "effective_date": self.effective_date,
            "version_date": self.version_date,
            "source": self.source,
            "source_url": self.source_url,
            "retrieval_source": "external_whitelist",
        }

    def summary(self, status: str = "verified") -> dict:
        return {
            "law_name": self.law_name,
            "article_no": self.article_no,
            "article_label": self.article_label,
            "source": self.source,
            "source_url": self.source_url,
            "version_date": self.version_date,
            "status": status,
        }


class ExternalWhitelist:
    def __init__(self, entries: Iterable[ExternalEntry] = ()) -> None:
        self._entries = list(entries)
        self._by_pair = {
            (normalize_law(entry.law_name), entry.article_no): entry
            for entry in self._entries
        }
        self._law_names = sorted({entry.law_name for entry in self._entries})

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def law_names(self) -> list[str]:
        return list(self._law_names)

    def get(
        self, law: str | None, article_no: int
    ) -> ExternalEntry | None:
        if not law:
            return None
        resolved = resolve_law(law, self._law_names) or law
        return self._by_pair.get((normalize_law(resolved), article_no))

    def verify(
        self, law: str | None, article_no: int, quote: str | None = None
    ) -> tuple[str, ExternalEntry | None]:
        """返回 (status, entry)。

        status:
          - verified_quote：白名单命中，且 quote 是原文子串
          - verified_exists：白名单命中，回答未给原文（只核对了条号存在）
          - quote_mismatch：白名单命中，但 quote 与原文不一致
          - not_found：白名单未收录
        """
        entry = self.get(law, article_no)
        if entry is None:
            return "not_found", None
        if quote:
            body = _normalize_text(entry.text)
            given = _normalize_text(quote)
            if given and given not in body:
                return "quote_mismatch", entry
            return "verified_quote", entry
        return "verified_exists", entry

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ExternalWhitelist":
        target = Path(path) if path is not None else DEFAULT_WHITELIST_PATH
        if not target.exists():
            return cls()
        entries: list[ExternalEntry] = []
        for line in target.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                number = int(item.get("article_no"))
            except (TypeError, ValueError):
                continue
            text = str(item.get("text") or "").strip()
            law = str(item.get("law_name") or "").strip()
            if not law or not text:
                continue
            entries.append(
                ExternalEntry(
                    law_name=law,
                    article_no=number,
                    article_label=str(item.get("article_label") or f"第{number}条"),
                    text=text,
                    source=str(item.get("source") or ""),
                    source_url=str(item.get("source_url") or ""),
                    version_date=str(item.get("version_date") or ""),
                    effective_date=str(item.get("effective_date") or ""),
                    doc_type=str(item.get("doc_type") or "法律"),
                )
            )
        return cls(entries)


def load_whitelist(
    path: str | Path | None = None,
) -> ExternalWhitelist:
    return ExternalWhitelist.load(path)
