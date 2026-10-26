"""法条语料构建：docx / txt → 结构化 JSONL。

对应设计文档 §6.4.2「按条切块」。

**为什么不用固定长度切块。** 法条的语义单元就是「一条」，一条内部各款互相限定
（「前款规定的……」「但有下列情形之一的除外」）。切碎了会出两种事故：
检索到半条 → 模型脑补后半条 → 引用编造；或者「但书」被切到另一个 chunk
→ 结论完全反向。

本模块只做解析，不碰向量。索引在 `legalqa.corpus.index`。
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence
from xml.etree import ElementTree as ET

from legalqa.pure.retrieval import cn_to_int

__all__ = [
    "Article",
    "ScannedPdfError",
    "read_docx_paragraphs",
    "read_pdf_paragraphs",
    "read_any",
    "parse_articles",
    "write_jsonl",
    "load_jsonl",
]

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# 页码 / 页眉行，PDF 提取时必然混入，必须滤掉否则会被当成条文续段
_PAGE_NOISE = (
    re.compile(r"^\d{1,4}$"),
    re.compile(r"^[-—–－]\s*\d{1,4}\s*[-—–－]$"),
    re.compile(r"^第\s*\d{1,4}\s*页(\s*/?\s*共\s*\d{1,4}\s*页)?$"),
    re.compile(r"^共\s*\d{1,4}\s*页$"),
)

# 第X条。
# **条号内部允许空格** —— PDF 提取经常把「第二十七条」拆成「第二十七 条」，
# 「第五十八条」拆成「第五十 八 条」。不允许空格就会静默漏条：
# 实测民用航空法 PDF 因此漏了 16 条，而且不报任何错。
# 以这些字符开头的「第X条」是正文里的引用/枚举，不是新条文
_FALSE_ARTICLE_HEAD = "至、，；：及和或（()）"
# 「第一款、」这种折行残留：以「第」开头**且很短**才算假条。
# 不能只看首字符 —— 真条文正文也可能以「第二款……」开头。
_FALSE_ARTICLE_PREFIX = "第"
_FALSE_ARTICLE_MAX_LEN = 8

_NUM_CHAR = r"[一二三四五六七八九十百千零〇两0-9]"
_ARTICLE_RE = re.compile(
    rf"^第\s*({_NUM_CHAR}(?:\s*{_NUM_CHAR}){{0,11}})\s*条[\s　]*(.*)$"
)
# 第X章 / 第X节
_CHAPTER_RE = re.compile(r"^(第[一二三四五六七八九十]{1,6}章)[\s　]*(.*)$")
_SECTION_RE = re.compile(r"^(第[一二三四五六七八九十]{1,6}节)[\s　]*(.*)$")
_TOC_RE = re.compile(r"^目[\s　]*录$")


@dataclass
class Article:
    """一条法条。字段与 §6.4.2 的元数据表一一对应。"""

    id: str
    law_name: str
    doc_type: str
    article_no: int
    article_label: str
    chapter: str
    section: str
    text: str
    # 现行有效 / 已废止。已废止的条文不参与检索（见 ReadTool）。
    is_valid: bool = True
    # 施行日期（手工填，可空）。回答里优先展示这个。
    effective_date: str = ""
    # 语料版本日期（从文件名解析，如 民法典_20200528 → 2020-05-28）。
    # 法律有强时效性：同一部法律不同版本的条文可能完全相反，
    # 不记版本就无法判断「这条现在还有效吗」。
    version_date: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------
def read_docx_paragraphs(path: str | Path) -> list[str]:
    """从 docx 抽取非空段落文本。

    直接读 `word/document.xml` 而不是用 python-docx：
    docx 本质是 zip + XML，标准库就够，少一个依赖，也少一层抽象。
    """
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)

    paragraphs: list[str] = []
    for node in root.iter(_W + "p"):
        text = "".join(t.text or "" for t in node.iter(_W + "t")).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


class ScannedPdfError(RuntimeError):
    """PDF 没有文本层（扫描件）。明确报错，不要静默产出空语料。"""


def read_pdf_paragraphs(path: str | Path, *, min_chars_per_page: int = 50) -> list[str]:
    """从文本版 PDF 抽取行列表。

    **扫描版 PDF 会被明确拒绝。** 实测一份 29 页的宪法 PDF：每页只有一张图、
    没有字体对象，`extract_text()` 返回空字符串。如果不过滤，解析器会产出
    0 条条文，而调用方以为「这份法律没有条文」——比报错恶劣得多。

    返回的是**行**列表而非段落：PDF 里一条法条常折成多行，交给 `parse_articles`
    按「第X条」开头切分更可靠（折行后的续行自然归到上一条）。
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError(
            "缺少 pypdf。请执行：pip install pypdf -i https://pypi.tuna.tsinghua.edu.cn/simple"
        ) from exc

    reader = PdfReader(str(path))
    page_count = max(len(reader.pages), 1)

    lines: list[str] = []
    total_chars = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        total_chars += len(text.strip())
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if any(pattern.match(line) for pattern in _PAGE_NOISE):
                continue
            lines.append(line)

    if total_chars < min_chars_per_page * page_count:
        raise ScannedPdfError(
            f"{Path(path).name} 是扫描件（{page_count} 页仅 {total_chars} 字符），"
            "没有文本层。请用 OCR 转换，或改用 docx 版本。"
        )
    if not lines:
        raise ScannedPdfError(f"{Path(path).name} 提取不到任何文本行")
    return lines


def read_any(path: str | Path) -> list[str]:
    """按扩展名分派。未知格式明确报错，不猜。"""
    suffix = Path(path).suffix.lower()
    if suffix in {".docx", ".docm"}:
        return read_docx_paragraphs(path)
    if suffix == ".pdf":
        return read_pdf_paragraphs(path)
    raise ValueError(f"不支持的格式：{suffix}（只接受 .docx / .pdf）")


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
_SENTENCE_END = ("。", "；", "：", "！", "？", "”")


def _append_text(buffer: list[str], line: str) -> None:
    """把续行并入当前条文。

    PDF 提取会把一句话拆成很多短行（「国务院民用航空主管部门对全国民」/
    「用航空活动实施统一监督管理」/「；」）。如果每行都独立成段，
    条文文本会碎片化，影响检索和引用展示。所以上一行没以句末标点结尾时
    直接拼接。
    """
    if buffer and not buffer[-1].endswith(_SENTENCE_END):
        buffer[-1] += line
    else:
        buffer.append(line)


def _norm(text: str) -> str:
    """全角空格归一，便于匹配标题行。"""
    return re.sub(r"[\s　]+", "", text)


def _find_body_start(paragraphs: Sequence[str]) -> int:
    """跳过封面与目录，返回正文起始下标。

    优先用「目 录」标记定位（目录项之后就是序言/正文）。
    没有目录时才退回「第一个条文行之前最近的章标题」。

    **不要用「第二个第一章」这种数数法。** 目录里有几个章，正文就有几个，
    一旦目录列了 4 个章，「第二个第一章」实际指向的是目录里的第二章。
    宪法这种多章法律会直接取错位置。
    """
    for i, line in enumerate(paragraphs):
        if _TOC_RE.match(_norm(line)):
            return i + 1

    for i, line in enumerate(paragraphs):
        if _ARTICLE_RE.match(line.strip()):
            # 回退到「连续的标题块」的起点，而不是最近的一个标题。
            # 只回退一层会把「第三章 → 第一节 → 第五十七条」中的第三章漏掉，
            # 导致第一章条文的 chapter 元数据为空。
            j = i - 1
            while j >= 0:
                flat = _norm(paragraphs[j])
                if _CHAPTER_RE.match(flat) or _SECTION_RE.match(flat):
                    j -= 1
                    continue
                break
            return j + 1
    return 0


def parse_articles(
    paragraphs: Iterable[str],
    *,
    law_name: str,
    doc_type: str = "法律",
    effective_date: str = "",
    version_date: str = "",
    source: str = "",
) -> list[Article]:
    """把段落序列解析为法条列表。

    规则：
      - 以「第X条」开头的行 = 新条文的起点
      - 以「第X章 / 第X节」开头的行 = 上下文标记，不产生条文
      - 其余行 = 上一条文的续段（款），用换行拼接保留款结构
      - 第一个「第X条」之前的内容（序言等）整体丢弃
    """
    lines = list(paragraphs)
    start = _find_body_start(lines)

    articles: list[Article] = []
    chapter = ""
    section = ""
    current_no: int | None = None
    current_label = ""
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current_no, current_label, buffer
        if current_no is None or not buffer:
            return
        articles.append(
            Article(
                id=f"{_slug(law_name)}-{current_no:04d}",
                law_name=law_name,
                doc_type=doc_type,
                article_no=current_no,
                article_label=current_label,
                chapter=chapter,
                section=section,
                text="\n".join(buffer).strip(),
                is_valid=True,
                effective_date=effective_date,
                version_date=version_date,
                source=source,
            )
        )
        buffer = []

    for line in lines[start:]:
        flat = _norm(line)

        chapter_match = _CHAPTER_RE.match(flat)
        if chapter_match:
            flush()
            current_no, current_label = None, ""
            chapter = flat
            section = ""
            continue

        section_match = _SECTION_RE.match(flat)
        if section_match:
            flush()
            current_no, current_label = None, ""
            section = flat
            continue

        # 用**原文**匹配而非归一化后的 flat：flat 把所有空白都删了，
        # 会把「工作 3 年」压成「工作3年」，法条文本保真度受损。
        # 正则里的 [\s　]* 已经能吃掉「第一条」后面的分隔空白。
        article_match = _ARTICLE_RE.match(line)
        if article_match:
            label, body = article_match.group(1), article_match.group(2)
            number = cn_to_int(label)
            if number is None:          # 解析不出条号就当续段，不丢内容
                if current_no is not None:
                    _append_text(buffer, line)
                continue

            # 条号必须严格递增。若出现「不大于当前条号」，说明这是正文里的
            # 回溯引用（「违反本法第十六条至第十九条、第二十一条规定的……」），
            # PDF 折行后恰好落在行首，被误判成新条文。
            # 实测商标法 PDF 因此多出 18 条假条，最短的正文只有 1 个字。
            if current_no is not None and number <= current_no:
                _append_text(buffer, line)
                continue

            # 另一类假条：条号后面跟的是枚举连接词，不是条文正文。
            # 例如「第十六条至第十九条……」「第十七条、」「第二十三条 第一款 、」。
            # 判据是**首字符**，不能用「正文长度」——PDF 里条号经常独占一行
            # （body 为空，正文在下一行），用长度判会把真条文全部误杀：
            # 实测民用航空法因此从 262 条掉到 29 条。
            stripped_body = body.strip()
            is_false = bool(stripped_body) and (
                stripped_body[0] in _FALSE_ARTICLE_HEAD
                or (
                    stripped_body.startswith(_FALSE_ARTICLE_PREFIX)
                    and len(stripped_body) < _FALSE_ARTICLE_MAX_LEN
                )
            )
            if current_no is not None and is_false:
                _append_text(buffer, line)
                continue

            flush()
            current_no = number
            current_label = f"第{label.strip()}条"
            buffer = [body] if body else []
            continue

        if current_no is not None:
            _append_text(buffer, line)

    flush()
    return articles


def _slug(name: str) -> str:
    """法律名 → id 前缀。

    去掉「中华人民共和国」这个共同前缀，得到可读的短名（民法典、劳动合同法）。
    不转拼音，避免引入分词依赖。前 12 字 + 全名前缀冲突时由调用方检测。
    """
    table = {
        "中华人民共和国宪法": "constitution",
    }
    if name in table:
        return table[name]
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "", name)
    for prefix in ("中华人民共和国", "中国"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    return cleaned[:12] or "law"


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def write_jsonl(articles: Iterable[Article], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for article in articles:
            handle.write(json.dumps(article.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def load_jsonl(path: str | Path) -> list[dict]:
    items: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items
