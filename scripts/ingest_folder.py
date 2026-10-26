"""批量导入法律文献：扫描 data/laws/ 下所有 docx / pdf，解析入库并重建索引。

用法：
    python scripts/ingest_folder.py
    python scripts/ingest_folder.py --dir data/laws --out data/corpus/all.jsonl

设计要点：
  - **单份文件失败不影响其他文件**，最后汇总报告
  - 扫描版 PDF 明确报错并跳过，不产出空语料
  - 条号缺号 / 重号会高亮提示，这是源文档格式不规整的信号
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legalqa.corpus.index import VectorIndex  # noqa: E402
from legalqa.env import load_dotenv  # noqa: E402
from legalqa.corpus.ingest import (  # noqa: E402
    Article,
    ScannedPdfError,
    parse_articles,
    read_any,
    write_jsonl,
)

SUPPORTED = {".docx", ".docm", ".pdf"}

# 从文件名推断效力层级，影响融合排序里的 authority_weight
_DOC_TYPE_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("宪法",), "宪法"),
    (("解释", "批复", "答复"), "司法解释"),
    (("条例",), "行政法规"),
    (("办法", "细则", "规则"), "部门规章"),
)


_DATE_SUFFIX = re.compile(r"[_\-\s]*(\d{8})$")
_VERSION_SUFFIX = re.compile(r"[（(][^）)]{0,20}(修正|修订|年版|文本)[^）)]{0,10}[）)]")


def law_name_from_path(path: Path) -> str:
    """文件名 → 法律全称。

    要剥掉三类噪声，否则同一部法律会变成好几份不同的 law_name：
      - 书名号：《中华人民共和国民法典》
      - 日期后缀：中华人民共和国劳动法_20181229
      - 版本说明：宪法（2018年修正文本）
    """
    stem = path.stem.replace("《", "").replace("》", "")
    stem = _DATE_SUFFIX.sub("", stem)
    stem = _VERSION_SUFFIX.sub("", stem)
    stem = re.sub(r"\s*[（(]\d+[）)]\s*$", "", stem)     # 民法典(1)
    stem = re.sub(r"\s*-\s*副本\s*$", "", stem)
    return stem.strip()


def parse_version_date(path: Path) -> str:
    """从文件名解析版本日期，返回 YYYY-MM-DD；没有则空串。

    法律有时效性：同一部法律不同版本的条文可能完全相反。文件名里的
    8 位日期（如 民法典_20200528）是最容易拿到的版本信号。
    """
    match = _DATE_SUFFIX.search(path.stem)
    if not match:
        return ""
    raw = match.group(1)
    try:
        year, month, day = raw[:4], raw[4:6], raw[6:]
        if not (1900 <= int(year) <= 2100 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31):
            return ""
        return f"{year}-{month}-{day}"
    except ValueError:
        return ""


def infer_doc_type(law_name: str) -> str:
    for keywords, doc_type in _DOC_TYPE_RULES:
        if any(keyword in law_name for keyword in keywords):
            return doc_type
    return "法律"


def ingest_one(path: Path) -> list[Article]:
    paragraphs = read_any(path)
    name = law_name_from_path(path)
    return parse_articles(
        paragraphs,
        law_name=name,
        doc_type=infer_doc_type(name),
        effective_date="",
        version_date=parse_version_date(path),
        source=path.name,
    )


def article_report(articles: list[Article]) -> str:
    numbers = [a.article_no for a in articles]
    if not numbers:
        return "0 条"
    expected = set(range(1, max(numbers) + 1))
    missing = sorted(expected - set(numbers))
    duplicated = sorted({n for n in numbers if numbers.count(n) > 1})
    parts = [f"{len(articles)} 条（第{min(numbers)}-{max(numbers)}条"]
    if missing:
        parts.append(f"缺 {len(missing)} 号: {missing[:8]}{'…' if len(missing) > 8 else ''}")
    if duplicated:
        parts.append(f"重号: {duplicated[:8]}")
    if not missing and not duplicated:
        parts.append("无缺号")
    return "，".join(parts) + "）"


def main() -> int:
    parser = argparse.ArgumentParser(description="批量导入法律文献")
    parser.add_argument("--dir", default="data/laws")
    parser.add_argument("--out", default="data/corpus/all.jsonl")
    parser.add_argument("--index", default="data/index/all")
    parser.add_argument(
        "--embedder", default="auto",
        choices=["auto", "hash", "dashscope", "dashscope_native"],
    )
    parser.add_argument("--full", action="store_true", help="强制全量重建索引（忽略已有向量）")
    args = parser.parse_args()

    load_dotenv()          # 云端 embedding 需要 LEGALQA_LLM_API_KEY
    source_dir = ROOT / args.dir
    if not source_dir.exists():
        print(f"[x] 目录不存在：{source_dir}", file=sys.stderr)
        return 2

    raw_files = sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    if not raw_files:
        print(f"[!] {source_dir.relative_to(ROOT)} 下没有 docx / pdf 文件")
        print("    把法律文件放进去，文件名用法律全称，例如「中华人民共和国民法典.docx」")
        return 1

    # 同名法律去重：docx 优先。
    # 实测 PDF 提取会有「假条」（正文里的条号引用折行到行首），
    # 同一部法律 docx 87 条 / pdf 105 条（含 18 条假条）。docx 质量明显更好。
    grouped: dict[str, list[Path]] = {}
    for path in raw_files:
        grouped.setdefault(law_name_from_path(path), []).append(path)

    # 同一部法律的多个版本：**按版本日期从新到旧依次尝试，取第一个能解析的**。
    #
    # 不能简单地「先按日期选，再解析」：实测宪法同时有 docx 和一份 2018 年
    # 的扫描版 PDF，按日期选会挑中 PDF，而扫描件提取不出文字，导致整部宪法
    # 从语料里消失，而且只在日志里留一行「扫描版，跳过」。
    #
    # 也不能简单地「docx 优先」：法律修订后旧版 docx 会覆盖新版 pdf，
    # 拿旧条文回答比答不出来更危险。
    #
    # 正确顺序：日期新 → 日期旧，同日期 docx 优先；解析失败就试下一个。
    all_articles: list[Article] = []
    slugs: dict[str, str] = {}
    failed: list[tuple[str, str]] = []

    def _rank(path: Path) -> tuple[str, int]:
        is_docx = 1 if path.suffix.lower() in {".docx", ".docm"} else 0
        return (parse_version_date(path), is_docx)

    for law, paths in grouped.items():
        ordered = sorted(paths, key=_rank, reverse=True)
        chosen: tuple[Path, list[Article]] | None = None
        attempts: list[tuple[Path, str]] = []

        for candidate in ordered:
            try:
                articles = ingest_one(candidate)
            except ScannedPdfError as exc:
                attempts.append((candidate, f"扫描版 PDF（{exc}）"))
                continue
            except Exception as exc:                  # noqa: BLE001 - 单份失败不影响整体
                attempts.append((candidate, f"{type(exc).__name__}: {exc}"))
                continue
            if not articles:
                attempts.append((candidate, "解析出 0 条"))
                continue
            chosen = (candidate, articles)
            break

        if chosen is None:
            for candidate, reason in attempts:
                print(f"  [x] {candidate.name}")
                print(f"      {reason}")
                failed.append((candidate.name, reason))
            continue

        path, articles = chosen
        for candidate, reason in attempts:
            print(f"  [-] {candidate.name} 跳过（{reason}）")
        # 用上了较低优先级的候选时，说明更优的那个坏了，要显式提示
        if attempts and path != ordered[0]:
            print(f"  [!] 已回退到 {path.name}")

        slug = articles[0].id.rsplit("-", 1)[0]
        if slug in slugs and slugs[slug] != law:
            print(f"  [!] id 前缀冲突：{law} 与 {slugs[slug]} 都归一到「{slug}」")
        slugs[slug] = law

        version = parse_version_date(path)
        print(f"  [+] {path.name}")
        print(
            f"      → {law}（{infer_doc_type(law)}，{version or '无版本日期'}）"
            f"{article_report(articles)}"
        )
        all_articles.extend(articles)

    if not all_articles:
        print("\n[x] 没有成功导入任何条文", file=sys.stderr)
        return 3

    out_path = ROOT / args.out
    count = write_jsonl(all_articles, out_path)
    laws = sorted({a.law_name for a in all_articles})
    # --out 允许指向仓库外（例如 D:/pip-cache/tmp），不能假设一定在 ROOT 下
    try:
        out_shown = out_path.relative_to(ROOT)
    except ValueError:
        out_shown = out_path
    print(f"\n[+] 合计 {count} 条，{len(laws)} 部法律 → {out_shown}")
    for law in laws:
        n = sum(1 for a in all_articles if a.law_name == law)
        print(f"      {law}: {n} 条")

    # ---- 重建索引 ----
    if args.embedder == "hash":
        from legalqa.llm_factory import HashEmbeddings

        embedder = HashEmbeddings()
        print("\n[!] 使用 HashEmbeddings：只验证链路，检索质量不代表真实效果")
    else:
        if args.embedder == "dashscope":
            os.environ["LEGALQA_EMBEDDING_PROVIDER"] = "dashscope"
            os.environ.setdefault("LEGALQA_EMBEDDING_MODEL", "text-embedding-v4")
        elif args.embedder == "dashscope_native":
            os.environ["LEGALQA_EMBEDDING_PROVIDER"] = "dashscope_native"
            os.environ.setdefault(
                "LEGALQA_EMBEDDING_MODEL", "qwen3.7-text-embedding"
            )
        from legalqa.llm_factory import get_embeddings

        embedder = get_embeddings()

    docs = [a.to_dict() for a in all_articles]
    index_path = ROOT / args.index

    # 增量：复用未变化条文的旧向量。全量重建 4000 条要十几分钟，
    # 而用户每加一部法律往往只新增几十到上千条。
    previous = None
    if not args.full and (index_path / "vectors.npy").exists():
        previous = VectorIndex.load(index_path)
        probe_dim = len(embedder.embed_query("维度探测"))
        if previous.dim != probe_dim:
            print(
                f"\n[!] 已有索引维度 {previous.dim} ≠ 当前模型 {probe_dim}"
                "（可能换了 embedding 模型），改为全量重建"
            )
            previous = None

    if previous is not None:
        print(f"\n[i] 增量索引：已有 {len(previous)} 条，目标 {len(docs)} 条")
    else:
        print(f"\n[i] 全量向量化 {len(docs)} 条（分批 32 条，避免单次请求打爆内存）")

    def _progress(done: int, total: int) -> None:
        if done == total or done % 320 == 0:
            print(f"      向量化 {done}/{total}")

    index, stats = VectorIndex.build_incremental(
        docs, embedder, previous, batch_size=32, progress=_progress
    )
    index.save(
        index_path,
        meta={
            "embedder_provider": os.environ.get("LEGALQA_EMBEDDING_PROVIDER", ""),
            "embedder_model": os.environ.get("LEGALQA_EMBEDDING_MODEL", ""),
            "embedder_dimension": index.dim,
        },
    )
    if previous is not None:
        print(
            f"[+] 复用 {stats.reused} 条，新增向量化 {stats.embedded} 条"
            f"（丢弃 {stats.removed} 条）"
        )
    print(f"[+] 索引 {len(index)} 条 × {index.dim} 维 → {index_path.relative_to(ROOT)}")

    if failed:
        print(f"\n[!] {len(failed)} 个文件未导入：")
        for name, reason in failed:
            print(f"      {name} —— {reason}")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
