"""构建法条语料与向量索引。

用法：
    python scripts/build_corpus.py ^
        --src "C:/path/中华人民共和国宪法.docx" ^
        --law "中华人民共和国宪法" ^
        --doc-type 宪法 ^
        --effective 2018-03-11

离线环境（无 bge 模型）加 `--embedder hash`，只验证链路，不验证语义。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legalqa.corpus.index import VectorIndex  # noqa: E402
from legalqa.corpus.ingest import (  # noqa: E402
    load_jsonl,
    parse_articles,
    read_docx_paragraphs,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="构建法条语料与索引")
    parser.add_argument("--src", required=True, help="docx 源文件路径")
    parser.add_argument("--law", required=True, help="法律全称")
    parser.add_argument("--doc-type", default="法律", help="效力层级")
    parser.add_argument("--effective", default="", help="施行日期 YYYY-MM-DD")
    parser.add_argument("--out", default="data/corpus", help="语料输出目录")
    parser.add_argument("--index-dir", default="data/index", help="索引输出目录")
    parser.add_argument(
        "--embedder", default="auto", choices=["auto", "hash"],
        help="auto=按 LEGALQA_EMBEDDING_PROVIDER；hash=确定性哈希（离线验证链路）",
    )
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"[x] 源文件不存在: {src}", file=sys.stderr)
        return 2

    paragraphs = read_docx_paragraphs(src)
    articles = parse_articles(
        paragraphs,
        law_name=args.law,
        doc_type=args.doc_type,
        effective_date=args.effective,
        source=src.name,
    )
    if not articles:
        print("[x] 未解析出任何条文，检查文档格式", file=sys.stderr)
        return 3

    numbers = [a.article_no for a in articles]
    missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
    duplicated = sorted({n for n in numbers if numbers.count(n) > 1})

    out_dir = ROOT / args.out
    out_file = out_dir / f"{articles[0].id.rsplit('-', 1)[0]}.jsonl"
    count = write_jsonl(articles, out_file)

    print(f"[+] 段落 {len(paragraphs)} → 条文 {count}")
    print(f"[+] 条号范围 {min(numbers)}-{max(numbers)}")
    if missing:
        print(f"[!] 缺失条号: {missing[:20]}")
    if duplicated:
        print(f"[!] 重复条号: {duplicated[:20]}")
    print(f"[+] 语料写入 {out_file.relative_to(ROOT)}")

    # ---- 索引 ----
    if args.embedder == "hash":
        from legalqa.llm_factory import HashEmbeddings

        embedder = HashEmbeddings()
        print("[!] 使用 HashEmbeddings：只验证链路，检索质量不代表真实效果")
    else:
        from legalqa.llm_factory import get_embeddings

        embedder = get_embeddings()

    docs = load_jsonl(out_file)
    index = VectorIndex.build(docs, embedder)
    index_dir = ROOT / args.index_dir / out_file.stem
    index.save(index_dir)
    print(f"[+] 索引 {len(index)} 条 × {index.dim} 维 → {index_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
