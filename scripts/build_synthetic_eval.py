"""从法条反向生成评测问题（合成检索评测集）。

思路：给一条法条，让模型扮演当事人提问，然后看检索能不能找回这条。

为什么值得做：
  - 手工用例只有几十条，覆盖不了 4000+ 条法条
  - 合成用例有**确定的标准答案**（原条文），不依赖人工标注
  - 只调一次模型生成问题，之后测检索零成本，可以反复跑

**关键约束：问题必须像普通人说话，不能照抄法条用词。**
照抄的话检索当然能命中，测出来的是「向量模型会不会复制粘贴」，
不是「能不能跨越口语和法条的词汇鸿沟」。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage  # noqa: E402

from legalqa.cli import build_llm, load_dotenv  # noqa: E402
from legalqa.corpus.ingest import load_jsonl  # noqa: E402

PROMPT = """你是普通老百姓，遇到了下面这条法律规定所涉及的情况。
用你自己的话问一个问题，20~40 字，像在咨询律师。

严格要求：
1. **不要照抄法条用词**，不要出现「本法」「前款」「依照」这类法条语言
2. 用生活化的说法（比如「老板」而不是「用人单位」，「赔钱」而不是「承担赔偿责任」）
3. 只输出问题本身，不要解释，不要加引号
4. 不要把条号写进问题里

法律规定（出自《{law}》{label}）：
{text}

你的问题："""


def sample_stratified(docs: list[dict], total: int, min_per_law: int = 3) -> list[dict]:
    """按法律分层抽样，保证每部法律都覆盖到。

    纯随机抽样会让生态环境法典（1242 条）占掉大半，其他法律只剩一两条，
    测出来的是「大法检得好不好」，不是「整体检得好不好」。
    """
    by_law: dict[str, list[dict]] = {}
    for doc in docs:
        by_law.setdefault(str(doc.get("law_name", "")), []).append(doc)

    rng = random.Random(20261026)
    quota: dict[str, int] = {}
    for law, items in by_law.items():
        quota[law] = max(min_per_law, round(total * len(items) / len(docs)))

    # 按比例分配后可能超出 total，从配额最大的往下削
    while sum(quota.values()) > total:
        biggest = max(quota, key=lambda k: quota[k])
        if quota[biggest] <= min_per_law:
            break
        quota[biggest] -= 1

    picked: list[dict] = []
    for law, items in by_law.items():
        take = min(quota[law], len(items))
        picked.extend(rng.sample(items, take))
    rng.shuffle(picked)
    return picked


def generate_question(llm, doc: dict) -> str | None:
    prompt = PROMPT.format(
        law=str(doc.get("law_name", "")).replace("中华人民共和国", ""),
        label=doc.get("article_label", ""),
        text=str(doc.get("text", ""))[:500],
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = str(getattr(response, "content", "") or "").strip()
    except Exception:
        return None
    text = text.strip().strip('"').strip("「」").strip()
    text = text.split("\n")[0].strip()
    if not (8 <= len(text) <= 120):
        return None
    # 问题里带条号会让检索走精确通道，测不出语义能力
    if "第" in text and "条" in text:
        return None
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="生成合成检索评测集")
    parser.add_argument("--corpus", default="data/corpus/all.jsonl")
    parser.add_argument("--out", default="data/eval/synthetic.jsonl")
    parser.add_argument("--total", type=int, default=120)
    args = parser.parse_args()

    load_dotenv()
    docs = load_jsonl(ROOT / args.corpus)
    if not docs:
        print("[x] 语料为空", file=sys.stderr)
        return 2

    sample = sample_stratified(docs, args.total)
    print(f"[i] 语料 {len(docs)} 条，分层抽样 {len(sample)} 条")
    print(f"    覆盖 {len({d['law_name'] for d in sample})} 部法律\n")

    llm = build_llm(argparse.Namespace(dry_run=False, provider=None, model=None))

    rows: list[dict] = []
    for i, doc in enumerate(sample, 1):
        question = generate_question(llm, doc)
        if question is None:
            continue
        rows.append(
            {
                "query": question,
                "law": doc["law_name"],
                "article_no": doc["article_no"],
                "source_text": str(doc["text"])[:120],
            }
        )
        if i % 20 == 0:
            print(f"    已生成 {len(rows)}/{i}")

    if not rows:
        print("[x] 一条都没生成出来", file=sys.stderr)
        return 3

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[+] 生成 {len(rows)} 条 → {out.relative_to(ROOT)}")
    counts = Counter(row["law"].replace("中华人民共和国", "") for row in rows)
    for law, n in counts.most_common():
        print(f"      {law}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
