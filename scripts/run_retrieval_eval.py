"""检索层评测：合成问题能不能找回原条文。

用法：
    python scripts/run_retrieval_eval.py
    python scripts/run_retrieval_eval.py --topk 6 --show-failures 10

指标：
    Recall@1   第一名是不是正确条文（最严）
    Recall@k   正确条文在不在前 k 条（k = RAG_TOP_K，决定模型能否看到）
    MRR        正确条文排名的倒数均值（衡量排序质量）

为什么单独测检索而不跑整图：
    整图跑一条 10 秒，检索只要 0.1 秒。同一个问题，
    「检索没找到」和「模型答错了」是两种完全不同的故障，
    混在端到端指标里分不清。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legalqa.cli import build_embedder, load_dotenv  # noqa: E402
from legalqa.corpus.index import VectorIndex  # noqa: E402
from legalqa.tools.read_tool import ReadTool  # noqa: E402


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _norm_law(name: str) -> str:
    return str(name or "").replace("中华人民共和国", "").strip()


def matches(doc: dict, case: dict) -> bool:
    """条号相同 **且** 是同一部法律。

    不能用「包含」判断：`劳动合同法` 是 `劳动合同法实施条例` 的子串，
    用包含会把实施条例的同号条文误判成命中，虚高指标。
    实测有 3 组这样的父子法名（劳动合同法/实施条例、药品管理法/实施条例、
    矿产资源法/实施条例），必须先规范化再精确比较。
    """
    if doc.get("article_no") != case.get("article_no"):
        return False
    return _norm_law(doc.get("law_name", "")) == _norm_law(case.get("law", ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="检索层评测")
    parser.add_argument("--cases", default="data/eval/synthetic.jsonl")
    parser.add_argument("--index", default="data/index/all")
    parser.add_argument("--embedder", default="auto", choices=["auto", "hash", "ollama", "dashscope", "hf"],
                        help="auto=跟随 .env（必须和建索引时一致，否则检索全 0）")
    parser.add_argument("--topk", type=int, default=0, help="默认用 config.RAG_TOP_K")
    parser.add_argument("--show-failures", type=int, default=8)
    parser.add_argument("--rerank", action="store_true", help="开启 LLM 重排")
    parser.add_argument("--rewrite", action="store_true", help="开启检索前 LLM 改写")
    args = parser.parse_args()

    load_dotenv()
    cases = load_cases(ROOT / args.cases)
    if not cases:
        print("[x] 评测集为空，先跑 scripts/build_synthetic_eval.py", file=sys.stderr)
        return 2

    embedder = build_embedder(args.embedder)
    index = VectorIndex.load(ROOT / args.index)

    reranker = rewriter = None
    if args.rerank or args.rewrite:
        from legalqa.cli import build_llm

        llm = build_llm(argparse.Namespace(dry_run=False, provider=None, model=None))
        if args.rerank:
            from legalqa.tools.reranker import make_llm_reranker

            reranker = make_llm_reranker(llm)
        if args.rewrite:
            from legalqa.tools.rewriter import make_llm_rewriter

            rewriter = make_llm_rewriter(llm)

    tool = ReadTool(
        index, embedder, reranker=reranker, rewriter=rewriter, always_expand=bool(rewriter)
    )

    k = args.topk or 6
    hit1 = hitk = 0
    mrr = 0.0
    per_law: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "h1": 0, "hk": 0})
    failures: list[dict] = []
    started = time.perf_counter()

    for case in cases:
        result = tool.run(case["query"])
        docs = result.docs
        rank = next((i + 1 for i, d in enumerate(docs) if matches(d, case)), None)

        law = str(case.get("law", "")).replace("中华人民共和国", "")
        per_law[law]["n"] += 1
        if rank == 1:
            hit1 += 1
            per_law[law]["h1"] += 1
        if rank is not None and rank <= k:
            hitk += 1
            per_law[law]["hk"] += 1
        if rank:
            mrr += 1.0 / rank
        else:
            failures.append(
                {
                    "query": case["query"],
                    "expect": f"{law} 第{case['article_no']}条",
                    "got": (docs[0].get("law_name", "").replace("中华人民共和国", "")
                            + f" 第{docs[0].get('article_no')}条") if docs else "无结果",
                    "score": round(result.score, 3),
                }
            )

    total = len(cases)
    elapsed = time.perf_counter() - started
    print(f"=== 检索评测 ===")
    print(f"用例 {total} 条 | 索引 {len(index)} 条 | 耗时 {elapsed:.1f}s "
          f"({elapsed/total*1000:.0f}ms/条)\n")
    print(f"  Recall@1   {hit1}/{total}  {hit1/total*100:>5.1f}%")
    print(f"  Recall@{k}   {hitk}/{total}  {hitk/total*100:>5.1f}%")
    print(f"  MRR        {mrr/total:.3f}")

    print(f"\n按法律:")
    for law, stat in sorted(per_law.items(), key=lambda kv: -kv[1]["n"]):
        print(
            f"  {law:<16} {stat['n']:>3} 条  "
            f"R@1 {stat['h1']/stat['n']*100:>5.1f}%  "
            f"R@{k} {stat['hk']/stat['n']*100:>5.1f}%"
        )

    if failures and args.show_failures:
        print(f"\n未命中 {len(failures)} 条，前 {min(args.show_failures, len(failures))} 条:")
        for item in failures[: args.show_failures]:
            print(f"  问：{item['query']}")
            print(f"    期望 {item['expect']}  实际 {item['got']}  (score {item['score']})")

    return 0 if hitk == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
