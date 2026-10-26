"""跑评测集，输出可对比的指标。

用法：
    python scripts/run_eval.py                    # 全部用例
    python scripts/run_eval.py --limit 5          # 先跑 5 条看看
    python scripts/run_eval.py --category labor   # 只跑劳动类
    python scripts/run_eval.py --json out.json    # 存明细

为什么要这个：改了这么多轮，如果没有客观指标，「改好了还是改坏了」只能靠
肉眼看几条 case。而肉眼会骗人——你会记得改好的那条，忘掉改坏的三条。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows 的 GBK 管道控制台打印 ✓/✗ 会直接 UnicodeEncodeError 中断评测。
# 统一按 UTF-8 输出，Git Bash / CI 都能读。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import HumanMessage  # noqa: E402

from legalqa.cli import build_embedder, build_llm, load_dotenv  # noqa: E402
from legalqa.corpus.index import VectorIndex  # noqa: E402
from legalqa.eval import (  # noqa: E402
    CaseReport,
    check_disclaimer,
    check_hallucination,
    check_keywords,
    check_length,
    check_expected_citation,
    check_refusal,
    check_traceability,
    extract_answer,
    strip_backfill_appendix,
)
from legalqa.graph import build_graph  # noqa: E402
from legalqa.state import coerce_state  # noqa: E402
from legalqa.tools.read_tool import ReadTool  # noqa: E402


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_one(app, index, case: dict) -> CaseReport:
    started = time.perf_counter()
    state = coerce_state(
        app.invoke({"messages": [HumanMessage(content=case["query"])], "turn_id": 0})
    )
    latency = int((time.perf_counter() - started) * 1000)

    answer = extract_answer(state.messages)
    # 所有基于文本的检查都只看模型自己写的正文：回捞附录里是法条原文，
    # 里面的交叉引用（「依照本法第三十八条」）不是模型的引用行为，
    # 算进去会既冤枉系统、又可能让没引对的回答蒙对。
    checked = strip_backfill_appendix(answer)
    # OutputGuard 会把「按条号回查到的」条文并进 retrieved_docs，并打上
    # retrieval_source=backfill。这里拆成两份：
    #   - all_docs：检索 + 回捞核验，用于「引用可溯源」的通过判定
    #     （回捞已经确认条文存在，不该因为没进 top-6 就判失败）
    #   - raw_docs：只有工具真正检索到的，用于「检索覆盖」与失败诊断
    # 混在一起会让诊断把「回捞到的」误认为「检索到了」。
    all_docs = list(state.retrieved_docs or [])
    raw_docs = [d for d in all_docs if d.get("retrieval_source") != "backfill"]
    backfilled_docs = [d for d in all_docs if d.get("retrieval_source") == "backfill"]

    def _brief(doc: dict) -> dict:
        return {
            "law": doc.get("law_name", ""),
            "no": doc.get("article_no"),
            "valid": doc.get("is_valid", True),
        }

    report = CaseReport(
        case_id=case["id"],
        query=case["query"],
        latency_ms=latency,
        iterations=state.iteration_count,
        channel=state.retrieval_channel,
        score=state.rag_score,
        answer=answer,
        retrieved=[_brief(d) for d in raw_docs],
        backfilled=[_brief(d) for d in backfilled_docs],
        retrieval_coverage=check_traceability(checked, raw_docs),
    )

    report.checks.append(check_disclaimer(checked))
    report.checks.append(check_length(checked))
    report.checks.append(check_traceability(checked, all_docs))
    report.checks.append(check_hallucination(checked, index))
    report.checks.append(check_refusal(checked, bool(case.get("should_answer"))))
    if case.get("should_answer") and case.get("must_mention"):
        report.checks.append(check_keywords(checked, case["must_mention"]))
    if case.get("expect_law") or case.get("accept_articles"):
        report.checks.append(
            check_expected_citation(
                checked,
                case.get("expect_law"),
                case.get("expect_article_no"),
                case.get("accept_articles"),
            )
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="跑法律问答评测集")
    parser.add_argument("--cases", default="data/eval/cases.jsonl")
    parser.add_argument("--index", default="data/index/all")
    parser.add_argument("--embedder", default="auto", choices=["auto", "hash", "ollama", "dashscope", "hf"],
                        help="auto=跟随 .env（必须和建索引时一致，否则检索全 0）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条")
    parser.add_argument("--category", default="", help="只跑某一类")
    parser.add_argument("--json", default="", help="把明细写到这个文件")
    parser.add_argument("--rerank", action="store_true", help="开启 LLM 重排")
    parser.add_argument("--rewrite", action="store_true", help="开启检索前 LLM 改写")
    parser.add_argument(
        "--supplement-user-query",
        action="store_true",
        help="工具检索后用用户原问题再检索一次，补最多 2 条模型 query 没覆盖的条文",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="并发数。每条用例 10 秒左右，串行跑 100 条要 20 分钟，并行能把评测压到可迭代的长度",
    )
    args = parser.parse_args()

    load_dotenv()
    cases = load_cases(ROOT / args.cases)
    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("[x] 没有可跑的用例", file=sys.stderr)
        return 2

    embedder = build_embedder(args.embedder)
    index = VectorIndex.load(ROOT / args.index)
    llm = build_llm(argparse.Namespace(dry_run=False, provider=None, model=None))

    # 让实验可复现：E8 的 rerank 对比最初是临时脚本跑的，这里补上开关
    reranker = None
    if args.rerank:
        from legalqa.tools.reranker import make_llm_reranker

        reranker = make_llm_reranker(llm)
    rewriter = None
    if args.rewrite:
        from legalqa.tools.rewriter import make_llm_rewriter

        rewriter = make_llm_rewriter(llm)

    read_tool = ReadTool(
        index, embedder, reranker=reranker, rewriter=rewriter, always_expand=bool(rewriter)
    )
    app = build_graph(
        llm=llm,
        read_tool=read_tool,
        supplement_user_query=args.supplement_user_query,
    )

    print(f"[i] 用例 {len(cases)} 条 | 索引 {len(index)} 条 | {len(index.law_names)} 部法律\n")

    started = time.perf_counter()
    workers = max(1, args.workers)

    if workers == 1:
        reports = []
        for case in cases:
            report = evaluate_one(app, index, case)
            reports.append(report)
            mark = "✓" if report.passed else "✗"
            print(
                f"  {mark} {report.case_id:<14} {report.latency_ms/1000:>5.1f}s "
                f"轮次{report.iterations} {report.channel or '-':<6} {case['query'][:24]}"
            )
    else:
        # 用例之间没有共享状态，`app.invoke` 每次都是独立的一次图执行，
        # 所以可以并发。单条 10 秒、100 条串行 20 分钟，串行跑不动迭代。
        with ThreadPoolExecutor(max_workers=workers) as pool:
            reports = list(pool.map(lambda c: evaluate_one(app, index, c), cases))
        for report in reports:
            mark = "✓" if report.passed else "✗"
            print(
                f"  {mark} {report.case_id:<14} {report.latency_ms/1000:>5.1f}s "
                f"轮次{report.iterations} {report.channel or '-':<6} {report.query[:24]}"
            )

    total_time = time.perf_counter() - started
    passed = sum(1 for r in reports if r.passed)
    rate = passed / len(reports) * 100

    print(f"\n{'='*60}")
    print(f"通过 {passed}/{len(reports)}  ({rate:.1f}%)")
    print(
        f"耗时 {total_time:.0f}s | 平均 {total_time/len(reports):.1f}s/条 | "
        f"平均轮次 {sum(r.iterations for r in reports)/len(reports):.1f}"
    )

    # 分项通过率
    by_name: dict[str, list[bool]] = {}
    for report in reports:
        for check in report.checks:
            by_name.setdefault(check.name, []).append(check.passed)
    print("\n分项通过率:")
    for name, results in by_name.items():
        ok = sum(results)
        print(f"  {name:<12} {ok:>3}/{len(results):<3} {ok/len(results)*100:>6.1f}%")

    # 检索覆盖与「引用可溯源」是两个指标：前者只看原始检索，后者允许回捞核验。
    # 不参与通过判定，单独打印，避免把「没召回但回查确认存在」算成检索成功。
    covered = sum(
        1 for r in reports if r.retrieval_coverage and r.retrieval_coverage.passed
    )
    print(
        f"  {'检索覆盖':<12} {covered:>3}/{len(reports):<3} "
        f"{covered/len(reports)*100:>6.1f}%  （仅原始检索，不参与通过判定）"
    )

    failures = [r for r in reports if not r.passed]
    if failures:
        print(f"\n失败用例 {len(failures)} 条:")
        for report in failures:
            print(f"  ✗ {report.case_id}  {report.query}")
            for check in report.failures:
                print(f"      · {check.name}: {check.detail}")

    if args.json:
        payload = [
            {
                "id": r.case_id,
                "query": r.query,
                "passed": r.passed,
                "latency_ms": r.latency_ms,
                "iterations": r.iterations,
                "channel": r.channel,
                "score": round(r.score, 4),
                "answer": r.answer,
                "retrieved": r.retrieved,
                "backfilled": r.backfilled,
                "retrieval_coverage": (
                    {
                        "passed": r.retrieval_coverage.passed,
                        "detail": r.retrieval_coverage.detail,
                    }
                    if r.retrieval_coverage
                    else None
                ),
                "checks": [
                    {"name": c.name, "passed": c.passed, "detail": c.detail}
                    for c in r.checks
                ],
            }
            for r in reports
        ]
        (ROOT / args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[i] 明细已写入 {args.json}")

    return 0 if passed == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
