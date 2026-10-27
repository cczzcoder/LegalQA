"""多轮对话评测：每个用例是一组连续提问，逐轮跑检查。

为什么需要：单轮评测覆盖不了「指代消解（那…）、承前引用、追问补充、
话题切换」——这些正是长对话产品的核心场景，也是会话落库后最容易被
用户打到的路径。

用法：
    python scripts/run_multiturn_eval.py                        # 全部用例
    python scripts/run_multiturn_eval.py --limit 2              # 先跑前 2 组
    python scripts/run_multiturn_eval.py --json out.json        # 存明细
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows 的 GBK 管道控制台打印 ✓/✗ 会 UnicodeEncodeError，统一 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from legalqa.cli import build_embedder, build_llm, load_dotenv  # noqa: E402
from legalqa.corpus.index import VectorIndex  # noqa: E402
from legalqa.eval import (  # noqa: E402
    check_disclaimer,
    check_expected_citation,
    check_hallucination,
    check_length,
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


def run_turn(app, index, history: list, turn: dict, turn_id: int) -> dict:
    started = time.perf_counter()
    messages = [*history, HumanMessage(content=turn["query"], id=f"h-{turn_id}-{len(history)}")]
    state = coerce_state(app.invoke({"messages": messages, "turn_id": turn_id}))
    latency = int((time.perf_counter() - started) * 1000)

    answer = extract_answer(state.messages)
    checked = strip_backfill_appendix(answer)
    all_docs = list(state.retrieved_docs or [])
    checks = [
        check_disclaimer(checked),
        check_length(checked),
        check_traceability(checked, all_docs),
        check_hallucination(checked, index),
        check_refusal(checked, bool(turn.get("should_answer", True))),
    ]
    if turn.get("expect_law") or turn.get("accept_articles"):
        checks.append(
            check_expected_citation(
                checked,
                turn.get("expect_law"),
                turn.get("expect_article_no"),
                turn.get("accept_articles"),
            )
        )
    return {
        "query": turn["query"],
        "passed": all(check.passed for check in checks),
        "latency_ms": latency,
        "iterations": state.iteration_count,
        "channel": state.retrieval_channel,
        "answer": answer,
        "retrieved": [
            {"law": doc.get("law_name", ""), "no": doc.get("article_no")}
            for doc in all_docs
            if doc.get("retrieval_source") != "backfill"
        ],
        "checks": [
            {"name": check.name, "passed": check.passed, "detail": check.detail}
            for check in checks
        ],
    }


def evaluate_case(app, index, case: dict) -> dict:
    history: list = []
    turns = []
    for turn_id, turn in enumerate(case["turns"]):
        result = run_turn(app, index, history, turn, turn_id)
        turns.append(result)
        # 与 Web 服务同构的 id 规则：下一轮的历史由「用户 + 助手答案」构成
        history.append(HumanMessage(content=turn["query"], id=f"h-{turn_id}-{turn_id}"))
        history.append(AIMessage(content=result["answer"], id=f"a-{turn_id}-{turn_id}"))
    return {
        "id": case["id"],
        "passed": all(turn["passed"] for turn in turns),
        "turns": turns,
        "latency_ms": sum(turn["latency_ms"] for turn in turns),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="跑多轮对话评测集")
    parser.add_argument("--cases", default="data/eval/multiturn.jsonl")
    parser.add_argument("--index", default="data/index/all")
    parser.add_argument("--embedder", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    load_dotenv()
    cases = load_cases(ROOT / args.cases)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("[x] 没有可跑的多轮用例", file=sys.stderr)
        return 2

    embedder = build_embedder(args.embedder)
    index = VectorIndex.load(ROOT / args.index)
    llm = build_llm(argparse.Namespace(dry_run=False, provider=None, model=None))
    read_tool = ReadTool(index, embedder)
    app = build_graph(llm=llm, read_tool=read_tool)

    turn_count = sum(len(case["turns"]) for case in cases)
    print(f"[i] 对话 {len(cases)} 组 / 轮次 {turn_count} | 索引 {len(index)} 条\n")

    started = time.perf_counter()
    workers = max(1, args.workers)
    if workers == 1:
        reports = [evaluate_case(app, index, case) for case in cases]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            reports = list(pool.map(lambda c: evaluate_case(app, index, c), cases))
    total_time = time.perf_counter() - started

    for report in reports:
        marks = " ".join("✓" if turn["passed"] else "✗" for turn in report["turns"])
        print(f"  {report['id']:<10} [{marks}]  {report['latency_ms']/1000:>5.1f}s")
        for turn_id, turn in enumerate(report["turns"]):
            if not turn["passed"]:
                print(f"      T{turn_id + 1} ✗ {turn['query'][:30]}")
                for check in turn["checks"]:
                    if not check["passed"]:
                        print(f"          · {check['name']}: {check['detail']}")

    passed_turns = sum(1 for report in reports for turn in report["turns"] if turn["passed"])
    passed_cases = sum(1 for report in reports if report["passed"])
    print(f"\n{'='*60}")
    print(f"轮次通过 {passed_turns}/{turn_count} | 对话通过 {passed_cases}/{len(reports)}")
    print(f"耗时 {total_time:.0f}s")

    by_name: dict[str, list[bool]] = {}
    for report in reports:
        for turn in report["turns"]:
            for check in turn["checks"]:
                by_name.setdefault(check["name"], []).append(check["passed"])
    print("分项通过率:")
    for name, results in by_name.items():
        ok = sum(results)
        print(f"  {name:<12} {ok:>3}/{len(results):<3} {ok/len(results)*100:>6.1f}%")

    if args.json:
        payload = [
            {
                "id": report["id"],
                "passed": report["passed"],
                "latency_ms": report["latency_ms"],
                "turns": report["turns"],
            }
            for report in reports
        ]
        with Path(args.json).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\n[i] 明细已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
