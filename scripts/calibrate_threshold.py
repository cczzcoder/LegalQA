"""用 golden set 校准检索阈值。

用法：
    python scripts/calibrate_threshold.py --golden data/golden/constitution.jsonl \
        --index data/index/constitution

为什么要单独跑这个：`RAG_THRESHOLD` 是**唯一**决定「答还是拒答」的开关。
凭感觉设一个 0.7，结果就是「国旗是什么」(0.662) 被拒、「国家主席任期」(0.688)
被拒，而正确答案就在 top-1 上。阈值不是超参，是需要用数据定的业务指标。

判据：F1 最高点，且优先选**平坦区**的中部 —— 单点最优往往过拟合小样本。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legalqa.corpus.index import VectorIndex  # noqa: E402
from legalqa.env import load_dotenv  # noqa: E402
from legalqa.llm_factory import get_embeddings  # noqa: E402
from legalqa.tools.read_tool import ReadTool  # noqa: E402


def load_golden(path: Path) -> list[dict]:
    """读取 golden set。

    多法源下必须同时标注 `law`：劳动法第 50 条和商标法第 50 条是两条完全
    不同的法条，只比条号会把「检索到错的法律」误判成命中。
    """
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def answers_of(row: dict) -> list[tuple[str, int]]:
    """取出该题的所有可接受答案。

    多法源下同一个问题常常在多部法律里都有对应条文，例如「公司拖欠工资」
    在劳动法第五十条和劳动合同法第三十条都规定了。只允许一个答案会把
    另一条正确的检索结果判成「错条」，人为压低召回率。

    支持两种写法：
      {"law": "...", "article_no": 50}                      单答案
      {"answers": [["法A", 50], ["法B", 30]]}                多答案
    """
    if "answers" in row:
        return [(str(law or ""), int(no)) for law, no in row["answers"]]
    if row.get("article_no") is None:
        return []
    return [(str(row.get("law") or ""), int(row["article_no"]))]


def is_relevant(row: dict) -> bool:
    return bool(answers_of(row))


def _is_hit(row: dict, top: dict | None) -> bool:
    """判定一条检索结果是否命中任一可接受答案。"""
    if top is None:
        return False
    actual_no = top.get("article_no")
    actual_law = str(top.get("law_name", ""))
    for law, no in answers_of(row):
        if actual_no != no:
            continue
        if not law:
            return True
        if law in actual_law or actual_law in law:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="校准检索阈值")
    parser.add_argument("--golden", default="data/golden/constitution.jsonl")
    parser.add_argument("--index", default="data/index/constitution")
    parser.add_argument("--lo", type=float, default=0.45)
    parser.add_argument("--hi", type=float, default=0.80)
    args = parser.parse_args()

    load_dotenv()
    golden_path = ROOT / args.golden
    if not golden_path.exists():
        print(f"[x] golden set 不存在：{golden_path}", file=sys.stderr)
        return 2

    embedder = get_embeddings()
    index = VectorIndex.load(ROOT / args.index)
    tool = ReadTool(index, embedder)
    gold = load_golden(golden_path)

    rows: list[tuple[dict, float, dict | None, bool]] = []
    for item in gold:
        result = tool.run(item["query"])
        top = result.docs[0] if result.docs else None
        rows.append((item, result.score, top, _is_hit(item, top)))

    print(f"{'查询':<26} {'标注':>6} {'分数':>6} {'top1':>6}  判定")
    for item, score, top, hit in rows:
        expected_list = answers_of(item)
        top_no = top["article_no"] if top else None
        top_law = (top.get("law_name", "").replace("中华人民共和国", "")) if top else "-"
        if not expected_list:
            mark = "无关"
            expected = None
        else:
            mark = "命中" if hit else "错条"
            expected = expected_list[0][1] if len(expected_list) == 1 else "多个"
        print(f"{item['query']:<26} {str(expected):>6} {score:>6.3f} "
              f"{top_law + str(top_no):>6}  {mark}")

    print(f"\n{'阈值':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'P':>7} {'R':>7} {'F1':>7}")
    best_f1 = -1.0
    curve: list[tuple[float, float, float, float, int, int, int, int]] = []
    step = 0.01
    t = args.lo
    while t <= args.hi + 1e-9:
        tp = sum(1 for item, s, _t, hit in rows if is_relevant(item) and s >= t and hit)
        fp = sum(1 for item, s, _t, _h in rows if not is_relevant(item) and s >= t)
        fn = sum(1 for item, s, _t, hit in rows if is_relevant(item) and not (s >= t and hit))
        tn = sum(1 for item, s, _t, _h in rows if not is_relevant(item) and s < t)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        curve.append((round(t, 2), f1, precision, recall, tp, fp, fn, tn))
        best_f1 = max(best_f1, f1)
        if abs(t * 100 - round(t * 100)) < 1e-6 and round(t * 100) % 3 == 0:
            print(f"{t:>5.2f} {tp:>4} {fp:>4} {fn:>4} {tn:>4} "
                  f"{precision:>7.3f} {recall:>7.3f} {f1:>7.3f}")
        t += step

    plateau = [row for row in curve if row[1] >= best_f1 - 1e-9]
    lo_p, hi_p = plateau[0][0], plateau[-1][0]
    # 取平坦区中部：单点最优容易过拟合小样本
    suggested = round((lo_p + hi_p) / 2, 2)

    print(f"\n最佳 F1 = {best_f1:.3f}，出现在 [{lo_p:.2f}, {hi_p:.2f}] 区间")
    print(f"建议阈值 = {suggested:.2f}（取平坦区中部，避免过拟合）")

    positives = [s for item, s, _t, _h in rows if is_relevant(item)]
    negatives = [s for item, s, _t, _h in rows if not is_relevant(item)]
    if positives and negatives:
        print(f"\n相关最低分 = {min(positives):.3f}")
        print(f"无关最高分 = {max(negatives):.3f}")
        print(f"分离间隔 = {min(positives) - max(negatives):.3f}")
    print(f"\n样本量 {len(rows)} 条。样本越少，阈值越应该往低取 —— "
          f"误拒的代价是「用户什么都得不到」，误放的代价是「模型可能答偏」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
