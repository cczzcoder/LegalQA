"""LLM 重排：从召回的 20 条里挑出最相关的 6 条。

对应设计文档 §6.4.14。

**为什么用 LLM 而不是 cross-encoder。**
bge-reranker-base 需要 sentence-transformers + torch（约 2 GB 依赖），
在 8 GB 机器上不划算。而 qwen-plus 一次调用能同时看 20 条候选、
理解问题意图、判断哪条真正相关——这是小 cross-encoder 做不到的。

**代价是延迟。** 一次调用约 1.5 秒，单轮问答从 10 秒变成 12 秒左右。
对法律问答来说准确性优先，接受这个代价。

**实测效果**（40 条口语用例，dense 召回 top-20）：

| 策略 | R@6 |
|---|---|
| dense 直接 top-6 | 62.5% |
| **+ LLM rerank** | **72.5%** |
| + 改写 + rerank | 75.0% |

改写只多贡献 2.5pt 却要多一次调用，所以只在 pipeline 模式保留
（loop 模式下模型自己会改写检索串）。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Sequence

from langchain_core.messages import HumanMessage

__all__ = ["RERANK_PROMPT", "make_llm_reranker"]

RERANK_PROMPT = """下面是一个法律问题，以及 {n} 条候选法条。
请选出与问题最相关的 6 条，按相关度从高到低输出编号，用英文逗号分隔。

要求：
1. 只输出编号，例如 `3,7,1,15,9,2`
2. 不要解释，不要输出其他任何内容
3. 必须选满 6 条（除非候选不足 6 条）

问题：{query}

候选：
{items}

输出："""


def _shorten(text: Any, limit: int = 80) -> str:
    return re.sub(r"\s+", "", str(text or ""))[:limit]


def make_llm_reranker(llm: Any, *, keep: int = 6, candidates: int = 20) -> Callable[..., list]:
    """构造重排函数。签名 `(query, hits) -> hits`。

    解析失败或调用异常时**原样返回前 keep 条**——重排是锦上添花，
    不能因为它挂了就让整个检索失败。
    """

    def rerank(query: str, hits: Sequence[Any]) -> list[Any]:
        items = list(hits)[:candidates]
        if len(items) <= keep:
            return items

        lines = []
        for i, hit in enumerate(items, start=1):
            doc = getattr(hit, "doc", {}) or {}
            law = str(doc.get("law_name", "")).replace("中华人民共和国", "")
            lines.append(f"{i}. 《{law}》{doc.get('article_label', '')}：{_shorten(doc.get('text'))}")

        prompt = RERANK_PROMPT.format(n=len(items), query=query, items="\n".join(lines))
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = str(getattr(response, "content", "") or "")
        except Exception:
            return items[:keep]

        picked: list[Any] = []
        seen: set[int] = set()
        for token in re.findall(r"\d+", raw):
            idx = int(token)
            if 1 <= idx <= len(items) and idx not in seen:
                seen.add(idx)
                picked.append(items[idx - 1])
            if len(picked) >= keep:
                break

        if len(picked) < keep:
            # 模型没选满（或输出了脏数据），用剩余候选按原顺序补齐
            for i, hit in enumerate(items, start=1):
                if i not in seen:
                    picked.append(hit)
                if len(picked) >= keep:
                    break
        return picked

    return rerank
