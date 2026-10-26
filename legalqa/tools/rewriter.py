"""检索用改写：把口语问题转成法律术语表述。

对应设计文档 §6.4.12。

**为什么不是词表扩展。** 试过口语→术语词表（`pure/terms.py`），实测有害：
120 条合成用例上 R@6 从 60.8% 掉到 57.5%。bge-m3 的跨表达能力强于人为注入术语，
追加术语会稀释原句语义。

**为什么 LLM 改写有效。** 实测 R@6 从 55% 提升到 75%（+20pt）。
模型能把「白送朋友一台电脑，说好帮我照看孩子」改写成
「赠与附义务，受赠人不履行约定义务，能否撤销赠与并请求返还」——
这是词表做不到的语义转换。

**代价是 R@1 从 35% 降到 25%。** 改写后 query 变成术语堆砌，
正确条文更容易进 top-k 但更难排第一。对 RAG 来说 R@6 才决定
「模型能否看到正确条文」，所以是净收益。

**只在 pipeline 模式启用。** loop 模式下模型自己就会改写检索串
（实测会传「建筑工地工人摔伤骨折是否认定为工伤」），再改写一次是重复劳动。
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from langchain_core.messages import HumanMessage

__all__ = ["REWRITE_FOR_RETRIEVAL_PROMPT", "make_llm_rewriter"]

REWRITE_FOR_RETRIEVAL_PROMPT = """把下面的问题改写成法律术语表述，用于检索法条。

要求：
1. 只输出改写后的句子，不要解释，不要加引号
2. **不要改变事实**：金额、时间、主体关系必须保留
3. 用规范的法律用语（如「用人单位」「劳动报酬」「赠与」），但不要编造条号
4. 保留原问题的核心争议点

问题：{query}
改写："""


def make_llm_rewriter(llm: Any) -> Callable[..., str | None]:
    """构造一个用于检索的改写函数，签名与 `ReadTool` 的 rewriter 兼容。"""

    def rewrite(query: str, history: Sequence[Any] | None = None) -> str | None:
        text = (query or "").strip()
        if not text:
            return None
        try:
            response = llm.invoke(
                [HumanMessage(content=REWRITE_FOR_RETRIEVAL_PROMPT.format(query=text))]
            )
        except Exception:
            return None
        result = str(getattr(response, "content", "") or "").strip()
        result = result.split("\n")[0].strip().strip('"').strip("「」").strip()
        if not (4 <= len(result) <= 200):
            return None
        return result

    return rewrite
