"""BM25 关键词检索，用于和向量检索做混合召回。

对应设计文档 §6.4.9 的 L1 阶段。

**为什么需要它。** 纯语义检索在专有名词上很弱：

| 用户问 | 法条写 | dense 能跨吗 |
|---|---|---|
| 白送朋友电脑 | 赠与 | 能（语义相近） |
| 建筑垃圾怎么处理 | 建筑垃圾 | **能，但常被更泛的条文挤掉** |
| 竞业限制补偿 | 竞业限制 | 词完全一样，dense 反而不如关键词匹配 |

dense 擅长「意思相近但用词不同」，BM25 擅长「用词完全一样」。
两者是互补的，不是替代关系。

**为什么不用 jieba 分词。** 中文 BM25 的分词质量对效果影响很大，但 jieba 要装包、
要维护词典、对法律术语还得加自定义词。**字 bigram 是不需要分词的近似**：
「竞业限制」切成「竞业」「业限」「限制」，query 和文档用同样的切法，
匹配依然成立，而且对未登录词（新法条术语）天然友好。

代价是索引变大（bigram 数量约为字数的 1 倍），4264 条法条约 20 MB，可以接受。

**为什么用 RRF 融合而不是加权求和。** dense 的余弦在 [0,1]，BM25 的分数没有上界，
两者量纲不同，加权求和需要归一化，而归一化参数很难调。RRF 只用**排名**，
天然免疫量纲问题，且是信息检索里的标准做法。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["BM25Index", "rrf_fuse", "RRF_K"]

# RRF 的平滑常数。60 是原论文的经验值，作用是压低头部排名的差异，
# 让多个通道的贡献更均衡。
RRF_K = 60

_NON_WORD = re.compile(r"[^\u4e00-\u9fff0-9a-zA-Z]+")


def _tokenize(text: str) -> list[str]:
    """字 bigram 切分。不需要分词器，对法律术语天然友好。"""
    cleaned = _NON_WORD.sub("", text or "")
    if len(cleaned) < 2:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


@dataclass(frozen=True)
class BM25Hit:
    doc: dict[str, Any]
    score: float


class BM25Index:
    """内存 BM25。4264 条法条构建约 1 秒，检索 < 5 ms。"""

    def __init__(
        self,
        docs: Sequence[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        text_key: str = "text",
    ) -> None:
        self._docs = list(docs)
        self._k1 = k1
        self._b = b

        tokenized = [_tokenize(str(doc.get(text_key, ""))) for doc in self._docs]
        self._lengths = [len(tokens) for tokens in tokenized]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

        # 倒排索引：token -> {doc_idx: tf}
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        for idx, tokens in enumerate(tokenized):
            for token, count in Counter(tokens).items():
                self._postings[token][idx] = count

        self._n = len(self._docs)
        # 预算 idf，避免每次检索重复算
        self._idf: dict[str, float] = {
            token: math.log(1 + (self._n - len(post) + 0.5) / (len(post) + 0.5))
            for token, post in self._postings.items()
        }

    def __len__(self) -> int:
        return self._n

    def search(self, query: str, k: int = 20) -> list[BM25Hit]:
        tokens = _tokenize(query)
        if not tokens or not self._n:
            return []

        scores: dict[int, float] = defaultdict(float)
        for token in set(tokens):
            postings = self._postings.get(token)
            if not postings:
                continue
            idf = self._idf[token]
            for idx, tf in postings.items():
                norm = self._k1 * (
                    1 - self._b + self._b * (self._lengths[idx] / (self._avg_len or 1))
                )
                scores[idx] += idf * (tf * (self._k1 + 1)) / (tf + norm)

        if not scores:
            return []
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: max(int(k), 1)]
        return [BM25Hit(doc=self._docs[idx], score=score) for idx, score in top]


def rrf_fuse(
    channels: Sequence[Sequence[Any]],
    *,
    k: int = RRF_K,
    top_k: int | None = None,
    id_key: str = "id",
) -> list[tuple[dict[str, Any], float]]:
    """Reciprocal Rank Fusion：按排名融合多路召回。

    `channels` 是若干条已排序的结果列表，每个元素需要能通过 `id_key` 取到唯一 id
    且带有 `.doc` 属性（`SearchHit` / `BM25Hit` 都满足）。

    公式：`score(d) = Σ 1 / (k + rank_i(d))`，rank 从 1 开始。
    只依赖排名，不需要归一化分数——这正是它适合融合 dense 和 BM25 的原因。
    """
    scores: dict[str, float] = defaultdict(float)
    pool: dict[str, dict[str, Any]] = {}

    for hits in channels:
        for rank, hit in enumerate(hits, start=1):
            doc = getattr(hit, "doc", None)
            if not doc:
                continue
            key = str(doc.get(id_key, ""))
            if not key:
                continue
            pool.setdefault(key, doc)
            scores[key] += 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        ranked = ranked[: max(int(top_k), 0)]
    return [(pool[key], score) for key, score in ranked]
