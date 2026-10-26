"""向量索引：内存暴力检索 + 条号字典。

对应设计文档 §6.4.3 的通道 A 与通道 B。

**为什么不用 FAISS。** 1 万条法条 × 512 维 float32 只有 20 MB，一次矩阵乘法
在 CPU 上约 5 ms，暴力检索比 ANN 更准且零额外依赖。FAISS 要到 5 万条以上
才有明显收益（§6.4.9 的 L1 阶段再考虑）。

索引做两件事：
  1. `search()`   —— 语义通道，返回 top-k 与原始余弦
  2. `by_article()` —— 条号通道，O(1) 精确命中
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from legalqa import config

__all__ = ["Embedder", "VectorIndex", "SearchHit", "IncrementalResult"]


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class IncrementalResult:
    """增量索引的结果统计。"""

    reused: int      # 复用旧向量的条数
    embedded: int    # 新向量化的条数
    removed: int     # 旧索引里已不存在、被丢弃的条数

    @property
    def total(self) -> int:
        return self.reused + self.embedded


@dataclass(frozen=True)
class SearchHit:
    doc: dict[str, Any]
    score: float


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0          # 零向量保持零，避免除零产生 NaN
    return matrix / norms


class VectorIndex:
    """法条向量索引。构建一次，之后只读。"""

    def __init__(self, docs: Sequence[dict[str, Any]], vectors: np.ndarray) -> None:
        if len(docs) != vectors.shape[0]:
            raise ValueError(f"docs({len(docs)}) 与 vectors({vectors.shape[0]}) 数量不一致")
        self._docs: list[dict[str, Any]] = list(docs)
        self._vectors: np.ndarray = _normalize(np.asarray(vectors, dtype=np.float32))
        # meta.json 的内容（provider / model / dim / content hash）。
        # 旧索引没有这些字段，读取时为空 dict 即可。
        self._meta: dict[str, Any] = {}
        # 条号通道：(law_name, article_no) -> 下标
        # 只用 article_no 会在多法源时串号，所以必须带法律名
        self._by_article: dict[tuple[str, int], int] = {}
        for i, doc in enumerate(self._docs):
            no = doc.get("article_no")
            if isinstance(no, int):
                self._by_article[(str(doc.get("law_name", "")), no)] = i

    # -- 属性 ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self._docs)

    @property
    def docs(self) -> list[dict[str, Any]]:
        return self._docs

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

    @property
    def dim(self) -> int:
        return int(self._vectors.shape[1]) if len(self._docs) else 0

    # -- 构建 ---------------------------------------------------------
    @classmethod
    def build(
        cls,
        docs: Sequence[dict[str, Any]],
        embedder: Embedder,
        *,
        batch_size: int = 32,
        progress: Any = None,
    ) -> "VectorIndex":
        """分批向量化。

        **不能一次性把所有文本丢给 embedder。** 实测 2558 条法条一次请求会让
        Ollama 崩溃（`/tokenize` 连接被拒），整份语料白跑 8 分钟。
        分批后单批内存可控，且失败只影响当前批次。

        `progress` 可选回调 `(done, total)`，用于打印进度。
        """
        if not docs:
            return cls([], np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32))

        texts = [str(doc.get("text", "")) for doc in docs]
        total = len(texts)
        vectors: list[list[float]] = []

        step = max(int(batch_size), 1)
        for start in range(0, total, step):
            chunk = texts[start : start + step]
            batch_vectors = embedder.embed_documents(chunk)
            if len(batch_vectors) != len(chunk):
                raise RuntimeError(
                    f"向量化数量不匹配：送入 {len(chunk)} 条，返回 {len(batch_vectors)} 条"
                )
            vectors.extend(batch_vectors)
            if progress is not None:
                progress(min(start + step, total), total)

        return cls(docs, np.asarray(vectors, dtype=np.float32))

    @classmethod
    def build_incremental(
        cls,
        docs: Sequence[dict[str, Any]],
        embedder: Embedder,
        previous: "VectorIndex | None" = None,
        *,
        batch_size: int = 32,
        progress: Any = None,
    ) -> tuple["VectorIndex", IncrementalResult]:
        """增量构建：只对**新增或内容变化**的条文做向量化。

        为什么需要：全量重建 4000 条要十几分钟，而用户每加一部法律可能只
        新增几十到上千条。增量能把「加一部法」的等待从十几分钟压到几十秒。

        判定依据是 `(id, text)`：id 相同但正文变了（法律修订）也要重新向量化，
        否则索引里会留着旧版本的向量，检索出来的却是新条文。

        没有 `previous` 时退化为全量构建。
        """
        if previous is None or len(previous) == 0:
            built = cls.build(docs, embedder, batch_size=batch_size, progress=progress)
            return built, IncrementalResult(reused=0, embedded=len(built), removed=0)

        old_by_id = {str(doc.get("id", "")): doc for doc in previous.docs}
        old_vectors = {
            str(doc.get("id", "")): previous._vectors[i]      # noqa: SLF001 - 同类内部访问
            for i, doc in enumerate(previous.docs)
        }

        reused_vectors: list[np.ndarray] = []
        pending: list[tuple[int, dict[str, Any]]] = []
        reused = 0
        for position, doc in enumerate(docs):
            key = str(doc.get("id", ""))
            cached_doc = old_by_id.get(key)
            same_text = cached_doc is not None and cached_doc.get("text") == doc.get("text")
            if same_text and key in old_vectors:
                reused_vectors.append(old_vectors[key])
                reused += 1
            else:
                reused_vectors.append(np.zeros(previous.dim, dtype=np.float32))
                pending.append((position, doc))

        new_ids = {str(doc.get("id", "")) for doc in docs}
        removed = sum(1 for key in old_by_id if key not in new_ids)

        texts = [str(doc.get("text", "")) for _pos, doc in pending]
        total = len(texts)
        new_vectors: list[list[float]] = []
        step = max(int(batch_size), 1)
        for start in range(0, total, step):
            chunk = texts[start : start + step]
            vectors = embedder.embed_documents(chunk)
            if len(vectors) != len(chunk):
                raise RuntimeError(
                    f"向量化数量不匹配：送入 {len(chunk)} 条，返回 {len(vectors)} 条"
                )
            new_vectors.extend(vectors)
            if progress is not None:
                progress(min(start + step, total), total)

        matrix = np.stack(reused_vectors) if reused_vectors else np.zeros((0, previous.dim), dtype=np.float32)
        for (position, _doc), vector in zip(pending, new_vectors):
            matrix[position] = np.asarray(vector, dtype=np.float32)

        index = cls(docs, matrix)
        return index, IncrementalResult(reused=reused, embedded=len(pending), removed=removed)

    # -- 通道 A：语义 -------------------------------------------------
    def search(
        self,
        query_vector: Sequence[float],
        k: int = config.RAG_RECALL_K,
        *,
        only_valid: bool = True,
    ) -> list[SearchHit]:
        """返回按原始余弦降序的 top-k。

        **返回的是原始余弦，不是归一化分。** 阈值判定只能吃这个值（BC-26）。

        `only_valid=True` 时屏蔽已废止条文。法律有时效性，引用一条已废止的
        旧条文比答不出来更危险 —— 用户会当真。
        """
        if not len(self._docs) or not query_vector:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                f"查询维度 {query.shape[0]} 与索引维度 {self._vectors.shape[1]} 不一致"
            )
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        query = query / norm
        scores = self._vectors @ query
        if only_valid:
            alive = np.array(
                [bool(doc.get("is_valid", True)) for doc in self._docs], dtype=bool
            )
            if not alive.any():
                return []
            scores = np.where(alive, scores, -np.inf)

        top_k = min(max(int(k), 1), len(self._docs))
        # argpartition 取 top-k 再排序，避免全量排序
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [
            SearchHit(doc=self._docs[int(i)], score=float(scores[int(i)]))
            for i in idx
            if np.isfinite(scores[int(i)])       # 屏蔽掉的失效条文分数为 -inf
        ]

    # -- 通道 B：条号 -------------------------------------------------
    def by_article(self, article_no: int, law_name: str | None = None) -> dict[str, Any] | None:
        """条号精确查找。未命中返回 None，不抛异常（BC-24）。

        **不传法律名时必须要求唯一。** 多法源下「第一条」几乎每部法律都有，
        随便返回一个会造成「看起来精确命中、实际答错法」——实测
        「宪法第一条」返回了「劳动法第一条」，score 还是 1.00。
        有歧义就返回 None，让调用方回落到语义通道。
        """
        if law_name:
            index = self._by_article.get((law_name, article_no))
            return self._docs[index] if index is not None else None

        matches = [i for (_, no), i in self._by_article.items() if no == article_no]
        if len(matches) == 1:
            return self._docs[matches[0]]
        return None

    @property
    def law_names(self) -> list[str]:
        """语料里出现的全部法律名，用于提问侧的法律名识别。"""
        return sorted({str(doc.get("law_name", "")) for doc in self._docs if doc.get("law_name")})

    # -- 持久化 -------------------------------------------------------
    @staticmethod
    def fingerprint_docs(docs: Sequence[dict[str, Any]]) -> str:
        """内容指纹：**包含正文**，不是只哈希 id。

        法律修订时 id（slug-条号）不变、正文变了；只比 id 会认为索引一致，
        继续检索旧条文——这是法律场景不能接受的静默错误。
        """
        payload = "|".join(
            f"{doc.get('id', '')}\x1f{doc.get('text', '')}" for doc in docs
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def fingerprint(self) -> str:
        return self.fingerprint_docs(self._docs)

    def save(
        self, directory: str | Path, meta: dict[str, Any] | None = None
    ) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "vectors.npy", self._vectors)
        (target / "docs.json").write_text(
            json.dumps(self._docs, ensure_ascii=False), encoding="utf-8"
        )
        payload: dict[str, Any] = dict(meta or {})
        # 基础字段由索引自己保证，不允许调用方覆盖
        payload.update(
            {"count": len(self._docs), "dim": self.dim, "fingerprint": self.fingerprint()}
        )
        (target / "meta.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "VectorIndex":
        source = Path(directory)
        vectors = np.load(source / "vectors.npy")
        docs = json.loads((source / "docs.json").read_text(encoding="utf-8"))
        index = cls(docs, vectors)
        meta_path = source / "meta.json"
        if meta_path.exists():
            try:
                index._meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                index._meta = {}
        return index
