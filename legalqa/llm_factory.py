"""模型与向量化客户端的单例工厂。

对应设计文档 §9.2 单例契约。

四条硬性规则（全部为了 8 GB 内存机器）：

1. **禁止在模块 import 时实例化** —— 否则 pytest 收集用例时就会拉起 1.2 GB 模型，
   CI 直接超时。所有重型 import 都写在 `_build_*` 函数内部。
2. **禁止在节点函数内部 new 客户端** —— 每次调用新建等于内存泄漏。
3. **双重检查锁** —— `lru_cache` 在首次并发调用时可能建出两个实例，
   8 GB 机器上双实例直接 OOM。这里用显式 Lock + 双检。
4. **可重置** —— `reset_singletons()` 供测试隔离使用。

环境变量：
    LEGALQA_LLM_PROVIDER        ollama | openai          (默认 ollama)
    LEGALQA_LLM_MODEL           qwen2.5:1.5b-instruct   (默认)
    LEGALQA_LLM_BASE_URL        ollama / openai 兼容端点
    LEGALQA_LLM_API_KEY         openai 或兼容服务密钥
    LEGALQA_EMBEDDING_PROVIDER  dashscope | ollama | huggingface | hash
    LEGALQA_EMBEDDING_MODEL     BAAI/bge-small-zh-v1.5  (默认)
    LEGALQA_EMBEDDING_DEVICE    cpu                     (默认)
    LEGALQA_MODEL_DIR           模型缓存目录（建议指向 D 盘）
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
from typing import Any

from legalqa import config

logger = logging.getLogger(__name__)


# OpenAI 兼容端点的厂商预设。
# 国内厂商基本都提供 /v1 兼容层，所以一套代码就能覆盖，不需要为每家写适配器。
# 只要厂商遵循 OpenAI 的 tool_calls 协议，loop 模式就能用。
OPENAI_COMPATIBLE_PRESETS: dict[str, dict[str, str]] = {
    "openai": {"base_url": "", "model": "gpt-4o-mini"},
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "zhipu": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-plus"},
    "moonshot": {"base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"},
}


class LLMConfigError(RuntimeError):
    """模型配置错误。带上可执行的修复建议，不要只抛原始 ImportError。"""


# ---------------------------------------------------------------------------
# 单例存储
# ---------------------------------------------------------------------------
_LLM_LOCK = threading.Lock()
_EMBEDDINGS_LOCK = threading.Lock()

_llm_instance: Any = None
_embeddings_instance: Any = None

# 供测试断言「只构建一次」。生产代码不要读它。
_BUILD_COUNT: dict[str, int] = {"llm": 0, "embeddings": 0}


def _llm_temperature() -> float:
    """采样温度。评测时设 LEGALQA_LLM_TEMPERATURE=0 以获得可重复结果。

    实测：temperature=0.1 时，同一批 34 条用例跑三次，
    稳定失败 3 条、偶发失败 5 条，通过率在 82.4%~88.2% 之间波动。
    **单次评测的数字不能当结论用。**
    """
    raw = os.environ.get("LEGALQA_LLM_TEMPERATURE", "").strip()
    if not raw:
        return config.LLM_TEMPERATURE
    try:
        return min(max(float(raw), 0.0), 2.0)
    except ValueError:
        return config.LLM_TEMPERATURE


def _llm_timeout() -> float:
    """LLM 超时。CPU 跑 7B 时 60 s 不够，允许用环境变量放宽。"""
    raw = os.environ.get("LEGALQA_LLM_TIMEOUT", "").strip()
    if not raw:
        return config.LLM_TIMEOUT_S
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return config.LLM_TIMEOUT_S


def _env(key: str, default: str) -> str:
    value = os.environ.get(key, "").strip()
    return value or default


def _prepare_model_dir() -> None:
    """把 HuggingFace 缓存指到指定目录。

    8 GB 机器上模型缓存动辄数 GB，默认落在 C 盘用户目录容易撑爆系统盘。
    只在用户显式配置时才改写，不硬编码任何绝对路径。
    """
    model_dir = os.environ.get("LEGALQA_MODEL_DIR", "").strip()
    if not model_dir:
        return
    for key in ("HF_HOME", "SENTENCE_TRANSFORMERS_HOME", "TRANSFORMERS_CACHE"):
        if not os.environ.get(key):          # 空字符串也要覆盖，setdefault 做不到
            os.environ[key] = model_dir


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
def _build_llm() -> Any:
    """构建 LLM 客户端。**不要在模块顶层调用。**"""
    provider = _env("LEGALQA_LLM_PROVIDER", "ollama").lower()
    model = _env("LEGALQA_LLM_MODEL", "qwen2.5:1.5b-instruct")
    base_url = os.environ.get("LEGALQA_LLM_BASE_URL", "").strip()

    common: dict[str, Any] = {
        "model": model,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
        "timeout": _llm_timeout(),
    }

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama  # 延迟 import
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise LLMConfigError(
                "缺少 langchain-ollama。请执行："
                "pip install langchain-ollama -i https://pypi.tuna.tsinghua.edu.cn/simple"
            ) from exc
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": _llm_temperature(),
            "num_predict": config.LLM_MAX_TOKENS,   # Ollama 用 num_predict 而非 max_tokens
            # timeout 不属于 ChatOllama 的显式参数，透传给底层 client，
            # 直接放在顶层在部分版本会被静默忽略。
            "client_kwargs": {"timeout": _llm_timeout()},
        }
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOllama(**kwargs)

    if provider in OPENAI_COMPATIBLE_PRESETS:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise LLMConfigError(
                "缺少 langchain-openai。请执行："
                "pip install langchain-openai -i https://pypi.tuna.tsinghua.edu.cn/simple"
            ) from exc

        preset = OPENAI_COMPATIBLE_PRESETS[provider]
        api_key = os.environ.get("LEGALQA_LLM_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigError(
                f"provider={provider} 必须设置 LEGALQA_LLM_API_KEY。\n"
                f"  推荐写进 .env 或直接 export，不要硬编码进代码。"
            )
        # 密钥必须是 ASCII：中文占位文本会让 HTTP 头编码直接抛 UnicodeEncodeError，
        # 错误信息完全看不出是「密钥没填」，排查会绕很远。
        if not api_key.isascii():
            raise LLMConfigError(
                "LEGALQA_LLM_API_KEY 含非 ASCII 字符，"
                "请确认已把 .env 里的占位文本替换成真实密钥。"
            )
        resolved_model = (
            model if model != "qwen2.5:1.5b-instruct" else preset["model"]
        )
        kwargs = {
            "model": resolved_model,
            "api_key": api_key,
            "temperature": _llm_temperature(),
            "max_tokens": config.LLM_MAX_TOKENS,
            "top_p": config.LLM_TOP_P,
            "timeout": _llm_timeout(),
        }
        resolved_base = base_url or preset["base_url"]
        if resolved_base:
            kwargs["base_url"] = resolved_base
        return ChatOpenAI(**kwargs)

    raise LLMConfigError(
        f"未知的 LEGALQA_LLM_PROVIDER={provider!r}，"
        f"可选：ollama / {' / '.join(OPENAI_COMPATIBLE_PRESETS)}"
    )


def get_llm() -> Any:
    """获取 LLM 单例。首次调用会构建，之后直接返回。"""
    global _llm_instance
    if _llm_instance is not None:          # 快路径，无锁
        return _llm_instance

    with _LLM_LOCK:
        if _llm_instance is None:          # 双检：并发下只有一个线程能进来
            _BUILD_COUNT["llm"] += 1
            logger.info(
                "building LLM provider=%s model=%s",
                _env("LEGALQA_LLM_PROVIDER", "ollama"),
                _env("LEGALQA_LLM_MODEL", "qwen2.5:1.5b-instruct"),
            )
            _llm_instance = _build_llm()
    return _llm_instance


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class HashEmbeddings:
    """确定性哈希向量化，**仅用于测试与无网络环境**。

    用字符 bigram 的 sha256 散列到固定维度并做 L2 归一化。
    性质：同输入必得同向量、跨进程稳定（不依赖 Python 随机化 hash）、
    相似文本有部分重叠维度 —— 足以让检索链路跑通，但**不代表真实语义相似度**。

    生产环境请用 LEGALQA_EMBEDDING_PROVIDER=huggingface。
    """

    def __init__(self, dim: int = config.EMBEDDING_DIM) -> None:
        self.dim = dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        if len(text) == 1:
            return [text]
        return [text[i : i + 2] for i in range(len(text) - 1)]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _build_embeddings() -> Any:
    """构建 Embedding 客户端。**不要在模块顶层调用。**"""
    provider = _env("LEGALQA_EMBEDDING_PROVIDER", "huggingface").lower()

    if provider == "hash":
        logger.warning("使用 HashEmbeddings（仅测试用，无真实语义能力）")
        return HashEmbeddings()

    if provider == "dashscope":
        # 云端向量化：本地零模型内存。8 GB 机器上 bge-m3 要占 1.16 GB，
        # 换到云端后这部分完全释放。
        # text-embedding-v4 的 CMTEB Retrieval 是 73.98（1024 维），
        # 高于 bge-m3 的量级；价格 0.0005 元/千 token，全量重建 4264 条约 0.2 元。
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise LLMConfigError(
                "缺少 langchain-openai。请执行："
                "pip install langchain-openai -i https://pypi.tuna.tsinghua.edu.cn/simple"
            ) from exc
        api_key = os.environ.get("LEGALQA_LLM_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigError("provider=dashscope 的 embedding 需要 LEGALQA_LLM_API_KEY")
        return OpenAIEmbeddings(
            model=_env("LEGALQA_EMBEDDING_MODEL", "text-embedding-v4"),
            api_key=api_key,
            base_url=_env(
                "LEGALQA_EMBEDDING_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            # 关掉本地 tiktoken 长度检查：它会尝试下载编码表，且对我们的
            # 分块逻辑没有帮助（我们自己控制单条长度）
            check_embedding_ctx_length=False,
            chunk_size=10,   # DashScope 单批上限
        )

    if provider == "ollama":
        # 走 Ollama 的向量接口：不需要 torch / sentence-transformers，
        # 8 GB 机器上省掉约 2 GB 依赖和一次模型加载。
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise LLMConfigError(
                "缺少 langchain-ollama。请执行："
                "pip install langchain-ollama -i https://pypi.tuna.tsinghua.edu.cn/simple"
            ) from exc
        model = _env("LEGALQA_EMBEDDING_MODEL", "bge-m3")
        kwargs: dict[str, Any] = {"model": model}
        base_url = os.environ.get("LEGALQA_OLLAMA_BASE_URL", "").strip()
        if base_url:
            kwargs["base_url"] = base_url
        return OllamaEmbeddings(**kwargs)

    if provider == "huggingface":
        _prepare_model_dir()
        model_name = _env("LEGALQA_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        device = _env("LEGALQA_EMBEDDING_DEVICE", "cpu")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            except ImportError as exc:  # pragma: no cover - 环境相关
                raise LLMConfigError(
                    "缺少 HuggingFace 向量化依赖。请执行："
                    "pip install langchain-huggingface sentence-transformers "
                    "-i https://pypi.tuna.tsinghua.edu.cn/simple"
                ) from exc
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},  # 归一化后内积即余弦
        )

    raise LLMConfigError(
        f"未知的 LEGALQA_EMBEDDING_PROVIDER={provider!r}，可选：huggingface / hash"
    )


def get_embeddings() -> Any:
    """获取 Embedding 单例。"""
    global _embeddings_instance
    if _embeddings_instance is not None:
        return _embeddings_instance

    with _EMBEDDINGS_LOCK:
        if _embeddings_instance is None:
            _BUILD_COUNT["embeddings"] += 1
            logger.info(
                "building embeddings provider=%s model=%s",
                _env("LEGALQA_EMBEDDING_PROVIDER", "huggingface"),
                _env("LEGALQA_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            )
            _embeddings_instance = _build_embeddings()
    return _embeddings_instance


# ---------------------------------------------------------------------------
# 生命周期
# ---------------------------------------------------------------------------
def probe_tool_calling(llm: Any, tool_spec: dict[str, Any], question: str = "公民有言论自由吗") -> dict[str, Any]:
    """探测模型是否真的会按工具协议调用。

    这是接入新模型时**必做**的一步。本地 7B 的教训：`bind_tools` 不报错、
    `tool_calls` 恒为空，模型直接用记忆作答，而且引错了条号。光看
    「接口能通」是发现不了的。

    返回 `{ok, calls, content_head, error}`。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from legalqa import config as _config

    try:
        bound = llm.bind_tools([tool_spec])
        response = bound.invoke(
            [
                SystemMessage(content=_config.SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
        )
    except Exception as exc:
        return {"ok": False, "calls": [], "content_head": "", "error": f"{type(exc).__name__}: {exc}"}

    calls = list(getattr(response, "tool_calls", None) or [])
    return {
        "ok": bool(calls),
        "calls": [(c.get("name"), c.get("args")) for c in calls],
        "content_head": str(getattr(response, "content", "") or "")[:120],
        "error": "",
    }


def warmup() -> None:
    """启动时预热，避免首个请求触发并发构建。

    8 GB 机器上两个实例同时加载模型会直接 swap，必须串行化。
    """
    get_llm()
    get_embeddings()


def reset_singletons() -> None:
    """清空单例。仅供测试使用。"""
    global _llm_instance, _embeddings_instance
    with _LLM_LOCK:
        _llm_instance = None
    with _EMBEDDINGS_LOCK:
        _embeddings_instance = None
    _BUILD_COUNT["llm"] = 0
    _BUILD_COUNT["embeddings"] = 0


def build_count(kind: str) -> int:
    """返回某类单例的构建次数。仅供测试断言使用。"""
    return _BUILD_COUNT.get(kind, 0)
