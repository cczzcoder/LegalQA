"""S1 验收测试：单例工厂。

对应设计文档 §9.2：
  - 100 次调用只实例化 1 次
  - import 时不实例化
  - reset 可隔离测试
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from legalqa import config, llm_factory

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _FakeLLM:
    """最小可用的假 LLM，只为验证单例行为。"""

    def __init__(self) -> None:
        self.calls = 0


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls = 0


@pytest.fixture(autouse=True)
def _clean_singletons():
    llm_factory.reset_singletons()
    yield
    llm_factory.reset_singletons()


def test_llm_built_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """核心验收项：100 次调用只构建 1 次，且返回同一对象。"""
    built: list[_FakeLLM] = []

    def fake_build() -> _FakeLLM:
        obj = _FakeLLM()
        built.append(obj)
        return obj

    monkeypatch.setattr(llm_factory, "_build_llm", fake_build)

    first = llm_factory.get_llm()
    for _ in range(99):
        assert llm_factory.get_llm() is first

    assert len(built) == 1
    assert llm_factory.build_count("llm") == 1


def test_embeddings_built_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[_FakeEmbeddings] = []

    def fake_build() -> _FakeEmbeddings:
        obj = _FakeEmbeddings()
        built.append(obj)
        return obj

    monkeypatch.setattr(llm_factory, "_build_embeddings", fake_build)

    first = llm_factory.get_embeddings()
    for _ in range(50):
        assert llm_factory.get_embeddings() is first

    assert len(built) == 1


def test_reset_clears_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_factory, "_build_llm", lambda: _FakeLLM())
    a = llm_factory.get_llm()
    llm_factory.reset_singletons()
    b = llm_factory.get_llm()
    assert a is not b
    assert llm_factory.build_count("llm") == 1


def test_no_heavy_import_on_module_import() -> None:
    """import 阶段不得拉起任何重型依赖。

    注意：不能在当前进程里断言 —— 测试进程早已 import 过 legalqa，
    那时的状态已经不可观测了。必须开子进程，从零 import 再检查。
    """
    code = (
        "import sys\n"
        "import legalqa.config, legalqa.state, legalqa.llm_factory\n"
        "heavy = [m for m in ('torch', 'transformers', 'sentence_transformers',\n"
        "                     'langchain_ollama', 'langchain_openai',\n"
        "                     'langchain_huggingface') if m in sys.modules]\n"
        "print('HEAVY=' + ','.join(heavy))\n"
        "print('BUILT=' + str(legalqa.llm_factory.build_count('llm')))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    heavy_line = next(
        (ln for ln in result.stdout.splitlines() if ln.startswith("HEAVY=")), None
    )
    assert heavy_line is not None, f"子进程输出异常: {result.stdout!r}"
    heavy = heavy_line.split("=", 1)[1].strip()
    assert heavy == "", f"import 时拉起了重型依赖: {heavy}"
    assert "BUILT=0" in result.stdout, "import 阶段不得构建模型"


def test_unknown_provider_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEGALQA_LLM_PROVIDER", "no-such-provider")
    with pytest.raises(llm_factory.LLMConfigError, match="未知的 LEGALQA_LLM_PROVIDER"):
        llm_factory._build_llm()


# ---------------------------------------------------------------------------
# HashEmbeddings
# ---------------------------------------------------------------------------
def test_hash_embeddings_shape_and_norm() -> None:
    emb = llm_factory.HashEmbeddings()
    vec = emb.embed_query("劳动合同到期不续签有补偿吗")
    assert len(vec) == config.EMBEDDING_DIM
    norm = math.sqrt(sum(v * v for v in vec))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_hash_embeddings_deterministic_across_instances() -> None:
    a = llm_factory.HashEmbeddings().embed_query("工伤认定")
    b = llm_factory.HashEmbeddings().embed_query("工伤认定")
    assert a == b


def test_hash_embeddings_handles_empty() -> None:
    vec = llm_factory.HashEmbeddings().embed_query("")
    assert len(vec) == config.EMBEDDING_DIM
    assert all(v == 0.0 for v in vec)


def test_hash_embeddings_batch_matches_single() -> None:
    emb = llm_factory.HashEmbeddings()
    batch = emb.embed_documents(["工资", "加班费"])
    assert batch[0] == emb.embed_query("工资")
    assert batch[1] == emb.embed_query("加班费")


# ---------------------------------------------------------------------------
# 云端（OpenAI 兼容端点）
# ---------------------------------------------------------------------------
def test_dashscope_preset_builds_correct_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEGALQA_LLM_PROVIDER", "dashscope")
    monkeypatch.setenv("LEGALQA_LLM_API_KEY", "sk-test-key")
    monkeypatch.delenv("LEGALQA_LLM_MODEL", raising=False)
    monkeypatch.delenv("LEGALQA_LLM_BASE_URL", raising=False)

    llm = llm_factory._build_llm()

    assert llm.model_name == "qwen-plus"
    assert "dashscope.aliyuncs.com" in str(llm.openai_api_base)


def test_cloud_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEGALQA_LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("LEGALQA_LLM_API_KEY", raising=False)

    with pytest.raises(llm_factory.LLMConfigError, match="LEGALQA_LLM_API_KEY"):
        llm_factory._build_llm()


def test_explicit_model_overrides_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEGALQA_LLM_PROVIDER", "dashscope")
    monkeypatch.setenv("LEGALQA_LLM_API_KEY", "sk-test-key")
    monkeypatch.setenv("LEGALQA_LLM_MODEL", "qwen-max")

    assert llm_factory._build_llm().model_name == "qwen-max"


def test_custom_base_url_overrides_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEGALQA_LLM_PROVIDER", "openai")
    monkeypatch.setenv("LEGALQA_LLM_API_KEY", "sk-test-key")
    monkeypatch.setenv("LEGALQA_LLM_BASE_URL", "https://my-proxy.example.com/v1")

    assert "my-proxy" in str(llm_factory._build_llm().openai_api_base)


def test_preset_list_covers_domestic_vendors() -> None:
    assert {"dashscope", "deepseek", "zhipu", "moonshot"} <= set(
        llm_factory.OPENAI_COMPATIBLE_PRESETS
    )


def test_non_ascii_api_key_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """占位文本里带中文时，HTTP 头编码会抛 UnicodeEncodeError，看不出是密钥问题。"""
    monkeypatch.setenv("LEGALQA_LLM_PROVIDER", "dashscope")
    monkeypatch.setenv("LEGALQA_LLM_API_KEY", "在此粘贴你的 Key")

    with pytest.raises(llm_factory.LLMConfigError, match="非 ASCII"):
        llm_factory._build_llm()
