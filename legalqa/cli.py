"""命令行入口。

用法：
    # 一次性提问（离线可跑，模型用占位实现）
    python -m legalqa.cli --query "宪法第一条是什么" --dry-run

    # 交互式，接真实模型
    python -m legalqa.cli --provider ollama --model qwen2.5:1.5b-instruct

    # 换 bge 向量（需要 sentence-transformers）
    python -m legalqa.cli --embedder hf --query "公民的言论自由"

`--dry-run` 的用途：不接模型，也能验证检索链路、窗口组装、护栏是否正常。
它会把**真正会发给模型的内容**打印出来 —— 排查「为什么模型答得不对」时，
先看这个比看模型输出有用得多。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from legalqa import config
from legalqa.corpus.index import VectorIndex
from legalqa.corpus.ingest import load_jsonl
from legalqa.graph import LOOP, PIPELINE, build_graph
from legalqa.state import LegalState, coerce_state
from legalqa.tools.read_tool import READ_TOOL_SPEC, ReadTool

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """极简 .env 加载。不引入 python-dotenv：15 行能解决的事不值得多一个依赖。

    规则：`KEY=VALUE`，`#` 开头为注释，已存在的环境变量优先（不覆盖）。
    """
    env_file = path or (ROOT / ".env")
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class DryRunModel(BaseChatModel):
    """占位模型：**会真的发起一次工具调用**，用来验证整条循环。

    只回显「收到几条消息」是没用的 —— 那样永远测不到检索、去重、预算、
    回边这些最容易出问题的地方。所以第一次调用固定产出 tool_calls，
    让图真的走一遍 `Agent → ToolNode → ContextCompress → Agent`。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: int = 0
    tools_bound: bool = False

    @property
    def _llm_type(self) -> str:
        return "dry-run"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "DryRunModel":
        self.tools_bound = bool(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls += 1
        tools_seen = sum(1 for m in messages if type(m).__name__ == "ToolMessage")

        if self.tools_bound and self.calls == 1 and tools_seen == 0:
            query = ""
            for message in reversed(messages):
                if isinstance(message, HumanMessage):
                    query = str(message.content)
                    break
            call = AIMessage(
                id="dry-call-1",
                content="",
                tool_calls=[{"id": "dry-1", "name": "read_law", "args": {"query": query}}],
            )
            return ChatResult(generations=[ChatGeneration(message=call)])

        text = (
            "（dry-run：模型未接入）\n"
            f"收到 {len(messages)} 条消息，其中 {tools_seen} 条工具结果。\n"
            "接上真实模型后，这里会是基于上述法条生成的回答。"
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def load_index(index_dir: Path, corpus_path: Path, embedder: Any) -> VectorIndex:
    """加载索引；语料变更时自动重建。

    只检查「文件存在」是不够的：重新构建过语料但没重建索引时，
    检索结果会与语料对不上，而且完全静默。条数比对是最便宜的一致性信号。
    """
    if not corpus_path.exists():
        raise SystemExit(f"[x] 语料不存在：{corpus_path}\n    先跑 scripts/build_corpus.py")

    docs = load_jsonl(corpus_path)

    if (index_dir / "vectors.npy").exists():
        index = VectorIndex.load(index_dir)
        expected = hashlib.sha256(
            "|".join(str(d.get("id", "")) for d in docs).encode("utf-8")
        ).hexdigest()[:16]
        if len(index) == len(docs) and index.fingerprint() == expected:
            return index
        print(f"[!] 索引与语料不一致（{len(index)} 条 / {len(docs)} 条），自动重建")

    index = VectorIndex.build(docs, embedder)
    index.save(index_dir)
    return index


def build_embedder(choice: str) -> Any:
    import os

    if choice == "hash":
        from legalqa.llm_factory import HashEmbeddings

        print("[!] 使用 HashEmbeddings：只验证链路，检索质量不代表真实效果")
        return HashEmbeddings()
    if choice == "ollama":
        os.environ["LEGALQA_EMBEDDING_PROVIDER"] = "ollama"
        os.environ.setdefault("LEGALQA_EMBEDDING_MODEL", "bge-m3")
    from legalqa.llm_factory import get_embeddings

    return get_embeddings()


def build_llm(args: argparse.Namespace) -> Any:
    if args.dry_run:
        return DryRunModel()
    if args.provider:
        import os

        os.environ["LEGALQA_LLM_PROVIDER"] = args.provider
    if args.model:
        import os

        os.environ["LEGALQA_LLM_MODEL"] = args.model
    from legalqa.llm_factory import LLMConfigError, get_llm

    try:
        return get_llm()
    except LLMConfigError as exc:
        raise SystemExit(f"[x] 模型配置错误：{exc}") from exc


def build_interrupted_state(
    base: LegalState | None,
    query: str,
    partial_text: str,
    turn_id: int,
) -> LegalState:
    """用户中途打断时，构造「已发生但未完成」的状态。

    两个要点：

    1. **半截回答必须入历史**，否则下一轮模型不知道刚才说到哪，会从头再答一遍。
    2. **必须打 `interrupted` 标记**。不标记的话，模型会把半句话当成完整结论，
       基于一个未完成的论断继续推理。
    """
    payload: dict[str, Any] = dict(base.to_dict()) if base is not None else {}
    messages = list(payload.get("messages") or [])
    messages.append(HumanMessage(id=f"h-{turn_id}", content=query))

    text = (partial_text or "").strip()
    if text:
        messages.append(
            AIMessage(
                id=f"a-{turn_id}-interrupted",
                content=text,
                additional_kwargs={"interrupted": True},
            )
        )

    payload["messages"] = messages
    payload["turn_id"] = turn_id
    # 用 interrupted 而不是 degraded：中断是用户主动行为，不是系统降级，
    # 混在一起会让「降级率」指标失真。
    payload["interrupted"] = True
    return LegalState(**payload)


def final_answer(state: LegalState) -> str:
    """取最后一条 AI 消息。不要直接取 messages[-1] —— 异常路径下它可能是
    HumanMessage（用户提问），打印出来像是「模型复读了你的话」。"""
    for message in reversed(state.messages):
        if isinstance(message, AIMessage):
            return str(message.content)
    return "（本轮没有产生回答）"


def print_turn(state: LegalState) -> None:
    print(f"\n{final_answer(state)}\n")
    print(
        f"  ── 通道={state.retrieval_channel or '-'} "
        f"score={state.rag_score:.2f} "
        f"轮次={state.iteration_count} "
        f"降级={'是' if state.degraded else '否'}"
    )
    if state.error:
        print(f"  ── error={state.error}")


def run_check(args: argparse.Namespace) -> int:
    """接入自检。

    接新模型时**必须**先跑这个。它回答的是「这个模型能不能用于 loop 模式」，
    而不是「接口通不通」——本地 7B 接口完全正常，但 tool_calls 恒为空，
    直接用记忆作答还引错了条号。
    """
    from legalqa.llm_factory import (
        LLMConfigError,
        get_embeddings,
        probe_tool_calling,
    )

    provider = os.environ.get("LEGALQA_LLM_PROVIDER", "ollama")
    model = os.environ.get("LEGALQA_LLM_MODEL", "(默认)")
    base = os.environ.get("LEGALQA_LLM_BASE_URL", "")
    key = os.environ.get("LEGALQA_LLM_API_KEY", "")
    masked = (key[:4] + "****" + key[-4:]) if len(key) > 8 else ("(未设置)" if not key else "****")

    print("── 配置 ──")
    print(f"  LLM   provider={provider} model={model}")
    if base:
        print(f"        base_url={base}")
    print(f"        api_key={masked}")
    print(f"  Embed provider={os.environ.get('LEGALQA_EMBEDDING_PROVIDER', 'huggingface')} "
          f"model={os.environ.get('LEGALQA_EMBEDDING_MODEL', '(默认)')}")

    ok = True

    # 1) LLM 连通性
    print("\n── 1. LLM 连通性 ──")
    try:
        llm = build_llm(args)
        reply = llm.invoke([HumanMessage(content="回答两个字：收到")])
        print(f"  [OK] 回复：{str(reply.content)[:40]}")
    except (LLMConfigError, Exception) as exc:      # noqa: BLE001 - 自检就是要兜住一切
        print(f"  [X] {type(exc).__name__}: {str(exc)[:200]}")
        return 1

    # 2) 工具调用能力（决定 loop 模式能不能用）
    print("\n── 2. 工具调用能力 ──")
    probe = probe_tool_calling(llm, READ_TOOL_SPEC)
    if probe["error"]:
        print(f"  [X] {probe['error'][:200]}")
        ok = False
    elif probe["ok"]:
        print(f"  [OK] tool_calls={probe['calls']}")
        print("  → 可以用 loop 模式")
    else:
        print(f"  [X] tool_calls 为空，模型选择直接作答")
        print(f"      回答片段：{probe['content_head']}")
        print("  → 只能用 pipeline 模式（检索前置，模型不绑工具）")
        ok = False

    # 3) Embedding（先按 --embedder 设定环境变量，否则会走默认 provider）
    print("\n── 3. Embedding ──")
    try:
        emb = build_embedder(args.embedder)
        vector = emb.embed_query("测试")
        print(f"  [OK] 维度 {len(vector)}")
    except Exception as exc:                        # noqa: BLE001
        print(f"  [X] {type(exc).__name__}: {str(exc)[:200]}")
        ok = False

    # 4) 索引
    print("\n── 4. 语料索引 ──")
    try:
        index = VectorIndex.load(ROOT / args.index)
        print(f"  [OK] {len(index)} 条 × {index.dim} 维")
    except Exception as exc:                        # noqa: BLE001
        print(f"  [X] {type(exc).__name__}: {str(exc)[:200]}")
        ok = False

    print("\n" + ("✅ 全部通过，可以跑 loop 模式" if ok else
                   "⚠️ 部分未通过，建议用 --mode pipeline"))
    return 0 if ok else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LangGraph 法律问答（阶段 0）")
    parser.add_argument("--query", help="一次性提问；省略则进入交互模式")
    parser.add_argument("--corpus", default="data/corpus/all.jsonl")
    parser.add_argument("--index", default="data/index/all")
    parser.add_argument(
        "--embedder", default="hash", choices=["hash", "ollama", "hf"],
        help="hash=离线验证链路；ollama=本地 bge-m3；hf=sentence-transformers",
    )
    parser.add_argument("--provider", choices=["ollama", "openai"])
    parser.add_argument("--model")
    parser.add_argument("--mode", default=LOOP, choices=[LOOP, PIPELINE])
    parser.add_argument("--dry-run", action="store_true", help="不接模型，只验证链路")
    parser.add_argument("--show-prompt", action="store_true", help="打印发给模型的内容")
    parser.add_argument("--check", action="store_true", help="接入自检：连通性 + 工具调用能力")
    args = parser.parse_args(argv)
    load_dotenv()

    if args.check:
        return run_check(args)

    embedder = build_embedder(args.embedder)
    index = load_index(ROOT / args.index, ROOT / args.corpus, embedder)
    llm = build_llm(args)

    # pipeline 模式下模型不参与检索，没人替它把「白送」翻成「赠与」。
    # 实测加改写后检索 R@6 从 55% 提升到 75%，代价是每次检索多一次 LLM 调用。
    read_tool = ReadTool(index, embedder)
    if args.mode == PIPELINE and not args.dry_run:
        from legalqa.tools.rewriter import make_llm_rewriter

        read_tool = ReadTool(index, embedder, rewriter=make_llm_rewriter(llm), always_expand=True)

    app = build_graph(llm=llm, read_tool=read_tool, mode=args.mode)

    print(f"[i] 语料 {len(index)} 条 | 模式 {args.mode} | 阈值 {config.RAG_THRESHOLD}")
    if args.dry_run:
        if args.provider or args.model:
            print("[!] --dry-run 已开启，--provider / --model 被忽略")
        print("[i] dry-run：不调用真实模型")

    def turn(query: str, state: LegalState | None, turn_id: int) -> LegalState:
        # 不用 checkpointer 时，每次 invoke 都是全新 state：只有显式传入的字段
        # 才会延续到下一轮。messages 必须传，trace 也要传（截尾避免无限增长）。
        payload: dict[str, Any] = {
            "messages": [
                *(list(state.messages) if state else []),
                HumanMessage(content=query),
            ],
            "turn_id": turn_id,
        }
        if state is not None:
            payload["trace"] = list(state.trace)[-50:]
            payload["conversation_summary"] = state.conversation_summary
            payload["summarized_count"] = state.summarized_count

        partial: list[str] = []
        final: dict[str, Any] | None = None
        shown_tools: set[str] = set()
        print()
        try:
            for mode, chunk in app.stream(payload, stream_mode=["messages", "values"]):
                if mode == "messages":
                    message, meta = chunk
                    node = meta.get("langgraph_node")
                    if node == "agent":
                        content = getattr(message, "content", "")
                        if isinstance(content, str) and content:
                            partial.append(content)
                            print(content, end="", flush=True)
                    elif node == "tool_node":
                        # 进入工具节点说明上一段 agent 输出是中间产物
                        # （「我来查一下」这类），不是最终回答。清掉它，
                        # 否则后面比较「流式文本 vs 最终回答」时会误判成
                        # 「回答被输出护栏修正」。
                        partial.clear()
                        call_id = str(getattr(message, "tool_call_id", "") or "")
                        if call_id and call_id not in shown_tools:
                            shown_tools.add(call_id)
                            print("  [检索中…]", end="", flush=True)
                elif mode == "values":
                    final = chunk
        except KeyboardInterrupt:
            print("\n\n[!] 已中断（半截回答已保留，可继续追问）")
            return build_interrupted_state(state, query, "".join(partial), turn_id)

        if final is None:
            print("\n[!] 本轮没有产生结果")
            return state if state is not None else LegalState(turn_id=turn_id)

        result = coerce_state(final)
        # 流式打印的是 Agent 的原始输出；OutputGuard 之后可能追加免责声明
        # 或整条替换（胜诉承诺）。必须把差异补打出来，否则用户看到的
        # 和 state 里的回答不一致 —— 实测流式输出会漏掉免责声明。
        streamed = "".join(partial)
        final_text = final_answer(result)
        if final_text != streamed:
            if streamed and final_text.startswith(streamed):
                print(final_text[len(streamed):], end="")
            else:
                print(f"\n[!] 回答已被输出护栏修正，完整版本：\n{final_text}")
        print()
        if args.show_prompt:
            print("\n──── 发给模型的窗口 ────")
            for message in result.llm_messages:
                print(f"[{type(message).__name__}] {str(message.content)[:300]}")
            print("────────────────────────")
        print(
            f"  ── 通道={result.retrieval_channel or '-'} "
            f"score={result.rag_score:.2f} 轮次={result.iteration_count} "
            f"降级={'是' if result.degraded else '否'}"
        )
        if result.error:
            print(f"  ── error={result.error}")
        return result

    if args.query:
        turn(args.query, None, 0)
        return 0

    state: LegalState | None = None
    turn_id = 0
    print("输入问题开始对话，/quit 退出，/reset 清空上下文")
    while True:
        try:
            line = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in {"/quit", "/exit", "/q"}:
            break
        if line == "/reset":
            state, turn_id = None, 0
            print("[i] 上下文已清空")
            continue
        state = turn(line, state, turn_id)
        turn_id += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
