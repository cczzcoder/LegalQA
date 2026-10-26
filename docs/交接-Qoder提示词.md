# 交接提示词（给 Qoder）

> 用法：把下面 `===` 之间的内容整段粘给 Qoder。
> 它是自包含的——对方没有本次对话的任何上下文。

---

======================================================================
【项目】LangGraph 长对话法律问答系统（阶段 0：只读链路）
【仓库】D:/git仓库/法律问答
【Python】./.venv/Scripts/python.exe（Windows，Git Bash 环境）
【当前状态】100 条端到端评测 60.0%，34 条手写回归集 97.1%，428 项测试全绿
======================================================================

## 0. 先做三件事（别跳过）

1. **先读这两份文档**，它们是这个项目的全部设计决策与实验证据：
   - `docs/阶段0-架构设计.md`（约 1900 行，含 ADR-01~06、不变量 INV-1~9、
     失效模式 CM/BC/T 编号表、检索 §6.4、压缩 §7、评测 §14）
   - `docs/实验记录.md`（E1~E13，每条都有「动机 → 数据 → 结论」，
     记录了**哪些看起来该做的优化实测有害**，别重复踩）
2. **确认工作区干净**：`git status`。上一轮改动尚未提交（13 个文件修改 + 若干
   eval JSON）。先 `git add -A && git commit` 打个基线，再动手。
3. **跑一遍基线确认环境可用**：
   ```bash
   ./.venv/Scripts/python.exe -m pytest tests/ -q                      # 应 428 passed
   ./.venv/Scripts/python.exe scripts/run_eval.py --cases data/eval/cases.jsonl \
       --workers 4 --json data/eval/baseline_check.json                # 应 33/34 (97.1%)
   ```

## 1. 项目在做什么

给普通用户用的中文法律问答。用户用大白话描述处境（「老板没跟我签合同干了快一年，
能要双倍工资吗」），系统检索中国法条并给出带引用的回答。

**核心约束：法律场景不能编。** 引用必须可溯源到具体条文，且必须标注版本
（法律有时效性）。宁可说「没查到」，也不能编一个条号。

**技术形态**：LangGraph 带环 Agent Loop，对标 Claude Code 的 harness 思路——
循环由模型驱动（它自己决定查不查、查几次、换什么词），但**终止和边界由确定性
代码保证，绝不依赖模型自觉**。

```
START → InputGuard → ContextCompress → Agent ⇄ ToolNode
                                          ↓
                                     OutputGuard → END
                     InputGuard ─(拦截)→ StaticFallback → SaveState → END
```

- `loop` 模式（默认）：Agent 绑工具，自己决定调不调 `read_law`
- `pipeline` 模式（降级）：不绑工具，检索前置一次

**环境**：阿里云百炼（OpenAI 兼容端点），`qwen-plus` + `text-embedding-v4`（1024 维），
`temperature=0`。语料 23 部法律 4264 条，索引在 `data/index/all`。
用户机器 15.2GB 内存但只剩 ~628MB 可用、纯 CPU——所以 embedding 也上了云
（E12，本地 bge-m3 要占 1.16GB）。

## 2. 代码地图

```
legalqa/
  state.py        LegalState dataclass，36 字段**全部有默认值**；带 .get()
  config.py       SYSTEM_PROMPT（硬编码、逐字节稳定）、全部阈值与上限、FALLBACK_TEXTS
  graph.py        build_graph(mode) + route_after_guard / route_after_agent
  nodes/
    guards.py     input_guard_node（含 TURN_SCOPED_RESET）、make_output_guard_node(index)
    agent.py      make_agent_node（循环核心）、make_tool_node
    context.py    make_context_compress_node(llm)，四级降级 L1~L4 + 滚动摘要
    fallback.py   static_fallback_node、save_state_node
  pure/           纯函数，无 IO，好测
    window.py     repair_tool_pairs / validate_window（INV-1~9）/ insert_before_last_human
    retrieval.py  extract_citations / cn_to_int / same_law / fuse_rank / safe_cosine
    rewrite.py    needs_rewrite（三层预判）/ entities_preserved
    summary.py    滚动摘要的 prompt 与 sanitize
    terms.py      ⚠️ 实验性，实测有害，**没有接进链路**
  corpus/
    ingest.py     docx/pdf 解析成 Article（version_date 来自文件名）
    index.py      VectorIndex：build / build_incremental / search / by_article
    bm25.py       ⚠️ 实验性，默认关闭，实测有害
  tools/
    read_tool.py  ReadTool（双通道检索）+ format_context + READ_TOOL_SPEC
    reranker.py   make_llm_reranker（LLM 重排 top-20 → top-6）
    rewriter.py   make_llm_rewriter（仅 pipeline 用）
  eval.py         规则化检查项 + CaseReport + strip_backfill_appendix
  llm_factory.py  多 provider 工厂，单例
scripts/
  run_eval.py           端到端评测（--workers / --rerank / --rewrite / --json）
  run_retrieval_eval.py 检索层评测（Recall@K / MRR）
  build_corpus.py / ingest_folder.py / calibrate_threshold.py / build_synthetic_eval.py
```

## 3. 必须守住的不变量（改任何代码前先读）

这些不是风格偏好，每条都对应一次真实事故：

1. **State 36 个字段全部有默认值，代码里一律 `state.get("x", 默认)`。**
   `LegalState` 刻意**不实现 `__getitem__`**——`state["x"]` 会抛 TypeError，
   用类型系统挡住 KeyError。
2. **`messages` 用 `add_messages` reducer**：合并而非替换。所以「从窗口里删掉」
   不等于「从历史里消失」。因此有**双轨**：`messages`（全量历史）+
   `llm_messages`（每轮由 ContextCompress 重算的视图，只读）。
3. **`trace` 用 `operator.add`**。忘了 reducer 会让后一个节点覆盖前一个，
   7 轮对话只剩 7 条 trace。
4. **SystemPrompt 物理外置**：`[SystemMessage(硬编码)] + trim(body)`，
   `trim_messages(..., include_system=False)`。system 参与 trim 会破坏前缀缓存，
   且压缩时可能被裁掉。
5. **`trim_messages` 的 `allow_partial=False` 只保证「不切断单条消息」，
   不保证不切断 tool_call/tool_call_id 配对。** 裁完必须 `repair_tool_pairs()`。
6. **写 `messages` 的只有 4 个节点**：Agent / ToolNode / OutputGuard / SaveState。
   InputGuard **只判定不写**，拦截文案统一由 SaveState 落盘。
7. **OutputGuard 用同 id 替换，不追加**（`add_messages` 同 id 是 update）。
   追加会让坏内容留在历史里自我污染。
8. **轮末最后一条必须是 AIMessage 且不带 `tool_calls`。** 出口停在 tool_calls 上
   会让用户收到 0 字回答（E13 事故）。已加四层防护，别拆。
9. **`TURN_SCOPED_RESET` 里 26 个字段每轮清零。** 漏一个就出跨轮残留 bug
   （`fallback_saved` 残留 → 第二轮降级不写历史；`iteration_count` 残留 →
   第二轮一进来就超预算）。这类 bug 单轮测试永远发现不了。
10. **阈值只吃原始余弦 `rag_score`，不吃融合分/归一化分。** 融合分是相对量、
    跨 query 不可比，拿它比阈值等于永远不 fallback。

## 4. 当前指标与噪声基线（**最重要的一节**）

| 指标 | 基线（E12） | 现在 |
|---|---|---|
| 100 条端到端通过率 | 44.0% | **60.0%** |
| 34 条手写回归集 | 91.2% | **97.1%** |
| 免责声明 | 95% | 100% |
| 长度合理 | 97% | 98% |
| 引用可溯源 | 77% | 97% |
| 无幻觉条号 | 100% | 99% |
| 命中期望法条 | 59% | 62% |

**噪声基线（务必记住）**：同一份代码连跑两次得 60.0% 和 59.0%，
总数只差 ±1pt，但**逐条有 15% 的用例翻转**。原因是模型每次生成的检索 query
不同 → 召回不同 → 下游全变。`temperature=0` 只固定采样，固定不了「同一个问题
会怎么措辞」。

**所以：±1~2pt 的差异一律视为噪声，不要据此做决策。**
这也是为什么历史上 20 条样本得出的结论（改写 +20pt、重排 -5pt）在 100 条上
完全翻转（改写 -4pt、重排 +4pt）。**小样本单次评测给出的结论是随机的。**

## 5. 现在的瓶颈在哪

60% 之外的 38 条失败**全部**卡在「命中期望法条」这一项，其余检查项全部 ≥97%。
用评测 JSON 里的 `retrieved` 字段逐条判定：

- **32 条：检索根本没捞到期望法条**（占 84%）
- 6 条：捞到了但模型没引（占 16%）

即：**检索 R@6 65.8% 是硬天花板**，端到端 62% ≈ 65.8%，两者已经贴在一起。

## 6. 下一步该做什么（按性价比排序）

### 方向 A（首选）：提升检索召回
1. **换更强 embedding**。现在 `text-embedding-v4`，CMTEB Retrieval 73.98。
   试 `qwen3-embedding` 系列。注意：换模型必须**重建索引**，
   且评测脚本 `--embedder` 必须跟着改（曾经因为默认值写死 `ollama`，
   拿 bge-m3 的 query 向量去查 text-embedding-v4 的索引，
   Recall 全 0 且不报错、0.7 秒跑完）。
2. **DashScope 原生 SDK 的 `text_type=query/document`**。现在走 OpenAI 兼容端点，
   没法区分 query/document 向量，这是已知损失。需要绕过 `llm_factory` 的
   OpenAI 兼容封装，直接用 dashscope SDK 调 embedding。
3. **不要重试已否决的方案**（都有数据）：
   - 口语→术语词表扩展：R@6 60.8% → 57.5%，**有害**
   - BM25 混合召回：60.8% → 53.3%，**有害**
   - loop 模式下强制 LLM 改写：端到端 -4pt，**有害**
     （pipeline 模式保留，那里没有别的改写机制）
   - `RAG_TOP_K` 6 → 8：更差

### 方向 B：生成侧（天花板 +6pt）
让模型把已经检索到的 6 条引对。现在 `format_context` 末尾有【引用约束】，
可以试试更强的约束、few-shot、或对「引了材料外条号」做一次纠正性重生成。

### 方向 C：功能扩展（阶段 1）
State 里已经预留了 `pending_action` / `requires_approval` / `approval_granted` /
`tool_permissions` 四个字段，但**都没接线**。阶段 1 是写操作 + 人工审批。

## 7. 改动的正确姿势

**任何改动都必须走这条链路，否则你无法知道自己是改好了还是改坏了：**

```bash
# 1) 单元/属性测试（秒级，428 项）
./.venv/Scripts/python.exe -m pytest tests/ -q

# 2) 34 条手写回归集（约 1.5 分钟）—— 必须 ≥33/34
./.venv/Scripts/python.exe scripts/run_eval.py --cases data/eval/cases.jsonl \
    --workers 4 --json data/eval/try_34.json

# 3) 100 条合成集（约 6 分钟）—— 必须 ≥59/100 才算没退化
./.venv/Scripts/python.exe scripts/run_eval.py --cases data/eval/synth100.jsonl \
    --workers 4 --json data/eval/try_100.json

# 4) 检索层（约 20 秒）—— 只有动检索才需要
./.venv/Scripts/python.exe scripts/run_retrieval_eval.py
```

**评测 JSON 里存了 `answer`（完整回答）和 `retrieved`（检索到的法条对），
诊断失败原因时直接读 JSON，不要为了看一条 case 重跑 6 分钟。**

`--workers 4` 是必要的：100 条串行要 20 分钟，跑不动迭代。

## 8. 已知坑（都是踩过的）

- **`parse_articles` 内部空格**：PDF 会输出「第二十七 条」，正则必须允许条号
  内部空格，否则静默丢条文（民用航空法曾因此丢 16 条）。
- **条号必须严格递增**：否则「第十六条至第十九条」这种回指会被解析成假条文
  （商标法曾多出 18 条假条文）。
- **扫描版 PDF**：`read_pdf_paragraphs` 会抛 `ScannedPdfError`。多版本选择时
  要按日期依次尝试直到解析成功，否则整个法律会消失（宪法曾整个丢失）。
- **slug 冲突**：「解释（一）」「解释（二）」截断到 12 字符会撞车，要保留尾部序号。
- **`retrieved_docs` 没有 reducer**：ToolNode 第二次调用会覆盖第一次的结果。
  必须从 `state.get("retrieved_docs")` 累积。
- **`call_signature` 用 sha256 不用内置 `hash`**：后者带随机盐，跨进程不稳定。
- **`extract_citations` 法律名与条号之间不能跨 `。`**：否则
  「《通知》……。依据《工伤保险条例》第十四条」会把 14 归到《通知》名下。
- **引用判断必须按 `(法律名, 条号)` 成对比较**：「38 在条号集合里」+
  「劳动合同法在法源集合里」≠「劳动合同法第 38 条在材料里」。
- **回捞附录里是法条原文**，其中的交叉引用（「依照本法第三十八条」）不是模型的
  引用行为，所有文本类检查必须先用 `strip_backfill_appendix()` 切掉附录。
- **`add_messages` 同 id 是 update**：模型复用历史 id 时新回答会顶掉旧消息位置，
  `messages[-1]` 变成 ToolMessage → OutputGuard 判「不是最终回答」→ 空白回答。

## 9. 交付要求

- **中文注释写「为什么」，不写「是什么」。** 代码本身能看出做了什么，
  注释要留下「为什么这么选、不这么选会怎样、对应哪次事故」。
- **改完更新 `docs/实验记录.md`**：加一条 E14，写清动机、数据、结论。
  数据不支持的结论不要写。
- **不要删 `.workbuddy-ai/` 目录**（项目记忆）。
- **不要在 C 盘写临时文件**。临时脚本/日志写到 `D:/pip-cache/tmp/`。
  Git Bash 的 `/tmp` 就是 `C:\Users\...\Local\Temp`，等于写 C 盘。
- **装 Python 包走清华镜像** `-i https://pypi.tuna.tsinghua.edu.cn/simple`，
  大包不要加 `-q`（静默模式下看不出是下载还是卡死）。

======================================================================

## 附：第一句可以直接这样说

> 我在 D:/git仓库/法律问答 有一个 LangGraph 法律问答项目。先读
> `docs/阶段0-架构设计.md` 和 `docs/实验记录.md`，再跑 `pytest tests/ -q`
> 和 `scripts/run_eval.py --cases data/eval/cases.jsonl --workers 4` 确认基线
> （应为 428 passed、33/34）。然后我们讨论提升检索召回——现在端到端 60%，
> 84% 的失败是检索没捞到期望法条。**先别改代码**，先告诉我你读完文档后
> 认为哪三个方向最值得试，以及各自的预期收益和验证方式。

======================================================================
