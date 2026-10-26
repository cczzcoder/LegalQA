# LegalMind · 中文法律问答（LangGraph + RAG）

给普通用户用的中文法律问答。用户用大白话描述处境（「老板没签劳动合同干了快一年，能要双倍工资吗」），系统检索中国法条并给出**带引用、带版本、可溯源**的回答。

核心约束：**法律场景不能编。** 引用必须能落到具体条文与版本；宁可说「没查到」，也不能编条号。

## 当前状态（2026-10-26）

| 指标 | 结果 | 数据 |
|---|---|---|
| 单元/属性测试 | **449 passed** | `pytest tests/ -q` |
| 34 条手工回归集 | **34/34（100%）** | `data/eval/qwen37_barefix_34.json` |
| 100 条合成端到端 | **89/100** | `data/eval/qwen37_fact_100.json` |
| 120 条检索集 R@1 / R@6 / MRR | **49.2% / 77.5% / 0.607** | `run_retrieval_eval.py` |
| 引用可溯源 | 97%→98%（裸引修复后；剩 2 条真编造） | 见上 |
| 免责声明 | 100% | 见上 |

> 100 条合成集有 15% 的单条翻转、±1~2pt 总数噪声；89 与 84 的差异不要当精确结论。合成金标已做过一轮审计（E15），仍不是人工标注。

## 架构

```
START → InputGuard → ContextCompress → Agent ⇄ ToolNode → OutputGuard → END
                     InputGuard ─(拦截)→ StaticFallback → SaveState → END
```

- `loop`（默认）：模型自己决定调不调 `read_law`、查几次、换什么词
- `pipeline`（降级）：检索前置一次，模型不绑工具
- 循环终止由确定性代码保证：工具解绑 + 预算 + 路由三层兜底
- 上下文压缩：system 物理外置 + 轮次/token 双约束 + 工具配对修复 + L1~L4 降级 + 滚动摘要
- 输出护栏：胜诉承诺拦截、引用回捞/核实、废止条文警示、免责声明

## 技术栈

| 项 | 选择 |
|---|---|
| 编排 | LangGraph（带环 Agent Loop） |
| 生成 | qwen-plus（阿里云百炼，OpenAI 兼容端点） |
| Embedding | qwen3.7-text-embedding（DashScope 原生接口，1024 维，`text_type=query/document` + 法律 instruct） |
| 语料 | 23 部法律 4264 条 |
| 检索 | 内存向量索引（余弦）+ 条号精确通道 + 融合排序 |
| 评测 | 规则化检查（引用可溯源、无幻觉、命中期望、拒答、长度、免责） |

## 快速开始

```bash
# 1) 环境（Windows Git Bash）
./.venv/Scripts/python.exe -m pip install -e ".[dev]"

# 2) 配置：复制 .env.example 为 .env，填 DashScope API Key
cp .env.example .env

# 3) 测试
./.venv/Scripts/python.exe -m pytest tests/ -q          # 449 passed

# 4) 接入自检（LLM 连通性 / 工具调用 / embedding / 索引）
./.venv/Scripts/python.exe -m legalqa.cli --check

# 5) 单次提问（loop 模式）
./.venv/Scripts/python.exe -m legalqa.cli --query "公司拖欠工资三个月了怎么办"

# 6) 评测
./.venv/Scripts/python.exe scripts/run_eval.py \
    --cases data/eval/cases.jsonl --workers 4 --json data/eval/run_34.json
./.venv/Scripts/python.exe scripts/run_eval.py \
    --cases data/eval/synth100.jsonl --workers 4 --json data/eval/run_100.json
./.venv/Scripts/python.exe scripts/run_retrieval_eval.py \
    --cases data/eval/synthetic.jsonl --index data/index/all
```

索引与 `.env` 不入库：`data/index/` 可用 `scripts/ingest_folder.py` 重建；`.env` 见 `.env.example`。

## 已知限制

- 100 条剩余 11 条失败：2 条模型显式编造条号（79/66）、2 条检索到没引、4 条模型工具 query 漏事实、2 条近邻排序、1 条语义 gap。
- 合成金标是弱标签；34 条手工集样本小。
- 当前只有 CLI 与评测脚本，**没有前端**。
- 阶段 0 是只读问答：不写文件、不调外部业务系统；阶段 1 的审批/写操作字段只预留未接线。

## 文档

- [阶段 0 架构设计](docs/阶段0-架构设计.md)：ADR、不变量、失效模式、压缩/检索/评测
- [实验记录](docs/实验记录.md)：E1~E22，每条含动机/数据/结论
- `.env.example`：模型与 embedding 配置

## 下一步

1. 前端 MVP：聊天界面 + 法条引用卡片/原文侧栏 + 流式输出 + 历史会话（见前端方案）。
2. 剩余 11 条逐案修复；不再做通用 prompt/补召回/换模型实验。
3. 评测集扩到人工标注；补多轮/中断场景。
