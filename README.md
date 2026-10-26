# LegalMind · 中文法律问答（LangGraph + RAG）

给普通用户用的中文法律问答。用户用大白话描述处境（「老板没签劳动合同干了快一年，能要双倍工资吗」），系统检索中国法条并给出**带引用、带版本、可溯源**的回答。

核心约束：**法律场景不能编。** 引用必须能落到具体条文与版本；宁可说「没查到」，也不能编条号。

## 当前状态（2026-10-26）

| 指标 | 结果 | 数据 |
|---|---|---|
| 单元/属性测试 | **459 passed** | `pytest tests/ -q` |
| 34 条手工回归集 | **34/34（100%）** | `data/eval/qwen37_barefix_34.json` |
| 100 条合成端到端 | **89/100** | `data/eval/qwen37_fact_100.json` |
| 120 条检索集 R@1 / R@6 / MRR | **49.2% / 77.5% / 0.607** | `run_retrieval_eval.py` |
| 引用可溯源 | 97%→98%（裸引修复后；剩 2 条真编造） | 见上 |
| 免责声明 | 100% | 见上 |
| Web 前端 | FastAPI SSE + Vue 3（登录/会话落库/限流/法条引用侧栏） | `frontend/`、`legalqa/api.py` |

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
./.venv/Scripts/python.exe -m pytest tests/ -q          # 459 passed

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

## Web 前端（FastAPI + Vue 3）

```bash
# 安装 Web 依赖
./.venv/Scripts/python.exe -m pip install -e ".[web]" \
    -i https://mirrors.aliyun.com/pypi/simple

# 启动 API（同时托管 frontend/dist）
./.venv/Scripts/python.exe -m uvicorn legalqa.api:app \
    --host 127.0.0.1 --port 8000

# 前端开发模式（另开终端；Vite 代理 /api → 8000）
cd frontend
npm install --registry=https://registry.npmmirror.com
npm run dev            # http://localhost:5173

# 生产构建：构建后直接访问 http://127.0.0.1:8000
npm run build
```

- 认证：`POST /api/auth/register` / `login` / `logout`，`GET /api/auth/me`
  - 密码 PBKDF2-SHA256 加盐（200k 次）；登录令牌只存 SHA-256 哈希；httpOnly Cookie，7 天
- 会话：`GET/POST /api/sessions`，`GET/DELETE /api/sessions/{id}`
- 聊天：`POST /api/sessions/{id}/chat`，SSE，事件类型 `status / delta / reset / replace / citations / meta / done / error`
- 健康检查：`GET /api/health`
- 会话落库：SQLite `data/app.db`（用户、令牌、会话、消息、引用、meta）；服务端无状态，历史来自数据库
- 限流（进程内滑动窗口，单进程 MVP）：
  - 聊天：默认 10 次/分钟、60 次/小时/用户
  - 登录：8 次/分钟/IP + 4 次/分钟/用户名；注册：5 次/分钟/IP
  - 环境变量见 `.env.example`；多 worker 部署时换 Redis
- 前端：Vue 3 + Vite + Pinia + Element Plus + marked/DOMPurify；登录/注册、流式聊天、法条引用侧栏、服务端历史会话、移动端抽屉
- API Key 只在服务端 `.env`，前端不接触密钥

移动端实测（764px 视口；桌面 >1024px 为侧栏 + 聊天 + 法条面板固定三栏）：

<p>
  <img src="docs/assets/web-login.png" width="24%" alt="登录页" />
  <img src="docs/assets/web-chat.png" width="24%" alt="聊天与流式回答" />
  <img src="docs/assets/web-sidebar.png" width="24%" alt="侧栏抽屉" />
  <img src="docs/assets/web-citations.png" width="24%" alt="法条依据面板" />
</p>

## 已知限制

- 100 条剩余 11 条失败：2 条模型显式编造条号（79/66）、2 条检索到没引、4 条模型工具 query 漏事实、2 条近邻排序、1 条语义 gap。
- 合成金标是弱标签；34 条手工集样本小。
- Web 认证是 MVP：开放注册，没有邮箱验证/找回密码/多租户；限流在进程内，多 worker 需换 Redis。
- 生产部署建议 HTTPS + `LEGALMIND_COOKIE_SECURE=1`，并补 CSRF token/反向代理限流。
- 阶段 0 是只读问答：不写文件、不调外部业务系统；阶段 1 的审批/写操作字段只预留未接线。

## 文档

- [阶段 0 架构设计](docs/阶段0-架构设计.md)：ADR、不变量、失效模式、压缩/检索/评测
- [实验记录](docs/实验记录.md)：E1~E22，每条含动机/数据/结论
- `.env.example`：模型与 embedding 配置

## 下一步

1. Web 生产化：Postgres/Redis、邮箱验证/找回密码、部署脚本与反向代理。
2. 前端补充：法条库浏览、评测看板、引用点击定位、反馈采集。
3. 剩余 11 条逐案修复；评测集扩到人工标注；补多轮/中断场景。
