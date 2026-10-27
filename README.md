# LegalQA · 中文法律问答（LangGraph + RAG）

> 独立项目说明：本仓库是「阶段 0 长对话法律问答」的重写版（LangGraph + RAG + Vue 全栈），
> 与原项目 **LegalMind**（律师/法务向法律信息辅助系统，`github.com/cczzcoder/LegalMind`）
> 是两个不同的项目，各自独立演进。

给普通用户用的中文法律问答。用户用大白话描述处境（「老板没签劳动合同干了快一年，能要双倍工资吗」），系统检索中国法条并给出**带引用、带版本、可溯源**的回答。

核心约束：**法律场景不能编。** 引用必须能落到具体条文与版本；宁可说「没查到」，也不能编条号。

## 当前状态（2026-10-27）

| 指标 | 结果 | 数据 |
|---|---|---|
| 单元/属性测试 | **495 passed** | `pytest tests/ -q` |
| 34 条手工回归集 | **34/34（100%）** | `data/eval/qwen37_barefix_34.json` |
| 100 条合成端到端 | **90/100**（历史最佳；单轮波动 84~90） | `data/eval/qwen37_partial_100.json` |
| 多轮对话（8 组 20 轮） | **20/20** | `scripts/run_multiturn_eval.py`、`data/eval/multiturn.jsonl` |
| 120 条检索集 R@1 / R@6 / MRR | **49.2% / 77.5% / 0.607** | `run_retrieval_eval.py` |
| 引用可溯源 | 裸引修复 + 编造引用剥离后，本轮 100 集无幻觉条号 100% | 见上 |
| 免责声明 | 100% | 见上 |
| Web 前端 | FastAPI SSE + Vue 3（登录/会话落库/限流/引用侧栏/法条库/评测看板/管理端） | `frontend/`、`legalqa/api.py` |

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
./.venv/Scripts/python.exe -m pytest tests/ -q          # 495 passed

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
./.venv/Scripts/python.exe scripts/run_multiturn_eval.py   # 多轮/追问/指代（8 组 20 轮）
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
  - **邮箱验证**：注册可带邮箱（`verify-email/resend`、`verify-email`）；未配 SMTP 时
    验证码打印到服务端控制台（`LEGALQA_EMAIL_CONSOLE=1` 时响应附带调试码），配
    `LEGALQA_SMTP_HOST` 后走邮件
  - **找回密码**：`password/forgot` + `password/reset`（验证码 10 分钟有效，重置后
    吊销全部登录令牌；响应不区分账号是否存在，防用户名枚举）
  - **CSRF 双提交**：`lm_csrf` cookie + `X-CSRF-Token` 头，所有不安全方法（POST/PATCH/DELETE）校验
  - **安全响应头**：CSP / X-Frame-Options / X-Content-Type-Options / Referrer-Policy / Permissions-Policy
  - **审计日志**：登录/注册/退出与管理员操作落 `audit_log`，`GET /api/admin/audit` 查看
- 会话：`GET/POST /api/sessions`，`GET/DELETE /api/sessions/{id}`
- 聊天：`POST /api/sessions/{id}/chat`，SSE，事件类型 `status / delta / reset / replace / citations / meta / done / error`
- 法条库：`GET /api/laws`（目录与统计）、`GET /api/laws/articles?law=&q=&offset=&limit=`
  - 支持按法律浏览、全文关键词（中文 bigram）、`第八十七条`/`87` 条号定位
- 评测看板：`GET /api/eval/runs`、`GET /api/eval/runs/{id}`、`GET /api/eval/runs/{id}/cases/{case_id}`、`GET /api/eval/compare?left=&right=`
  - 只读 `data/eval/*.json`，不重跑模型；失败用例可展开看答案与检索材料；支持两个批次并排对比（通过率/耗时/失败数与各检查项差量，绿=变好、红=变差）
- 反馈：`POST /api/feedback`（登录用户对某条回答提交反馈，含回答快照）
- 管理端（仅 `role=admin`，`/api/admin/*` 前缀）：
  - 概览：`GET /api/admin/overview`（未读反馈、待处理导入、索引落后、版本年限提醒）
  - 反馈：`GET /api/admin/feedback`、`PATCH /api/admin/feedback/{id}`
  - 导入：`POST/DELETE /api/admin/imports`，`POST /api/admin/imports/{id}/analyze|apply`
    - 上传 .docx/.pdf（≤20MB）→ 分析（查重：法名+版本+条文重合率；时效：版本日期/新旧比较）→ 确认入库
    - 入库 = 语料备份 + 同名新版替换旧版 + **增量 embedding**（只向量化新条文），服务热重载索引
- 健康检查：`GET /api/health`
- 会话落库：SQLite `data/app.db`（用户、令牌、会话、消息、引用、meta）；服务端无状态，历史来自数据库
- 限流（进程内滑动窗口，单进程 MVP）：
  - 聊天：默认 10 次/分钟、60 次/小时/用户
  - 登录：8 次/分钟/IP + 4 次/分钟/用户名；注册：5 次/分钟/IP
  - 环境变量见 `.env.example`；多 worker 部署时换 Redis
- 前端：Vue 3 + Vite + Pinia + Element Plus + marked/DOMPurify；登录/注册、流式聊天、法条引用侧栏、**回答内引用点击定位**（《法》第X条可点击，面板滚动高亮对应条文）、服务端历史会话、法条库浏览、评测看板（含批次对比）、回答反馈、管理端（反馈/导入/语料/提醒）、移动端抽屉
- API Key 只在服务端 `.env`，前端不接触密钥

移动端实测（764px 视口；桌面 >1024px 为侧栏 + 聊天 + 法条面板固定三栏）：

<p>
  <img src="docs/assets/web-login.png" width="24%" alt="登录页" />
  <img src="docs/assets/web-chat.png" width="24%" alt="聊天与流式回答" />
  <img src="docs/assets/web-sidebar.png" width="24%" alt="侧栏抽屉" />
  <img src="docs/assets/web-citations.png" width="24%" alt="法条依据面板" />
</p>

<p>
  <img src="docs/assets/web-library.png" width="49%" alt="法条库浏览" />
  <img src="docs/assets/web-eval.png" width="49%" alt="评测看板" />
</p>

<p>
  <img src="docs/assets/web-admin.png" width="49%" alt="管理端概览" />
</p>

### 管理端怎么用

```bash
# 1) 先注册一个普通账号（如 demo），再在 .env 里指定它为管理员
LEGALQA_ADMIN_USERNAME=demo
./.venv/Scripts/python.exe -m uvicorn legalqa.api:app --host 127.0.0.1 --port 8000
```

- 登录后顶栏出现「管理」标签，普通用户看不到。
- 反馈：用户在回答下方点「反馈」，管理端「反馈」页可筛选并标记已处理/忽略。
- 导入：「导入」页上传 .docx/.pdf → 点「分析」看查重/时效报告 → 「确认入库（替换旧版）」。入库会备份语料到 `data/backups/`、增量更新向量索引；若 embedding 暂不可用，语料仍入库，概览页会提示「索引落后」，重启服务后自动全量重建。
- 概览页提醒：未读反馈数、待处理导入、索引落后、无版本日期/版本超 5 年的法律。

### Docker 部署（app + Postgres + Redis）

```bash
docker compose up -d --build    # 读取 .env（可用 .env.production.example 改出）
```

- `Dockerfile`：多阶段构建（前端构建 + 后端运行时）；语料/评测打进镜像，`/app/data` 卷存运行数据；启动时按 `.env` 自动重建向量索引
- `docker-compose.yml`：app + postgres(pg17) + redis；配了 `LEGALQA_DATABASE_URL`/`LEGALQA_REDIS_URL` 即启用外置存储与多 worker 共享限流
- 本地不跑 compose 时也可单独起数据库容器跑集成测试：

```bash
docker run -d --name legalqa-pg-test -e POSTGRES_USER=legalqa \
    -e POSTGRES_PASSWORD=legalqa -e POSTGRES_DB=legalqa \
    -p 127.0.0.1:5433:5432 pgvector/pgvector:pg17
docker run -d --name legalqa-redis-test -p 127.0.0.1:6380:6379 redis:7-alpine
LEGALQA_TEST_DATABASE_URL=postgresql://legalqa:legalqa@127.0.0.1:5433/legalqa \
LEGALQA_TEST_REDIS_URL=redis://127.0.0.1:6380/0 \
./.venv/Scripts/python.exe -m pytest tests/test_postgres.py -v
```

- 生产要求：HTTPS 反向代理后置 `LEGALQA_COOKIE_SECURE=1`，并指定 `LEGALQA_ADMIN_USERNAME`
- 一键部署：`bash scripts/deploy.sh`（首次生成 .env 模板并校验密钥占位，然后 compose 构建启动）
- HTTPS 反向代理示例：`deploy/Caddyfile`（自动证书）与 `deploy/nginx-legalqa.conf`（certbot；含 SSE 关缓冲、上传 25MB 上限）
- 多 worker：`LEGALQA_WORKERS=4`（必须已配 `LEGALQA_DATABASE_URL` + `LEGALQA_REDIS_URL`，否则限流计数不共享）

## 已知限制

- 100 条剩余失败全部是「未引用期望法条」（检索未召回或模型未选择引用），属逐案检索/选择问题；编造条号已被输出护栏剥离，过度拒答已由「部分覆盖先作答」提示词修复。
- 合成金标是弱标签；34 条手工集样本小。
- Web 认证是 MVP：开放注册，没有多租户；邮箱验证/找回密码已支持（SMTP 可选，控制台模式先行）；限流默认进程内，配 `LEGALQA_REDIS_URL` 即切 Redis 滑动窗口（多 worker 共享）。
- 法条库读 `data/index/qwen37/docs.json`（索引缺失时回落到 `data/corpus/all.jsonl`）；评测看板读 `data/eval/*.json`，两者都不调用模型。
- 文献导入的「自动更新」目前只做站内提醒（反馈/索引/版本年限）+ 导入时新旧比对，未联网抓取官方数据库（有反爬与网页改版风险，列为二期）。
- 生产部署建议 HTTPS + `LEGALQA_COOKIE_SECURE=1`，并补 CSRF token/反向代理限流。
- 阶段 0 是只读问答：不写文件、不调外部业务系统；阶段 1 的审批/写操作字段只预留未接线。

## 文档

- [阶段 0 架构设计](docs/阶段0-架构设计.md)：ADR、不变量、失效模式、压缩/检索/评测
- [实验记录](docs/实验记录.md)：E1~E23，每条含动机/数据/结论
- `.env.example`：模型与 embedding 配置

## 下一步

1. 部署脚本与反向代理示例；多租户（如需）。
2. 前端增强：自动更新二期（官方库联网检查提醒）。
3. 剩余 10 条逐案修复；评测集扩到人工标注。
