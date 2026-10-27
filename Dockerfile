# LegalQA 生产镜像：先构建前端，再打包后端。
# 数据（语料/评测）打进镜像；运行期产生的 app.db / 导入文件 / 备份放 /app/data 卷。

# -- 前端构建 ----------------------------------------------------------
FROM node:22-alpine AS frontend
WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm install --registry=https://registry.npmmirror.com
COPY frontend/ ./
RUN npm run build

# -- 后端运行时 ----------------------------------------------------------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml ./
COPY legalqa ./legalqa
RUN pip install --no-cache-dir -e ".[web]" -i https://mirrors.aliyun.com/pypi/simple

COPY --from=frontend /build/frontend/dist ./frontend/dist
COPY data/corpus ./data/corpus
COPY data/eval ./data/eval
COPY .env.example ./.env.example

# 向量索引不入镜像（体积大）：启动时按 .env 的 embedding 配置自动重建。
RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

# 多 worker 需要 LEGALQA_DATABASE_URL + LEGALQA_REDIS_URL（见 .env.example）
CMD ["sh", "-c", "uvicorn legalqa.api:app --host 0.0.0.0 --port 8000 --workers ${LEGALQA_WORKERS:-1}"]
