#!/usr/bin/env bash
# LegalQA 生产部署脚本：首次准备 .env，再构建并启动 compose。
# 用法：bash scripts/deploy.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    cp .env.production.example .env
    echo "[!] 已生成 .env（模板），请先填写 LEGALQA_LLM_API_KEY 与 LEGALQA_ADMIN_USERNAME，再重跑本脚本。"
    exit 1
fi

if grep -qE 'LEGALQA_LLM_API_KEY=sk-xx' .env; then
    echo "[x] .env 里还是占位 API Key，请填写真实密钥后再部署。"
    exit 1
fi

echo "[*] 构建并启动（app + postgres + redis）…"
docker compose up -d --build

echo "[*] 容器状态："
docker compose ps

echo ""
echo "完成。检查：http://127.0.0.1:8000/api/health"
echo "公网 HTTPS：见 deploy/Caddyfile 或 deploy/nginx-legalqa.conf，"
echo "并记得把 .env 的 LEGALQA_COOKIE_SECURE 设为 1。"
