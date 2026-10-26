"""极简 .env 加载。

不引入 python-dotenv：十几行能解决的事不值得多一个依赖。
所有脚本（CLI / 建索引 / 评测）都要用它，所以单独成模块——
之前只有 CLI 里有一份，导致 `ingest_folder.py` 读不到 API Key。
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["load_dotenv", "PROJECT_ROOT"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | str | None = None) -> None:
    """加载项目根目录的 .env。

    规则：`KEY=VALUE`，`#` 开头为注释，**已存在的环境变量优先**（不覆盖）。
    """
    env_file = Path(path) if path else (PROJECT_ROOT / ".env")
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
