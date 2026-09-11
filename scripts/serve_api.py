"""P6 常驻后端：进程内起 uvicorn，前台阻塞（供浏览器验收 / 用户手测）。

本环境 Bash 后台 `&` 进程在工具返回后会被杀，所以本脚本**前台阻塞**运行，
由调用方用 run_in_background 交给工具管理。

用法：
    ATTEST_HUMAN_CONFIRM=1 python scripts/serve_api.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn  # noqa: E402

from app.main import create_app  # noqa: E402

if __name__ == "__main__":
    port = int(os.environ.get("ATTEST_PORT", "8000"))
    print(f"[serve_api] http://127.0.0.1:{port}  human_confirm={os.environ.get('ATTEST_HUMAN_CONFIRM', '0')}", flush=True)
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")
