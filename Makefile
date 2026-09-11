# Attest 质证 · 开发便捷命令
#
# Windows 用户请用 scripts\start.bat（本机无 make）。
# 本 Makefile 面向 macOS / Linux，同时给出等价的底层命令便于排查。

PY := .venv/bin/python
PY_WIN := .venv/Scripts/python.exe

.PHONY: help install dev api web test smoke verify clean tag-p6

help:
	@echo "make install  安装前后端依赖"
	@echo "make dev      同时起后端(8000) + 前端(5173)"
	@echo "make api      只起后端  http://127.0.0.1:8000"
	@echo "make web      只起前端  http://127.0.0.1:5173"
	@echo "make test     跑 pytest（离线，不耗额度）"
	@echo "make smoke    跑端到端探针 + 前端契约探针"
	@echo "make tag-p6   打 p6 标签（需先 make smoke 全绿）"

install:
	python -m venv .venv
	$(PY) -m pip install -r requirements.txt
	cd frontend && npm install

# 同时起两个服务；Ctrl-C 一并退出
dev:
	@echo "后端 -> http://127.0.0.1:8000   前端 -> http://127.0.0.1:5173"
	@trap 'kill 0' EXIT INT TERM; \
	 $(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload & \
	 (cd frontend && npm run dev) & \
	 wait

api:
	$(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

web:
	cd frontend && npm run dev

test:
	$(PY) -m pytest -q

# 三层验证：单测 → 真 ASGI 栈冒烟 → 真 TCP 端口端到端
smoke:
	$(PY) -m pytest -q
	$(PY) scripts/api_smoke.py
	$(PY) scripts/p6_e2e_probe.py
	$(PY) scripts/p6_confirm_probe.py   # 需 ATTEST_HUMAN_CONFIRM=1
	@echo "== 前端契约探针需先起服务：make dev，然后另开终端跑 =="
	@echo "   $(PY) scripts/p6_frontend_probe.py"

tag-p6:
	git tag -a p6 -m "P6 Web 工作台：FastAPI/SSE 后端 + React 前端（求职展示版本）"
	@echo "已打标签 p6。推送：git push origin main --tags"

clean:
	rm -rf frontend/dist .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
