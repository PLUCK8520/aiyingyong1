@echo off
REM ============================================================
REM  Attest 质证 · 一键启动（Windows）
REM
REM  起两个服务：
REM    后端  http://127.0.0.1:8000   （FastAPI + SSE）
REM    前端  http://127.0.0.1:5173   （Vite dev server，/api 代理到 8000）
REM
REM  用法：
REM    scripts\start.bat              默认不开人工确认断点
REM    set ATTEST_HUMAN_CONFIRM=1 && scripts\start.bat    开大纲确认
REM
REM  关闭：直接关掉两个新开的窗口即可。
REM ============================================================
setlocal

cd /d "%~dp0.."

REM ---- 前置检查 ----
if not exist ".venv\Scripts\python.exe" (
  echo [start] 未找到 .venv\Scripts\python.exe
  echo        请先创建虚拟环境并安装依赖：
  echo          python -m venv .venv
  echo          .venv\Scripts\pip install -r requirements.txt
  exit /b 1
)

if not exist "frontend\node_modules" (
  echo [start] 未找到 frontend\node_modules
  echo        请先安装前端依赖：
  echo          cd frontend ^&^& npm install
  exit /b 1
)

echo [start] 后端 -> http://127.0.0.1:8000   （人工确认=%ATTEST_HUMAN_CONFIRM%）
start "attest-api" cmd /k ".venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

echo [start] 前端 -> http://127.0.0.1:5173
start "attest-web" cmd /k "cd frontend && npm run dev"

echo.
echo [start] 两个窗口已打开。等前端编译完成后，浏览器打开：
echo         http://127.0.0.1:5173
echo.
echo [start] 提示：首次启动后端要装配 Chroma 与检查点，约 3~10 秒。
echo [start] 关闭服务：直接关掉那两个窗口。

endlocal
