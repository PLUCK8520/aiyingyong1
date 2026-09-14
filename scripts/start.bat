@echo off
REM ============================================================
REM  Attest 质证 · 一键启动（Windows）
REM
REM  起两个服务：
REM    后端  http://127.0.0.1:8000   （FastAPI + SSE）
REM    前端  http://127.0.0.1:5173   （Vite dev server，/api 代理到 8000）
REM
REM  用法：
REM    scripts\start.bat              本机访问（只绑 127.0.0.1，默认、最安全）
REM    scripts\start.bat --lan        局域网访问（绑 0.0.0.0，同网段设备可打开）
REM    set ATTEST_HUMAN_CONFIRM=1 && scripts\start.bat    开大纲确认
REM
REM  关闭：直接关掉两个新开的窗口即可。
REM
REM  ⚠️ --lan 绑 0.0.0.0 的含义是「同网段的任何人都能访问」——包括同事、室友，
REM     以及你在咖啡厅/图书馆连的公共 WiFi 里的陌生人。**仅用于演示，用完切回默认模式。**
REM     另外还需要放行防火墙（见 scripts\lan_firewall.ps1）。
REM ============================================================
setlocal enabledelayedexpansion

cd /d "%~dp0.."

REM ---- 模式判定：--lan 参数或 ATTEST_LAN=1 都进局域网模式 ----
set "LANMODE=0"
if /i "%~1"=="--lan" set "LANMODE=1"
if "%ATTEST_LAN%"=="1" set "LANMODE=1"

set "BIND=127.0.0.1"
if "%LANMODE%"=="1" set "BIND=0.0.0.0"

REM ---- 前置检查 ----
if not exist ".venv\Scripts\python.exe" (
  echo [start] 未找到 .venv\Scripts\python.exe
  echo         请先创建虚拟环境并安装依赖：
  echo           python -m venv .venv
  echo           .venv\Scripts\pip install -r requirements.txt
  exit /b 1
)

if not exist "frontend\node_modules" (
  echo [start] 未找到 frontend\node_modules
  echo         请先安装前端依赖：
  echo           cd frontend ^&^& npm install
  exit /b 1
)

echo [start] 后端 -> http://127.0.0.1:8000   （人工确认=%ATTEST_HUMAN_CONFIRM%）
start "attest-api" cmd /k ".venv\Scripts\python.exe -m uvicorn app.main:app --host %BIND% --port 8000 --reload"

echo [start] 前端 -> http://127.0.0.1:5173
start "attest-web" cmd /k "cd frontend && npm run dev -- --host %BIND%"

echo.
echo [start] 两个窗口已打开。等前端编译完成后，浏览器打开：
if "%LANMODE%"=="1" (
  set "LANIP="
  for /f %%i in ('powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\lan_ip.ps1"') do set "LANIP=%%i"
  echo         本机        http://127.0.0.1:5173
  if defined LANIP (
    echo         同网段设备  http://!LANIP!:5173
    echo.
    echo [start] 其他设备打不开？九成是防火墙没放行。用**管理员身份**跑一次：
    echo           powershell -ExecutionPolicy Bypass -File scripts\lan_firewall.ps1
    echo         或手动执行：
    echo           New-NetFirewallRule -DisplayName "Attest LAN Demo" -Direction Inbound -Protocol TCP -LocalPort 5173,8000 -Action Allow -Profile Private
  ) else (
    echo         未识别到内网地址，请运行 ipconfig 自行查看 IPv4
  )
) else (
  echo         http://127.0.0.1:5173
)
echo.
echo [start] 提示：首次启动后端要装配 Chroma 与检查点，约 3~10 秒。
echo [start] 关闭服务：直接关掉那两个窗口。

endlocal
