"""pytest 共享夹具（T0.7）。

原则：**默认离线、不联网、不调真实 API**。
需要真实外部服务的用例必须打 `@pytest.mark.online`，并在 CI / 本地默认跳过。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from attest.config import Settings

from attest.config import Settings


@pytest.fixture(autouse=True)
def _hermetic_from_local_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """让测试**不受开发者本地 `.env` 影响**。

    为什么必须这样（2026-09-12 实测暴露）：`Settings` 配了 `env_file=.env`，
    而 `.env` 是 gitignore 的本地文件——一旦开发者把它切成真实厂商
    （`ATTEST_LLM_MODE=siliconflow` + 真 key），所有"缺 key 应报错""默认走 mock"
    之类的断言会**在本机静默失效**（CI 上却仍是绿的），这种"只在别人机器上红"的差别最难查。

    关键：光删环境变量不够——pydantic-settings 是**独立去读 dotenv 文件**的，
    `delenv("SILICONFLOW_API_KEY")` 挡不住 `.env` 里的那行。必须把 `env_file` 摘掉。
    用例若显式传参（`Settings(llm_mode=...)`）仍然优先，不受影响。
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setenv("ATTEST_LLM_MODE", "mock")
    monkeypatch.setenv("ATTEST_SEARCH_MODE", "mock")
    for key in ("SILICONFLOW_API_KEY", "DASHSCOPE_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _offline_by_default(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """默认把 API key 抹掉——防止用例意外打到真实接口、烧额度。

    需要真实调用的用例显式打 `@pytest.mark.online` 即可豁免。
    """
    if request.node.get_closest_marker("online"):
        return
    for key in ("DASHSCOPE_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _research_memory_isolated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """T7.4：测试默认**关闭**研究闭环，并把存储目录指到 tmp——

    否则任何走 `build_context` 的用例都会把 supported 结论写进仓库的
    `data/index/research_memory`（已实测发生过一次污染，2026-09-12），
    而且用例之间互相检索到对方的沉淀，断言变得不可复现。
    需要测闭环本身的用例（test_p7_research_memory.py）显式传
    `research_memory_enabled=...` / `research_memory_dir=...` 覆盖，不受本 fixture 影响。
    """
    monkeypatch.setenv("ATTEST_RESEARCH_MEMORY_ENABLED", "0")
    monkeypatch.setenv("ATTEST_RESEARCH_MEMORY_DIR", str(tmp_path / "research_memory"))
