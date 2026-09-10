"""pytest 共享夹具（T0.7）。

原则：**默认离线、不联网、不调真实 API**。
需要真实外部服务的用例必须打 `@pytest.mark.online`，并在 CI / 本地默认跳过。
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _offline_by_default(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """默认把 API key 抹掉——防止用例意外打到真实接口、烧额度。

    需要真实调用的用例显式打 `@pytest.mark.online` 即可豁免。
    """
    if request.node.get_closest_marker("online"):
        return
    for key in ("DASHSCOPE_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(key, raising=False)
