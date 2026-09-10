"""T0.4 · Agent 日志（loguru 封装）。

格式复刻课程模板：`[intent] 开始 | agent=intent_router | data={'query': '...'}`
统一出口，节点不自己 print——日志格式一变，全链路一起变。
"""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger

_CONFIGURED = False
_DEFAULT_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
    "<cyan>{extra[mod]}</cyan> | {message}"
)


def setup_logging(level: str = "INFO", *, sink: Any = None) -> None:
    """幂等配置。多次调用只生效一次（避免重复 handler 导致日志翻倍）。"""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logger.remove()
    logger.add(
        sink if sink is not None else sys.stderr,
        level=level.upper(),
        format=_DEFAULT_FORMAT,
        colorize=sink is None,
        backtrace=False,
        diagnose=False,
    )
    logger.configure(extra={"mod": "attest"})
    _CONFIGURED = True


def _fmt(data: dict[str, Any]) -> str:
    if not data:
        return "{}"
    inner = ", ".join(f"{k}={v!r}" for k, v in data.items())
    return "{" + inner + "}"


class _Bound:
    """绑定了模块名的轻量包装，暴露 loguru 常用级别 + 结构化的节点日志。"""

    def __init__(self, mod: str) -> None:
        self._log = logger.bind(mod=mod)

    def debug(self, msg: str, **kw: Any) -> None:
        self._log.debug(msg)

    def info(self, msg: str, **kw: Any) -> None:
        self._log.info(msg)

    def warning(self, msg: str, **kw: Any) -> None:
        self._log.warning(msg)

    def error(self, msg: str, **kw: Any) -> None:
        self._log.error(msg)

    def node(self, tag: str, agent: str, phase: str, **data: Any) -> None:
        """节点生命周期日志。tag 是短标签（如 intent / planner），agent 是节点名。"""
        self._log.info(f"[{tag}] {phase} | agent={agent} | data={_fmt(data)}")


def get_logger(name: str) -> _Bound:
    setup_logging()
    short = name.split(".")[-1]
    return _Bound(short)
