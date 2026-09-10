"""Attest（质证）· 逐句质证的多 Agent 行业深度调研系统。

分层（依赖只能向下，见《架构设计.md》§2/§11）：
    ① 接入层   app/ · frontend/ · scripts/
    ② 编排层   graph/
    ③ 能力层   agents/
    ④ 服务层   retrieval/ · llm/ · memory/
    ⑤ 存储层   （由 retrieval/memory 的 Adapter 落地）
    横切治理   budget/ · trace/ · quality/

禁止跨层反向引用（例如 retrieval/ 里 import graph/）。
"""

__version__ = "0.0.1"
__all__ = ["__version__"]
