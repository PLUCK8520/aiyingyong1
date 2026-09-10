"""② 编排层：StateGraph 组装、GraphState 定义、Send 扇出、interrupt/resume、条件边。

**不负责**：具体业务逻辑（③ agents/）、任何外部 IO（④ 服务层的 Port）。

⚠️ 编码约定（P-1 实测，langgraph 1.2.11）：
    Send / interrupt / Command 一律 `from langgraph.types import ...`；
    `from langgraph.graph import Send` 在 1.x 会 ImportError。
"""
