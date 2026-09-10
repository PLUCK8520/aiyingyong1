"""③ 能力层：9 个图节点（纯函数 + 依赖注入）。

包含：intent_router / direct_responder / planner / scout_web / scout_local /
evidence_judge / reflect / analyst / citation_auditor。

**不负责**：图组装（② graph/）、外部调用实现（④ retrieval·llm）、预算熔断判定（横切 budget/）。
节点只接收注入的 Port 与 state，返回增量字典。
"""
