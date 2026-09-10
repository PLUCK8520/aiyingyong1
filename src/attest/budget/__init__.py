"""横切治理：派生式预算与分级熔断。

**不变式（ADR-02）**：state 里只有 `cost_incurred: Annotated[float, operator.add]` 是可变预算字段；
`total` / 阈值来自静态配置 `budget_cfg`；`fuse_level = f(cost_incurred, budget_cfg)` 随时派生，**不写入 state**。

**不变式（方案 §6）**：报告必须产出——熔断只降级，不杀进程。

**不负责**：LLM 调用（④ llm/gateway.py 负责上报增量）。
"""
