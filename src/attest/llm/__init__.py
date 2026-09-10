"""④ 服务层：LLM 接入。

包含：
    gateway.py —— 统一出口（chat / embed / 重试 / reasoning 分离），也是**唯一记账点**；
    router.py  —— 模型路由策略（T4.4），独立关注点，勿与网关混写。

**不负责**：业务 prompt 组装（③ agents/）、预算熔断判定（横切 budget/，只上报花费增量）。
"""
