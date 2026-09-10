"""④ 服务层：Port 的生产 Adapter 与测试 Adapter（Mock 是正式 Adapter，不是补丁）。

对应《架构设计.md》§3 的 6 个 Port：
    LLMGateway / SearchClient / Reranker / VectorStore / Store / Clock（示例）

默认走离线 Mock Adapter；只有录 fixture 与验收才打真实 API（额度保护线）。
"""
