"""④/⑤ 记忆层：checkpointer 接入、用户画像、research_memory 沉淀。

**硬约束**：research_memory **只沉淀 audit_verdict=supported 的结论**（防自我投毒，见方案 §7）。

**不负责**：检索实现（④ retrieval/）、审计判定（横切 quality/）。
"""
