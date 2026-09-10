"""④ 服务层：检索相关能力。

包含：tavily 客户端 / 混合检索(BM25+向量+RRF) / rerank / chroma 向量库 /
citations 编号器 / ports.py（Ports Protocol，依赖倒置落点）。

**不负责**：节点编排（② graph/）、任务级预算决策（横切 budget/）。
"""
