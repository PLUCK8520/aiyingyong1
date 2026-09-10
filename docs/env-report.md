# P-1 环境预检报告 · env-report.md

> 产出时间：2026-09-10 · 执行机：用户本机（Windows）· 对应任务 T-1.1 ~ T-1.4
> **证据等级**：本报告全部结论为 **①实测**（在本项目的 `.venv` 里真跑出来的），不是文档推断。
> 与《审查报告》H 节对照：H 节中"Python 3.13 可装性""Chroma 三坑""LangGraph 具体版本行为"三条 ④未核项，本报告已闭环。

---

## 0. 一句话结论

**"Windows + Python 3.13 + 本项目技术栈"这个组合跑得通。** 9 个直接依赖全部安装可导入，Chroma 自传向量路径无静默下载，LangGraph 的 Send 扇出 / reducer / interrupt-resume 三大核心机制在本机实跑通过。
唯一需要修正的是 **`Send` 的导入路径**（见 §5），以及 **GitHub 远端未建**（缺 `gh` CLI）。

---

## 1. 环境基线

| 项 | 值 |
|---|---|
| OS | Windows（win32） |
| Python | **3.13.14**（`MSC v.1944 64 bit (AMD64)`） |
| 解释器 | 项目内 venv：`<项目根>/.venv/Scripts/python.exe` |
| pip | 26.1.2 |
| venv 隔离 | ✅ 全局环境零污染，依赖只落在 `.venv/` |

---

## 2. T-1.1 · 依赖实测安装与导入（9/9 通过）

`scripts/env_probe.py` 退出码 0，逐包 import + 版本核对：

| 包 | import 名 | 实测版本 | 结果 |
|---|---|---|---|
| langgraph | `langgraph` | **1.2.11** | ✅ |
| langgraph-checkpoint-sqlite | `langgraph.checkpoint.sqlite` | **3.1.1** | ✅ |
| chromadb | `chromadb` | **1.5.9** | ✅ |
| jieba | `jieba` | **0.42.1** | ✅ |
| rank-bm25 | `rank_bm25` | **0.2.2** | ✅ |
| aiosqlite | `aiosqlite` | **0.22.1** | ✅ |
| pypdf | `pypdf` | **6.18.0** | ✅ |
| fastapi | `fastapi` | **0.141.1** | ✅ |
| pydantic-settings | `pydantic_settings` | **2.15.0** | ✅ |

**关键传递依赖**（影响设计，非随手记）：

| 传递依赖 | 版本 | 为什么记它 |
|---|---|---|
| langchain-core | 1.6.2 | LangGraph 1.x 的内核，与旧 0.x 教程不兼容 |
| langgraph-checkpoint | 4.2.0 | checkpointer 基类；P5 换 saver 时看它 |
| pydantic | 2.13.5 | 结构化输出（Planner JSON）依赖 |
| onnxruntime | **1.29.0** | chromadb 的**硬依赖**，即使我们不用默认 embedding 也会被装 |
| numpy | 2.5.3 | —— |
| sqlite-vec | 0.1.9 | Chroma 1.x 的向量索引后端 |
| tokenizers | 0.23.2 | chromadb 带来 |
| starlette / uvicorn | 1.6.0 / 0.52.4 | FastAPI 运行时（P6） |
| kubernetes | 36.0.3 | chromadb 带来（体积大户，用不上） |

> ⚠️ **体积提示**：chromadb 会拉进 `onnxruntime`(~1.29) + `kubernetes` + `tokenizers` + `numpy`，是安装耗时大户（本次全栈安装约 8 分钟）。这是"用 Chroma 换 Milvus"的隐性代价，记下来免得下次以为是网络问题。

---

## 3. T-1.2 · 依赖锁文件

- `requirements.txt` —— 9 个直接依赖，已按实测版本 `==` 锁定。
- `requirements.lock.txt` —— `pip freeze` 全量 **106** 条传递依赖，可复现环境。

---

## 4. T-1.3 · Chroma 冒烟（PASS）

`scripts/chroma_smoke.py`，退出码 0：

```
chromadb   : 1.5.9
count      : 3
query top2 : [('d1', '苹果'), ('d3', '红富士')]
onnx cache : C:\Users\32519\.cache\chroma\onnx_models (before=False, after=False)
RESULT     : PASS
```

逐条对照 T3.1 记录的"三个静默坑"：

| 坑 | 本机实测 | 结论 |
|---|---|---|
| ① 不显式 `embedding_function=None` 会静默下载 ~200MB 模型 | 显式传 `None` + 自传向量，**`~/.cache/chroma/onnx_models` 全程未创建** | ✅ 坑成立且已绕开 |
| ② `PersistentClient` 不支持多进程并发 | **未测**（P6 才有并发场景） | ⚠️ 待 P6 验证，代码上先按"单例 client"写 |
| ③ 默认外发 telemetry | `ANONYMIZED_TELEMETRY=False` + `Settings(anonymized_telemetry=False)` 生效，正常读写 | ✅ 已关 |

查询命中正确（`[1,0,0]` 最相似 `d1`，其次 `d3`），说明**自传向量的写入与检索链路可用**，无需依赖 Chroma 自带的 embedding。

---

## 5. 补充 · LangGraph 核心机制探针（PASS，含一处 API 修正）

`scripts/langgraph_probe.py`，退出码 0。这一步把 H 节标 ④未核 的"LangGraph 具体版本行为（Send 归并细节、state schema 写法）"补成了 ①实测。

| 验证项 | 结果 |
|---|---|
| `from langgraph.graph import StateGraph, START, END` | ✅ |
| **`Send` 不在 `langgraph.graph`** | ✅（1.2.11 确实不再导出） |
| `from langgraph.types import Send, interrupt, Command` | ✅ **正确路径** |
| `SqliteSaver` + `AsyncSqliteSaver` 均可导入 | ✅（P5 同步 → P6 异步的切换可行） |
| Send 3 路扇出 + `Annotated[list, operator.add]` 归并 | ✅ `hits=['a','b','c']`（3 条，未互相覆盖） |
| `Annotated[float, operator.add]` 累加 | ✅ `cost=0.3` |
| `interrupt()` 暂停 → `Command(resume=...)` 恢复跑完 | ✅ 暂停态含 `__interrupt__`，恢复后 `approved=True` |

### ⚠️ 需要修正的一处：`Send` 的导入路径

- **事实**：langgraph **1.2.11** 下 `from langgraph.graph import Send` → `ImportError`。
- **正确写法**：`from langgraph.types import Send`（顺带 `interrupt` / `Command` 也在这个模块）。
- **旁证**：`from langgraph.constants import Send` 仍可用，但已抛 `LangGraphDeprecatedSinceV10` 弃用告警，**V2.0 将移除**，不要用。
- **对文档的影响**：7 份文档只写了"Send API/扇出"的语义，**没有写过完整 import 语句**，所以**不算 bug、无需改文档**；但写代码时（T1.1 / T3.5）必须用 `langgraph.types`。此条记入本报告，作为编码约定。

### 一处数值细节

`cost=0.30000000000000004` —— 浮点累加的正常误差（~1e-16 量级）。对 FR-20 的"记账误差 < 5%"无影响；但建议**成本以 token（int）为准记账，元做展示时再换算**，避免"金额对不上"的观感问题。

---

## 6. 未完成 / 未核实（诚实清单）

| 项 | 状态 | 阻塞点 |
|---|---|---|
| GitHub 公开仓库（T0.1） | ❌ 未做 | **本机未安装 `gh` CLI**；`git` 2.55 已就绪。需用户决定：装 `gh` 自动建仓，还是手动建后我配远端 |
| Chroma 多进程并发坑 | ⚠️ 未测 | 无并发场景，P6 前不阻塞 |
| `TAVILY_API_KEY` / `DASHSCOPE_API_KEY` | ❌ 缺失 | P2 才需要；`.env.example` 将在 P0 建 |
| 真实 LLM 调用（网关连通性） | ❌ 未测 | 需 DashScope key；P0 的 T0.3/T0.6 会验 |
| Tavily 免费额度 1000 credits/月 | ⚠️ 仍未复核 | H 节遗留项，需用户注册后在控制台确认 |
| 单价表（方案 T0.5） | ⚠️ 仍是③第三方 | 需在百炼控制台价格页核对后回填 |

---

## 7. 项目不足（本报告视角）

1. **P-1 只证明了"能装能导入能跑最小机制"，没证明"能跑通链路"**——真正的风险在 LLM 网关连通性、Tavily 录制回放、Rerank 接口结构差异，那些要 P0–P3 才碰。
2. **依赖体积失控的隐患**：为 Chroma 背了 `onnxruntime` + `kubernetes`，而理想解法（直接走本地 `sqlite-vec` 或交付时裁剪依赖）没评估。
3. **`requirements.txt` 用固定 `==` 锁死**，虽保证复现，但 P0 之后每加一个依赖都要手动维护两份文件（`requirements.txt` + `lock`），容易漂移。建议 P0 建 `pyproject.toml` 时统一由它管理，`lock` 交给 `pip freeze` 自动生成。

---

## 8. 我的观点与下一步

**观点**：
- P-1 的价值不在于"装成功了"，而在于把 H 节里 3 条 ④未核降成 ①实测，并且**在写第一行业务代码之前**就抓到了 `Send` 导入路径这个真会踩的坑。这正是文档里那条"P0 装包后按实际版本文档再确认"的兑现。
- 目前最该警惕的不是技术，而是**节奏**：文档已 1800+ 行，代码 0 行，两者严重失衡。P-1 半天做完，就该立刻进 P0 出可运行代码，别再回头改文档。

**下一步（P0，建议顺序）**：
1. `git init` + 首次提交（把 7 份文档 + 本报告入库）——⚠️ 仓库远端需先解决 `gh`；
2. `pyproject.toml` + 目录骨架（`src/attest/{graph,llm,retrieval,quality,...}`）+ `tests/`；
3. T0.7 pytest 骨架 + **把 §5 的两条机制探针固化成 pytest 用例**（Send 归并 + reducer + interrupt/resume），这是 L2 亮点的第一批证据；
4. 再做 T0.2 config / T0.3 网关 / T0.5 记账。

---

*本报告由 P-1 实跑生成；所有版本号、退出码、输出均来自本机真实执行，未做任何推断或填充。*
