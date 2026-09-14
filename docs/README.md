# Attest（质证）· 多 Agent 行业深度调研系统

> 逐句质证：**每句结论都要经得起证据回查。**
> 从一句自然语言提问，到一份逐句可质证、经过反幻觉审计的结构化调研报告。

`Python 3.13` · `LangGraph` · `Qwen3` · `Chroma` · `FastAPI` · `React`

---

## 它解决什么问题

通用 LLM 直答调研类问题有四大硬伤：**结论无法溯源、复杂问题不会拆解、多源信息不会综合**（信源打架时要么装看不见要么瞎调和）、**成本不可控**（单次消耗无法预估，链路跑飞无法叫停）。

Attest 用一条多 Agent 流水线解决：意图分流 → 任务规划 → 双源并行检索 → 证据判别 → 反思补检 → 撰写 → **引用质证**，全程 Token 预算治理，报告成果自动沉淀形成研究闭环。

## 核心特性（区别于一般 RAG / Agent Demo）

| 特性 | 说明 |
|---|---|
| 🕵️ 引用忠实性审计 | 报告每句带引用的话逐句回查证据原文，不支撑自动降级标注或重写——反幻觉是流水线节点，不是口号 |
| ⚔️ 矛盾检测与分歧呈现 | 信源冲突不强行调和，报告单列"争议与分歧"章节摆明双方口径 |
| 🔀 混合检索 + Rerank | BM25 + jieba 分词 + 向量语义双路召回，RRF 融合，qwen3.7-text-rerank 精排 |
| 🕸 动态并行 | LangGraph Send API 按子问题数量扇出 N 路 map-reduce 检索 |
| 🙋 人机协同 | Planner 大纲 interrupt 暂停确认（可修改/跳过），任意断点可恢复续跑 |
| 💰 Token 预算治理 | 预算账户 + 两级熔断降级（切小模型 → 停补检 + 截断上下文） |
| ♻️ 研究闭环 | 报告结论与证据自动入向量库，后续调研可命中——越用越聪明 |
| 📏 评测流水线 | Token 成本（每份报告）/ 引用有效率 / 大纲覆盖度 / 完成率 / 延迟五指标，版本回归对比 + **baseline 对照组** |
| 📡 全链路 Trace | 每节点 start/end/token/成本写 JSONL，前端渲染执行时间线 + 成本面板 |
| 📚 知识库自助管理 | Web 端上传 / 移除本地文档（md / txt / pdf），入库即时生效——**不接网络检索也能出带 `[LOC*]` 引用的真报告** |

## 架构

```
用户提问
 └─ Intent Router 意图路由
     ├─ direct → Direct Responder（秒回）
     └─ research → Planner（目标/子问题/大纲 JSON）
          └─ [interrupt] 人工确认大纲（可跳过）
               ├─ Web Scout（Tavily + 正文抓取）  ┐ Send API 按子问题
               ├─ Local Scout（混合检索+Rerank）  ┘ 动态扇出并行
               ├─ Evidence Judge（置信度/缺口/矛盾检测）
               ├─ Reflect ──证据不足──→ 补检循环（≤2轮 + 预算熔断）
               ├─ Analyst（含"争议与分歧"章节）
               └─ Citation Auditor（逐句核验引用）
                    └─ 结构化报告 [WEB1-1-1]/[LOC1-1-1] 全程可溯源
横切层：Thread 检查点 · 用户画像记忆 · Token 预算 · JSONL Trace · 研究闭环沉淀
```

> 这里是**运行时流水线视图**。分层架构（5 层 + 横切治理）、依赖方向、端口-适配器、ADR 决策与故障域划分见 [docs/架构设计.md](docs/架构设计.md)。

## 技术栈

| 层 | 选型 |
|---|---|
| 编排 | LangGraph（Send API / interrupt-resume / checkpointer） |
| 模型 | **默认目标档**：Qwen3 系列（DashScope：qwen3.8-max 12/36 元每百万 token、qwen3.8-flash 0.8/2.7，以控制台为准）+ Ollama 本地（离线兜底）。**当前实跑档**：智谱 `glm-4-flash` 免费档（走通用 OpenAI 兼容档接入）——⚠️ 该档**指令服从性不足以支撑引用契约**，见 P8/T8.6 |
| 接入 | base_url 锁华北2(北京) —— 百炼免费额度**仅支持北京地域实时推理** |
| 检索 | Tavily（网络）+ Chroma（本地向量，接口可切 Milvus）+ rank-bm25 + jieba + RRF + qwen3.7-text-rerank |
| 存储 | SQLite（检查点/画像）→ 可切 PostgreSQL |
| 后端 | FastAPI + SSE |
| 前端 | Vite + React + TS + Tailwind（工作台 / 执行时间线 / 成本面板） |

## 快速开始

> **状态（2026-09-14）：P-1 ~ P8 全部交付**，CLI 与 Web 工作台均可用。
> ⚠️ 两处如实说明：① P8 批次（知识库 Web 管理 / 报告导出 / 计时心跳 / 局域网演示 + 两个 P0 修复）
> 已于 2026-09-14 **提交并推送**（`v1.0` 之后新增 4 个提交），尚未打 tag；② **当前账号下只有智谱 `glm-4-flash` 免费档可用**
> （强模型档余额不足、SiliconFlow 无额度），报告可用但**质量不足以交付**，详见
> [docs/进度快照-2026-09-14.md](docs/进度快照-2026-09-14.md)。
> 开工前先跑 **P-1 环境预检**（Python 3.13 + chromadb + jieba + rank-bm25 实测安装，避坑记录见 `docs/env-report.md`）。
> 开发与测试默认走**离线 mock**（Tavily 录制回放），只有录 fixture 与验收才打真实 API。理由：百炼 LLM 额度每模型 100 万 token，按单份报告 30 万 token 估算**只够约 3 次**，Tavily 也仅 1000 credits/月——**额度瓶颈在开发期，不在演示期**。

```bash
# 1. 配置密钥
cp .env.example .env   # 填入 DASHSCOPE_API_KEY / TAVILY_API_KEY

# 2. 启动
# Windows（无 make）:
scripts\start.bat
# macOS / Linux:
make dev

# 3. 提问
# 浏览器打开工作台，输入：
# "调研'企业知识库 Agent 平台'市场，按市场规模/竞品/收费模式三部分输出，附溯源链接"
```

CLI 方式：`python scripts/chat.py "你的调研问题"`

CLI 会话与记忆（P5）：
```bash
python scripts/chat.py --confirm --thread my-session "调研…"   # 打开大纲确认（人工介入）
python scripts/chat.py --thread my-session --resume            # 从断点续跑（进程重启后）
python scripts/chat.py --remember "输出语言=中文"               # 写入用户画像（跨会话生效）
python scripts/chat.py --profile                               # 查看当前画像
```
> ⚠️ 大纲确认走**编译期静态断点**：续跑是 `update_state` + `invoke(None)`，
> 不是 `Command(resume=...)`（后者在静态断点下无处投递，会静默沿用旧大纲）。详见 `agents/human_confirm.py`。

测试：`pytest`（默认离线、不消耗 API 额度）

知识库（P8）：把自己的资料变成证据源——**不接网络检索（`ATTEST_SEARCH_MODE=mock`）也能出带
`[LOC*]` 引用的真报告**。两条路径：
```bash
# ① Web：工作台左栏切到「知识库」→ 拖拽上传（.md / .markdown / .txt / .pdf，单文件 ≤ 8MB）
# ② CLI：
python scripts/ingest_local.py <文件或目录>     # 入库
python scripts/chat.py "你的问题"                # 入库后立即可被检索召回
```
> ⚠️ 上传目录 = `ATTEST_LOCAL_DOCS`（默认 `data/fixtures/local`，**与仓库自带示例语料同目录**）。
> 该目录已在 `.gitignore` 里白名单化（只放行 `local_*.md`），**你上传的资料不会被提交到仓库**；
> 但如果你改用其它目录，请自行确认它也在忽略范围内。
> ⚠️ 入库的文档**不会自动删除**：`GET /api/kb` 可查清单，`DELETE /api/kb/{name}` 或页面上 hover「移除」可摘除。

## Roadmap

**P6 已交付**（2026-09-11）：后端 `app/session.py` + `app/main.py`（SSE 支持
`Last-Event-ID` 断线重连）+ 前端 `frontend/`（Vite + React + TS + Tailwind，
会话侧栏 / 流式进度 / 报告双栏 + 引用悬浮卡 / 大纲确认 / 时间线 + 成本面板）。

**P7 已交付**（2026-09-12，`v1.0`）：评测流水线（五指标 + baseline 对照组）/ 研究闭环 / md 导出 + 一分钟展示稿。
**P8 已交付**（2026-09-14）：知识库 Web 管理（上传 / 移除 / 重建，`src/attest/kb.py`
+ `KBView.tsx`）/ 报告导出（复制 / 下载 md / 打印）/ 运行计时心跳 / 局域网演示（`scripts/start.bat --lan`）
+ 两个 P0 修复（判别编号丢方括号导致全部查询误判拒编；零引用时引用契约失效被伪装成"审计通过"）。

一键启动（Windows）：`scripts\start.bat`；macOS/Linux：`make dev`。
手动起：
```bash
# 后端（终端 1）
.venv/Scripts/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
# 前端（终端 2）→ 浏览器打开 http://127.0.0.1:5173
cd frontend && npm run dev
```
开大纲确认：起后端前设 `ATTEST_HUMAN_CONFIRM=1`。API 文档：`http://127.0.0.1:8000/docs`。

验证（最近一次实跑，2026-09-14）：`pytest` **273 用例 / 268 passed**，5 个失败**全部是运行环境
沙箱的删除守卫所致**（`safe-delete` 拦下测试自身要删的临时文件，非代码缺陷；无守卫环境为 273 passed）/
`scripts/api_smoke.py` 36 项 / `scripts/p6_e2e_probe.py` 11 项 / `scripts/p6_coldstart_probe.py`
（跨进程恢复 start 4/4 · recover 10/10）/ `python -m eval.mini` M1~M6 全绿 /
`npm run build`（tsc + vite）0 错误。
✅ **前端视觉与交互已于 2026-09-12 用真实浏览器（agent-browser + Chrome）实测通过**
（首页 / 全流程 / 报告双栏 / 引用悬浮卡 / 时间线 / 控制台零报错；知识库视图与上传另于 09-13 实测通过）。

| 阶段 | 内容 | 状态 |
|---|---|---|
| P-1 | 环境预检（Python 3.13 依赖实测） | ✅ |
| P0 | 脚手架 / LLM 网关 / 日志 / Trace / 预算记账 / pytest 骨架 | ✅ |
| P1 | 意图二分流 + 快速回答 + 结构化规划 | ✅ |
| P2 | 网络检索 MVP + 证据判别 + 引用报告 + Tavily 录制回放 | ✅ |
| P3 | 本地混合检索 + 动态并行 + 反思循环 + 矛盾检测 + **最小评测尺子** | ✅ |
| P4 | 引用审计 + 预算熔断 + 模型路由 | ✅ |
| P5 | 三层记忆 + 人工确认 + 断点续跑 | ✅ |
| P6 | Web 工作台（SSE / 时间线 / 成本面板） | ✅（浏览器实测已补） |
| P7 | 评测流水线（含 baseline）/ 研究闭环 / md 导出 + 展示稿（追问与图表**已砍**） | ✅ `v1.0` |
| P8 | 知识库 Web 管理（上传/移除/重建）· 报告导出 · 运行计时心跳 · 局域网演示 · 两处 P0 修复 | ✅ |

> 图例：✅ 已交付（有测试/冒烟证据，tag 见 `p0`–`p6` + `v1.0`）· ⬜ 待开发。
> 说明：P-1/P0/P1/P2 四阶段在早期合并于同一提交（`3bb03c2`），故 `git tag` 中 `p0`/`p1`/`p2` 指向同一提交，标注为补打；`p3`/`p4`/`p5` 为各自阶段的独立交付点。**所有已交付阶段的验证均在离线 mock 下完成（真实 API 未消耗额度）。**
> **P8 已入库并推送**（2026-09-14，`v1.0` 之后的提交）：功能批次 `e6c8834` · 两处 P0 修复 `7467a92` ·
> 文档与配置 `07485a4` · 清理误入库文件 `d82b632`。**尚未打 tag**（`v1.0` 之后无新标签）。

**交付线**：P2 可演示 / **P5 可讲**（面试技术面底线，CLI + 录屏）/ **P6 可投递**（写进简历）。

详见 [docs/进度快照-2026-09-14.md](docs/进度快照-2026-09-14.md)（**当前状态一页纸**：做到哪一步 / 验证证据 / 未完成与未验证清单）· [docs/架构设计.md](docs/架构设计.md)（分层与关键决策）· [docs/需求分析.md](docs/需求分析.md)（要满足什么需求）· [docs/功能设计.md](docs/功能设计.md)（每个功能怎么设计、失败怎么办）· [docs/开发任务清单.md](docs/开发任务清单.md)（分几期、怎么验收）。

## 说明

架构蓝图受课程《DeepResearch 多 Agent 行业深度研究助手》启发；Citation Auditor、矛盾检测、预算治理、研究闭环、评测流水线等 9 个模块为自主设计（见方案 v2.4 §3 技术亮点）。选型经 **2026-09-08 与 2026-09-10 两轮联网核查**（见 `docs/审查报告-2026-09-08.md` 及其 F / G 节）。

**仓库**：[github.com/teachmehowtouse/aiyingyong1](https://github.com/teachmehowtouse/aiyingyong1)（私有）。
> 说明：已按方案 §10.1 落到 **GitHub 私有仓库**（属主 `teachmehowtouse`，仓库名沿用创建时的 `aiyingyong1`）。⚠️ **投递展示前需把仓库转为 Public，或邀请面试官为协作者**，否则对方看不到。历史曾短暂托管于 Gitee（国内网络备选），现已迁移至 GitHub，`main` 与 `p0`–`p4` 标签齐全。

**命名**：**Attest / 质证**——"对证据的质疑与核实"。原名 Orchestr（灵枢）描述的是"调度"，而调度不是本项目的差异化；本项目的差异化是"核验"（逐句回查引用 / 冲突不调和 / 成本对账）。更名于 2026-09-10，设计与内容未改动。
