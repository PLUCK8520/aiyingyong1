/**
 * T6.3 ~ T6.7 · Attest Web 工作台主视图。
 *
 * **状态归属（重要，避免双份真相）**：
 *   - `timeline`：SSE 事件聚合出的**唯一**进度来源，同时喂给 ProgressStream 与
 *     TimelinePanel。两处不再各自累积（否则两者会对不上）。
 *   - `report`：终态后从 `GET /api/report/{id}` 拉一次（而不是从 SSE 里拼），
 *     因为报告是**大对象**，走 SSE 既浪费带宽又要处理分片。
 *   - `lastEventIdRef`：重连游标。SSE 用 fetch 手写（因为 /api/chat 是 POST，
 *     原生 EventSource 只支持 GET），所以没有自动重连——自己记游标，断了用它续订。
 *
 * **T9.3 · 布局决策（改之前先读这段）**：
 *   中间这一列是**一张面板**，页眉（会话身份 + 视图切换）、内容区、页脚（输入框）
 *   都在同一张纸内，靠分隔线分段。旧版是"header 一个 panel + 内容一个 panel +
 *   输入区一个 panel"三个盒子叠着——加上左右两栏就是五个等重盒子，
 *   层级彻底消失，观感像贴纸墙而不是工作台。
 *   与之配套：左右两栏用 `.rail`（背景层），内容视图（Progress / Report / KB）
 *   **自己不再套 panel**（它们已经在纸上了，再套就是"框里套框"）。
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  fmtCny,
  fmtTokens,
  STATUS_META,
  streamChat,
  type AnyEvent,
  type ReportResult,
  type SessionMeta,
  type SessionStatus,
  type TimelineRow,
} from "./api";
import { Sidebar } from "./Sidebar";
import { ProgressStream } from "./ProgressStream";
import { TimelinePanel } from "./TimelinePanel";
import { ReportView } from "./ReportView";
import { ConfirmModal, type ConfirmPayload } from "./ConfirmModal";
import { KBView } from "./KBView";

type Tab = "progress" | "report" | "kb";

/**
 * Tab 显示名。
 *
 * 用查表而不是 `t === "progress" ? "进度" : "报告"`：加到第三个 tab 时，
 * 嵌套三元就会开始难读、且新增成员不会报错（TS 不会告诉你漏了分支）——
 * `Record<Tab, string>` 会在漏键时直接编译失败。
 */
const TAB_LABEL: Record<Tab, string> = {
  progress: "进度",
  report: "报告",
  kb: "知识库",
};

/**
 * 空态示例问题。
 *
 * ⚠️ 这三条**不是随便编的文案**——它们都来自项目自带的离线语料（`data/fixtures`），
 * 所以点下去**真的能跑出带引用的报告**。写一个语料覆盖不到的问题，
 * 会让用户第一次使用就一头撞进「证据不足 · 系统主动拒编」，
 * 那是非常差的初体验（他会以为系统坏了）。
 * 换言之：示例问题必须**跑得通**，这是一条硬约束，不是文案选择。
 */
const EXAMPLE_QUERIES = [
  "调研国产数据库替换的驱动因素与落地情况",
  "调研新能源汽车出海的竞争格局与政策环境",
  "调研大模型推理成本的构成与优化趋势",
];

export default function App() {
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [status, setStatus] = useState<SessionStatus>("idle");
  const [timeline, setTimeline] = useState<TimelineRow[]>([]);
  const [cost, setCost] = useState(0);
  const [tokens, setTokens] = useState(0);
  const [report, setReport] = useState<ReportResult | null>(null);
  const [confirm, setConfirm] = useState<ConfirmPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingList, setLoadingList] = useState(true);
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState<Tab>("progress");
  const [busy, setBusy] = useState(false);
  const [mockMode, setMockMode] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  /** 窄屏（<lg）的会话抽屉开关。宽屏侧栏常驻在流内，此值不生效。 */
  const [navOpen, setNavOpen] = useState(false);

  const lastEventIdRef = useRef(-1);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // ---------------------------------------------------------- 初始化
  useEffect(() => {
    void (async () => {
      try {
        const h = await api.health();
        setMockMode(h.llm_mode === "mock");
      } catch {
        /* 健康检查失败不阻塞（下面 listSessions 会给更明确的错误） */
      }
      try {
        setSessions(await api.listSessions());
      } catch (e) {
        setError(`无法连接后端：${(e as Error).message}`);
      } finally {
        setLoadingList(false);
      }
    })();
  }, []);

  // ---------------------------------------------------------- SSE 聚合
  const handleEvent = useCallback((ev: AnyEvent, id: number) => {
    lastEventIdRef.current = Math.max(lastEventIdRef.current, id);

    switch (ev.event) {
      case "run_start":
        setStatus("running");
        setError(null);
        break;

      case "agent_start":
        setStatus("running");
        setTimeline((prev) => [
          ...prev,
          {
            index: ev.index,
            node: ev.node,
            start: ev.ts,
            duration_ms: null, // 未闭合：UI 显示"进行中"
            brief: "",
            outputs: ev.outputs,
          },
        ]);
        break;

      case "agent_end":
        setTimeline((prev) =>
          prev.map((row) =>
            row.index === ev.index
              ? {
                  ...row,
                  duration_ms: ev.duration_ms,
                  brief: ev.brief,
                  end: ev.ts,
                  // 产出字段名只有结束事件才带（开始事件里恒为空）——
                  // 不在这里合并的话，实时视图看不到 outputs，而刷新后 `GET /api/trace`
                  // 又把它补上了，两处对不上（2026-09-13 核对契约时发现）。
                  // 缺失时保留原值：异常中断补发的结束事件不带 outputs，别把它清空。
                  outputs: ev.outputs ?? row.outputs,
                }
              : row,
          ),
        );
        break;

      case "awaiting_confirm":
        setStatus("awaiting_confirm");
        setConfirm({
          node: ev.node,
          objective: ev.objective,
          outlines: ev.outlines,
          sub_questions: ev.sub_questions,
        });
        break;

      case "report_ready":
        setCost(ev.cost_cny);
        setTokens(ev.tokens);
        setStatus("done");
        break;

      case "error":
        setError(ev.message);
        setStatus("error");
        break;

      case "stream_end":
        setBusy(false);
        setSubmitting(false);
        break;

      default:
        break;
    }
  }, []);

  /** 跑一次调研（首跑或续跑），订阅事件流。 */
  const run = useCallback(
    async (threadId: string, q: string, opts: { resume?: boolean; reset?: boolean } = {}) => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;

      if (opts.reset) {
        setTimeline([]);
        setReport(null);
        setCost(0);
        setTokens(0);
      }
      // 🔴 必须在这里就把状态置为 running（2026-09-12 实测坑）：
      // 上一轮跑完时 status 是 "done"，而 setReport(null) 会立刻触发"取报告"effect——
      // 那个 effect 只看 status==="done" 就去 GET /api/report，此刻新会话其实还在跑，
      // 后端回 409「尚未产出结果（当前 running）」，界面上就挂着一条刺眼的红字报错，
      // 而且跑完取到报告后**没人清它**（成功路径没 setError(null)），一直留在屏幕上。
      // 所以：请求一发出就认领 running，别等 SSE 的 run_start 事件回来。
      setStatus("running");
      if (!opts.resume) lastEventIdRef.current = -1;

      setBusy(true);
      try {
        await streamChat(
          { session_id: threadId, query: q, resume: opts.resume },
          handleEvent,
          { lastEventId: lastEventIdRef.current, signal: ac.signal },
        );
      } catch (e) {
        if ((e as Error).name !== "AbortError") {
          setError(`事件流中断：${(e as Error).message}`);
          setStatus("error");
        }
      } finally {
        setBusy(false);
        setSubmitting(false);
        void api.listSessions().then(setSessions).catch(() => undefined);
      }
    },
    [handleEvent],
  );

  /** 终态后把报告拉回来（报告是大对象，不走 SSE）。 */
  useEffect(() => {
    if (status !== "done" || !activeId || report) return;
    void api
      .getReport(activeId)
      .then((r) => {
        setReport(r);
        setCost(r.cost_incurred);
        setTokens(r.tokens_incurred);
        setTab("report");
        // 成功取到报告 = 这一轮没问题：清掉可能残留的旧报错（含上面那条 409 误报）
        setError(null);
      })
      .catch((e) => {
        // 409「尚未产出结果」/ 404「会话不存在」都是**暂态**，不是故障：
        // 前者说明还在跑（等 report_ready 会再来一次），后者说明会话已被清理。
        // 都不该弹红字——否则用户会以为跑挂了，实际报告马上就到。
        const st = (e as { status?: number }).status;
        if (st === 409 || st === 404 || st === 401) return;
        setError(`取报告失败：${(e as Error).message}`);
      });
  }, [status, activeId, report]);

  // ---------------------------------------------------------- 动作
  const startNew = useCallback(async () => {
    const q = query.trim();
    if (!q) return;
    setSubmitting(true);
    try {
      const s = await api.createSession(null, q);
      setSessions((prev) => [s, ...prev.filter((p) => p.thread_id !== s.thread_id)]);
      setActiveId(s.thread_id);
      setTab("progress");
      setQuery("");
      await run(s.thread_id, q, { reset: true });
    } catch (e) {
      setError(`新建会话失败：${(e as Error).message}`);
      setSubmitting(false);
    }
  }, [query, run]);

  const selectSession = useCallback(
    async (id: string) => {
      if (id === activeId) return;
      abortRef.current?.abort();
      setActiveId(id);
      setError(null);
      setTimeline([]);
      setReport(null);
      setCost(0);
      setTokens(0);
      setConfirm(null);
      lastEventIdRef.current = -1;
      try {
        // 会话详情带"待确认/可续跑"信息 —— 冷启动恢复路径也走这里
        const s = await api.getSession(id);
        setStatus(s.status);
        if (s.pending_confirm) setConfirm(s.pending_confirm as ConfirmPayload);
        if (s.has_report || s.status === "done") {
          try {
            const r = await api.getReport(id);
            setReport(r);
            setCost(r.cost_incurred);
            setTokens(r.tokens_incurred);
            setTab("report");
          } catch {
            /* 内存态已丢且检查点无报告：只是没报告可看，不是错误 */
          }
        }
        // 拿回已有事件（重连语义），让时间线不至于空白
        const t = await api.getTrace(id);
        setTimeline(t.timeline);
      } catch (e) {
        setError(`加载会话失败：${(e as Error).message}`);
      }
    },
    [activeId],
  );

  const decide = useCallback(
    async (
      action: "approve" | "skip" | "edit" | "abort",
      plan?: { outlines: string[] },
    ) => {
      if (!activeId) return;
      setBusy(true);
      try {
        await api.confirm(activeId, action, plan);
        setConfirm(null);
        if (action === "abort") {
          setStatus("aborted");
          setBusy(false);
          return;
        }
        await run(activeId, query || "（续跑）", { resume: true });
      } catch (e) {
        setError(`确认失败：${(e as Error).message}`);
        setBusy(false);
      }
    },
    [activeId, query, run],
  );

  useEffect(() => {
    const el = scrollRef.current;
    if (el && status === "running") el.scrollTop = el.scrollHeight;
  }, [timeline.length, status]);

  const activeSession = sessions.find((s) => s.thread_id === activeId);
  const running = status === "running" || busy;

  return (
    /* 外框留白 14px（旧版 12px）：暗色界面里面板与窗口边缘贴太近会显得"挤"，
       多 2px 就能让整块区域读起来是"浮在窗口里"而不是"撑满窗口"。 */
    <div className="flex h-full gap-3.5 p-3.5">
      <Sidebar
        sessions={sessions}
        activeId={activeId}
        onSelect={(id) => void selectSession(id)}
        onNew={() => {
          abortRef.current?.abort();
          setActiveId(null);
          setTimeline([]);
          setReport(null);
          setCost(0);
          setTokens(0);
          setConfirm(null);
          setStatus("idle");
          setError(null);
          setTab("progress");
          lastEventIdRef.current = -1;
          // 窄屏：新建后收起抽屉，让用户直接看到主列底部的输入框
          setNavOpen(false);
        }}
        loading={loadingList}
        open={navOpen}
        onClose={() => setNavOpen(false)}
      />

      {/* ============================ 主列：一整张纸 ============================ */}
      <main className="panel flex min-w-0 flex-1 flex-col overflow-hidden">
        {/* 页眉：会话身份（左）+ 视图切换（右）。
            标题用 base 字号而非 sm——它是这一屏的"我在看哪次调研"的唯一标识。 */}
        <header className="no-print flex shrink-0 items-center gap-2 border-b border-border px-3 py-3 sm:gap-4 sm:px-4">
          {/* 抽屉开关：只在 lg 以下出现（lg 以上侧栏常驻在流内，不需要入口） */}
          <button
            type="button"
            onClick={() => setNavOpen(true)}
            className="btn-icon -ml-0.5 text-fg-muted hover:text-fg lg:hidden"
            aria-label="打开会话列表"
            title="会话列表"
          >
            <svg
              width="15"
              height="15"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              aria-hidden="true"
            >
              <path d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>

          <div className="min-w-0 flex-1">
            <h1 className="truncate text-base font-semibold tracking-tight text-fg">
              {activeSession?.query || "新会话"}
            </h1>
            <div className="mt-1 flex items-center gap-2 text-2xs text-fg-subtle">
              {activeId ? (
                <span className="truncate font-mono" title={activeId}>
                  {activeId}
                </span>
              ) : (
                <span>尚未创建会话</span>
              )}
              {mockMode && (
                <span className="chip-warn" title="当前 LLM 走离线 mock，产出仅供链路验证">
                  离线 mock
                </span>
              )}
              <StatusPill status={status} />
            </div>
          </div>

          {/* 紧凑成本：xl 以下右栏不显示，成本必须有别处落脚——
              否则它在窄屏上是**彻底丢失**，而不是"折叠起来"。 */}
          <span
            className="chip-neutral hidden shrink-0 font-mono lg:inline-flex xl:hidden"
            title="累计费用 · 累计 token（宽屏时右栏有完整面板）"
          >
            {fmtCny(cost)}
            <span className="text-fg-subtle/60">·</span>
            {fmtTokens(tokens)}
          </span>

          {/* 视图切换（segmented）：槽内嵌 + 选中滑块抬起。
              选中态用"抬升底色 + 投影"两个信号，而不只是变个色——
              暗色里单靠变色，余光扫不到"当前在哪一屏"。 */}
          <nav
            className="flex shrink-0 items-center gap-0.5 rounded-control border border-border/70
                       bg-bg/70 p-[3px]"
            aria-label="视图切换"
          >
            {(["progress", "report", "kb"] as Tab[]).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setTab(t)}
                aria-current={tab === t ? "page" : undefined}
                className={`cursor-pointer rounded-[6px] px-2.5 py-1.5 text-xs font-medium
                            transition-colors duration-fast sm:px-3 ${
                              tab === t
                                ? "bg-elevated text-fg shadow-segmented"
                                : "text-fg-muted hover:text-fg"
                            }`}
              >
                {TAB_LABEL[t]}
              </button>
            ))}
          </nav>
        </header>

        {/* 错误条：纸内的一段（不圆角、不描边，只跟上下内容用线分开），
            否则它自己又成了一个盒子，跟前后的内容层级打架。 */}
        {error && (
          <div
            role="alert"
            className="no-print flex shrink-0 animate-fade-in items-start gap-2.5 border-b
                       border-danger/30 bg-danger-soft/40 px-4 py-2.5"
          >
            <span className="mt-1.5 dot bg-danger" aria-hidden="true" />
            <p className="flex-1 text-xs leading-relaxed text-fg">{error}</p>
            <button type="button" className="btn-ghost btn-xs" onClick={() => setError(null)}>
              关闭
            </button>
          </div>
        )}

        {/* 内容区。三个视图**自己不再套 panel**（它们已经在这张纸上了）。 */}
        <div ref={scrollRef} className="flex min-h-0 flex-1 flex-col overflow-hidden">
          {tab === "kb" ? (
            /* 知识库与会话无关（是全局语料池），所以放在最前面判定——
               否则"没有活跃会话"时会掉进最后那个"还没有可查看的报告"空态。 */
            <KBView />
          ) : tab === "progress" ? (
            <ProgressStream
              timeline={timeline}
              running={running}
              phase={timeline.find((r) => r.duration_ms === null)?.node ?? null}
            />
          ) : report ? (
            <ReportView
              markdown={report.report || report.direct_answer}
              references={report.references}
              sufficiency={report.evidence_sufficiency}
            />
          ) : (
            <EmptyReport status={status} />
          )}
        </div>

        {/* 页脚：输入区。把"快捷键 + 落盘位置"提示做进来——
            旧版只有一个 placeholder，用户不知道可以 Ctrl/⌘+Enter 直接提交。 */}
        <form
          className="no-print flex shrink-0 flex-col gap-2 border-t border-border px-3.5 py-3"
          onSubmit={(e) => {
            e.preventDefault();
            void startNew();
          }}
        >
          {/* 空态引导：给几个**可点**的示例。
              用户面对空输入框时，最难的不是"怎么问"而是"问什么"——
              给三条真跑得通的例子，比一句 placeholder 有用得多。 */}
          {!activeId && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="shrink-0 text-2xs text-fg-subtle">试试</span>
              {EXAMPLE_QUERIES.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => setQuery(q)}
                  className="chip-neutral max-w-full cursor-pointer truncate transition-colors
                             duration-fast hover:border-border-strong hover:bg-elevated hover:text-fg"
                >
                  {q}
                </button>
              ))}
            </div>
          )}

          <div className="flex items-end gap-2">
            <div className="min-w-0 flex-1">
              <label htmlFor="q" className="sr-only">
                调研问题
              </label>
              <textarea
                id="q"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                    e.preventDefault();
                    void startNew();
                  }
                }}
                rows={2}
                placeholder="输入调研问题，例如：调研 企业知识库 Agent 平台 市场，按市场规模/竞品/收费模式三部分输出"
                className="input resize-none !text-sm"
                disabled={submitting}
              />
            </div>
            <button
              type="submit"
              className="btn-primary mb-0.5 shrink-0"
              disabled={!query.trim() || submitting || running}
            >
              {submitting || running ? (
                <>
                  <span className="dot animate-breathe bg-black/45" aria-hidden="true" />
                  运行中
                </>
              ) : (
                "开始调研"
              )}
            </button>
          </div>
          <p className="flex items-center gap-1.5 text-2xs text-fg-subtle">
            <kbd className="kbd">Ctrl</kbd>
            <span className="text-fg-subtle/70">+</span>
            <kbd className="kbd">Enter</kbd>
            <span>提交 · 报告自动落盘到 data/reports/</span>
          </p>
        </form>
      </main>

      <div className="no-print hidden w-[316px] shrink-0 xl:block">
        <TimelinePanel timeline={timeline} cost={cost} tokens={tokens} running={running} />
      </div>

      {confirm && (
        <ConfirmModal payload={confirm} busy={busy} onDecide={(a, p) => void decide(a, p)} />
      )}
    </div>
  );
}

/**
 * 会话状态徽标。文案与配色取自 `api.ts` 的 `STATUS_META`（**唯一**口径）——
 * 这里只负责渲染，不再自己定义一份（否则 Sidebar 的改动会悄悄落后）。
 */
function StatusPill({ status }: { status: SessionStatus }) {
  const meta = STATUS_META[status];
  if (!meta || status === "idle") return null;
  return (
    <span className={meta.chip}>
      {meta.pulse && <span className="dot animate-breathe bg-current" aria-hidden="true" />}
      {meta.label}
    </span>
  );
}

/** 报告空态：把"为什么没有报告"讲清楚（旧版两句灰字，用户不知道下一步做什么）。 */
function EmptyReport({ status }: { status: SessionStatus }) {
  const hint =
    status === "awaiting_confirm"
      ? "请先在上方弹窗中确认大纲，确认后即开始检索与撰写。"
      : status === "running"
        ? "调研进行中——完成后报告会自动出现在这里。"
        : "在下方输入问题并「开始调研」，完成后报告会出现在这里。";
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 p-8 text-center">
      {/* 报告页空态用**中性提亮**而不是内凹：它表达的是"等待中"，
          不该像进度页空态那样带 accent 光晕（那会暗示"可以开始了"）。 */}
      <div
        className="flex h-12 w-12 items-center justify-center rounded-panel border
                   border-border-strong/60 bg-elevated/70 text-fg-muted
                   shadow-[inset_0_1px_0_rgba(255,255,255,.06)]"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M4 5.5A1.5 1.5 0 0 1 5.5 4h9A1.5 1.5 0 0 1 16 5.5v13A1.5 1.5 0 0 1 14.5 20h-9A1.5 1.5 0 0 1 4 18.5v-13Z"
            stroke="currentColor"
            strokeWidth="1.5"
          />
          <path
            d="M7.5 8h5M7.5 11h5M7.5 14h3"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
          <path d="M18 8v9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      </div>
      <p className="text-sm font-semibold tracking-tight text-fg">还没有可查看的报告</p>
      <p className="max-w-[40ch] text-xs leading-relaxed text-fg-muted">{hint}</p>
    </div>
  );
}
