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
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
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
    <div className="flex h-full gap-3 p-3">
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
        }}
        loading={loadingList}
      />

      <main className="flex min-w-0 flex-1 flex-col gap-3">
        {/* 顶栏：会话身份（左）+ 视图切换（右）。
            标题用 base 字号而非 sm——它是这一屏的"我在看哪次调研"的唯一标识。 */}
        <header className="panel no-print flex items-center justify-between gap-4 px-4 py-3">
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-base font-semibold tracking-tight text-fg">
              {activeSession?.query || "新会话"}
            </h1>
            <div className="mt-1 flex items-center gap-2 text-2xs text-fg-subtle">
              {activeId ? (
                <span className="font-mono" title={activeId}>
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

          {/* 视图切换：选中项带边框 + 抬升底色（不只是变色），
              让"当前在哪一屏"在余光里也能看出来。 */}
          <nav
            className="flex shrink-0 items-center gap-0.5 rounded-control border border-border bg-bg/60 p-0.5"
            aria-label="视图切换"
          >
            {(["progress", "report", "kb"] as Tab[]).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setTab(t)}
                aria-current={tab === t ? "page" : undefined}
                className={`cursor-pointer rounded-[6px] px-3 py-1.5 text-xs font-medium
                            transition-colors duration-fast ${
                              tab === t
                                ? "bg-elevated text-fg shadow-card"
                                : "text-fg-muted hover:bg-elevated/60 hover:text-fg"
                            }`}
              >
                {TAB_LABEL[t]}
              </button>
            ))}
          </nav>
        </header>

        {error && (
          <div
            role="alert"
            className="status-bar no-print animate-fade-in border-danger/40 bg-danger-soft/60"
          >
            <span className="mt-1.5 dot bg-danger" aria-hidden="true" />
            <p className="flex-1 text-fg">{error}</p>
            <button type="button" className="btn-ghost btn-xs" onClick={() => setError(null)}>
              关闭
            </button>
          </div>
        )}

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

        {/* 输入区：把"提示 + 快捷键"做进来。旧版只有一个 placeholder，
            用户不知道可以 Ctrl/⌘+Enter 直接提交。 */}
        <div className="panel no-print p-3">
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
              type="button"
              className="btn-primary mb-0.5 shrink-0"
              onClick={() => void startNew()}
              disabled={!query.trim() || submitting || running}
            >
              {submitting || running ? (
                <>
                  <span className="dot animate-breathe bg-[#06240F]" aria-hidden="true" />
                  运行中
                </>
              ) : (
                "开始调研"
              )}
            </button>
          </div>
          <p className="mt-2 flex items-center gap-1.5 px-0.5 text-2xs text-fg-subtle">
            <kbd className="rounded border border-border bg-bg px-1 font-mono">Ctrl</kbd>
            <span>+</span>
            <kbd className="rounded border border-border bg-bg px-1 font-mono">Enter</kbd>
            <span>提交 · 报告会自动落盘到 data/reports/</span>
          </p>
        </div>
      </main>

      <div className="no-print hidden w-[320px] shrink-0 xl:block">
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
    <div className="panel flex flex-1 flex-col items-center justify-center gap-2 p-8 text-center">
      <div className="mb-1 flex h-10 w-10 items-center justify-center rounded-pill border border-border bg-elevated text-fg-subtle">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M4 5.5A1.5 1.5 0 0 1 5.5 4h9A1.5 1.5 0 0 1 16 5.5v13A1.5 1.5 0 0 1 14.5 20h-9A1.5 1.5 0 0 1 4 18.5v-13Z"
            stroke="currentColor"
            strokeWidth="1.5"
          />
          <path d="M7.5 8h5M7.5 11h5M7.5 14h3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          <path d="M18 8v9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      </div>
      <p className="text-sm font-medium text-fg-muted">还没有可查看的报告</p>
      <p className="max-w-[38ch] text-xs leading-relaxed text-fg-subtle">{hint}</p>
    </div>
  );
}
