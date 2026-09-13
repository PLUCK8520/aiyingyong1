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

type Tab = "progress" | "report";

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
        <header className="panel flex items-center justify-between gap-4 px-4 py-3">
          <div className="min-w-0">
            <h2 className="truncate text-sm font-medium text-fg">
              {activeSession?.query || "新会话"}
            </h2>
            <p className="mt-0.5 flex items-center gap-2 text-[11px] text-fg-muted">
              {activeId ? (
                <span className="font-mono">{activeId}</span>
              ) : (
                <span>尚未创建会话</span>
              )}
              {mockMode && (
                <span className="rounded bg-warn/15 px-1.5 py-0.5 text-[10px] text-warn">
                  离线 mock 模式
                </span>
              )}
            </p>
          </div>

          <div className="flex shrink-0 items-center gap-1 rounded-lg border border-border p-0.5">
            {(["progress", "report"] as Tab[]).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setTab(t)}
                className={`cursor-pointer rounded-md px-2.5 py-1 text-[12px] transition-colors duration-150 ${
                  tab === t ? "bg-muted text-fg" : "text-fg-muted hover:text-fg"
                }`}
              >
                {t === "progress" ? "进度" : "报告"}
              </button>
            ))}
          </div>
        </header>

        {error && (
          <div
            role="alert"
            className="panel flex items-start gap-2.5 border-danger/40 bg-danger/10 px-4 py-2.5"
          >
            <span className="mt-1.5 dot bg-danger" aria-hidden="true" />
            <p className="flex-1 text-[12.5px] leading-relaxed text-fg">{error}</p>
            <button
              type="button"
              className="btn-ghost !px-2 !py-0.5 !text-[11px]"
              onClick={() => setError(null)}
            >
              关闭
            </button>
          </div>
        )}

        <div ref={scrollRef} className="flex min-h-0 flex-1 flex-col overflow-hidden">
          {tab === "progress" ? (
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
            <div className="panel flex flex-1 flex-col items-center justify-center p-8">
              <p className="text-[13px] text-fg-muted">还没有可查看的报告</p>
              <p className="mt-1.5 text-[12px] text-fg-muted/70">
                {status === "awaiting_confirm"
                  ? "请先在上方弹窗中确认大纲"
                  : "完成一次调研后将在此显示"}
              </p>
            </div>
          )}
        </div>

        <div className="panel flex items-end gap-2 p-3">
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
              className="input resize-none !text-[13px]"
              disabled={submitting}
            />
          </div>
          <button
            type="button"
            className="btn-primary mb-0.5 shrink-0"
            onClick={() => void startNew()}
            disabled={!query.trim() || submitting || running}
          >
            {submitting || running ? "运行中…" : "开始调研"}
          </button>
        </div>
      </main>

      <div className="hidden w-[320px] shrink-0 xl:block">
        <TimelinePanel timeline={timeline} cost={cost} tokens={tokens} running={running} />
      </div>

      {confirm && (
        <ConfirmModal payload={confirm} busy={busy} onDecide={(a, p) => void decide(a, p)} />
      )}
    </div>
  );
}
