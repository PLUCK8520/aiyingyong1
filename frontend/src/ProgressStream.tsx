/**
 * T6.4 · 流式进度渲染：agent 状态滚动。
 *
 * 输入是**已聚合的**时间线行（来自 `App` 的 SSE 聚合），这里只负责展示。
 * 把聚合逻辑留在 `App` 里是为了让「进度」与「时间线」共享同一份真相——
 * 两处各自累积会让它们对不上（经典的双份状态 bug）。
 */

import { fmtDuration, nodeLabel, type TimelineRow } from "./api";
import { fmtElapsed, useNow } from "./useNow";

interface Props {
  timeline: TimelineRow[];
  running: boolean;
  phase: string | null;
}

export function ProgressStream({ timeline, running, phase }: Props) {
  // ⚠️ hooks 必须在任何 early return **之前**调用（下面有一个提前返回的空态分支）
  const now = useNow(running);

  if (timeline.length === 0 && !running) {
    return (
      <div className="panel flex flex-1 flex-col items-center justify-center gap-2 p-8 text-center">
        <div className="mb-1 flex h-11 w-11 items-center justify-center rounded-pill border border-border bg-elevated text-fg-subtle">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="11" cy="11" r="6.2" stroke="currentColor" strokeWidth="1.6" />
            <path d="m15.6 15.6 3.4 3.4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </div>
        <p className="text-sm font-medium text-fg-muted">还没有开始调研</p>
        <p className="max-w-[42ch] text-xs leading-relaxed text-fg-subtle">
          在右下输入一个调研问题开始。系统会先规划大纲、向你确认，
          再并行检索、逐句质证，最后产出带可回查引用的报告。
        </p>
      </div>
    );
  }

  return (
    <div className="panel flex min-h-0 flex-1 flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-2.5">
          <span
            className={`dot ${running ? "bg-info animate-breathe" : "bg-accent"}`}
            aria-hidden="true"
          />
          <h2 className="text-sm font-semibold tracking-tight text-fg">
            {running ? "调研进行中" : "调研已结束"}
          </h2>
          <span className="chip-neutral">{timeline.length} 个节点</span>
        </div>
        {phase && (
          <span className="chip-info" title="当前执行中的节点">
            {running && <span className="dot animate-breathe bg-current" aria-hidden="true" />}
            当前：{nodeLabel(phase)}
          </span>
        )}
      </div>

      {/* 节点列表：左侧一条竖线把节点串成"时间轴"，比旧版纯列表更能传达**顺序**与**进度** */}
      <div className="relative min-h-0 flex-1 overflow-y-auto p-3">
        <ol className="relative space-y-0.5">
          <span
            className="pointer-events-none absolute left-[15px] top-3 bottom-3 w-px bg-border"
            aria-hidden="true"
          />
          {timeline.map((row) => {
            const open = row.duration_ms === null;
            // open 有两种成因：真的在跑 / 异常中断留下的未闭合行。只有会话在运行时
            // 才能断定是前者——否则计时会一直涨，看着像永远跑不完。
            const live = open && running;
            return (
              <li
                key={row.index}
                className="relative flex animate-fade-up items-start gap-3 rounded-control px-2 py-2
                           transition-colors duration-fast hover:bg-elevated/50"
              >
                {/* 节点标记压在竖线上；运行中用 breathe 动画（含蓄但能看出"活的"） */}
                <span
                  className={`relative z-10 mt-1 flex h-[9px] w-[9px] shrink-0 items-center justify-center
                              rounded-pill ring-4 ring-surface ${
                                open ? "bg-info" : "bg-accent"
                              } ${live ? "animate-breathe" : ""}`}
                  aria-hidden="true"
                />
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline justify-between gap-3">
                    <span
                      className={`text-xs ${live ? "font-medium text-fg" : "text-fg-muted"}`}
                    >
                      {nodeLabel(row.node)}
                    </span>
                    <span
                      className={`shrink-0 font-mono text-2xs ${
                        live ? "text-info" : "text-fg-subtle"
                      }`}
                    >
                      {live
                        ? `已运行 ${fmtElapsed(now - row.start)}`
                        : open
                          ? "未正常结束"
                          : fmtDuration(row.duration_ms)}
                    </span>
                  </div>
                  {row.brief && (
                    <p className="mt-0.5 text-2xs leading-relaxed text-fg-subtle">{row.brief}</p>
                  )}
                </div>
              </li>
            );
          })}
        </ol>

        {running && (
          <p className="mt-2 flex items-center gap-2 px-2 text-2xs text-fg-subtle">
            <span className="dot animate-pulse-soft bg-fg-subtle" aria-hidden="true" />
            {timeline.length === 0 ? "正在启动…" : "等待下一个节点…"}
          </p>
        )}
      </div>
    </div>
  );
}
