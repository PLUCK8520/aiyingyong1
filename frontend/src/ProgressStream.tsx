/**
 * T6.4 · 流式进度渲染：agent 状态滚动。
 *
 * 输入是**已聚合的**时间线行（来自 `App` 的 SSE 聚合），这里只负责展示。
 * 把聚合逻辑留在 `App` 里是为了让「进度」与「时间线」共享同一份真相——
 * 两处各自累积会让它们对不上（经典的双份状态 bug）。
 */

import { fmtDuration, nodeLabel, type TimelineRow } from "./api";

interface Props {
  timeline: TimelineRow[];
  running: boolean;
  phase: string | null;
}

export function ProgressStream({ timeline, running, phase }: Props) {
  if (timeline.length === 0 && !running) {
    return (
      <div className="panel flex flex-1 flex-col items-center justify-center p-8 text-center">
        <p className="text-[13px] text-fg-muted">还没有开始调研</p>
        <p className="mt-1.5 max-w-md text-[12px] leading-relaxed text-fg-muted/70">
          在下方输入一个调研问题开始。系统会先规划大纲、向你确认，
          再并行检索、逐句质证，最后产出带可回查引用的报告。
        </p>
      </div>
    );
  }

  return (
    <div className="panel flex min-h-0 flex-1 flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <span
            className={`dot ${running ? "bg-info animate-pulse-soft" : "bg-accent"}`}
            aria-hidden="true"
          />
          <h2 className="text-sm font-medium text-fg">
            {running ? "调研进行中" : "调研已结束"}
          </h2>
        </div>
        {phase && (
          <span className="font-mono text-[11px] text-fg-muted">
            当前：{nodeLabel(phase)}
          </span>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        <ol className="space-y-0.5">
          {timeline.map((row) => {
            const open = row.duration_ms === null;
            return (
              <li
                key={row.index}
                className="flex animate-fade-up items-start gap-3 rounded-lg px-2.5 py-2 hover:bg-muted/40"
              >
                <span
                  className={`mt-1.5 dot ${open ? "bg-info animate-pulse-soft" : "bg-accent"}`}
                  aria-hidden="true"
                />
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-[13px] text-fg">{nodeLabel(row.node)}</span>
                    <span className="shrink-0 font-mono text-[11px] text-fg-muted">
                      {open ? "进行中…" : fmtDuration(row.duration_ms)}
                    </span>
                  </div>
                  {row.brief && (
                    <p className="mt-0.5 text-[11.5px] leading-relaxed text-fg-muted">
                      {row.brief}
                    </p>
                  )}
                </div>
              </li>
            );
          })}
        </ol>

        {running && (
          <p className="mt-2 px-2.5 text-[11px] text-fg-muted/70">
            {timeline.length === 0 ? "正在启动…" : "等待下一个节点…"}
          </p>
        )}
      </div>
    </div>
  );
}
