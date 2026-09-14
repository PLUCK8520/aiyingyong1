/**
 * T6.7 · 执行时间线 + 成本面板。
 *
 * **数据源**：SSE 事件聚合（`agent_start` / `agent_end`）与 `report_ready`
 * 里的成本汇总。**不读 trace.jsonl**——架构设计 §7 的硬决策。
 *
 * 成本面板的分级条（《功能设计》§8）：级别 1 黄条、级别 2 红条。
 * 后端预算默认 ¥2.0，告警 70% / 硬熔断 90%（`config.py`）。
 */

import { fmtCny, fmtDuration, fmtTokens, nodeLabel, type TimelineRow } from "./api";
import { fmtElapsed, useNow } from "./useNow";

const WARN_RATIO = 0.7;
const HARD_RATIO = 0.9;
const BUDGET_CNY = 2.0;

interface Props {
  timeline: TimelineRow[];
  cost: number;
  tokens: number;
  running: boolean;
}

export function TimelinePanel({ timeline, cost, tokens, running }: Props) {
  // 运行中每秒滴答一次，用于"已运行 N 秒"（见 useNow 的说明）
  const now = useNow(running);
  const ratio = Math.min(cost / BUDGET_CNY, 1);
  const level = ratio >= HARD_RATIO ? 2 : ratio >= WARN_RATIO ? 1 : 0;
  const barColor = level === 2 ? "bg-danger" : level === 1 ? "bg-warn" : "bg-accent";

  return (
    <div className="flex h-full flex-col gap-3 overflow-hidden">
      <section className="panel p-4">
        <h2 className="mb-3 text-sm font-medium text-fg">成本</h2>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <p className="text-[11px] text-fg-muted">累计费用</p>
            <p className="mt-0.5 font-mono text-[15px] text-fg">{fmtCny(cost)}</p>
          </div>
          <div>
            <p className="text-[11px] text-fg-muted">累计 token</p>
            <p className="mt-0.5 font-mono text-[15px] text-fg">{fmtTokens(tokens)}</p>
          </div>
        </div>

        <div className="mt-3">
          <div className="mb-1.5 flex items-center justify-between text-[11px]">
            <span className="text-fg-muted">预算占比（¥{BUDGET_CNY.toFixed(2)}）</span>
            <span
              className={
                level === 2 ? "text-danger" : level === 1 ? "text-warn" : "text-fg-muted"
              }
            >
              {(ratio * 100).toFixed(1)}%
            </span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className={`h-full rounded-full transition-all duration-300 ${barColor}`}
              style={{ width: `${Math.max(ratio * 100, cost > 0 ? 2 : 0)}%` }}
            />
          </div>
          {level === 2 && (
            <p className="mt-2 text-[11px] leading-relaxed text-danger">
              已触发硬熔断（≥90%），后续调用将降级到轻量模型，报告可能不完整。
            </p>
          )}
          {level === 1 && (
            <p className="mt-2 text-[11px] leading-relaxed text-warn">
              已超过告警线（≥70%），接近预算上限。
            </p>
          )}
        </div>
      </section>

      <section className="panel flex min-h-0 flex-1 flex-col overflow-hidden">
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <h2 className="text-sm font-medium text-fg">执行时间线</h2>
          <span className="text-[11px] text-fg-muted">
            {timeline.length} 个节点{running ? " · 进行中" : ""}
          </span>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-2">
          {timeline.length === 0 ? (
            <p className="px-3 py-8 text-center text-xs text-fg-muted">
              还没有执行记录
            </p>
          ) : (
            <ul className="space-y-0.5">
              {timeline.map((row) => {
                const open = row.duration_ms === null;
                // `open` 有两种成因：① 真的在跑；② 异常中断留下的未闭合行。
                // 只有会话在运行时才能断定是前者——否则显示计时会一直涨，骗人。
                const live = open && running;
                return (
                  <li
                    key={row.index}
                    className="animate-fade-up rounded-lg px-3 py-2 hover:bg-muted/50"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex min-w-0 items-center gap-2">
                        <span
                          className={`dot ${open ? "bg-info animate-pulse-soft" : "bg-accent"}`}
                          aria-hidden="true"
                        />
                        <span className="truncate text-[12.5px] text-fg">
                          {nodeLabel(row.node)}
                        </span>
                        <span className="font-mono text-[10px] text-fg-muted/60">
                          {row.node}
                        </span>
                      </div>
                      <span
                        className={`shrink-0 font-mono text-[11px] ${
                          live ? "text-info" : "text-fg-muted"
                        }`}
                        title={live ? "已运行时长（每秒刷新）" : undefined}
                      >
                        {live
                          ? fmtElapsed(now - row.start)
                          : open
                            ? "—"
                            : fmtDuration(row.duration_ms)}
                      </span>
                    </div>
                    {row.brief && (
                      <p className="mt-0.5 truncate pl-4 text-[11px] text-fg-muted">
                        {row.brief}
                      </p>
                    )}
                    {row.outputs && row.outputs.length > 0 && (
                      <p className="mt-0.5 truncate pl-4 font-mono text-[10px] text-fg-muted/60">
                        → {row.outputs.join(", ")}
                      </p>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </section>
    </div>
  );
}
