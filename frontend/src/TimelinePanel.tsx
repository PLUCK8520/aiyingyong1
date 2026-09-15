/**
 * T6.7 · 执行时间线 + 成本面板。
 *
 * **数据源**：SSE 事件聚合（`agent_start` / `agent_end`）与 `report_ready`
 * 里的成本汇总。**不读 trace.jsonl**——架构设计 §7 的硬决策。
 *
 * 成本面板的分级条（《功能设计》§8）：级别 1 黄条、级别 2 红条。
 * 后端预算默认 ¥2.0，告警 70% / 硬熔断 90%（`config.py`）。
 *
 * T9.3 · 视觉定位：右栏整体是**一块 rail**（背景层），内部分段靠分隔线，
 * 而不是"两个 panel 摞起来"——后者会跟主内容区抢"这是主体"的位置。
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
  const barColor =
    level === 2
      ? "from-danger to-danger/70"
      : level === 1
        ? "from-warn to-warn/70"
        : "from-accent to-accent/60";
  const pctText = level === 2 ? "text-danger" : level === 1 ? "text-warn" : "text-fg-muted";

  return (
    <div className="rail flex h-full flex-col overflow-hidden">
      {/* -------------------------------------------------- 成本 */}
      <section className="shrink-0 border-b border-border/60 p-4">
        <h2 className="mb-3 text-2xs font-medium uppercase tracking-wider text-fg-subtle">
          成本
        </h2>

        {/* 两个读数并排。数字用 xl + semibold：它们是这一栏里唯一需要"一眼读到"的东西，
            其余全是说明文字。tabular-nums（全局）保证每秒刷新时不左右抖。 */}
        <div className="grid grid-cols-2 gap-4">
          <div className="min-w-0">
            <p className="text-2xs text-fg-muted">累计费用</p>
            <p className="mt-1 truncate font-mono text-xl font-semibold tracking-tight text-fg">
              {fmtCny(cost)}
            </p>
          </div>
          <div className="min-w-0">
            <p className="text-2xs text-fg-muted">累计 token</p>
            <p className="mt-1 truncate font-mono text-xl font-semibold tracking-tight text-fg">
              {fmtTokens(tokens)}
            </p>
          </div>
        </div>

        <div className="mt-4">
          <div className="mb-1.5 flex items-baseline justify-between gap-2 text-2xs">
            <span className="text-fg-muted">预算占比（¥{BUDGET_CNY.toFixed(2)}）</span>
            <span className={`font-mono font-medium ${pctText}`}>{(ratio * 100).toFixed(1)}%</span>
          </div>
          {/* 槽用 inset 内凹（阴影朝内），条用渐变 + 上高光——
              这两个方向相反的明暗让"条嵌在槽里"成立，比两根纯色条有实体感。 */}
          <div className="h-1.5 w-full overflow-hidden rounded-pill bg-bg shadow-[inset_0_1px_2px_rgba(0,0,0,.5)]">
            <div
              className={`h-full rounded-pill bg-gradient-to-b ${barColor}
                          shadow-[inset_0_1px_0_rgba(255,255,255,.25)]
                          transition-[width] duration-500 ease-in-out`}
              style={{ width: `${Math.max(ratio * 100, cost > 0 ? 2 : 0)}%` }}
            />
          </div>
          {level === 2 && (
            <p className="mt-2 text-2xs leading-relaxed text-danger">
              已触发硬熔断（≥90%），后续调用将降级到轻量模型，报告可能不完整。
            </p>
          )}
          {level === 1 && (
            <p className="mt-2 text-2xs leading-relaxed text-warn">
              已超过告警线（≥70%），接近预算上限。
            </p>
          )}
        </div>
      </section>

      {/* -------------------------------------------------- 时间线 */}
      <section className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <div className="shrink-0 px-4 pb-1.5 pt-3.5">
          <div className="flex items-center justify-between gap-2">
            <h2 className="text-2xs font-medium uppercase tracking-wider text-fg-subtle">
              执行时间线
            </h2>
            <span className="flex items-center gap-1.5 text-2xs text-fg-subtle">
              {running && <span className="dot animate-breathe bg-info" aria-hidden="true" />}
              <span className="font-mono">{timeline.length}</span>
              <span>节点</span>
            </span>
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
          {timeline.length === 0 ? (
            <p className="px-3 py-8 text-center text-xs text-fg-subtle/80">还没有执行记录</p>
          ) : (
            <ul className="space-y-0.5">
              {timeline.map((row, i) => {
                const open = row.duration_ms === null;
                // `open` 有两种成因：① 真的在跑；② 异常中断留下的未闭合行。
                // 只有会话在运行时才能断定是前者——否则显示计时会一直涨，骗人。
                const live = open && running;
                return (
                  <li
                    key={row.index}
                    style={{ animationDelay: `${Math.min(i, 10) * 18}ms` }}
                    className="animate-fade-up rounded-control px-2.5 py-2 transition-colors duration-fast
                               hover:bg-elevated/40"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex min-w-0 items-center gap-2">
                        <span
                          className={`dot ${live ? "bg-info animate-breathe" : open ? "bg-warn" : "bg-accent/70"}`}
                          aria-hidden="true"
                        />
                        <span className={`truncate text-xs ${live ? "text-fg" : "text-fg-muted"}`}>
                          {nodeLabel(row.node)}
                        </span>
                      </div>
                      <span
                        className={`shrink-0 font-mono text-2xs ${
                          live ? "text-info" : "text-fg-subtle"
                        }`}
                        title={live ? "已运行时长（每秒刷新）" : undefined}
                      >
                        {live
                          ? fmtElapsed(now - row.start)
                          : open
                            ? "未结束"
                            : fmtDuration(row.duration_ms)}
                      </span>
                    </div>
                    {/* 副行缩进对齐到节点文字（dot 2px + gap 8px = 10px → pl-[18px] 减去自身 padding） */}
                    {row.brief && (
                      <p className="mt-0.5 truncate pl-[18px] text-2xs leading-relaxed text-fg-subtle/90">
                        {row.brief}
                      </p>
                    )}
                    {row.outputs && row.outputs.length > 0 && (
                      <p className="mt-0.5 truncate pl-[18px] font-mono text-2xs text-fg-subtle/70">
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
