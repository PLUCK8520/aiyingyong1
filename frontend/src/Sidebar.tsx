/**
 * T6.3 · 会话侧栏：会话列表 / 新建 / 切换 thread。
 *
 * 交互态（《功能设计》§8）：
 *   空态给引导文案；会话项显示状态点（运行中/已完成/已中断）。
 */

import { STATUS_META, type SessionMeta } from "./api";

interface Props {
  sessions: SessionMeta[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  loading: boolean;
}

export function Sidebar({ sessions, activeId, onSelect, onNew, loading }: Props) {
  return (
    <aside className="panel no-print flex h-full w-[264px] shrink-0 flex-col overflow-hidden">
      {/* 品牌头。用 accent 小方点 + 字重对比建立识别度——
          旧版只有两行同色文字，和列表项混在一起分不出"这是标题区"。 */}
      <div className="flex items-center justify-between gap-2 border-b border-border px-3.5 py-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <span
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-control bg-accent-soft text-accent"
            aria-hidden="true"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
              <path
                d="M12 3 4.5 6.5v5c0 4.2 3.1 7.9 7.5 9.5 4.4-1.6 7.5-5.3 7.5-9.5v-5L12 3Z"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinejoin="round"
              />
              <path
                d="m9 11.8 2.1 2.2L15.2 9.6"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
          <div className="min-w-0">
            <h2 className="truncate text-sm font-semibold tracking-tight text-fg">Attest 质证</h2>
            <p className="truncate text-2xs text-fg-subtle">逐句质证的调研工作台</p>
          </div>
        </div>
        <button
          type="button"
          onClick={onNew}
          className="btn-ghost !px-2 !py-1.5"
          aria-label="新建会话"
          title="新建会话"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
            <path d="M12 5v14M5 12h14" />
          </svg>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {loading && sessions.length === 0 ? (
          <p className="px-3 py-6 text-center text-xs text-fg-subtle">加载中…</p>
        ) : sessions.length === 0 ? (
          <div className="px-3 py-8 text-center">
            <p className="text-xs font-medium text-fg-muted">还没有会话</p>
            <p className="mt-1.5 text-2xs leading-relaxed text-fg-subtle">
              点右上角 + 新建，
              <br />
              或在右下输入框直接提问
            </p>
          </div>
        ) : (
          <ul className="space-y-0.5">
            {sessions.map((s) => {
              const meta = STATUS_META[s.status];
              const active = s.thread_id === activeId;
              return (
                <li key={s.thread_id}>
                  {/* 选中态用"左侧 2px 色条 + 抬升底色"两个信号，
                      而不是只换背景色——只换底色在暗色里几乎看不出来。 */}
                  <button
                    type="button"
                    onClick={() => onSelect(s.thread_id)}
                    aria-current={active ? "true" : undefined}
                    className={`group relative w-full cursor-pointer rounded-control py-2 pl-3.5 pr-2.5 text-left
                                transition-colors duration-fast ${
                                  active
                                    ? "bg-elevated"
                                    : "hover:bg-elevated/50"
                                }`}
                  >
                    {active && (
                      <span
                        className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-pill bg-accent"
                        aria-hidden="true"
                      />
                    )}
                    <div className="flex items-start gap-2">
                      <span
                        className={`dot mt-1.5 ${meta.color} ${
                          meta.pulse ? "animate-pulse-soft" : ""
                        }`}
                        title={meta.label}
                        aria-hidden="true"
                      />
                      <div className="min-w-0 flex-1">
                        <p
                          className={`truncate text-xs leading-snug ${
                            active ? "font-medium text-fg" : "text-fg-muted group-hover:text-fg"
                          }`}
                        >
                          {s.query || "（未命名会话）"}
                        </p>
                        <p className="mt-1 flex items-center gap-1.5 text-2xs text-fg-subtle">
                          <span>{meta.label}</span>
                          <span aria-hidden="true">·</span>
                          <span className="truncate font-mono opacity-80">
                            {s.thread_id.slice(0, 12)}
                          </span>
                        </p>
                      </div>
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* 底部图例：改成"点 + 文字"横排，比旧版一句话更省高度、更好扫读 */}
      <div className="border-t border-border px-3.5 py-2.5">
        <ul className="flex flex-wrap items-center gap-x-3 gap-y-1 text-2xs text-fg-subtle">
          {(
            [
              ["bg-info", "运行中"],
              ["bg-warn", "待确认"],
              ["bg-accent", "已完成"],
              ["bg-danger", "失败"],
            ] as const
          ).map(([cls, label]) => (
            <li key={label} className="flex items-center gap-1.5">
              <span className={`dot ${cls}`} aria-hidden="true" />
              {label}
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
