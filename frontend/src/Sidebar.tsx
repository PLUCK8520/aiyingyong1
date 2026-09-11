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
    <aside className="panel flex h-full w-[280px] shrink-0 flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div>
          <h1 className="text-sm font-medium text-fg">Attest 质证</h1>
          <p className="mt-0.5 text-[11px] text-fg-muted">逐句质证的调研工作台</p>
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
          <p className="px-3 py-6 text-center text-xs text-fg-muted">加载中…</p>
        ) : sessions.length === 0 ? (
          <div className="px-3 py-8 text-center">
            <p className="text-xs text-fg-muted">还没有会话</p>
            <p className="mt-1 text-[11px] leading-relaxed text-fg-muted/70">
              点右上角 + 新建，
              <br />
              或在下方直接提问开始调研
            </p>
          </div>
        ) : (
          <ul className="space-y-1">
            {sessions.map((s) => {
              const meta = STATUS_META[s.status];
              const active = s.thread_id === activeId;
              return (
                <li key={s.thread_id}>
                  <button
                    type="button"
                    onClick={() => onSelect(s.thread_id)}
                    className={`w-full cursor-pointer rounded-lg px-3 py-2 text-left transition-colors duration-150 ${
                      active ? "bg-muted" : "hover:bg-muted/60"
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      <span
                        className={`dot mt-1.5 ${meta.color} ${
                          meta.pulse ? "animate-pulse-soft" : ""
                        }`}
                        title={meta.label}
                        aria-hidden="true"
                      />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-[12.5px] text-fg">
                          {s.query || "（未命名会话）"}
                        </p>
                        <p className="mt-0.5 flex items-center gap-1.5 text-[11px] text-fg-muted">
                          <span>{meta.label}</span>
                          <span className="text-fg-muted/50">·</span>
                          <span className="font-mono text-[10px]">
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

      <div className="border-t border-border px-4 py-2.5">
        <p className="text-[10.5px] leading-relaxed text-fg-muted/70">
          状态点：蓝=运行中 / 黄=待确认 / 绿=已完成 / 红=失败
        </p>
      </div>
    </aside>
  );
}
