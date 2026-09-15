/**
 * T6.6 · 大纲确认模态（前端编辑 + resume）。
 *
 * 对应后端的 `POST /api/confirm`：`action` ∈ approve | skip | edit | abort。
 *
 * ⚠️ **为什么必须调 `/api/confirm` 而不是复用 SSE**：
 *   T5.4 是编译期静态断点，续跑要 `update_state(plan_approval)` + `invoke(None)`，
 *   而 SSE 是单向推送通道投不进决策。所以确认动作走独立端点，
 *   前端确认后再开一条 SSE 订阅后续事件。
 *
 * T9.3：遮罩用**模糊**而不是纯半透明黑。纯遮罩只是"压暗背景"，
 * 背景文字仍然参与视觉竞争；加一层 blur 后它彻底退成"底"，模态本身才立得住。
 */

import { useEffect, useRef, useState } from "react";

export interface ConfirmPayload {
  node: string;
  objective: string;
  outlines: string[];
  sub_questions: string[];
}

interface Props {
  payload: ConfirmPayload;
  busy: boolean;
  onDecide: (action: "approve" | "skip" | "edit" | "abort", plan?: { outlines: string[] }) => void;
}

export function ConfirmModal({ payload, busy, onDecide }: Props) {
  const [outlines, setOutlines] = useState<string[]>(payload.outlines);
  const [editing, setEditing] = useState(false);
  const firstRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    setOutlines(payload.outlines);
    setEditing(false);
  }, [payload]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Escape 不直接中止（中止是破坏性动作），只退出编辑态
      if (e.key === "Escape" && editing) {
        setEditing(false);
        setOutlines(payload.outlines);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [editing, payload.outlines]);

  const changed =
    outlines.some((o, i) => o !== payload.outlines[i]) ||
    outlines.length !== payload.outlines.length;
  const valid = outlines.length > 0 && outlines.every((o) => o.trim().length > 0);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-[#04070D]/70 p-4
                 backdrop-blur-[3px]"
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-title"
    >
      <div
        className="panel-raised flex max-h-[85vh] w-full max-w-xl animate-rise-in flex-col
                   overflow-hidden"
      >
        <div className="shrink-0 border-b border-border px-5 py-4">
          <div className="flex items-center gap-2.5">
            <span
              className="flex h-6 w-6 shrink-0 items-center justify-center rounded-[7px]
                         border border-warn/30 bg-warn-soft text-warn"
              aria-hidden="true"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
                <path
                  d="M12 9v4.5M12 17h.01"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
                <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.6" />
              </svg>
            </span>
            <h2 id="confirm-title" className="text-sm font-semibold tracking-tight text-fg">
              大纲确认
            </h2>
            <span className="chip-warn ml-auto shrink-0">待你决策</span>
          </div>
          <p className="mt-1.5 text-xs leading-relaxed text-fg-muted">
            已生成调研大纲。确认后开始并行检索；也可修改章节后再继续。
          </p>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          <div className="mb-4">
            <p className="text-2xs font-medium uppercase tracking-wider text-fg-subtle">调研目标</p>
            <p className="mt-1 text-sm leading-relaxed text-fg">
              {payload.objective || "（未给出目标）"}
            </p>
          </div>

          <div className="divider-fade mb-4" />

          <div className="mb-4">
            <div className="mb-2 flex items-center justify-between gap-2">
              <p className="text-2xs font-medium uppercase tracking-wider text-fg-subtle">
                报告大纲
                <span className="ml-1.5 font-mono text-fg-muted normal-case">
                  {outlines.length} 章
                </span>
              </p>
              {!editing && (
                <button
                  type="button"
                  className="btn-ghost btn-xs"
                  onClick={() => {
                    setEditing(true);
                    setTimeout(() => firstRef.current?.focus(), 0);
                  }}
                >
                  编辑
                </button>
              )}
            </div>

            {editing ? (
              <div className="space-y-2">
                <textarea
                  ref={firstRef}
                  value={outlines.join("\n")}
                  onChange={(e) => setOutlines(e.target.value.split("\n"))}
                  rows={Math.max(4, outlines.length + 1)}
                  className="input resize-y font-mono !text-xs leading-relaxed"
                  placeholder="每行一个章节标题"
                  aria-label="编辑报告大纲，每行一个章节"
                />
                <div className="flex items-center gap-3 text-2xs text-fg-muted">
                  <span>每行一章，空行会被忽略</span>
                  {changed && <span className="text-warn">已修改</span>}
                  {!valid && <span className="text-danger">章节不能为空</span>}
                </div>
              </div>
            ) : (
              <ol className="space-y-1">
                {outlines.map((o, i) => (
                  <li
                    key={i}
                    style={{ animationDelay: `${Math.min(i, 10) * 20}ms` }}
                    className="flex animate-fade-up gap-2.5 rounded-control px-2 py-1.5 text-sm
                               leading-relaxed transition-colors duration-fast hover:bg-elevated/40"
                  >
                    <span className="mt-px shrink-0 font-mono text-2xs text-fg-subtle">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <span className="text-fg">{o}</span>
                  </li>
                ))}
              </ol>
            )}
          </div>

          {payload.sub_questions.length > 0 && (
            <div>
              <div className="divider-fade mb-4" />
              <p className="mb-2 text-2xs font-medium uppercase tracking-wider text-fg-subtle">
                检索子问题
                <span className="ml-1.5 font-mono text-fg-muted normal-case">
                  {payload.sub_questions.length} 个
                </span>
              </p>
              <ul className="space-y-1">
                {payload.sub_questions.map((q, i) => (
                  <li
                    key={i}
                    className="flex gap-2.5 text-xs leading-relaxed text-fg-muted"
                  >
                    <span className="mt-px shrink-0 font-mono text-2xs text-fg-subtle/80">
                      {i + 1}.
                    </span>
                    <span>{q}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="flex shrink-0 flex-wrap items-center justify-between gap-3
                        border-t border-border bg-bg/30 px-5 py-3.5">
          <button
            type="button"
            className="btn-ghost !text-danger hover:!border-danger/40 hover:!bg-danger-soft"
            disabled={busy}
            onClick={() => onDecide("abort")}
          >
            中止
          </button>

          <div className="flex items-center gap-2">
            <button
              type="button"
              className="btn-ghost"
              disabled={busy}
              onClick={() => onDecide("skip")}
            >
              跳过确认
            </button>
            {editing ? (
              <>
                <button
                  type="button"
                  className="btn-ghost"
                  disabled={busy}
                  onClick={() => {
                    setEditing(false);
                    setOutlines(payload.outlines);
                  }}
                >
                  取消编辑
                </button>
                <button
                  type="button"
                  className="btn-primary"
                  disabled={busy || !valid}
                  onClick={() =>
                    onDecide("edit", { outlines: outlines.map((o) => o.trim()).filter(Boolean) })
                  }
                >
                  {busy ? "提交中…" : "提交并继续"}
                </button>
              </>
            ) : (
              <button
                type="button"
                className="btn-primary"
                disabled={busy}
                onClick={() => onDecide("approve")}
              >
                {busy ? "启动中…" : "确认并开始检索"}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
