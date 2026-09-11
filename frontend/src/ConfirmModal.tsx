/**
 * T6.6 · 大纲确认模态（前端编辑 + resume）。
 *
 * 对应后端的 `POST /api/confirm`：`action` ∈ approve | skip | edit | abort。
 *
 * ⚠️ **为什么必须调 `/api/confirm` 而不是复用 SSE**：
 *   T5.4 是编译期静态断点，续跑要 `update_state(plan_approval)` + `invoke(None)`，
 *   而 SSE 是单向推送通道投不进决策。所以确认动作走独立端点，
 *   前端确认后再开一条 SSE 订阅后续事件。
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

  const changed = outlines.some((o, i) => o !== payload.outlines[i])
    || outlines.length !== payload.outlines.length;
  const valid = outlines.length > 0 && outlines.every((o) => o.trim().length > 0);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-bg/80 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-title"
    >
      <div className="panel flex max-h-[85vh] w-full max-w-xl flex-col overflow-hidden">
        <div className="border-b border-border px-5 py-4">
          <h2 id="confirm-title" className="text-sm font-medium text-fg">
            大纲确认
          </h2>
          <p className="mt-1 text-[12px] leading-relaxed text-fg-muted">
            已生成调研大纲。确认后开始并行检索；也可修改章节后再继续。
          </p>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          <div className="mb-4">
            <p className="text-[11px] uppercase tracking-wide text-fg-muted">调研目标</p>
            <p className="mt-1 text-[13px] leading-relaxed text-fg">
              {payload.objective || "（未给出目标）"}
            </p>
          </div>

          <div className="mb-4">
            <div className="mb-2 flex items-center justify-between">
              <p className="text-[11px] uppercase tracking-wide text-fg-muted">
                报告大纲（{outlines.length} 章）
              </p>
              {!editing && (
                <button
                  type="button"
                  className="btn-ghost !px-2 !py-1 !text-[11px]"
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
                  className="input resize-y font-mono !text-[12.5px] leading-relaxed"
                  placeholder="每行一个章节标题"
                  aria-label="编辑报告大纲，每行一个章节"
                />
                <div className="flex items-center gap-3 text-[11px] text-fg-muted">
                  <span>每行一章，空行会被忽略</span>
                  {changed && <span className="text-warn">已修改</span>}
                  {!valid && <span className="text-danger">章节不能为空</span>}
                </div>
              </div>
            ) : (
              <ol className="space-y-1.5">
                {outlines.map((o, i) => (
                  <li key={i} className="flex gap-2.5 text-[13px] leading-relaxed">
                    <span className="shrink-0 font-mono text-[11px] text-fg-muted">
                      {i + 1}.
                    </span>
                    <span className="text-fg">{o}</span>
                  </li>
                ))}
              </ol>
            )}
          </div>

          {payload.sub_questions.length > 0 && (
            <div>
              <p className="mb-2 text-[11px] uppercase tracking-wide text-fg-muted">
                检索子问题（{payload.sub_questions.length} 个）
              </p>
              <ul className="space-y-1">
                {payload.sub_questions.map((q, i) => (
                  <li
                    key={i}
                    className="flex gap-2.5 text-[12px] leading-relaxed text-fg-muted"
                  >
                    <span className="shrink-0 font-mono text-[11px]">{i + 1}.</span>
                    <span>{q}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border px-5 py-3.5">
          <button
            type="button"
            className="btn-ghost !text-danger hover:!bg-danger/10"
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
