/**
 * T8.1 · 知识库管理视图（2026-09-13）。
 *
 * **为什么值得单独做一屏**：系统的本地检索（`scout_local`）从 P3 起就一直可用，
 * 但入库只能敲命令行。用户问一个语料覆盖不到的问题会被拒编——**拒编这个结果本身是对的**
 * （零造假铁律），只是用户不知道"只要把自己的资料传进来，就能出带引用的真报告"。
 * 这一屏就是那个入口：把"系统凭什么拒编"和"怎么让它不拒编"放在同一个地方。
 *
 * **设计取舍**：
 *   - 上传**逐个串行**（不是 Promise.all）：单文件失败要能精确报出是哪一个，
 *     并发上传时错误会糊成一团，而且后端分块是 CPU 密集，并发只会互相抢。
 *   - 部分失败**必须逐条列出**：只提示"上传完成"是在骗人——3 个文件里坏了 1 个，
 *     用户会以为都进去了。
 *   - `sample: true` 的文档打「示例」标：那 7 个是仓库自带语料，不是用户传的，
 *     不标出来用户会纳闷"我一个都没传，怎么已经有 7 个了"。
 *
 * T9.3：本视图在 App 的"纸面"内，各段用分隔线分，**不再各自套 panel**。
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { api, type KBDocument, type KBStats } from "./api";

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}

function fmtTime(ts: number): string {
  const d = new Date(ts * 1000);
  if (Number.isNaN(d.getTime())) return "—";
  const p = (v: number) => String(v).padStart(2, "0");
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

export function KBView() {
  const [docs, setDocs] = useState<KBDocument[]>([]);
  const [stats, setStats] = useState<KBStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [pending, setPending] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const load = useCallback(async () => {
    try {
      const r = await api.kbOverview();
      setDocs(r.documents);
      setStats(r.stats);
      setError(null);
    } catch (e) {
      setError(`读取知识库失败：${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const upload = useCallback(
    async (files: File[]) => {
      if (!files.length) return;
      setBusy(true);
      setError(null);
      setNotice(null);
      const ok: string[] = [];
      const bad: string[] = [];
      for (const f of files) {
        try {
          const r = await api.kbUpload(f);
          ok.push(`${r.name}（${r.chunks} 块${r.replaced ? "，已覆盖同名" : ""}）`);
        } catch (e) {
          bad.push(`${f.name} → ${(e as Error).message}`);
        }
      }
      await load();
      setBusy(false);
      const parts: string[] = [];
      if (ok.length) parts.push(`已入库 ${ok.length} 个：${ok.join("；")}`);
      if (bad.length) parts.push(`失败 ${bad.length} 个：${bad.join("；")}`);
      setNotice(parts.join("　·　") || "没有可处理的文件");
    },
    [load],
  );

  const remove = useCallback(
    async (name: string) => {
      setPending(name);
      setError(null);
      setNotice(null);
      try {
        const r = await api.kbDelete(name);
        setNotice(`已移除 ${r.name}（同步摘除索引 ${r.removed_chunks} 块）`);
        await load();
      } catch (e) {
        setError(`移除失败：${(e as Error).message}`);
      } finally {
        setPending(null);
      }
    },
    [load],
  );

  const rebuild = useCallback(async () => {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const r = await api.kbRebuild();
      setNotice(
        r.removed_index
          ? "已丢弃持久索引，下次检索会从文档目录全量重建"
          : "已失效检索缓存（当前是内存索引档，本就每次重建）",
      );
      await load();
    } catch (e) {
      setError(`重建失败：${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }, [load]);

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
      {/* ------------------------------------------------ 概览 */}
      <section className="shrink-0 border-b border-border px-6 py-5">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <h2 className="text-sm font-semibold tracking-tight text-fg">本地知识库</h2>
            <p className="mt-1 max-w-[70ch] text-2xs leading-relaxed text-fg-muted">
              这里的文档会被「本地检索」节点召回，作为报告里 <code className="font-mono">[LOC*]</code>{" "}
              引用的来源。不需要联网、不消耗额度。
            </p>
          </div>
          <button
            type="button"
            className="btn-ghost btn-xs shrink-0"
            onClick={() => void rebuild()}
            disabled={busy}
            title="换了 embedding 模型、或索引与磁盘不一致时使用"
          >
            重建索引
          </button>
        </div>

        {/* 统计网格用 `gap-px` + 容器底色做**发丝线**：
            比"四张各自带边框的小卡"轻得多（后者在暗色里会变成四个盒子），
            但依然读得出是四个独立的数。 */}
        <div
          className="mt-4 grid grid-cols-2 gap-px overflow-hidden rounded-card border
                     border-border/70 bg-border/40 sm:grid-cols-4"
        >
          <Stat label="文档" value={stats ? String(stats.documents) : "—"} />
          <Stat label="可检索片段" value={stats ? String(stats.chunks) : "—"} />
          <Stat
            label="索引档位"
            value={stats ? (stats.store === "chroma" ? "持久化" : "内存") : "—"}
            hint={stats ? (stats.store === "chroma" ? "Chroma" : "每次检索重建") : undefined}
          />
          <Stat
            label="证据来源"
            value={stats ? (stats.web_search_ready ? "网络 + 本地" : "仅本地") : "—"}
            hint={stats ? (stats.web_search_ready ? "Tavily 已就绪" : "网络检索未接入") : undefined}
          />
        </div>

        {/* 下面三条是"这个库能不能支撑出一份真报告"的关键前提，必须显式暴露 */}
        {stats && !stats.local_enabled && (
          <Notice tone="danger">
            本地知识库已被配置关闭（<Code>ATTEST_LOCAL_ENABLED=0</Code>
            ），这里的文档<Strong>不会</Strong>被任何节点检索到。改配置后重启后端即可生效。
          </Notice>
        )}
        {stats && !stats.web_search_ready && (
          <Notice tone="warn">
            网络检索当前未接入（<Code>ATTEST_SEARCH_MODE={stats.search_mode}</Code>
            ），<Strong>本地库是唯一的真实证据来源</Strong>
            。库里没有的主题，系统会按「零造假」原则拒编而不是编造内容——这不是故障。
          </Notice>
        )}
        {stats && stats.embed_fallback === "hashing" && (
          <Notice tone="neutral">
            向量检索走的是<Strong>词法兜底</Strong>（<Code>ATTEST_EMBED_FALLBACK=hashing</Code>
            ）：当前厂商无 embedding 接口时，用散列词袋代替语义向量。检索仍可用，但
            <Strong>只按字面匹配、不懂同义改写</Strong>——文档一多，召回质量会明显下降。
          </Notice>
        )}
      </section>

      {/* ------------------------------------------------ 提示 / 错误 */}
      {error && (
        <div
          role="alert"
          className="flex shrink-0 animate-fade-in items-start gap-2.5 border-b border-danger/30
                     bg-danger-soft/40 px-6 py-2.5"
        >
          <span className="mt-1.5 dot bg-danger" aria-hidden="true" />
          <p className="flex-1 text-xs leading-relaxed text-fg">{error}</p>
          <button type="button" className="btn-ghost btn-xs" onClick={() => setError(null)}>
            关闭
          </button>
        </div>
      )}
      {notice && (
        <div className="flex shrink-0 animate-fade-in items-start gap-2.5 border-b border-accent/25 bg-accent/[0.07] px-6 py-2.5">
          <span className="mt-1.5 dot bg-accent" aria-hidden="true" />
          <p className="flex-1 text-xs leading-relaxed text-fg">{notice}</p>
          <button type="button" className="btn-ghost btn-xs" onClick={() => setNotice(null)}>
            关闭
          </button>
        </div>
      )}

      {/* ------------------------------------------------ 上传 */}
      <section className="shrink-0 border-b border-border px-6 py-5">
        <div
          className={`rounded-panel border border-dashed px-6 py-7 text-center
                      transition-colors duration-fast ${
                        dragging
                          ? "border-accent/70 bg-accent/[0.06]"
                          : "border-border-strong/70 bg-bg/30 hover:border-border-strong"
                      }`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            void upload(Array.from(e.dataTransfer.files));
          }}
        >
          <input
            ref={fileRef}
            type="file"
            multiple
            accept=".md,.markdown,.txt,.pdf"
            className="hidden"
            onChange={(e) => {
              void upload(Array.from(e.target.files ?? []));
              // 清空 value：否则连续选同一个文件不会触发 change（浏览器认为值没变）
              e.target.value = "";
            }}
          />
          <p className="text-sm font-medium text-fg">
            {busy ? "正在入库…" : dragging ? "松开即可导入" : "把文档拖到这里"}
          </p>
          <p className="mt-1 text-2xs text-fg-subtle">支持 md / txt / pdf，单文件 ≤ 8MB</p>
          <button
            type="button"
            className="btn-ghost mx-auto mt-3.5"
            onClick={() => fileRef.current?.click()}
            disabled={busy}
          >
            选择文件
          </button>
        </div>
      </section>

      {/* ------------------------------------------------ 文档列表（最后一段，无下边框） */}
      <section className="flex min-h-0 flex-col">
        <div className="panel-head shrink-0">
          <h2 className="text-sm font-semibold tracking-tight text-fg">文档清单</h2>
          <span className="chip-neutral">
            {loading ? (
              "加载中"
            ) : (
              <>
                <span className="font-mono">{docs.length}</span> 个
              </>
            )}
          </span>
        </div>

        <div className="min-h-0 flex-1 px-3 py-2">
          {loading ? (
            <p className="px-3 py-8 text-center text-xs text-fg-subtle/80">正在读取…</p>
          ) : docs.length === 0 ? (
            <div className="px-3 py-8 text-center">
              <p className="text-xs text-fg-muted">知识库还是空的</p>
              <p className="mt-1.5 text-2xs text-fg-subtle">
                传一份你自己的资料进来，再问一个相关问题试试
              </p>
            </div>
          ) : (
            <ul className="space-y-0.5">
              {docs.map((d, i) => (
                <li
                  key={d.name}
                  style={{ animationDelay: `${Math.min(i, 10) * 16}ms` }}
                  className="group flex animate-fade-up items-center gap-3 rounded-control px-2.5 py-2
                             transition-colors duration-fast hover:bg-elevated/40"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-xs text-fg">{d.title}</span>
                      {d.sample && <span className="chip-neutral shrink-0">示例</span>}
                    </div>
                    <p className="mt-0.5 truncate font-mono text-2xs text-fg-subtle">
                      {d.name} · {fmtBytes(d.size)} · {d.chunks} 块 · {fmtTime(d.modified)}
                    </p>
                  </div>
                  <button
                    type="button"
                    className="btn-ghost btn-xs shrink-0 opacity-0 transition-opacity duration-fast
                               focus-visible:opacity-100 group-hover:opacity-100"
                    onClick={() => void remove(d.name)}
                    disabled={pending === d.name}
                    aria-label={`移除 ${d.name}`}
                  >
                    {pending === d.name ? "移除中…" : "移除"}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {stats && (
          <div className="shrink-0 border-t border-border/60 px-5 py-2">
            <p className="truncate font-mono text-2xs text-fg-subtle/80" title={stats.docs_dir}>
              目录 {stats.docs_dir}
            </p>
          </div>
        )}
      </section>
    </div>
  );
}

/** 统计格：靠容器的 `gap-px` 形成发丝线，自身只负责内容与底色 */
function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="bg-bg/40 px-3.5 py-3">
      <p className="text-2xs text-fg-muted">{label}</p>
      <p className="mt-1 truncate font-mono text-lg font-semibold tracking-tight text-fg">
        {value}
      </p>
      {hint && <p className="mt-0.5 truncate text-2xs text-fg-subtle/80">{hint}</p>}
    </div>
  );
}

function Notice({
  tone,
  children,
}: {
  tone: "danger" | "warn" | "neutral";
  children: React.ReactNode;
}) {
  const cls =
    tone === "danger"
      ? "border-danger/30 bg-danger-soft/40"
      : tone === "warn"
        ? "border-warn/30 bg-warn-soft/40"
        : "border-border/70 bg-bg/40";
  return (
    <p className={`mt-3 rounded-card border px-3.5 py-2.5 text-xs leading-relaxed text-fg ${cls}`}>
      {children}
    </p>
  );
}

function Code({ children }: { children: React.ReactNode }) {
  return (
    <code className="rounded-[4px] bg-muted px-1 py-0.5 font-mono text-2xs text-accent">
      {children}
    </code>
  );
}

function Strong({ children }: { children: React.ReactNode }) {
  return <strong className="font-semibold text-fg">{children}</strong>;
}
