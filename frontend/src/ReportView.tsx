/**
 * T6.5 · 报告双栏视图 + 引用悬浮卡片。
 *
 * 左：正文（markdown 渲染）；右：引用列表。
 * 正文里 `[WEB1-1-1]` 这类引用编号可 hover / focus，弹出证据原文与来源链接。
 *
 * **T7.10 · 拒编态**：`sufficiency.sufficient === false` 时，正文是 analyst 产出的
 * **拒编页**（"查了什么 / 为什么不出结论 / 怎么办"），`reference_list` 必然为空。
 * 此时：① 顶部挂一条提示条，让用户一眼看出"是主动拒编、不是故障"；
 * ② **不渲染右栏**——否则会显示"没有引用来源"，被误读成抓取失败。
 *
 * ⚠️ **不用第三方 markdown 库**（`react-markdown` 会带 20+ 依赖，
 * 而本项目的报告格式是自己生成的、可控）。用一个小的行级渲染器即可，
 * 代价是不支持嵌套列表等复杂结构——本项目报告用不到，够用就行。
 * 这个取舍写在这里，避免下一个人以为"漏了"。
 */

import { useMemo, useState } from "react";
import type { EvidenceSufficiency, ReferenceItem } from "./api";

const CITE_RE = /(\[(?:WEB|LOC)\d+-\d+-\d+\])/g;

interface Props {
  markdown: string;
  references: Record<string, ReferenceItem>;
  /** T7.10：证据充足性。缺省（老后端 / 非调研路径）视为"充足"，不显示提示条。 */
  sufficiency?: EvidenceSufficiency;
}

export function ReportView({ markdown, references, sufficiency }: Props) {
  const [active, setActive] = useState<string | null>(null);
  const refEntries = useMemo(() => Object.entries(references), [references]);

  const blocks = useMemo(() => parseBlocks(markdown), [markdown]);

  // 拒编：`sufficient` 显式为 false 才算（undefined 是"老后端没这字段"，不能当成拒编）
  const refused = sufficiency?.sufficient === false;

  return (
    <div
      className={`grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-hidden ${
        refused ? "" : "xl:grid-cols-[minmax(0,1fr)_320px]"
      }`}
    >
      <section className="panel flex min-h-0 flex-col overflow-hidden">
        <ExportBar markdown={markdown} />
        {refused && sufficiency && <RefusalBanner info={sufficiency} />}
        <div className="report-body min-h-0 flex-1 overflow-y-auto p-5">
          {blocks.map((b, i) => (
            <Block key={i} block={b} references={references} onCite={setActive} />
          ))}
        </div>
      </section>

      {!refused && (
        <aside className="panel print-show hidden min-h-0 flex-col overflow-hidden xl:flex">
          <div className="border-b border-border px-4 py-3">
            <h2 className="text-sm font-medium text-fg">引用来源</h2>
            <p className="mt-0.5 text-[11px] text-fg-muted">{refEntries.length} 条证据</p>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto p-2">
            {refEntries.length === 0 ? (
              <p className="px-3 py-8 text-center text-xs text-fg-muted">没有引用来源</p>
            ) : (
              <ul className="space-y-1.5">
                {refEntries.map(([id, ref]) => (
                  <li
                    key={id}
                    id={`ref-${id}`}
                    onMouseEnter={() => setActive(id)}
                    onMouseLeave={() => setActive(null)}
                    className={`scroll-mt-2 rounded-lg border px-3 py-2 transition-colors duration-150 ${
                      active === id
                        ? "border-accent/50 bg-muted"
                        : "border-transparent hover:bg-muted/50"
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      <span className="mt-0.5 shrink-0 rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-accent">
                        {id}
                      </span>
                      {ref.url ? (
                        <a
                          href={ref.url}
                          target="_blank"
                          rel="noreferrer noopener"
                          className="text-[12px] leading-snug text-info underline decoration-info/40 underline-offset-2 hover:decoration-info"
                        >
                          {ref.title || ref.url}
                        </a>
                      ) : (
                        <span className="text-[12px] leading-snug text-fg">{ref.title}</span>
                      )}
                    </div>
                    {ref.snippet && (
                      <p className="mt-1.5 line-clamp-3 text-[11px] leading-relaxed text-fg-muted">
                        {ref.snippet}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </aside>
      )}
    </div>
  );
}

// ============================================================ 导出（T8.2）

/**
 * 从报告正文里推断一个像样的文件名。
 *
 * 取正文第一个 H1 当标题——**不要用 thread_id**：用户下载下来是想归档或交出去，
 * `web-1789286556753.md` 这种名字对人是噪音。标题里的非法字符要替换掉，
 * 否则 Windows 上会保存失败（且各浏览器报错方式不一致，很难查）。
 */
function reportFileName(md: string): string {
  const m = /^#\s+(.+)$/m.exec(md);
  const base = (m?.[1] ?? "调研报告")
    .trim()
    .replace(/[\\/:*?"<>|\s]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 60);
  const d = new Date();
  const p = (v: number) => String(v).padStart(2, "0");
  return `${base || "调研报告"}-${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(
    d.getHours(),
  )}${p(d.getMinutes())}.md`;
}

/**
 * 复制文本到剪贴板（带非安全上下文的回退）。
 *
 * ⚠️ **为什么不能直接用 `navigator.clipboard`**：它在**安全上下文**才存在
 * （https 或 localhost）。局域网演示时页面是 `http://192.168.x.x:5173`——
 * 不是安全上下文，`navigator.clipboard` 直接是 `undefined`，
 * 调用会抛 TypeError，"复制"按钮在演示现场静默失灵，而开发机上一切正常。
 * 这条分支就是为那个场景写的。
 */
async function copyText(text: string): Promise<void> {
  if (window.isSecureContext && navigator.clipboard) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.top = "-9999px";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } finally {
    document.body.removeChild(ta);
  }
  if (!ok) throw new Error("浏览器拒绝了复制操作，请手动全选复制");
}

/**
 * 报告导出的操作条。
 *
 * 三种导出方式，覆盖不同用途：
 *   - **复制**：贴进飞书/Notion/聊天窗口，最常用；
 *   - **下载 .md**：归档、进 Git、二次编辑（后端 `export_report` 落盘的也是 md）；
 *   - **打印**：需要 PDF 时的现实解法。T7.7 已经决策过——中文 PDF 要内嵌 CJK 字体，
 *     与"依赖最小"冲突，所以交给浏览器打印（打印样式见 `index.css` 的 `@media print`）。
 */
function ExportBar({ markdown }: { markdown: string }) {
  const [copied, setCopied] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const onCopy = async () => {
    try {
      await copyText(markdown);
      setErr(null);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      setErr((e as Error).message);
    }
  };

  const onDownload = () => {
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = reportFileName(markdown);
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    // 立刻 revoke 在部分浏览器上会打断下载；下一个宏任务再回收
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  return (
    <div className="no-print flex shrink-0 items-center gap-1 border-b border-border px-3 py-2">
      <span className="mr-auto text-[11px] text-fg-muted">导出</span>
      {err && <span className="mr-1 text-[11px] text-danger">{err}</span>}
      <button type="button" className="btn-ghost !px-2 !py-1 !text-[11px]" onClick={() => void onCopy()}>
        {copied ? "已复制" : "复制正文"}
      </button>
      <button type="button" className="btn-ghost !px-2 !py-1 !text-[11px]" onClick={onDownload}>
        下载 .md
      </button>
      <button
        type="button"
        className="btn-ghost !px-2 !py-1 !text-[11px]"
        onClick={() => window.print()}
        title="用浏览器打印（可选另存为 PDF）"
      >
        打印
      </button>
    </div>
  );
}

// ============================================================ T7.10 拒编提示条

/**
 * 证据不足时的提示条。
 *
 * 放在**滚动区之外**（`section` 的 flex 首子元素）：正文很长时提示条仍常驻可见，
 * 用户不会因为滑到文章底部而错过"这一页不是结论"这个前提。
 *
 * 措辞刻意与后端 `build_insufficient_report` 的口径一致——两处说法不一致的话，
 * 用户会怀疑是两套东西。
 */
function RefusalBanner({ info }: { info: EvidenceSufficiency }) {
  const nEv = info.n_evidence ?? 0;
  const nSel = info.n_selected ?? 0;
  const nGrounded = info.n_grounded ?? 0;
  const nSub = info.sub_questions?.length ?? 0;
  const nCov = info.covered_sub_questions?.length ?? 0;

  return (
    <div className="shrink-0 border-b border-warn/30 bg-warn/10 px-5 py-3.5">
      <div className="flex items-start gap-2.5">
        <span className="dot mt-1.5 bg-warn" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <h2 className="flex flex-wrap items-center gap-2 text-[13px] font-medium text-fg">
            本次未产出调研结论
            <span className="rounded bg-warn/20 px-1.5 py-0.5 text-[10px] font-normal text-warn">
              证据不足 · 系统主动拒编
            </span>
          </h2>

          <p className="mt-1 text-[12px] leading-relaxed text-fg-muted">
            {info.reasons?.[0] || "本轮检索没有获得能支撑结论的相关证据。"}
            按本系统的「零造假」原则，没有证据支撑的数字与判断一律不写——所以不给结论，
            而不是用不相关的资料拼一份看着完整、实则张冠李戴的报告。
          </p>

          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-fg-muted">
            <Stat label="检索条目" value={`${nEv} 条`} />
            <Stat label="通过判据" value={`${nSel} 条`} />
            <Stat label="真实命中" value={`${nGrounded} 条`} />
            <Stat label="子问题覆盖" value={`${nCov}/${nSub}`} />
            {info.basis && <Stat label="判据" value={info.basis} />}
          </div>

          <p className="mt-2 text-[11.5px] leading-relaxed text-fg-muted">
            <span className="text-fg">这不是故障</span>，而是系统在证据不足时的主动拦截。
            要拿到真正的报告：接入真实检索（配置{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px] text-accent">
              TAVILY_API_KEY
            </code>{" "}
            并把{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px] text-accent">
              ATTEST_SEARCH_MODE
            </code>{" "}
            设为{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px] text-accent">
              tavily
            </code>
            ）后重跑本次问题。
          </p>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-baseline gap-1">
      <span className="text-fg-muted/70">{label}</span>
      <span className="font-mono text-fg">{value}</span>
    </span>
  );
}

// ============================================================ 极简 markdown

type Block =
  | { kind: "h"; level: number; text: string }
  | { kind: "p"; text: string }
  | { kind: "ul"; items: string[] }
  | { kind: "ol"; items: string[] }
  | { kind: "quote"; text: string }
  | { kind: "hr" };

function parseBlocks(md: string): Block[] {
  const out: Block[] = [];
  const lines = md.split("\n");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const t = line.trim();

    if (!t) {
      i++;
      continue;
    }
    if (/^(-{3,}|\*{3,})$/.test(t)) {
      out.push({ kind: "hr" });
      i++;
      continue;
    }
    const h = /^(#{1,4})\s+(.*)$/.exec(t);
    if (h) {
      out.push({ kind: "h", level: h[1].length, text: h[2] });
      i++;
      continue;
    }
    if (t.startsWith("> ")) {
      out.push({ kind: "quote", text: t.slice(2) });
      i++;
      continue;
    }
    if (/^[-*]\s+/.test(t)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*]\s+/, ""));
        i++;
      }
      out.push({ kind: "ul", items });
      continue;
    }
    if (/^\d+\.\s+/.test(t)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ""));
        i++;
      }
      out.push({ kind: "ol", items });
      continue;
    }
    out.push({ kind: "p", text: t });
    i++;
  }
  return out;
}

function Inline({
  text,
  references,
  onCite,
}: {
  text: string;
  references: Record<string, ReferenceItem>;
  onCite: (id: string | null) => void;
}) {
  const parts = text.split(CITE_RE);
  return (
    <>
      {parts.map((part, i) => {
        const m = part.match(/^\[((?:WEB|LOC)\d+-\d+-\d+)\]$/);
        if (!m) return <InlineMd key={i} text={part} />;
        const id = m[1];
        // 后端 `_reference_index` 的键是带方括号的完整编号（`[LOC1-1-1]`），
        // 而这里捕获组剥掉了括号——不补回的话 known 恒为 false，
        // 悬浮卡永不渲染、右栏高亮永不命中（2026-09-12 浏览器验收实测发现）。
        const key = `[${id}]`;
        const ref = references[key];
        const known = Boolean(ref);
        return (
          <span key={i} className="group relative inline-block">
            <button
              type="button"
              onMouseEnter={() => onCite(key)}
              onMouseLeave={() => onCite(null)}
              onFocus={() => onCite(key)}
              onBlur={() => onCite(null)}
              className={`mx-0.5 cursor-pointer rounded px-1 font-mono text-[10.5px] align-middle transition-colors duration-150 ${
                known
                  ? "bg-muted text-accent hover:bg-accent/20"
                  : "bg-danger/15 text-danger"
              }`}
              aria-label={known ? `引用 ${id}` : `引用 ${id}（无对应证据）`}
            >
              {id}
            </button>
            {known && <HoverCard id={id} ref_={ref} />}
          </span>
        );
      })}
    </>
  );
}

function HoverCard({
  id,
  ref_,
}: {
  id: string;
  ref_: ReferenceItem | undefined;
}) {
  if (!ref_) return null;
  return (
    <span
      role="tooltip"
      className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-2 w-72 -translate-x-1/2
                 rounded-lg border border-border bg-surface p-3 opacity-0
                 transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100"
    >
      <span className="block font-mono text-[10px] text-accent">{id}</span>
      <span className="mt-1 block text-[12px] font-medium leading-snug text-fg">
        {ref_.title || "（无标题）"}
      </span>
      <span className="mt-1.5 block line-clamp-4 text-[11px] leading-relaxed text-fg-muted">
        {ref_.snippet}
      </span>
      {ref_.url && (
        <span className="mt-1.5 block truncate font-mono text-[10px] text-info">
          {ref_.url}
        </span>
      )}
    </span>
  );
}

/** 行内的粗体 / 代码 / 链接（不含引用编号，引用在上面单独处理） */
function InlineMd({ text }: { text: string }) {
  const nodes: React.ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) {
      nodes.push(
        <strong key={k++} className="font-medium text-fg">
          {tok.slice(2, -2)}
        </strong>,
      );
    } else if (tok.startsWith("`")) {
      nodes.push(<code key={k++}>{tok.slice(1, -1)}</code>);
    } else {
      const lm = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(tok);
      if (lm) {
        nodes.push(
          <a key={k++} href={lm[2]} target="_blank" rel="noreferrer noopener">
            {lm[1]}
          </a>,
        );
      } else nodes.push(tok);
    }
    last = m.index + tok.length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return <>{nodes}</>;
}

function Block({
  block,
  references,
  onCite,
}: {
  block: Block;
  references: Record<string, ReferenceItem>;
  onCite: (id: string | null) => void;
}) {
  const inline = (t: string) => (
    <Inline text={t} references={references} onCite={onCite} />
  );
  switch (block.kind) {
    case "h": {
      const Tag = (`h${Math.min(block.level, 3)}`) as "h1" | "h2" | "h3";
      return <Tag>{inline(block.text)}</Tag>;
    }
    case "p":
      return <p>{inline(block.text)}</p>;
    case "ul":
      return (
        <ul>
          {block.items.map((it, i) => (
            <li key={i}>{inline(it)}</li>
          ))}
        </ul>
      );
    case "ol":
      return (
        <ol>
          {block.items.map((it, i) => (
            <li key={i}>{inline(it)}</li>
          ))}
        </ol>
      );
    case "quote":
      return <blockquote>{inline(block.text)}</blockquote>;
    case "hr":
      return <hr />;
  }
}
