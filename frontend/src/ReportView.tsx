/**
 * T6.5 · 报告双栏视图 + 引用悬浮卡片。
 *
 * 左：正文（markdown 渲染）；右：引用列表。
 * 正文里 `[WEB1-1-1]` 这类引用编号可 hover / focus，弹出证据原文与来源链接。
 *
 * ⚠️ **不用第三方 markdown 库**（`react-markdown` 会带 20+ 依赖，
 * 而本项目的报告格式是自己生成的、可控）。用一个小的行级渲染器即可，
 * 代价是不支持嵌套列表等复杂结构——本项目报告用不到，够用就行。
 * 这个取舍写在这里，避免下一个人以为"漏了"。
 */

import { useMemo, useState } from "react";
import type { ReferenceItem } from "./api";

const CITE_RE = /(\[(?:WEB|LOC)\d+-\d+-\d+\])/g;

interface Props {
  markdown: string;
  references: Record<string, ReferenceItem>;
}

export function ReportView({ markdown, references }: Props) {
  const [active, setActive] = useState<string | null>(null);
  const refEntries = useMemo(() => Object.entries(references), [references]);

  const blocks = useMemo(() => parseBlocks(markdown), [markdown]);

  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-hidden xl:grid-cols-[minmax(0,1fr)_320px]">
      <section className="panel min-h-0 overflow-y-auto p-5">
        <div className="report-body">
          {blocks.map((b, i) => (
            <Block key={i} block={b} references={references} onCite={setActive} />
          ))}
        </div>
      </section>

      <aside className="panel hidden min-h-0 flex-col overflow-hidden xl:flex">
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
    </div>
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
        const known = Boolean(references[id]);
        return (
          <span key={i} className="group relative inline-block">
            <button
              type="button"
              onMouseEnter={() => onCite(id)}
              onMouseLeave={() => onCite(null)}
              onFocus={() => onCite(id)}
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
            {known && <HoverCard id={id} ref_={references[id]} />}
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
