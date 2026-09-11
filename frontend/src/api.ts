/**
 * T6.3 · 后端契约的类型定义与 API 客户端。
 *
 * **契约来源**：`app/main.py`（P6 批次 1+2，已冻结）。改这里之前先改后端，
 * 或先确认后端没变——类型是照着 `GET /docs` 的 OpenAPI 手写的，
 * 不是从代码生成（本项目规模不值得引入代码生成器）。
 */

// ============================================================ 事件协议（T6.2）

/** SSE 事件类型。`message` 是 SSE 帧的 event 名，真实类型在 `data.event` 里。 */
export type EventKind =
  | "run_start"
  | "agent_start"
  | "agent_end"
  | "awaiting_confirm"
  | "report_ready"
  | "error"
  | "ping"
  | "stream_end";

export interface BaseEvent {
  event: EventKind;
  [key: string]: unknown;
}

export interface RunStartEvent extends BaseEvent {
  event: "run_start";
  thread_id: string;
  query: string;
  llm_mode: string;
}

export interface AgentStartEvent extends BaseEvent {
  event: "agent_start";
  node: string;
  index: number;
  ts: number;
  outputs: string[];
}

export interface AgentEndEvent extends BaseEvent {
  event: "agent_end";
  node: string;
  index: number;
  ts: number;
  /** 真实耗时（ms）。后端从 trace 镜像；取不到为 null，UI 显示 "—" 而不是 0。 */
  duration_ms: number | null;
  brief: string;
}

export interface AwaitingConfirmEvent extends BaseEvent {
  event: "awaiting_confirm";
  node: string;
  objective: string;
  outlines: string[];
  sub_questions: string[];
}

export interface ReportReadyEvent extends BaseEvent {
  event: "report_ready";
  route: string | null;
  cost_cny: number;
  tokens: number;
  audit_summary: Record<string, unknown>;
  citation_check: Record<string, unknown>;
  llm_calls: number;
  node_pairs_ok: boolean;
  mock: boolean;
}

export interface ErrorEvent extends BaseEvent {
  event: "error";
  message: string;
}

export interface StreamEndEvent extends BaseEvent {
  event: "stream_end";
  status: SessionStatus;
}

export type AnyEvent =
  | RunStartEvent
  | AgentStartEvent
  | AgentEndEvent
  | AwaitingConfirmEvent
  | ReportReadyEvent
  | ErrorEvent
  | StreamEndEvent;

// ============================================================ 会话

export type SessionStatus =
  | "idle"
  | "running"
  | "awaiting_confirm"
  | "done"
  | "error"
  | "aborted";

export interface SessionMeta {
  thread_id: string;
  query: string;
  status: SessionStatus;
  current_node: string | null;
  pending_confirm: {
    node: string;
    objective: string;
    outlines: string[];
    sub_questions: string[];
  } | null;
  event_count: number;
  result_keys: string[];
  error: string | null;
  created_at: number;
  /** 仅冷启动恢复路径（`GET /api/session/{id}` 回查检查点）会出现 */
  source?: "checkpoint";
  next?: string[];
  has_report?: boolean;
  cost_incurred?: number;
  tokens_incurred?: number;
}

/** 引用编号 → 证据（报告双栏悬浮卡的数据源） */
export interface ReferenceItem {
  title: string;
  url: string | null;
  source: string | null;
  snippet: string;
}

export interface ReportResult {
  query: string;
  route: string | null;
  direct_answer: string;
  report: string;
  reference_list: string;
  citation_check: Record<string, unknown>;
  audit_summary: Record<string, unknown>;
  audit_items: unknown[];
  profile: Record<string, string>;
  plan: { objective?: string; outlines?: string[]; sub_questions?: string[] };
  cost_incurred: number;
  tokens_incurred: number;
  references: Record<string, ReferenceItem>;
}

export interface TimelineRow {
  index: number;
  node: string;
  start: number;
  end?: number;
  duration_ms: number | null;
  brief: string;
  outputs?: string[];
}

// ============================================================ 客户端

/**
 * 后端地址。
 *
 * 开发期走 Vite 代理（`vite.config.ts` 里把 `/api` 转给 8000），
 * 所以默认值用**相对路径空串**——同源请求，不需要 CORS，也不用写死端口。
 * 若前端独立部署，设 `VITE_API_BASE=http://host:8000`。
 */
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    // 后端错误体是 FastAPI 的 {detail: string}，把 detail 提出来给人看
    let detail = `HTTP ${res.status}`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* 非 JSON 错误体：保留状态码 */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () =>
    jsonFetch<{ ok: boolean; llm_mode: string; search_mode: string; human_confirm: boolean }>(
      "/api/health",
    ),

  listSessions: () =>
    jsonFetch<{ ok: boolean; sessions: SessionMeta[] }>("/api/session").then((r) => r.sessions),

  createSession: (threadId: string | null, query: string) =>
    jsonFetch<{ ok: boolean; session: SessionMeta }>("/api/session", {
      method: "POST",
      body: JSON.stringify({ thread_id: threadId, query }),
    }).then((r) => r.session),

  getSession: (threadId: string) =>
    jsonFetch<{ ok: boolean; session: SessionMeta }>(`/api/session/${threadId}`).then(
      (r) => r.session,
    ),

  getReport: (threadId: string) =>
    jsonFetch<{ ok: boolean; report: ReportResult }>(`/api/report/${threadId}`).then(
      (r) => r.report,
    ),

  getTrace: (threadId: string) =>
    jsonFetch<{
      ok: boolean;
      status: SessionStatus;
      count: number;
      events: AnyEvent[];
      timeline: TimelineRow[];
    }>(`/api/trace/${threadId}`),

  confirm: (threadId: string, action: string, plan?: { outlines: string[] }) =>
    jsonFetch<{ ok: boolean; status: SessionStatus; resumed: boolean }>("/api/confirm", {
      method: "POST",
      body: JSON.stringify({ session_id: threadId, action, plan }),
    }),
};

// ============================================================ SSE

/**
 * 订阅某个会话的事件流（`POST /api/chat` 的 SSE 响应）。
 *
 * ⚠️ **为什么不用浏览器原生 `EventSource`**：
 *   `EventSource` 只支持 GET，而 `/api/chat` 是 POST（要带 body），所以必须用
 *   `fetch` + `ReadableStream` 手工解析 SSE。
 *   代价：**失去 `EventSource` 的自动重连**，重连逻辑得自己写——这正是
 *   `lastEventId` 参数的用途（对应后端的 `Last-Event-ID` 头）。
 *
 * @param body      请求体（session_id + query + resume）
 * @param onEvent   每收到一个事件回调一次
 * @param lastEventId 重连游标（已收到的最后事件 id）。后端据此从事件缓冲重放。
 * @param signal    用于中止（组件卸载时传 AbortSignal）
 */
export async function streamChat(
  body: { session_id: string; query: string; resume?: boolean },
  onEvent: (ev: AnyEvent, id: number) => void,
  opts: { lastEventId?: number; signal?: AbortSignal } = {},
): Promise<void> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (opts.lastEventId !== undefined && opts.lastEventId >= 0) {
    headers["Last-Event-ID"] = String(opts.lastEventId);
  }

  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal: opts.signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(`SSE 连接失败：HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let lastId = opts.lastEventId ?? -1;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE 用空行分隔帧；按 "\n\n" 切，最后一段可能不完整，留在 buffer 里
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      if (!frame.trim()) continue;
      let id: number | undefined;
      let data: string | undefined;
      for (const line of frame.split("\n")) {
        if (line.startsWith("id: ")) id = Number(line.slice(4));
        else if (line.startsWith("data: ")) data = line.slice(6);
      }
      if (data === undefined) continue;
      if (id !== undefined) lastId = id;
      try {
        const ev = JSON.parse(data) as AnyEvent;
        onEvent(ev, id ?? lastId);
      } catch {
        // 单帧解析失败不该中断整条流（可能是心跳等非 JSON 帧）
        console.warn("[sse] 无法解析事件帧", data);
      }
    }
  }
}

// ============================================================ 展示辅助

/** 节点名 → 中文标签（与 CLI 的 NODE_LABEL 同源语义） */
export const NODE_LABEL: Record<string, string> = {
  memory_loader: "记忆注入",
  intent_router: "意图分流",
  direct_responder: "快速回答",
  planner: "任务规划",
  human_confirm: "大纲确认",
  scout_web: "并行检索",
  scout_local: "本地检索",
  evidence_judge: "证据判别",
  reflect: "反思补检",
  analyst: "撰写报告",
  auditor: "引用审计",
};

export function nodeLabel(node: string): string {
  return NODE_LABEL[node] ?? node;
}

export const STATUS_META: Record<
  SessionStatus,
  { label: string; color: string; pulse?: boolean }
> = {
  idle: { label: "待启动", color: "bg-fg-muted" },
  running: { label: "运行中", color: "bg-info", pulse: true },
  awaiting_confirm: { label: "待确认", color: "bg-warn", pulse: true },
  done: { label: "已完成", color: "bg-accent" },
  error: { label: "失败", color: "bg-danger" },
  aborted: { label: "已中止", color: "bg-fg-muted" },
};

export function fmtCny(v: number): string {
  return `¥${v.toFixed(4)}`;
}

export function fmtTokens(v: number): string {
  return v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(v);
}

export function fmtDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1000) return `${ms.toFixed(1)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}
