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
  /**
   * 节点实例序号（**全局递增**，由后端在任务开始时分配）。
   *
   * ⚠️ 配对 `agent_start`/`agent_end` 必须用它，不能用节点名：`scout_web` 会扇出多路
   * 并行（同名同时开始、**乱序结束**），按名配对会把它们的耗时混成一条。
   */
  index: number;
  /** LangGraph `debug` 流的任务 id。同一节点的多路并行靠它区分（后端按此配对）。 */
  task_id: string;
  ts: number;
  /** 开始事件里恒为空数组——节点返回值只有结束时才知道，见 `AgentEndEvent.outputs`。 */
  outputs: string[];
}

export interface AgentEndEvent extends BaseEvent {
  event: "agent_end";
  node: string;
  index: number;
  task_id: string;
  ts: number;
  /** 真实耗时（ms）。后端从 trace 镜像；取不到为 null，UI 显示 "—" 而不是 0。 */
  duration_ms: number | null;
  brief: string;
  /**
   * 该节点真实返回的产出字段名（取结束事件里的，**不是**开始事件里的空数组）。
   * 异常/中断补发的结束事件不带此字段。
   */
  outputs?: string[];
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
  /** T7.10：见 `EvidenceSufficiency`。老后端不返回时为 undefined，前端按"充足"处理。 */
  evidence_sufficiency?: EvidenceSufficiency;
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

/**
 * T7.10 · 证据充足性评估（`analyst` 节点产出）。
 *
 * `sufficient === false` 表示本轮**拒编**：检索拿不到能支撑结论的相关证据，
 * `report` 是一页如实说明（"查了什么 / 为什么不出结论 / 怎么办"），**不是调研结论**。
 * 前端据此显示提示条——否则用户看到一份"结论"却没有任何引用，只会以为系统坏了。
 *
 * 契约来源：`src/attest/agents/analyst.py` 的 `assess_evidence()`。
 */
export interface EvidenceSufficiency {
  sufficient: boolean;
  objective?: string;
  /** 检索到的原始条目数（含跨主题兜底填充） */
  n_evidence?: number;
  /** 通过可用性判据（判别相关度 ≥ 3）的条目数 */
  n_selected?: number;
  /** 其中**真实命中**（检索计分 > 0）的条目数 */
  n_grounded?: number;
  n_judged?: number;
  /** 本次用的判据描述，例如"证据判别相关度 ≥ 3" */
  basis?: string;
  sub_questions?: string[];
  covered_sub_questions?: string[];
  uncovered_sub_questions?: string[];
  /** 人话原因（拒编时非空），直接可展示 */
  reasons?: string[];
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
  /** T7.10：证据充足性。`sufficient === false` ⇒ 本报告是拒编页，不是调研结论。 */
  evidence_sufficiency?: EvidenceSufficiency;
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

/**
 * 带 HTTP 状态码的请求错误。
 *
 * 为什么需要状态码：`409 尚未产出结果` 是**运行中的正常状态**，不是故障——
 * 调用方要能区分"还在跑，稍后再取"与"真出错了"，光看文案不可靠（后端改一个字就失效）。
 */
export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

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
    throw new ApiError(detail, res.status);
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
  memory_writer: "结论沉淀",
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
