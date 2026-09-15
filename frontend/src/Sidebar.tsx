/**
 * T6.3 · 会话侧栏：会话列表 / 新建 / 切换 thread。
 *
 * 交互态（《功能设计》§8）：
 *   空态给引导文案；会话项显示状态点（运行中/已完成/已中断）。
 *
 * T9.3 · 视觉定位：侧栏是**背景层**（`.rail` 而非 `.panel`）。
 * 它与主内容区用同一种卡片样式时，整屏会读成"三个等重的盒子"，
 * 眼睛失去落点；退成 rail 之后，中间的"纸面"才浮得起来。
 *
 * T9.4 · 响应式：`lg` 以上常驻在流内；`lg` 以下变成**浮起的抽屉**（配遮罩）。
 * 抽屉而不是"直接隐藏"的理由：会话列表是导航，隐藏了用户就换不了会话、
 * 也开不了新会话——那是功能丢失，不是布局取舍。
 */

import { STATUS_META, type SessionMeta } from "./api";

interface Props {
  sessions: SessionMeta[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  loading: boolean;
  /** `lg` 以下时的抽屉开关；`lg` 以上侧栏常驻，此值不生效。 */
  open?: boolean;
  /** 抽屉关闭回调：点会话项 / 点遮罩 / 点 × 都走它。 */
  onClose?: () => void;
}

export function Sidebar({
  sessions,
  activeId,
  onSelect,
  onNew,
  loading,
  open = false,
  onClose,
}: Props) {
  return (
    <>
      {/* 抽屉遮罩：只在 lg 以下存在。
          用 opacity 而不是条件渲染，是为了让开合**有过渡**（条件渲染是瞬变，没有动画）。 */}
      <div
        className={`fixed inset-0 z-30 bg-[#04070D]/60 backdrop-blur-[2px]
                    transition-opacity duration-200 ease-out lg:hidden ${
                      open ? "opacity-100" : "pointer-events-none opacity-0"
                    }`}
        onClick={onClose}
        aria-hidden="true"
      />

      <aside
        aria-label="会话列表"
        /* 响应式（T9.4）：lg 以上常驻在流内；lg 以下改为**浮起的抽屉**。
           注意 `max-lg:h-auto`：fixed + inset-y 会自动拉伸高度，
           留着 `h-full` 反而跟 inset 打架。 */
        className={`rail no-print flex h-full w-[264px] shrink-0 flex-col overflow-hidden
                    max-lg:fixed max-lg:inset-y-3.5 max-lg:left-3.5 max-lg:z-40 max-lg:h-auto
                    max-lg:shadow-raised max-lg:transition-transform max-lg:duration-300
                    max-lg:ease-out ${open ? "max-lg:translate-x-0" : "max-lg:-translate-x-[110%]"}`}
      >
        {/* 品牌头：accent 图标 + 两行文字。图标做"内发光 + 上高光"，
            比纯色块更像一枚有厚度的标记。 */}
        <div className="flex items-center justify-between gap-2 px-3.5 py-3.5">
          <div className="flex min-w-0 items-center gap-2.5">
            <span
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[10px]
                         border border-accent/25 bg-gradient-to-b from-accent/20 to-accent/[0.04]
                         text-accent shadow-[inset_0_1px_0_rgba(255,255,255,.09)]"
              aria-hidden="true"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
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
              {/* 用 h2 而不是 h1：页面主标题是主列里的"当前会话"（App.tsx）。
                  一页只能有一个 h1，两个平级标题会让屏幕阅读器的文档大纲
                  读起来是"两个并列的东西"。 */}
              <h2 className="truncate text-sm font-semibold tracking-tight text-fg">
                Attest 质证
              </h2>
              <p className="truncate text-2xs text-fg-subtle">逐句质证的调研工作台</p>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-0.5">
            <button
              type="button"
              onClick={onNew}
              className="btn-icon text-fg-muted hover:text-fg"
              aria-label="新建会话"
              title="新建会话"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                aria-hidden="true"
              >
                <path d="M12 5v14M5 12h14" />
              </svg>
            </button>
            {/* 抽屉模式的关闭键：只在 lg 以下有意义 */}
            {onClose && (
              <button
                type="button"
                onClick={onClose}
                className="btn-icon text-fg-muted hover:text-fg lg:hidden"
                aria-label="关闭会话列表"
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  aria-hidden="true"
                >
                  <path d="M18 6 6 18M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        </div>

        {/* 分组标题：给列表一个"从哪开始"的起点，否则会话项直接贴着品牌区 */}
        <p className="px-4 pb-1.5 pt-0.5 text-2xs font-medium uppercase tracking-wider text-fg-subtle/80">
          会话
        </p>

        <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
          {loading && sessions.length === 0 ? (
            <ListSkeleton />
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
              {sessions.map((s, i) => {
                const meta = STATUS_META[s.status];
                const active = s.thread_id === activeId;
                return (
                  <li key={s.thread_id}>
                    {/* 选中态用"左侧渐变竖条 + 抬升底色"两个信号，而不是只换背景色——
                        只换底色在暗色里几乎看不出来。竖条用渐变（上实下虚）与
                        报告正文 h2 的左侧标记同一套语言，全站"当前项"看着像一路的。

                        逐项 stagger：列表出现时有从下往上的节奏，
                        比整块一起闪现更像"内容被加载进来"。上限 8 项，
                        否则长列表末尾要等一秒多才出现。 */}
                    <button
                      type="button"
                      onClick={() => {
                        onSelect(s.thread_id);
                        // 抽屉模式下选完会话就该收起来，否则用户还得再点一次遮罩
                        onClose?.();
                      }}
                      aria-current={active ? "true" : undefined}
                      style={{ animationDelay: `${Math.min(i, 8) * 22}ms` }}
                      className={`group relative w-full animate-fade-up cursor-pointer rounded-control
                                  py-2 pl-3.5 pr-2.5 text-left transition-colors duration-fast ${
                                    active ? "bg-elevated" : "hover:bg-elevated/50"
                                  }`}
                    >
                      {active && (
                        <span
                          className="absolute bottom-2 left-0 top-2 w-[2px] rounded-pill
                                     bg-gradient-to-b from-accent to-accent/20"
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
                            <span aria-hidden="true" className="text-fg-subtle/50">
                              ·
                            </span>
                            <span className="truncate font-mono opacity-75">
                              {s.thread_id.slice(0, 10)}
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

        {/* 底部图例：点 + 文字横排。用分隔线跟列表隔开，避免图例被读成会话项 */}
        <div className="border-t border-border/60 px-3.5 py-2.5">
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
    </>
  );
}

/**
 * 加载骨架屏。
 *
 * 比"加载中…"这行字更好的地方不在于好看：**它预先占住了最终布局的位置**，
 * 列表回来时不会整块往下跳（CLS）。骨架本身带 shimmer，也明确传达"在拿数据"。
 */
function ListSkeleton() {
  return (
    <ul className="space-y-1.5 px-1 py-1" aria-hidden="true">
      {[0.72, 0.55, 0.64, 0.48].map((w, i) => (
        <li key={i} className="flex items-start gap-2 px-2.5 py-2">
          <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-muted" />
          <div className="min-w-0 flex-1 space-y-1.5">
            <span className="skeleton block h-2.5 rounded-pill" style={{ width: `${w * 100}%` }} />
            <span className="skeleton block h-2 w-16 rounded-pill" />
          </div>
        </li>
      ))}
    </ul>
  );
}
