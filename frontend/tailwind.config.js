/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        /**
         * T9.2 · 暗色分层（Dark Mode / OLED 友好）。
         *
         * 分层是这次改造的核心：旧版只有 `bg` / `surface` 两级 + `muted`，
         * 面板套面板时层级立刻糊掉（同色叠同色分不出边界）。
         * 现在按"离用户多近"排：bg（最远）→ surface（面板）→ elevated（抬升：hover/选中）
         * → muted（次级填充：标签/代码底）。
         */
        bg: "#0B1220",
        surface: "#131C2E",
        elevated: "#1A2438",
        muted: "#222E45",
        border: "#243149",
        "border-strong": "#33425F",
        fg: "#E9EEF8",
        "fg-muted": "#8A99B4",
        //: 三级文字（时间戳、计数、路径这类"扫一眼"的信息）。旧版一律用 fg-muted，
        //: 结果所有次要信息都一个亮度，视觉上没有主次。
        "fg-subtle": "#5E6E8A",
        //: ⚠️ accent 保持**绿色**：本项目里它承载语义——「核验通过 / 证据成立」，
        //: 换成通用的蓝紫会把语义抹掉（`citation_check.pass`、supported 计数都用它）。
        accent: "#22C55E",
        "accent-soft": "#1B3A2A", // 绿色低饱和底（用于成功态背景）
        danger: "#F87171",
        "danger-soft": "#3A1D22",
        warn: "#FBBF24",
        "warn-soft": "#3A2E14",
        info: "#60A5FA",
        "info-soft": "#16283F",
      },
      fontFamily: {
        /**
         * 字体栈（T9.2）：**自托管** IBM Plex Sans 在前（`src/fonts.css`），
         * 后面是系统栈兜底——中文不由 IBM Plex Sans 提供，必须显式列中文字体，
         * 否则中文会掉到浏览器默认（Windows 上是宋体，观感立刻垮掉）。
         */
        sans: [
          '"IBM Plex Sans"',
          "system-ui",
          "-apple-system",
          '"Segoe UI"',
          '"PingFang SC"',
          '"Hiragino Sans GB"',
          '"Microsoft YaHei"',
          '"Noto Sans SC"',
          "sans-serif",
        ],
        mono: [
          '"JetBrains Mono"',
          "ui-monospace",
          "SFMono-Regular",
          "Consolas",
          '"Cascadia Mono"',
          "monospace",
        ],
      },
      fontSize: {
        //: 收口字号阶梯（旧版大量裸写 text-[11px]/text-[12.5px]/text-[13.5px]）——
        //: 报价目表而不是魔法数字，改一次全局生效。
        "2xs": ["11px", { lineHeight: "1.5" }],
        xs: ["12px", { lineHeight: "1.55" }],
        sm: ["13px", { lineHeight: "1.6" }],
        base: ["14px", { lineHeight: "1.65" }],
        lg: ["16px", { lineHeight: "1.5" }],
        xl: ["18px", { lineHeight: "1.45" }],
        "2xl": ["22px", { lineHeight: "1.35" }],
      },
      borderRadius: {
        //: 统一圆角阶梯：控件 8px、卡片 12px、容器 16px、胶囊 999px
        control: "8px",
        card: "12px",
        panel: "16px",
        pill: "999px",
      },
      boxShadow: {
        /**
         * ⚠️ 暗色主题下**传统阴影几乎不可见**（深色叠深色没有明暗差）。
         * 所以这里用两种手段替代：① 极暗的投影拉一层"离地感"；
         * ② 高光内描边（inset 1px 亮线）勾出上边缘——这是暗色 UI 里让卡片"立起来"的关键，
         * 单靠 border 会显得平。
         */
        card: "0 1px 2px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.03)",
        raised: "0 8px 24px -6px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.05)",
        glow: "0 0 0 1px rgba(34,197,94,.35), 0 0 18px -4px rgba(34,197,94,.35)",
      },
      transitionDuration: {
        //: 统一过渡时长（旧版 150ms 散落各处，缺少"快/常规"两档区分）
        fast: "120ms",
        DEFAULT: "180ms",
      },
      keyframes: {
        "fade-up": {
          from: { opacity: "0", transform: "translateY(4px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "fade-in": {
          from: { opacity: "0" },
          to: { opacity: "1" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.45" },
        },
        //: 运行中的"呼吸"指示（比 pulse 更含蓄；用于时间线当前节点）
        breathe: {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.55", transform: "scale(0.85)" },
        },
        //: 横向流光（用于"进行中"的进度条，替代生硬的闪烁）
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "scale-in": {
          from: { opacity: "0", transform: "scale(.97)" },
          to: { opacity: "1", transform: "scale(1)" },
        },
      },
      animation: {
        "fade-up": "fade-up 180ms ease-out",
        "fade-in": "fade-in 180ms ease-out",
        "pulse-soft": "pulse-soft 1.4s ease-in-out infinite",
        breathe: "breathe 1.6s ease-in-out infinite",
        shimmer: "shimmer 1.8s linear infinite",
        "scale-in": "scale-in 160ms ease-out",
      },
    },
  },
  plugins: [],
};
