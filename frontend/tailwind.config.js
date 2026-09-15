/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        /**
         * T9.3 · 暗色**四层**表面体系。
         *
         * 层次不是"调一个好看的颜色"，而是解决一个具体观感问题：
         * T9.2 时侧栏 / 主区 / 右栏三者都是同一个 `surface` + 同样的边框，
         * 结果整屏是"三个等重的盒子"——**没有主次，眼睛不知道该落在哪**。
         *
         * 现在按"离用户多近"排：
         *   bg      —— 页面底（最远）
         *   rail    —— 侧栏 / 右栏：**只比 bg 亮一点点**。作用不是"被看见"，
         *              而是在滚动时能感知到区域边界，从而让中间那块浮起来。
         *   surface —— 主内容面板：**全屏唯一的"纸面"**
         *   elevated—— 抬升元素（hover / 选中 / 段控件滑块）
         *   muted   —— 次级填充（标签、代码底、进度槽）
         */
        bg: "#0A101C",
        rail: "#0E1626",
        surface: "#131D2F",
        elevated: "#1B2740",
        muted: "#24314B",
        border: "#22304A",
        "border-strong": "#31415F",
        fg: "#E9EEF8",
        "fg-muted": "#8A99B4",
        //: 三级文字（时间戳、计数、路径这类"扫一眼"的信息）。旧版一律用 fg-muted，
        //: 结果所有次要信息都一个亮度，视觉上没有主次。
        "fg-subtle": "#5E6E8A",
        //: ⚠️ accent 保持**绿色**：本项目里它承载语义——「核验通过 / 证据成立」，
        //: 换成通用的蓝紫会把语义抹掉（`citation_check.pass`、supported 计数都用它）。
        accent: "#22C55E",
        "accent-soft": "#16301F", // 绿色低饱和底（用于成功态背景）
        danger: "#F87171",
        "danger-soft": "#331A1F",
        warn: "#FBBF24",
        "warn-soft": "#33290F",
        info: "#60A5FA",
        "info-soft": "#15243A",
      },
      fontFamily: {
        /**
         * 字体栈（T9.2）：**自托管** IBM Plex Sans 在前（`src/fonts.css`），
         * 后面是系统栈兜底——中文不由 IBM Plex Sans 提供，必须显式列中文字体，
         * 否则中文会掉到浏览器默认（Windows 上是宋体，观感立刻垮掉）。
         *
         * 这套配对（IBM Plex Sans + JetBrains Mono）就是设计库里
         * "Developer Tool" 品类的最佳配对（T9.3 复核确认），不换。
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
        //: 收口字号阶梯（旧版大量裸写 text-[11px]/text-[12.5px]/text-[13.5px]）。
        //: 大字号带**负字距**：IBM Plex Sans 在小尺寸是紧凑的，放大后默认间距会显松，
        //: 标题层级因此"散"。这是排版惯例，不是随手加的。
        "2xs": ["11px", { lineHeight: "1.5" }],
        xs: ["12px", { lineHeight: "1.55" }],
        sm: ["13px", { lineHeight: "1.6" }],
        base: ["14px", { lineHeight: "1.65" }],
        lg: ["16px", { lineHeight: "1.5", letterSpacing: "-0.011em" }],
        xl: ["18px", { lineHeight: "1.45", letterSpacing: "-0.014em" }],
        "2xl": ["22px", { lineHeight: "1.35", letterSpacing: "-0.019em" }],
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
         *
         * T9.3 精调：高光 alpha 从 .03 提到 .045 并**只在顶边**保留，
         * 底部去掉亮线（真实光从上方来，底边发亮会显假）。
         */
        card: "0 1px 2px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.045)",
        raised: "0 10px 30px -8px rgba(0,0,0,.6), inset 0 1px 0 rgba(255,255,255,.06)",
        /** 段控件里"被选中"的滑块：抬起来 + 一圈弱光 */
        segmented: "0 1px 3px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.07)",
        glow: "0 0 0 1px rgba(34,197,94,.35), 0 0 18px -4px rgba(34,197,94,.35)",
      },
      transitionDuration: {
        //: 统一过渡时长（旧版 150ms 散落各处，缺少"快/常规"两档区分）
        fast: "120ms",
        DEFAULT: "180ms",
        slow: "260ms",
      },
      transitionTimingFunction: {
        /**
         * T9.3：把默认 `ease`（对称的缓入缓出）换成**减速曲线**。
         * 界面动效的物理直觉是"起步快、到位慢"（像物体被推一下再停下），
         * 对称曲线会显得"匀速平移"，这是廉价感的主要来源之一。
         *   - out（默认）：进场 / hover / 位置变化
         *   - in-out：需要"经过中点"的对称变化（进度条）
         */
        DEFAULT: "cubic-bezier(0.22, 1, 0.36, 1)",
        "in-out": "cubic-bezier(0.65, 0, 0.35, 1)",
      },
      keyframes: {
        "fade-up": {
          from: { opacity: "0", transform: "translateY(6px)" },
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
          "50%": { opacity: "0.5", transform: "scale(0.8)" },
        },
        //: 横向流光（用于"进行中"的进度条，替代生硬的闪烁）
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "scale-in": {
          from: { opacity: "0", transform: "scale(.98)" },
          to: { opacity: "1", transform: "scale(1)" },
        },
        //: 模态进场：从下方略微升起 + 放大（比纯淡入有"被召唤"的方向感）
        "rise-in": {
          from: { opacity: "0", transform: "translateY(10px) scale(.99)" },
          to: { opacity: "1", transform: "translateY(0) scale(1)" },
        },
      },
      animation: {
        "fade-up": "fade-up 260ms cubic-bezier(0.22,1,0.36,1) both",
        "fade-in": "fade-in 200ms cubic-bezier(0.22,1,0.36,1) both",
        "pulse-soft": "pulse-soft 1.4s ease-in-out infinite",
        breathe: "breathe 1.6s ease-in-out infinite",
        shimmer: "shimmer 1.8s linear infinite",
        "scale-in": "scale-in 180ms cubic-bezier(0.22,1,0.36,1) both",
        "rise-in": "rise-in 240ms cubic-bezier(0.22,1,0.36,1) both",
      },
    },
  },
  plugins: [],
};
