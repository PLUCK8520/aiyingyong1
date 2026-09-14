/**
 * T8.3 · 秒级时间源与耗时格式化（2026-09-13）。
 *
 * **为什么需要它**：一次调研里 `evidence_judge` / `analyst` 这类节点实测单次跑
 * 60~190 秒。此前界面上只显示一个静止的「进行中…」——用户等待的几分钟里屏幕毫无变化，
 * 观感与**卡死**无异（这是真实反馈过的体验问题，不是假设）。
 * 有秒级跳动的计时，用户就能判断"它在动"而不是"它挂了"。
 *
 * 计时**不用后端推送**：那样每个运行中的节点都要产生事件，把事件缓冲塞满、
 * 还要处理断线漏推。前端自己按 `agent_start.ts` 算差值，零额外流量。
 */

import { useEffect, useState } from "react";

/**
 * 每秒钟返回一次当前时间（秒级，与后端 `time.time()` 同基准）。
 *
 * ⚠️ `active` 为 false 时**不挂 interval**：不做这个判断的话，
 * 面板在空闲状态也会每秒重渲染一次，白白耗电（笔记本上肉眼可见）。
 */
export function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now() / 1000);

  useEffect(() => {
    if (!active) return;
    setNow(Date.now() / 1000); // 立刻对齐一次，避免挂上 interval 后 1 秒内显示旧值
    const t = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(t);
  }, [active]);

  return now;
}

/** 把秒数格式化成紧凑可读形态：< 60s 用秒，≥ 60s 用「分+秒」。 */
export function fmtElapsed(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
}
