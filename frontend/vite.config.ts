import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    // ⚠️ 必须显式绑 IPv4。Vite 默认 host=localhost 在 Windows 上只监听 [::1]（IPv6），
    //    于是 `http://127.0.0.1:5173` 直接连接被拒——而 scripts/start.bat 提示用户打开的
    //    正是这个地址（后端 uvicorn 也绑的 127.0.0.1）。绑到 127.0.0.1 后，
    //    127.0.0.1 与 localhost 两个地址都能打开，不会再出现"按提示打开却是白屏"。
    //
    // 局域网演示：`npm run dev -- --host 0.0.0.0` 覆盖（scripts\start.bat --lan 就是这么起的）。
    // 命令行参数优先于这里的值，所以**默认仍然是只监听本机**——绑 0.0.0.0 等于同网段人人可访问，
    // 不该是默认行为。
    host: "127.0.0.1",
    port: 5173,
    // 把 /api 代理到后端：前端用**相对路径**请求，开发期不需要 CORS，
    // 也不用在前端写死后端端口（改端口只改这里）。
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        // SSE 必须关掉代理缓冲，否则事件会被攒着一起发（流式就白做了）
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes) => {
            if (String(proxyRes.headers["content-type"] ?? "").includes("text/event-stream")) {
              proxyRes.headers["cache-control"] = "no-cache";
              delete proxyRes.headers["content-encoding"];
            }
          });
        },
      },
    },
  },
});
