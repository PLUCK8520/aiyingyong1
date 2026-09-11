import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
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
