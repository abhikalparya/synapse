import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "VITE_");
  const apiProxyTarget = env.VITE_API_PROXY_TARGET || "http://127.0.0.1:8000";

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/graph": { target: apiProxyTarget, changeOrigin: true },
        "/stats": { target: apiProxyTarget, changeOrigin: true },
        "/lint": { target: apiProxyTarget, changeOrigin: true },
        "/ingest": { target: apiProxyTarget, changeOrigin: true },
        "/ai": { target: apiProxyTarget, changeOrigin: true },
        "/topics": { target: apiProxyTarget, changeOrigin: true },
        "/obsidian": { target: apiProxyTarget, changeOrigin: true },
        "/settings": { target: apiProxyTarget, changeOrigin: true },
        "/zones": { target: apiProxyTarget, changeOrigin: true },
        "/dependencies": { target: apiProxyTarget, changeOrigin: true },
        "/proposals": { target: apiProxyTarget, changeOrigin: true },
        "/apply": { target: apiProxyTarget, changeOrigin: true },
        "/discard": { target: apiProxyTarget, changeOrigin: true },
        "/rollback": { target: apiProxyTarget, changeOrigin: true },
      },
    },
  };
});
