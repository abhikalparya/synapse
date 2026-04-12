import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/graph": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/query": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/stats": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/ingest": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/ingest/upload": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/ingest/upload/batch": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/generate": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/generate/from-raw": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/refactor": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
