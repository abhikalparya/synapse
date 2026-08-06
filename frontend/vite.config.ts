import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/graph": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/stats": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/lint": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/ingest": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/generate": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/topics": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/dependencies": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/proposals": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/apply": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/discard": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/rollback": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
