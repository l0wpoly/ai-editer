import { defineConfig } from "vite";

export default defineConfig({
  root: ".",
  base: "/",
  build: {
    outDir: "../static",
    emptyOutDir: true,
    assetsDir: "assets",
    sourcemap: false,
    target: "es2022"
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:10000",
        changeOrigin: true
      }
    }
  }
});
