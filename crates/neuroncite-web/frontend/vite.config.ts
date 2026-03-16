import { defineConfig } from "vite";
import solid from "vite-plugin-solid";
import { resolve } from "path";
import { visualizer } from "rollup-plugin-visualizer";

export default defineConfig({
  plugins: [
    solid(),
    // Bundle size visualization plugin. Generates an interactive treemap
    // HTML file (stats.html) in the project root when the ANALYZE environment
    // variable is set to "true". Run with: ANALYZE=true npm run build
    // The generated file is gitignored and only used during development
    // to identify large dependencies and code-splitting opportunities.
    ...(process.env.ANALYZE === "true"
      ? [
          visualizer({
            filename: "stats.html",
            open: true,
            gzipSize: true,
            brotliSize: true,
          }),
        ]
      : []),
  ],
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api/v1": {
        target: "http://127.0.0.1:3030",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // Target ES2022 for broader browser compatibility (Safari 15.4+, Chrome 94+,
    // Firefox 93+, Edge 94+). ESNext may emit syntax that WebView2 or older
    // browsers do not support, while ES2022 provides top-level await, class
    // fields, and all features used by the codebase.
    target: "es2022",
  },
  test: {
    environment: "jsdom",
    globals: true,
  },
});
