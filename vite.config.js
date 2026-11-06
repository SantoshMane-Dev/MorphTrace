import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const root = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [svelte()],
  base: "/static/dist/",
  build: {
    outDir: resolve(root, "static/dist"),
    emptyOutDir: true,
    rollupOptions: {
      input: resolve(root, "frontend/main.js"),
      output: {
        entryFileNames: "assets/morphtrace.js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: "assets/[name][extname]",
      },
    },
  },
});
