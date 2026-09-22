import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: {
      // Tauri watches the Rust sources; Vite watching the build output crashes
      // with EBUSY on Windows while cargo links (recommended by Tauri).
      ignored: ["**/src-tauri/**"],
    },
  },
  build: {
    target: "es2022",
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});
