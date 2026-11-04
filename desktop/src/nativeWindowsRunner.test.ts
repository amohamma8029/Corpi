import { EventEmitter } from "node:events";
import { createHash } from "node:crypto";
import path from "node:path";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

const io = vi.hoisted(() => ({
  read: vi.fn(), write: vi.fn(), mkdir: vi.fn(), remove: vi.fn(), spawn: vi.fn(), exec: vi.fn(),
  construct: vi.fn(), prepare: vi.fn(), start: vi.fn(), stop: vi.fn(), delay: vi.fn(), toPath: vi.fn(),
  events: [] as string[], records: [] as Record<string, unknown>[],
}));
vi.mock("node:fs", () => ({
  default: { readFileSync: io.read, writeFileSync: io.write, mkdirSync: io.mkdir, rmSync: io.remove },
  readFileSync: io.read, writeFileSync: io.write, mkdirSync: io.mkdir, rmSync: io.remove,
}));
vi.mock("node:child_process", () => ({
  default: { spawn: io.spawn, execFileSync: io.exec }, spawn: io.spawn, execFileSync: io.exec,
}));
vi.mock("node:timers/promises", () => ({ default: { setTimeout: io.delay }, setTimeout: io.delay }));
// Vite rewrites asset URLs in jsdom. Path discovery is unchanged production code
// already exercised by hosted runs; this unit supplies its own inert project root.
vi.mock("node:url", () => ({ default: { fileURLToPath: io.toPath }, fileURLToPath: io.toPath }));
vi.mock("../scripts/windows-webdriver.mjs", () => ({ WindowsWebDriver: class {
  constructor(options: unknown) { io.construct(options); }
  prepare(...args: unknown[]) { return io.prepare(...args); }
  start() { return io.start(); }
  stop() { return io.stop(); }
} }));

const argv = process.argv;
const platform = Object.getOwnPropertyDescriptor(process, "platform")!;
// Deliberately not a native executable. This unit tests runner wiring only.
const bytes = Buffer.from("unit-only native binary stand-in");
const binding = { debugPort: 45000, identifier: "app.corpi.desktop.e2e.t12345678-1234-1234-1234-123456789abc" };
const configuration = { identifier: binding.identifier, app: { windows: [{ label: "main", additionalBrowserArgs: "unit-only API arguments" }] } };
let previous: Record<string, unknown>;
let lock: string;

beforeEach(() => {
  vi.resetAllMocks();
  vi.resetModules();
  io.events.length = 0;
  io.records.length = 0;
  Object.defineProperty(process, "platform", { ...platform, value: "win32" });
  process.argv = [process.execPath, "unit-native-runner.mjs"];
  vi.stubEnv("npm_execpath", path.resolve("cached-pnpm.cjs"));
  vi.stubEnv("CORPI_EDGE_DRIVER", path.resolve("verified-test-driver.exe"));
  vi.stubEnv("CORPI_RUN_MODEL_DOWNLOAD", "0");
  vi.stubEnv("GITHUB_SHA", "unit-current-source");
  vi.spyOn(console, "log").mockImplementation(() => undefined);
  io.toPath.mockReturnValue(path.resolve("unit-native-desktop"));
  lock = '[[package]]\nname = "wry"\nversion = "0.55.1"\n';
  previous = { provider: "external", sourceCommit: "unit-original-source", windowsBuild: binding,
    testBinarySha256: createHash("sha256").update(bytes).digest("hex") };
  io.read.mockImplementation((filename: string) => {
    switch (path.basename(filename)) {
      case "package.json": return JSON.stringify({ dependencies: { "pdfjs-dist": "unit-pdfjs" },
        devDependencies: { "@wdio/tauri-service": "unit-service" } });
      case "Cargo.lock": return lock;
      case "tauri.conf.json": return JSON.stringify({ app: { windows: [{ label: "main", width: 1024, height: 768 }] } });
      case "tauri.e2e.conf.json": return JSON.stringify({ identifier: "app.corpi.desktop.e2e" });
      case "run.json": return JSON.stringify(previous);
      case "corpi-desktop.exe": return bytes;
      default: throw new Error(`Unexpected unit input: ${filename}`);
    }
  });
  io.write.mockImplementation((_filename: string, content: string) => { io.records.push(JSON.parse(content)); });
  io.exec.mockReturnValue(""); // No core PID before or after; never searches/kills a real process.
  io.prepare.mockImplementation(async () => {
    io.events.push("prepare"); return { binding, configuration };
  });
  io.start.mockImplementation(async () => {
    io.events.push("start");
    return { CORPI_E2E_DEBUGGER_ADDRESS: "127.0.0.1:45000", CORPI_E2E_DRIVER_PORT: "45001" };
  });
  io.stop.mockImplementation(async () => { io.events.push("stop"); });
  io.spawn.mockImplementation((_command: string, args: string[]) => {
    if (args.includes("build")) io.events.push("build");
    if (args.includes("wdio")) io.events.push("viewer");
    const child = Object.assign(new EventEmitter(), {
      pid: 101, exitCode: null as number | null, signalCode: null,
    });
    queueMicrotask(() => { child.exitCode = 0; child.emit("exit", 0, null); });
    return child;
  });
});
afterEach(() => {
  process.argv = argv;
  Object.defineProperty(process, "platform", platform);
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

it("builds with the prepared JSON before launch and passes only its returned endpoints to WDIO", async () => {
  await import("../scripts/run-native-e2e.mjs");
  expect(io.events).toEqual(["prepare", "build", "start", "viewer", "stop"]);
  const build = io.spawn.mock.calls.find((call) => call[1].includes("build"))!;
  const configIndex = build[1].indexOf("--config");
  expect(JSON.parse(build[1][configIndex + 1])).toEqual(configuration);
  expect(io.prepare).toHaveBeenCalledWith({
    baseConfig: { app: { windows: [{ label: "main", width: 1024, height: 768 }] } },
    e2eConfig: { identifier: "app.corpi.desktop.e2e" }, wryVersion: "0.55.1",
  }, undefined);
  const viewer = io.spawn.mock.calls.find((call) => call[1].includes("wdio"))!;
  expect(viewer[2].env).toMatchObject({ CORPI_E2E_DEBUGGER_ADDRESS: "127.0.0.1:45000", CORPI_E2E_DRIVER_PORT: "45001" });
  expect(io.records.at(-1)).toMatchObject({ windowsBuild: binding, status: "passed", reusedBuild: false,
    sourceCommit: "unit-current-source", orphanCorePids: [] });
});

it("reuses the original binding and source identity without rebuilding", async () => {
  process.argv.push("--reuse-build");
  await import("../scripts/run-native-e2e.mjs");
  expect(io.events).toEqual(["prepare", "start", "viewer", "stop"]);
  expect(io.prepare.mock.calls[0][1]).toEqual(binding);
  expect(io.records.at(-1)).toMatchObject({ windowsBuild: binding, reusedBuild: true,
    sourceCommit: "unit-original-source", status: "passed" });
});

it("rejects an older retained binary without API settings instead of inventing a new binding", async () => {
  process.argv.push("--reuse-build");
  delete previous.windowsBuild;
  await expect(import("../scripts/run-native-e2e.mjs")).rejects.toThrow("rebuild without --reuse-build");
  expect(io.prepare).not.toHaveBeenCalled();
  expect(io.start).not.toHaveBeenCalled();
  expect(io.records.at(-1)).toMatchObject({ status: "failed" });
});

it("rejects an ambiguous locked Wry version before preparing or launching", async () => {
  lock += lock;
  await expect(import("../scripts/run-native-e2e.mjs")).rejects.toThrow("exactly one locked Wry");
  expect(io.prepare).not.toHaveBeenCalled();
  expect(io.start).not.toHaveBeenCalled();
});

it("keeps startup failure primary while preserving an additional cleanup failure", async () => {
  io.start.mockRejectedValue(new Error("unit debugging endpoint not ready"));
  io.stop.mockRejectedValue(new AggregateError([new Error("unit close refused")], "unit cleanup failed"));
  await expect(import("../scripts/run-native-e2e.mjs")).rejects.toThrow("unit debugging endpoint not ready");
  expect(io.stop).toHaveBeenCalledOnce();
  expect(io.records.at(-1)).toMatchObject({ status: "failed", error: "unit debugging endpoint not ready",
    cleanupErrors: ["unit close refused"], orphanCorePids: [] });
});

it("does not report a successful viewer run when cleanup fails", async () => {
  io.stop.mockRejectedValue(new AggregateError([new Error("unit close refused")], "unit cleanup failed"));
  await expect(import("../scripts/run-native-e2e.mjs")).rejects.toThrow("unit cleanup failed");
  expect(io.records.at(-1)).toMatchObject({ status: "failed", cleanupErrors: ["unit close refused"] });
});
