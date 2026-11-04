import { EventEmitter } from "node:events";
import path from "node:path";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

const io = vi.hoisted(() => ({
  spawn: vi.fn(), exec: vi.fn(), mkdir: vi.fn(), temporary: vi.fn(),
  server: vi.fn(), delay: vi.fn(),
}));
vi.mock("node:child_process", () => ({
  default: { spawn: io.spawn, execFileSync: io.exec }, spawn: io.spawn, execFileSync: io.exec,
}));
vi.mock("node:fs", () => ({
  default: { mkdirSync: io.mkdir, mkdtempSync: io.temporary }, mkdirSync: io.mkdir, mkdtempSync: io.temporary,
}));
vi.mock("node:net", () => ({ default: { createServer: io.server }, createServer: io.server }));
vi.mock("node:timers/promises", () => ({ default: { setTimeout: io.delay }, setTimeout: io.delay }));

import { WindowsWebDriver, windowsAttachment } from "../scripts/windows-webdriver.mjs";

class Child extends EventEmitter {
  exitCode: number | null = null;
  signalCode: string | null = null;
  constructor(public pid: number) { super(); }
  finish(code = 0) { this.exitCode = code; this.emit("exit", code, null); }
}

const root = path.resolve("owned-native-session");
const options = {
  application: path.join(root, "corpi-desktop.exe"), driver: path.join(root, "msedgedriver.exe"),
  cwd: root, scratch: root, env: { CORPI_PDF_RENDERED_PAGE_PIXELS: "20000000" },
};
let app: Child;
let driver: Child;
let clock: number;
let nextPort: number;
const fetchOwned = vi.fn();

beforeEach(() => {
  vi.resetAllMocks();
  clock = 0;
  nextPort = 45000;
  vi.spyOn(Date, "now").mockImplementation(() => clock);
  vi.stubGlobal("fetch", fetchOwned);
  app = new Child(101);
  driver = new Child(102);
  io.spawn.mockImplementation((command: string) => command === options.application ? app : driver);
  io.delay.mockImplementation(async (milliseconds: number) => { clock += milliseconds; });
  io.temporary.mockReturnValue(path.join(root, "unique-profile-root"));
  io.server.mockImplementation(() => {
    const port = nextPort++;
    const server = {
      once: vi.fn().mockReturnThis(),
      listen: vi.fn((_port: number, host: string, ready: () => void) => {
        expect(host).toBe("127.0.0.1"); ready();
      }),
      address: () => ({ port }),
      close: (closed: () => void) => { closed(); },
    };
    return server;
  });
  fetchOwned.mockImplementation(async (url: string) => ({
    ok: true,
    json: async () => url.endsWith("/json/list")
      ? [{ type: "page", url: "http://tauri.localhost/" }] : { value: { ready: true } },
  }));
  io.exec.mockImplementation((command: string, args: string[]) => {
    if (command === "powershell.exe") app.finish();
    else if (args[1] === String(app.pid)) app.finish();
    else if (args[1] === String(driver.pid)) driver.finish();
    else throw new Error("An unrelated PID was targeted");
  });
});
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers(); });

it("starts one app before its separate driver and exposes only attachment endpoints", async () => {
  const session = new WindowsWebDriver(options);
  const environment = await session.start();
  expect(environment).toEqual({
    CORPI_E2E_DEBUGGER_ADDRESS: "127.0.0.1:45000", CORPI_E2E_DRIVER_PORT: "45001",
  });
  expect(windowsAttachment(environment)).toEqual({ debuggerAddress: "127.0.0.1:45000", port: 45001 });
  expect(io.spawn.mock.calls.map((call) => call[0])).toEqual([options.application, options.driver]);
  expect(io.spawn.mock.calls[0][1]).toEqual([]);
  expect(io.spawn.mock.calls[0][2].env).toMatchObject({
    ...options.env, TAURI_WEBVIEW_AUTOMATION: "true",
    WEBVIEW2_USER_DATA_FOLDER: path.join(root, "unique-profile-root", "profile"),
    WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: "--remote-debugging-port=45000 --remote-debugging-address=127.0.0.1",
  });
  expect(io.spawn.mock.calls[1][1]).toEqual(["--port=45001", "--host=127.0.0.1"]);
  expect(io.spawn.mock.calls[1][2].env).toEqual(options.env);
  expect(fetchOwned.mock.calls.map((call) => call[0])).toEqual([
    "http://127.0.0.1:45000/json/list", "http://127.0.0.1:45001/status",
  ]);
  expect(fetchOwned.mock.calls.every((call) => call[1].redirect === "error")).toBe(true);
  await session.stop();
  expect(io.exec.mock.calls.map((call) => [call[0], call[1]])).toEqual([
    ["powershell.exe", ["-NoProfile", "-Command",
      "if (-not (Get-Process -Id 101 -ErrorAction Stop).CloseMainWindow()) { throw 'Owned test window did not accept close' }"]],
    ["taskkill.exe", ["/PID", "102", "/T", "/F"]],
  ]);
});

it.each(["", "localhost:45000", "0.0.0.0:45000", "192.0.2.1:45000", "127.0.0.1:0", "127.0.0.1:65536"])(
  "rejects an unowned or invalid debugger endpoint: %s", (address) => {
    expect(() => windowsAttachment({ CORPI_E2E_DEBUGGER_ADDRESS: address, CORPI_E2E_DRIVER_PORT: "45001" })).toThrow();
  },
);

it.each(["", "0", "-1", "65536", "45001/remote"])("rejects an invalid native-driver port: %s", (port) => {
  expect(() => windowsAttachment({ CORPI_E2E_DEBUGGER_ADDRESS: "127.0.0.1:45000", CORPI_E2E_DRIVER_PORT: port })).toThrow();
});

it("does not treat about:blank or an unrelated page as a ready Corpi WebView", async () => {
  fetchOwned.mockResolvedValue({ ok: true, json: async () => [{ type: "page", url: "about:blank" }] });
  const session = new WindowsWebDriver(options);
  await expect(session.start()).rejects.toThrow("within 30 seconds");
  expect(clock).toBe(30000);
  expect(io.spawn).toHaveBeenCalledTimes(1);
  await session.stop();
  expect(io.exec.mock.calls.some((call) => call[0] === "taskkill.exe")).toBe(false);
});

it("reaps the app and driver when the driver never becomes ready", async () => {
  fetchOwned.mockImplementation(async (url: string) => ({ ok: true, json: async () =>
    url.endsWith("/json/list") ? [{ type: "page", url: "http://tauri.localhost/" }] : { value: { ready: false } } }));
  const session = new WindowsWebDriver(options);
  await expect(session.start()).rejects.toThrow("within 30 seconds");
  expect(clock).toBe(30000);
  await session.stop();
  expect(app.exitCode).toBe(0);
  expect(driver.exitCode).toBe(0);
});

it("does not retry app launch after a startup failure", async () => {
  io.spawn.mockImplementation(() => { app.finish(1); return app; });
  const session = new WindowsWebDriver(options);
  await expect(session.start()).rejects.toThrow("exited during startup");
  await expect(session.start()).rejects.toThrow("cannot be started twice");
  expect(io.spawn).toHaveBeenCalledTimes(1);
  await session.stop();
  expect(io.exec).not.toHaveBeenCalled();
});

it("retains graceful-close failure while force-reaping only its exact live children", async () => {
  const session = new WindowsWebDriver(options);
  await session.start();
  io.exec.mockImplementation((command: string, args: string[]) => {
    if (command === "powershell.exe") throw new Error("Owned close failed");
    if (args[1] === "101") app.finish();
    else if (args[1] === "102") driver.finish();
    else throw new Error("An unrelated PID was targeted");
  });
  const failure = await session.stop().catch((error: AggregateError) => error);
  expect(failure).toBeInstanceOf(AggregateError);
  expect((failure as AggregateError).errors[0].message).toBe("Owned close failed");
  expect(io.exec.mock.calls.filter((call) => call[0] === "taskkill.exe").map((call) => call[1])).toEqual([
    ["/PID", "101", "/T", "/F"], ["/PID", "102", "/T", "/F"],
  ]);
  io.exec.mockClear();
  await session.stop();
  expect(io.exec).not.toHaveBeenCalled();
});

it("rejects relative process or scratch paths before creating anything", () => {
  expect(() => new WindowsWebDriver({ ...options, application: "corpi.exe" })).toThrow("absolute owned paths");
  expect(io.spawn).not.toHaveBeenCalled();
  expect(io.temporary).not.toHaveBeenCalled();
});

it("does not relabel a process crash after session startup as successful cleanup", async () => {
  const session = new WindowsWebDriver(options);
  await session.start();
  app.finish(1);
  const failure = await session.stop().catch((error: AggregateError) => error);
  expect(failure).toBeInstanceOf(AggregateError);
  expect((failure as AggregateError).errors[0].message).toContain("exited unexpectedly");
  expect(io.exec.mock.calls.map((call) => call[1])).toEqual([["/PID", "102", "/T", "/F"]]);
});

it("retains an app crash during normal window close and still reaps its driver", async () => {
  const session = new WindowsWebDriver(options);
  await session.start();
  io.exec.mockImplementation((command: string, args: string[]) => {
    if (command === "powershell.exe") app.finish(1);
    else if (args[1] === String(driver.pid)) driver.finish();
    else throw new Error("An unrelated PID was targeted");
  });
  const failure = await session.stop().catch((error: AggregateError) => error);
  expect(failure).toBeInstanceOf(AggregateError);
  expect((failure as AggregateError).errors[0].message).toContain("failed while closing");
  expect(driver.exitCode).toBe(0);
  expect(io.exec.mock.calls.filter((call) => call[0] === "taskkill.exe").map((call) => call[1])).toEqual([
    ["/PID", "102", "/T", "/F"],
  ]);
});
