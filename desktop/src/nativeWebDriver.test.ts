import { afterEach, expect, it, vi } from "vitest";

vi.mock("@wdio/globals", () => ({ browser: {} }));
vi.mock("@wdio/tauri-service", () => ({ launcher: class {} }));
vi.mock("node:fs", async (original) => ({
  ...await original<typeof import("node:fs")>(),
  appendFileSync: vi.fn(),
  mkdirSync: vi.fn(),
}));

const platformDescriptor = Object.getOwnPropertyDescriptor(process, "platform")!;
afterEach(() => {
  Object.defineProperty(process, "platform", platformDescriptor);
  vi.unstubAllEnvs();
  vi.resetModules();
});

it.each([
  { platform: "win32", provider: "external", classic: true, browserName: "webview2" },
  { platform: "win32", provider: "embedded", classic: undefined, browserName: "wry" },
  { platform: "linux", provider: "external", classic: undefined, browserName: undefined },
  { platform: "darwin", provider: "embedded", classic: undefined, browserName: "wry" },
])("configures the native protocol for $platform/$provider without changing other providers", async ({
  platform, provider, classic, browserName,
}) => {
  Object.defineProperty(process, "platform", { ...platformDescriptor, value: platform });
  vi.stubEnv("CORPI_E2E_PROVIDER", provider);
  vi.stubEnv("CORPI_E2E_DEBUGGER_ADDRESS", "127.0.0.1:45231");
  vi.stubEnv("CORPI_E2E_DRIVER_PORT", "45232");
  const { config } = await import("../wdio.conf");
  if (!Array.isArray(config.capabilities)) throw new Error("Native tests require a single capabilities list");
  expect(config.capabilities).toHaveLength(1);
  const capability = config.capabilities[0];
  if (classic) expect(capability).toHaveProperty("wdio:enforceWebDriverClassic", true);
  else expect(capability).not.toHaveProperty("wdio:enforceWebDriverClassic");
  if (browserName) expect(capability).toHaveProperty("browserName", browserName);
  else expect(capability).not.toHaveProperty("browserName");
  if (platform === "win32" && provider === "external") {
    expect(capability).toHaveProperty("ms:edgeOptions", { debuggerAddress: "127.0.0.1:45231" });
    expect(capability).not.toHaveProperty("tauri:options");
  } else expect(capability).toHaveProperty("tauri:options.application");
  expect(config.port).toBe(classic ? 45232 : provider === "embedded" ? 4445 : 4444);
});

it("rejects Windows external sessions without the runner-owned attachment endpoint", async () => {
  Object.defineProperty(process, "platform", { ...platformDescriptor, value: "win32" });
  vi.stubEnv("CORPI_E2E_PROVIDER", "external");
  vi.stubEnv("CORPI_E2E_DEBUGGER_ADDRESS", "");
  await expect(import("../wdio.conf")).rejects.toThrow("runner-owned loopback");
});
