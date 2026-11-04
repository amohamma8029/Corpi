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
  { platform: "win32", provider: "external", classic: true, browserName: "wry" },
  { platform: "win32", provider: "embedded", classic: undefined, browserName: "wry" },
  { platform: "linux", provider: "external", classic: undefined, browserName: undefined },
  { platform: "darwin", provider: "embedded", classic: undefined, browserName: "wry" },
])("configures the native protocol for $platform/$provider without changing other providers", async ({
  platform, provider, classic, browserName,
}) => {
  Object.defineProperty(process, "platform", { ...platformDescriptor, value: platform });
  vi.stubEnv("CORPI_E2E_PROVIDER", provider);
  const { config } = await import("../wdio.conf");
  if (!Array.isArray(config.capabilities)) throw new Error("Native tests require a single capabilities list");
  expect(config.capabilities).toHaveLength(1);
  const capability = config.capabilities[0];
  if (classic) expect(capability).toHaveProperty("wdio:enforceWebDriverClassic", true);
  else expect(capability).not.toHaveProperty("wdio:enforceWebDriverClassic");
  if (browserName) expect(capability).toHaveProperty("browserName", browserName);
  else expect(capability).not.toHaveProperty("browserName");
  expect(config.port).toBe(provider === "embedded" ? 4445 : 4444);
});
