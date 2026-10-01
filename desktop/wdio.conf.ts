import { browser } from "@wdio/globals";
import { launcher as TauriLauncher, type TauriServiceOptions } from "@wdio/tauri-service";
import { appendFileSync, mkdirSync } from "node:fs";
import path from "node:path";

const APPLICATION = path.resolve(`./src-tauri/target/release/corpi-desktop${process.platform === "win32" ? ".exe" : ""}`);
const embedded = process.env.CORPI_E2E_PROVIDER === "embedded";
const results = process.env.CORPI_E2E_RESULTS ?? path.resolve("e2e-results");
mkdirSync(results, { recursive: true });
const launcherOptions: TauriServiceOptions = {
  appBinaryPath: APPLICATION,
  driverProvider: "embedded",
  embeddedPort: 4445,
  startTimeout: 60000,
  captureBackendLogs: true,
  logDir: path.join(results, "logs"),
};
// WDIO accepts launcher-only custom classes at runtime, but ServiceEntry's
// constructor type describes worker services. Keep the cast at this boundary.
const launcherService = [TauriLauncher, launcherOptions] as unknown as
  NonNullable<WebdriverIO.Config["services"]>[number];

export const config: WebdriverIO.Config = {
  specs: ["./e2e/specs/**/*.ts"],
  runner: "local",
  maxInstances: 1,
  logLevel: "warn",
  outputDir: path.join(results, "logs"),
  framework: "mocha",
  mochaOpts: { timeout: 180000, bail: true },
  reporters: ["spec"],
  waitforTimeout: 15000,
  connectionRetryCount: 3,
  connectionRetryTimeout: 30000,
  hostname: "127.0.0.1",
  port: embedded ? 4445 : 4444,
  // Only app/driver ownership is needed. The full worker service installs IPC
  // window-management/mocking hooks, which require an additional bridge plugin.
  services: embedded ? [launcherService] : [],
  capabilities: [
    {
      browserName: "wry",
      "tauri:options": {
        application: APPLICATION,
      },
    } as WebdriverIO.Capabilities,
  ],
  afterTest: async (test, _context, result) => {
    const name = test.title.replace(/[^a-zA-Z0-9]+/g, "-").toLowerCase();
    appendFileSync(path.join(results, "tests.jsonl"), `${JSON.stringify({
      test: test.title,
      passed: result.passed,
      durationMs: result.duration,
      error: result.error?.message,
    })}\n`);
    await browser.saveScreenshot(path.join(results, `${name}.png`));
  },
};
