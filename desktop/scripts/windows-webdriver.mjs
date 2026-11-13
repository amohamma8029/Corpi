import { execFileSync, spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { createServer } from "node:net";
import path from "node:path";
import { setTimeout as delay } from "node:timers/promises";

function portNumber(value) {
  if (!/^[1-9]\d{0,4}$/.test(value ?? "")) throw new Error("Invalid runner-owned WebDriver port.");
  const port = Number(value);
  if (port > 65535) throw new Error("Invalid runner-owned WebDriver port.");
  return port;
}

export function windowsAttachment(env) {
  const address = env.CORPI_E2E_DEBUGGER_ADDRESS;
  const match = /^127\.0\.0\.1:([1-9]\d{0,4})$/.exec(address ?? "");
  if (!match) throw new Error("Windows E2E requires its runner-owned loopback WebView2 endpoint.");
  portNumber(match[1]);
  return { debuggerAddress: address, port: portNumber(env.CORPI_E2E_DRIVER_PORT) };
}

async function freePort(port = 0) {
  const server = createServer();
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        server.close(() => reject(new Error("No loopback test port was assigned.")));
        return;
      }
      server.close((error) => error ? reject(error) : resolve(address.port));
    });
  });
}

function running(child) {
  return Boolean(child?.pid && child.exitCode === null && child.signalCode === null);
}

async function waitForExit(child) {
  if (!running(child)) return;
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      child.removeListener("exit", exited);
      reject(new Error("Runner-owned test process did not exit within 10 seconds."));
    }, 10_000);
    function exited() { clearTimeout(timer); resolve(); }
    child.once("exit", exited);
  });
}

export class WindowsWebDriver {
  constructor({ application, driver, cwd, env }) {
    if (![application, driver, cwd].every((value) => path.isAbsolute(value))) {
      throw new Error("Windows E2E requires absolute owned paths.");
    }
    this.options = { application, driver, cwd, env };
    this.binding = null;
    this.app = null;
    this.driver = null;
    this.environment = null;
    this.spawnErrors = new Map();
    this.stopped = false;
  }

  launch(command, args, env) {
    const child = spawn(command, args, { cwd: this.options.cwd, env, stdio: "inherit" });
    child.once("error", (error) => { this.spawnErrors.set(child, error); });
    return child;
  }

  async prepare({ baseConfig, e2eConfig, wryVersion }, previous) {
    if (this.binding) throw new Error("Windows test build settings were already prepared.");
    // Explicit arguments replace Wry's defaults. This exact pinned version uses
    // these feature switches, autoplay=true and no proxy for Corpi's main window.
    if (wryVersion !== "0.55.1") throw new Error("Review Windows test browser defaults for the locked Wry version.");
    const windows = baseConfig.app?.windows;
    if (!Array.isArray(windows) || windows.length !== 1 || windows[0].label !== "main"
      || e2eConfig.identifier !== "app.corpi.desktop.e2e" || e2eConfig.app?.windows) {
      throw new Error("Windows E2E requires the unchanged single main window and test-only identifier.");
    }
    const window = windows[0];
    if (["additionalBrowserArgs", "dataDirectory", "proxyUrl"].some((key) => window[key] != null)) {
      throw new Error("Review configured Windows browser arguments, data directory or proxy before overriding them.");
    }
    const binding = previous ?? {
      debugPort: await freePort(), identifier: `app.corpi.desktop.e2e.t${randomUUID()}`,
    };
    if (!Number.isInteger(binding.debugPort)
      || !/^app\.corpi\.desktop\.e2e\.t[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(binding.identifier ?? "")) {
      throw new Error("Invalid previous Windows test build binding; rebuild without --reuse-build.");
    }
    portNumber(String(binding.debugPort));
    this.binding = { debugPort: binding.debugPort, identifier: binding.identifier };
    return {
      binding: { ...this.binding },
      configuration: {
        ...e2eConfig,
        // The pinned Tauri manager passes local-app-data/<identifier> directly
        // to WebView2. A new test identifier gives each build fresh profile and
        // core data without relying on environment overrides or touching old data.
        identifier: binding.identifier,
        app: { ...e2eConfig.app, windows: [{
          ...window,
          additionalBrowserArgs: "--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection"
            + " --autoplay-policy=no-user-gesture-required"
            + ` --remote-debugging-port=${binding.debugPort} --remote-debugging-address=127.0.0.1`,
        }] },
      },
    };
  }

  async waitFor(child, endpoint, accepts) {
    const deadline = Date.now() + 30_000;
    while (Date.now() < deadline) {
      const spawnError = this.spawnErrors.get(child);
      if (spawnError) throw spawnError;
      if (!running(child)) throw new Error("Runner-owned test process exited during startup.");
      try {
        const response = await fetch(endpoint, { signal: AbortSignal.timeout(2000), redirect: "error" });
        if (response.ok && accepts(await response.json())) return;
      } catch { /* Poll only this owned loopback endpoint until its fixed deadline. */ }
      await delay(200);
    }
    throw new Error("Runner-owned native endpoint did not become ready within 30 seconds.");
  }

  async start() {
    if (this.app || this.driver) throw new Error("A Windows test session cannot be started twice.");
    if (!this.binding) throw new Error("Prepare Windows test build settings before building and launching the app.");
    const { debugPort } = this.binding;
    // The binary embeds this port. Fail if it is occupied; never silently attach
    // to a different process or invent another port for an already-built app.
    await freePort(debugPort);
    const env = {
      ...this.options.env,
      TAURI_WEBVIEW_AUTOMATION: "true",
    };
    // Runtime 150+ ignores these on elevated hosts. Both settings now reach the
    // native WebView2 API through the E2E-only build configuration instead.
    delete env.WEBVIEW2_USER_DATA_FOLDER;
    delete env.WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS;
    this.app = this.launch(this.options.application, [], env);
    await this.waitFor(this.app, `http://127.0.0.1:${debugPort}/json/list`, (targets) =>
      Array.isArray(targets) && targets.some((target) => target.type === "page"
        && /^(https?:\/\/tauri\.localhost|tauri:\/\/localhost)\//.test(target.url ?? "")));
    // Allocate this only after the app owns its debugging port, so the two
    // temporary endpoints cannot accidentally receive the same free port.
    const driverPort = await freePort();
    // The driver is a sibling, not the app's owner. Session retries can only
    // reattach to this one WebView; they cannot launch another core/library owner.
    this.driver = this.launch(this.options.driver,
      [`--port=${driverPort}`, "--host=127.0.0.1"], this.options.env);
    await this.waitFor(this.driver, `http://127.0.0.1:${driverPort}/status`,
      (status) => status.value?.ready === true);
    this.environment = {
      CORPI_E2E_DEBUGGER_ADDRESS: `127.0.0.1:${debugPort}`,
      CORPI_E2E_DRIVER_PORT: String(driverPort),
    };
    return this.environment;
  }

  async stop() {
    if (this.stopped) return;
    const failures = [];
    if (this.environment) {
      for (const child of [this.app, this.driver]) {
        if (child?.signalCode || (child?.exitCode !== null && child?.exitCode !== 0)) {
          failures.push(new Error("A runner-owned native process exited unexpectedly before cleanup."));
        }
      }
    }
    if (running(this.app)) {
      try {
        execFileSync("powershell.exe", ["-NoProfile", "-Command",
          `if (-not (Get-Process -Id ${this.app.pid} -ErrorAction Stop).CloseMainWindow()) { throw 'Owned test window did not accept close' }`],
        { timeout: 15_000, stdio: "ignore" });
        await waitForExit(this.app);
        if (this.app.exitCode !== 0 || this.app.signalCode) {
          throw new Error("The runner-owned native app failed while closing its window.");
        }
      } catch (error) { failures.push(error); }
    }
    // A failed graceful close remains a failure. The fallback only reaps exact
    // still-running children; it never searches for or kills an unrelated core.
    for (const child of [this.app, this.driver]) {
      if (!running(child)) continue;
      try {
        execFileSync("taskkill.exe", ["/PID", String(child.pid), "/T", "/F"],
          { timeout: 15_000, stdio: "ignore" });
        await waitForExit(child);
      } catch (error) { failures.push(error); }
    }
    this.stopped = !running(this.app) && !running(this.driver);
    if (failures.length) throw new AggregateError(failures, "Runner-owned Windows test cleanup failed.");
  }
}
