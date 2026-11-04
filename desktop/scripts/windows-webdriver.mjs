import { execFileSync, spawn } from "node:child_process";
import { mkdirSync, mkdtempSync } from "node:fs";
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

async function freePort() {
  const server = createServer();
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
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
  constructor({ application, driver, cwd, env, scratch }) {
    if (![application, driver, cwd, scratch].every((value) => path.isAbsolute(value))) {
      throw new Error("Windows E2E requires absolute owned paths.");
    }
    this.options = { application, driver, cwd, env, scratch };
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
    const debugPort = await freePort();
    const root = mkdtempSync(path.join(this.options.scratch, "corpi-native-webview-"));
    const profile = path.join(root, "profile");
    mkdirSync(profile);
    const env = {
      ...this.options.env,
      TAURI_WEBVIEW_AUTOMATION: "true",
      WEBVIEW2_USER_DATA_FOLDER: profile,
      WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS:
        `--remote-debugging-port=${debugPort} --remote-debugging-address=127.0.0.1`,
    };
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
