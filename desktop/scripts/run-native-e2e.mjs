import { execFileSync, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { arch, release, tmpdir, version } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";

import { WindowsWebDriver } from "./windows-webdriver.mjs";

const desktop = fileURLToPath(new URL("../", import.meta.url));
const repo = path.dirname(desktop);
const windows = process.platform === "win32";
if (!["win32", "darwin", "linux"].includes(process.platform)) {
  throw new Error(`Native E2E is unsupported on ${process.platform}.`);
}
if (process.argv.slice(2).some((arg) => !["--embedded", "--reuse-build"].includes(arg))) {
  throw new Error("Supported native E2E options are --embedded and --reuse-build.");
}
const provider = process.platform === "darwin" || process.argv.includes("--embedded")
  ? "embedded" : "external";
const results = path.join(desktop, "e2e-results", provider);
const binary = path.join(desktop, `src-tauri/target/release/corpi-desktop${windows ? ".exe" : ""}`);
const reuseBuild = process.argv.includes("--reuse-build");
const modelDownloads = process.env.CORPI_RUN_MODEL_DOWNLOAD === "1";
let previousBuild = null;
if (reuseBuild) {
  try {
    previousBuild = JSON.parse(readFileSync(path.join(results, "run.json"), "utf8"));
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    const firstRun = provider === "embedded" ? "test:e2e:embedded" : "test:e2e";
    throw new Error(`No previous test build found. Run without --reuse-build first: pnpm --dir desktop run ${firstRun}`);
  }
}
if (reuseBuild && (previousBuild.provider !== provider || !previousBuild.testBinarySha256
  || createHash("sha256").update(readFileSync(binary)).digest("hex") !== previousBuild.testBinarySha256)) {
  throw new Error("The previous test binary is unavailable or changed; rerun without --reuse-build.");
}
const env = {
  ...process.env,
  CORPI_CORE_DIR: path.join(repo, "core"),
  CORPI_E2E_PROVIDER: provider,
  CORPI_E2E_RESULTS: results,
  CORPI_PDF_MAX_BYTES: "25000000",
  CORPI_PDF_RENDERED_PAGE_PIXELS: "20000000",
};
// Every test binary uses a separate Tauri identifier/app-data directory.
// Only this generated evidence directory is replaced between runs.
rmSync(results, { recursive: true, force: true });
mkdirSync(results, { recursive: true });
const packageJson = JSON.parse(readFileSync(path.join(desktop, "package.json"), "utf8"));
const evidence = {
  sourceCommit: reuseBuild ? previousBuild.sourceCommit : process.env.GITHUB_SHA ?? null,
  platform: process.platform,
  architecture: arch(),
  osVersion: version(),
  osRelease: release(),
  nodeVersion: process.version,
  pdfjsVersion: packageJson.dependencies["pdfjs-dist"],
  tauriServiceVersion: packageJson.devDependencies["@wdio/tauri-service"],
  provider,
  reusedBuild: reuseBuild,
  modelDownloads,
  startedAt: new Date().toISOString(),
  stage: "setup",
  status: "running",
};
function saveEvidence() {
  writeFileSync(path.join(results, "run.json"), `${JSON.stringify(evidence, null, 2)}\n`);
}
saveEvidence();

function corePids() {
  if (windows) {
    const output = execFileSync("powershell.exe", ["-NoProfile", "-Command",
      "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object { $_.CommandLine -like '*core.api.server*' } | ForEach-Object { $_.ProcessId }"],
    { encoding: "utf8", timeout: 15_000 });
    return output.trim().split(/\s+/).filter(Boolean).map(Number);
  }
  const output = execFileSync("ps", ["-axo", "pid=,args="], { encoding: "utf8", timeout: 15_000 });
  return output.split("\n").filter((line) => /\s-m\s+core\.api\.server(?:\s|$)/.test(line))
    .map((line) => Number(line.trim().split(/\s+/)[0]));
}

async function stopProcess(child) {
  if (!child?.pid || child.exitCode !== null || child.signalCode !== null) return;
  if (windows) {
    execFileSync("taskkill.exe", ["/PID", String(child.pid), "/T", "/F"], { timeout: 15_000, stdio: "ignore" });
  } else {
    // The external driver and its native driver are in a runner-owned group.
    process.kill(-child.pid, "SIGTERM");
  }
}

async function run(command, args, timeout = 1_200_000) {
  console.log(`Running ${command} ${args.join(" ")}`);
  // Invoke pnpm's actual JS entry on Windows instead of executing a .cmd shell shim.
  let executable = command;
  let commandArgs = args;
  if (windows && command === "pnpm") {
    const entry = process.env.npm_execpath;
    if (!entry) throw new Error("On Windows, start this runner with pnpm run test:e2e.");
    executable = /\.(c?js|mjs)$/i.test(entry) ? process.execPath : entry;
    commandArgs = executable === process.execPath ? [entry, ...args] : args;
  }
  const child = spawn(executable, commandArgs, {
    cwd: desktop, env, stdio: "inherit",
    detached: !windows,
  });
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      void stopProcess(child).catch(() => undefined);
      reject(new Error(`${command} exceeded its ${timeout} ms timeout.`));
    }, timeout);
    child.once("error", (error) => { clearTimeout(timer); reject(error); });
    child.once("exit", (code, signal) => {
      clearTimeout(timer);
      if (code === 0) resolve();
      else reject(new Error(`${command} failed (exit ${code}, signal ${signal ?? "none"}).`));
    });
  });
}

async function waitForDriver(child) {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null || child.signalCode !== null) throw new Error("tauri-driver exited during startup.");
    try {
      const response = await fetch("http://127.0.0.1:4444/status", { signal: AbortSignal.timeout(2000) });
      if (response.ok && (await response.json()).value?.ready) return;
    } catch { /* Retry while the native driver starts. */ }
    await delay(200);
  }
  throw new Error("tauri-driver did not become ready within 30 seconds.");
}

const existingCorePids = new Set();
let driver;
let windowsDriver;
let failure;
try {
  for (const pid of corePids()) existingCorePids.add(pid);
  if (process.platform === "darwin") {
    evidence.webviewVersion = execFileSync("plutil", ["-extract", "CFBundleShortVersionString", "raw", "-o", "-",
      "/System/Library/Frameworks/WebKit.framework/Resources/Info.plist"], { encoding: "utf8", timeout: 15_000 }).trim();
    evidence.macOSVersion = execFileSync("sw_vers", ["-productVersion"], { encoding: "utf8", timeout: 15_000 }).trim();
  } else if (process.platform === "linux") {
    evidence.webviewVersion = execFileSync("dpkg-query", ["-W", "-f", "${Version}", "libwebkit2gtk-4.1-0"],
      { encoding: "utf8", timeout: 15_000 }).trim();
  }
  evidence.stage = "production-isolation";
  saveEvidence();
  await run("pnpm", ["run", "check:webdriver-isolation"], 180_000);
  await run("pnpm", ["run", "check:e2e"], 120_000);
  evidence.stage = "build";
  saveEvidence();
  const buildArgs = ["tauri", "build", "--no-bundle", "--config",
    `src-tauri/tauri.${provider === "embedded" ? "webdriver" : "e2e"}.conf.json`];
  if (provider === "embedded") buildArgs.push("--features", "webdriver");
  if (!reuseBuild) await run("pnpm", buildArgs);
  evidence.testBinarySha256 = createHash("sha256").update(readFileSync(binary)).digest("hex");

  if (provider === "external") {
    if (windows) {
      const edgeDriver = process.env.CORPI_EDGE_DRIVER ?? execFileSync("powershell.exe", ["-NoProfile", "-Command",
        "$driver = (Get-Command msedgedriver.exe -ErrorAction SilentlyContinue).Source; if (-not $driver) { $driver = Get-ChildItem \"$env:LOCALAPPDATA\\Microsoft\\WinGet\\Packages\" -Recurse -Filter msedgedriver.exe | Select-Object -First 1 -ExpandProperty FullName }; if (-not $driver) { throw 'Microsoft Edge WebDriver was not found' }; $driver"],
      { encoding: "utf8", timeout: 15_000 }).trim();
      evidence.stage = "native-startup";
      saveEvidence();
      windowsDriver = new WindowsWebDriver({
        application: binary, driver: edgeDriver, cwd: desktop, env,
        scratch: env.RUNNER_TEMP ?? tmpdir(),
      });
      Object.assign(env, await windowsDriver.start());
      evidence.windowsSession = { mode: "prelaunched-classic", loopbackOnly: true };
    } else {
      driver = spawn("tauri-driver", ["--port", "4444"], { cwd: desktop, env, stdio: "inherit", detached: true });
      // Capture spawn failure before polling, avoiding an unhandled child-process error.
      let spawnError;
      driver.once("error", (error) => { spawnError = error; });
      await delay(50);
      if (spawnError) throw spawnError;
      await waitForDriver(driver);
    }
  }
  evidence.stage = "viewer";
  saveEvidence();
  // Allow the bounded setup test plus the ordinary viewer/startup checks.
  await run("pnpm", ["exec", "wdio", "run", "wdio.conf.ts"], modelDownloads ? 1_800_000 : 300_000);
} catch (error) {
  failure = error;
} finally {
  try {
    if (windowsDriver) await windowsDriver.stop();
    else await stopProcess(driver);
  } catch (error) {
    evidence.cleanupErrors = (error.errors ?? [error]).map((item) => item.message);
    failure ??= error;
  }
  // Observe shutdown before any core cleanup. Never kill a core to make this pass.
  let orphans = [];
  const deadline = Date.now() + 10_000;
  do {
    try { orphans = corePids().filter((pid) => !existingCorePids.has(pid)); }
    catch (error) { failure ??= error; break; }
    if (orphans.length === 0) break;
    await delay(200);
  } while (Date.now() < deadline);
  evidence.orphanCorePids = orphans;
  if (orphans.length) failure ??= new Error(`Core survived app exit (PIDs: ${orphans.join(", ")}).`);
  evidence.status = failure ? "failed" : "passed";
  evidence.finishedAt = new Date().toISOString();
  if (failure) evidence.error = failure.message;
  saveEvidence();
}
if (failure) throw failure;
console.log(`Native ${provider} viewer checks passed; no new core process survived app exit.`);
