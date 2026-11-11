import assert from "node:assert/strict";
import childProcess from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import { createRequire, syncBuiltinESMExports } from "node:module";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import test from "node:test";
import tls from "node:tls";
import { setTimeout } from "node:timers/promises";
import { fileURLToPath, pathToFileURL } from "node:url";

const desktop = fileURLToPath(new URL("../", import.meta.url));
const require = createRequire(new URL("../package.json", import.meta.url));

function installationError(detail, cause) {
  return new Error(
    `Security checks require the locked, patched dependencies: ${detail}\n` +
    "From the repository root, run: pnpm --dir desktop install --frozen-lockfile\n" +
    "No amplification tests were started.", { cause });
}

async function resolveTestTools() {
  try {
    // Follow real consumer edges, not pnpm's generated peer/patch directory names.
    const cli = require.resolve("@wdio/cli");
    // WDIO already depends on this resolver for import-only package exports.
    const { resolve } = await import(pathToFileURL(createRequire(cli).resolve("import-meta-resolve")).href);
    const from = (name, parent) => fileURLToPath(resolve(name, pathToFileURL(parent).href));
    const config = from("@wdio/config", cli);
    const glob = createRequire(config).resolve("glob");
    const minimatch9 = createRequire(glob).resolve("minimatch");
    const webdriver = from("webdriverio", cli);
    const archiver = from("archiver", webdriver);
    const readdirGlob = createRequire(archiver).resolve("readdir-glob");
    const minimatch10 = createRequire(readdirGlob).resolve("minimatch");
    const utils = from("@wdio/utils", cli);
    const browsers = createRequire(utils).resolve("@puppeteer/browsers");
    const proxy = createRequire(browsers).resolve("proxy-agent");
    const pac = createRequire(proxy).resolve("pac-proxy-agent");
    const getUriEntry = createRequire(pac).resolve("get-uri");
    return { cli, glob, minimatch9, minimatch10, webdriver, browsers, pac, getUriEntry };
  } catch (cause) {
    throw installationError("could not resolve the test-tool dependencies", cause);
  }
}

const { cli, glob, minimatch9, minimatch10, webdriver, browsers, pac, getUriEntry } = await resolveTestTools();

function moduleFile(entry, format) {
  return format === "commonjs" ? entry : path.join(path.dirname(entry), "../esm/index.js");
}

async function moduleAt(entry, format) {
  return format === "commonjs" ? require(entry) : import(pathToFileURL(moduleFile(entry, format)).href);
}

// Reviewed patched-file fingerprints, not a substitute for the behavioral tests.
// Update these alongside patch changes; see patches/README.md.
const patchedFiles = [
  ["glob@10.5.0 CommonJS", path.join(path.dirname(glob), "walker.js"),
    "8dc8d70dda5219f6351d8267acf30bc17b0a97b353862db7edb8e234c4a38479"],
  ["glob@10.5.0 ESM", path.join(path.dirname(moduleFile(glob, "esm")), "walker.js"),
    "9e07bcdb1fe1ad47fd593cd9c89a1dc4e91d49e21392f19f9b62797e010852dd"],
  ["minimatch@9.0.9 CommonJS", minimatch9,
    "fe0b8b021737a57cb80467d261eb5da6fa8f4a2fa27ea230ee656591e6557df6"],
  ["minimatch@9.0.9 ESM", moduleFile(minimatch9, "esm"),
    "d07c175c5575980682c90950a1209cd126ebe9069eb7fc5b7b533080c3a75ad9"],
  ["minimatch@10.2.6 CommonJS", minimatch10,
    "287848814812667e45c2b3a8998ab7d50a5509b4c1bd3a05167f6bc1a84e0a71"],
  ["minimatch@10.2.6 ESM", moduleFile(minimatch10, "esm"),
    "e75554b52846ddbe3976cf3e75741e22b19d9e30b75918e32825b23adc300acf"],
  ["get-uri@6.0.5 dispatch", getUriEntry,
    "b88be9661ff8e441742d435ad892c6aef8842ec5095a7d1ec9153b64eea60cd1"],
  ["get-uri@6.0.5 file reader", path.join(path.dirname(getUriEntry), "file.js"),
    "eea3451fcc97c38ee0e204802877130266e04d0170496ab08943c09b5c24a9b8"],
  ["get-uri@6.0.5 public types", path.join(path.dirname(getUriEntry), "index.d.ts"),
    "e5b0c24ed5e120acfad397f41217bcd224ca6ce544eedb2389692c8ba0cdc171"],
];

function assertPatchedInstall() {
  for (const [label, filename, expected] of patchedFiles) {
    let source;
    try {
      source = fs.readFileSync(filename, "utf8").replaceAll("\r\n", "\n");
    } catch (cause) {
      throw installationError(`could not read ${label}`, cause);
    }
    if (createHash("sha256").update(source).digest("hex") !== expected) {
      throw installationError(`${label} does not match the reviewed patch`);
    }
  }
}

// Stop stale installs before importing the affected code or registering stress tests.
assertPatchedInstall();

for (const failure of ["stale", "missing"]) {
  test(`installation preflight: each ${failure} patched file fails promptly`, (t) => {
    const readFile = fs.readFileSync;
    for (const [index, [label, filename]] of patchedFiles.entries()) {
      let reads = 0;
      const missing = Object.assign(new Error("synthetic missing patched file"), { code: "ENOENT" });
      t.mock.method(fs, "readFileSync", (file, ...options) => {
        reads++;
        if (file === filename) {
          if (failure === "missing") throw missing;
          return "synthetic stale dependency source\n";
        }
        return readFile(file, ...options);
      });
      assert.throws(assertPatchedInstall, (error) =>
        error.message.includes(label) &&
        error.message.includes("pnpm --dir desktop install --frozen-lockfile") &&
        error.message.includes("No amplification tests were started.") &&
        (failure !== "missing" || error.cause === missing));
      assert.equal(reads, index + 1, "A failed preflight must not read later files");
      t.mock.restoreAll();
    }
  });
}

test("installation preflight: CRLF-normalized reviewed sources are accepted", (t) => {
  const readFile = fs.readFileSync;
  t.mock.method(fs, "readFileSync", (...args) =>
    readFile(...args).replaceAll("\r\n", "\n").replaceAll("\n", "\r\n"));
  assert.doesNotThrow(assertPatchedInstall);
});

function fixture() {
  const parent = path.join(os.tmpdir(), "opencode");
  fs.mkdirSync(parent, { recursive: true });
  const root = fs.mkdtempSync(path.join(parent, "corpi-security-unit-"));
  // Small, explicitly synthetic files. Never walk a personal library or cache.
  for (const name of ["a/b/one.txt", "a/c/two.txt", "d/e/three.txt"]) {
    fs.mkdirSync(path.dirname(path.join(root, name)), { recursive: true });
    fs.writeFileSync(path.join(root, name), "synthetic security unit input\n");
  }
  return root;
}

function denySockets(t) {
  let calls = 0;
  const deny = () => {
    calls++;
    throw new Error("Security unit attempted an outbound socket");
  };
  t.mock.method(net.Socket.prototype, "connect", deny);
  t.mock.method(tls, "connect", deny);
  return () => assert.equal(calls, 0, "No outbound socket may be attempted");
}

function denyProcesses(t) {
  let calls = 0;
  for (const name of ["spawn", "spawnSync", "exec", "execSync", "execFile", "execFileSync", "fork"]) {
    t.mock.method(childProcess, name, () => {
      calls++;
      throw new Error("Security unit attempted to start a child process");
    });
  }
  syncBuiltinESMExports();
  t.after(() => { t.mock.restoreAll(); syncBuiltinESMExports(); });
  return () => assert.equal(calls, 0, "No child process may be attempted");
}

for (const format of ["commonjs", "esm"]) {
  test(`glob ${format}: abort stops directory reads`, { timeout: 5_000 }, async () => {
    const api = await moduleAt(glob, format);
    const controller = new AbortController();
    const reason = new Error("intentional unit cancellation");
    let calls = 0;
    const count = () => { if (++calls === 1) controller.abort(reason); };
    const countedFs = {
      ...fs,
      readdir(...args) { count(); return fs.readdir(...args); },
      readdirSync(...args) { count(); return fs.readdirSync(...args); },
    };
    await assert.rejects(api.glob("**", {
      cwd: fixture(), signal: controller.signal, fs: countedFs,
    }), (error) => error === reason);
    const atRejection = calls;
    await setTimeout(100);
    assert.equal(calls, atRejection);
    assert.equal(calls, 1);
  });

  test(`glob ${format}: cancelled synchronous stream does not keep walking`, { timeout: 5_000 }, async () => {
    const api = await moduleAt(glob, format);
    const controller = new AbortController();
    const reason = new Error("intentional synchronous cancellation");
    let calls = 0;
    const stream = api.globStreamSync("**", {
      cwd: fixture(), signal: controller.signal,
      fs: { ...fs, readdirSync(...args) {
        if (++calls === 1) controller.abort(reason);
        return fs.readdirSync(...args);
      } },
    });
    await new Promise((resolve, reject) => {
      stream.on("error", (error) => error === reason ? resolve() : reject(error));
      stream.on("end", () => reject(new Error("Cancelled stream unexpectedly completed")));
      stream.resume();
    });
    assert.equal(calls, 1);
  });

  test(`glob ${format}: a buffered stream resumes without losing matches`, { timeout: 5_000 }, async () => {
    const api = await moduleAt(glob, format);
    const cwd = fixture();
    const stream = api.globStream("**", { cwd });
    // Let matches buffer before attaching a reader, as in the upstream control.
    await setTimeout(50);
    assert.deepEqual((await Array.fromAsync(stream)).sort(), api.globSync("**", { cwd }).sort());
  });

  for (const [version, entry] of [[9, minimatch9], [10, minimatch10]]) {
    test(`minimatch ${version} ${format}: skip redundant large-set comparisons`, async () => {
      const { Minimatch } = await moduleAt(entry, format);
      const matcher = new Minimatch("x", { optimizationLevel: 2 });
      matcher.partsMatch = () => assert.fail("Large sets must not enter quadratic deduplication");
      const parts = Array.from({ length: 1001 }, (_, i) => [String(i)]);
      assert.equal(matcher.secondPhasePreProcess(parts), parts);
    });

    test(`minimatch ${version} ${format}: ordinary matching is preserved`, async () => {
      const { Minimatch } = await moduleAt(entry, format);
      const matcher = new Minimatch("**/../a/b/**/../c/d", { optimizationLevel: 2 });
      assert.equal(matcher.set.length, 4);
      assert.equal(matcher.match("../a/b/../c/d"), true);
      assert.equal(matcher.match("x/a/b/y/c/d"), true);
      assert.equal(matcher.match("not/a/match"), false);
      const small = new Minimatch("{x,*}/{0..9}", { optimizationLevel: 2 });
      assert.equal(small.set.length, 10, "Small sets must still be deduplicated");
      const large = new Minimatch("{x,*}/{0..999}", { optimizationLevel: 2 });
      assert.equal(large.set.length, 2000);
      assert.equal(large.match("y/999"), true);
      assert.equal(large.match(".y/5"), false);
    });

    test(`minimatch ${version} ${format}: amplification bound`, { timeout: 15_000 }, async () => {
      const { Minimatch } = await moduleAt(entry, format);
      const matcher = new Minimatch("x", { optimizationLevel: 2, braceExpandMax: 64 });
      // Test the preprocessing pass directly: do not compile 100,000 regexps.
      const parts = [Array(20).fill("**/../a/b").join("/").split("/")];
      const expanded = matcher.firstPhasePreProcess(parts);
      assert.equal(expanded.length, version === 9 ? 100_000 : 64);
    });
  }

  test(`glob ${format}: exact native spec discovery`, async () => {
    const api = await moduleAt(glob, format);
    const actual = api.globSync("./e2e/specs/**/*.ts", { cwd: desktop, matchBase: true })
      .map((name) => name.replaceAll("\\", "/")).sort();
    const expected = fs.readdirSync(path.join(desktop, "e2e/specs"), { recursive: true })
      .filter((name) => name.endsWith(".ts"))
      .map((name) => `e2e/specs/${name.replaceAll("\\", "/")}`).sort();
    assert.deepEqual(actual, expected);
    assert.ok(actual.length > 0);
  });
}

test("file URI stream retains FileHandle ownership", async (t) => {
  const api = require(getUriEntry);
  const handle = { fd: 123, stat: async () => ({}), close: async () => {} };
  const result = Readable.from(["synthetic"]);
  t.mock.method(fs.promises, "open", async () => handle);
  t.mock.method(fs, "createReadStream", (_path, options) => {
    assert.equal(options.fd, handle, "The stream must retain the handle, not just its numeric fd");
    return result;
  });
  assert.equal(await api.getUri(pathToFileURL(path.join(desktop, "unused-unit-file")).href), result);
});

for (const phase of ["stat", "stream"]) {
  test(`file URI ${phase} failure closes the opened handle`, async (t) => {
    const api = require(getUriEntry);
    const failure = new Error(`synthetic ${phase} failure`);
    let closed = 0;
    const handle = {
      fd: 123, stat: async () => { if (phase === "stat") throw failure; return {}; },
      close: async () => { closed++; },
    };
    t.mock.method(fs.promises, "open", async () => handle);
    t.mock.method(fs, "createReadStream", () => { throw failure; });
    await assert.rejects(api.getUri(pathToFileURL(path.join(desktop, "unused-unit-file")).href),
      (error) => error === failure);
    assert.equal(closed, 1);
  });
}

test("FTP rejection preserves non-FTP protocol dispatch and file/data reads", async (t) => {
  const noSockets = denySockets(t);
  const api = require(getUriEntry);
  assert.deepEqual(Object.keys(api.protocols).sort(), ["data", "file", "http", "https"]);
  await assert.rejects(api.getUri("ftp://127.0.0.1/unit"),
    (error) => error instanceof TypeError && error.message.includes('Unsupported protocol "ftp"'));
  const root = fixture();
  const opened = [];
  const open = fs.promises.open;
  t.mock.method(fs.promises, "open", async (...args) => {
    const handle = await open(...args);
    opened.push(handle);
    return handle;
  });
  const url = pathToFileURL(path.join(root, "a/b/one.txt")).href;
  const text = await api.getUri(url);
  assert.equal((await Array.fromAsync(text)).map((chunk) => chunk.toString()).join(""),
    "synthetic security unit input\n");
  assert.equal(opened[0].fd, -1, "Stream completion must close its owning handle");
  await assert.rejects(api.getUri(url, { cache: text }), { code: "ENOTMODIFIED" });
  assert.equal(opened[1].fd, -1, "Cache rejection must close its handle");
  const stopped = await api.getUri(url);
  const closed = new Promise((resolve, reject) => { stopped.once("close", resolve); stopped.once("error", reject); });
  stopped.destroy();
  await closed;
  assert.equal(opened[2].fd, -1, "Early stream destruction must close its handle");
  const data = await api.getUri("data:text/plain,synthetic%20proxy%20configuration");
  assert.equal((await Array.fromAsync(data)).map((chunk) => chunk.toString()).join(""),
    "synthetic proxy configuration");
  for (const scheme of ["http", "https"]) {
    t.mock.method(api.protocols, scheme, async (url) => {
      assert.equal(url.protocol, `${scheme}:`);
      return Readable.from(["synthetic dispatcher result"]);
    });
    const stream = await api.getUri(`${scheme}://example.invalid/unit`);
    assert.deepEqual(await Array.fromAsync(stream), ["synthetic dispatcher result"]);
  }
  // Dispatch is tested without a real HTTP connection; those implementations are unchanged.
  for (const name of ["ftp.js", "ftp.d.ts", "ftp.js.map"]) {
    assert.equal(fs.existsSync(path.join(path.dirname(getUriEntry), name)), false);
  }
  assert.throws(() => createRequire(getUriEntry).resolve("basic-ftp"), { code: "MODULE_NOT_FOUND" });
  noSockets();
});

test("PAC FTP rejection happens before a direct-connection fallback", async (t) => {
  const noSockets = denySockets(t);
  const { PacProxyAgent } = require(pac);
  const agent = new PacProxyAgent("pac+ftp://127.0.0.1/unit.pac", { fallbackToDirect: true });
  // Exercise actual file loading + connect failure ordering without evaluating PAC JavaScript/WASM.
  t.mock.method(agent, "getResolver", () => agent.loadPacFile());
  await assert.rejects(agent.connect({ path: "/", getHeader: () => "example.invalid" }, {
    host: "example.invalid", port: 443, secureEndpoint: true,
  }), (error) => error instanceof TypeError && error.message.includes('Unsupported protocol "ftp"'));
  agent.destroy();
  noSockets();
});

test("ZIP rejection happens before extraction or an output-directory write", async (t) => {
  const noSockets = denySockets(t);
  const api = require(path.join(path.dirname(browsers), "fileUtil.js"));
  const root = fixture();
  const output = path.join(root, "must-not-be-created");
  // No archive fixture is needed: the disabled module must fail before opening it.
  await assert.rejects(api.unpackArchive(path.join(root, "absent.zip"), output),
    (error) => ["MODULE_NOT_FOUND", "ERR_MODULE_NOT_FOUND"].includes(error.code));
  assert.equal(fs.existsSync(output), false);
  assert.throws(() => createRequire(browsers).resolve("extract-zip"), { code: "MODULE_NOT_FOUND" });
  noSockets();
});

test("tooling entry points load without starting drivers, browsers or network connections", async (t) => {
  const noSockets = denySockets(t);
  const noProcesses = denyProcesses(t);
  for (const entry of [cli, fileURLToPath(import.meta.resolve("@wdio/mocha-framework")), webdriver, browsers]) {
    await import(pathToFileURL(entry).href);
  }
  noSockets();
  noProcesses();
});
