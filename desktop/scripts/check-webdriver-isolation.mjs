import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const tauriDirectory = fileURLToPath(new URL("../src-tauri/", import.meta.url));
const tree = execFileSync("cargo", ["tree", "--locked", "--edges", "normal", "--prefix", "none"], {
  cwd: tauriDirectory,
  encoding: "utf8",
  timeout: 120_000,
});
if (tree.includes("tauri-plugin-wdio")) {
  throw new Error("The default app dependency graph includes test automation.");
}
const config = JSON.parse(readFileSync(new URL("../src-tauri/tauri.conf.json", import.meta.url), "utf8"));
const capability = JSON.parse(readFileSync(new URL("../src-tauri/capabilities/default.json", import.meta.url), "utf8"));
if (JSON.stringify([config, capability]).includes("wdio")) {
  throw new Error("Default app configuration grants test automation capabilities.");
}
console.log("Default app dependencies and capabilities exclude WebDriver automation.");
