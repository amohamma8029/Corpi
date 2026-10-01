import { cp, mkdir, rm, copyFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const entry = fileURLToPath(import.meta.resolve("pdfjs-dist"));
const packageRoot = resolve(dirname(entry), "..");
const publicRoot = fileURLToPath(new URL("../public/pdfjs/", import.meta.url));

await rm(publicRoot, { recursive: true, force: true });
await mkdir(publicRoot, { recursive: true });
for (const directory of ["cmaps", "standard_fonts", "wasm", "iccs"]) {
  await cp(resolve(packageRoot, directory), resolve(publicRoot, directory), { recursive: true });
}
await copyFile(resolve(packageRoot, "LICENSE"), resolve(publicRoot, "LICENSE"));
