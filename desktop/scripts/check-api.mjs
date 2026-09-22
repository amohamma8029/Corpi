import { execFileSync } from "node:child_process";
import { readFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const desktopRoot = dirname(scriptDir);
const coreRoot = join(desktopRoot, "..", "core");
const committedSchema = join(coreRoot, "openapi.json");

const tempDir = mkdtempSync(join(tmpdir(), "corpi-schema-"));
try {
  const freshSchema = join(tempDir, "openapi.json");
  execFileSync("uv", ["run", "python", "scripts/export_openapi.py", freshSchema], {
    cwd: coreRoot,
    stdio: "inherit",
  });

  const committed = JSON.parse(readFileSync(committedSchema, "utf8"));
  const fresh = JSON.parse(readFileSync(freshSchema, "utf8"));
  if (JSON.stringify(committed) !== JSON.stringify(fresh)) {
    console.error(
      "API contract drift: core/openapi.json does not match the live core routes.",
    );
    console.error("Re-export with: just schema-export, then regenerate desktop types.");
    process.exit(1);
  }
  console.log("OpenAPI schema matches the live core routes.");
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}
