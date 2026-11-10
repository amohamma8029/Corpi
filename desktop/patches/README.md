# Local tooling security patches

These exact-version pnpm patches backport selected fixes without upgrading the
current native-test tooling's dependency majors. They are development-tool changes, not
a PDF sandbox or a security certification of Corpi. Run from the repository root:

```powershell
pnpm --dir desktop run check:security
```

Use the repository's pinned tools and a frozen lockfile. `pnpm-workspace.yaml`
selects the four patches and two dependency removals; `pnpm-lock.yaml` records
their hashes. Do not disable supply-chain checks or lifecycle restrictions to
make an installation pass.

## Changes and provenance

| Exact package | Change | Source |
| --- | --- | --- |
| `glob@10.5.0` | Stop directory traversal after cancellation. Four early returns and the `onResume` correction, in CommonJS and ESM. Addresses `GHSA-3g3m-2fjr-6pw4`. | [Upstream fix 4682e8c](https://github.com/isaacs/node-glob/commit/4682e8c9babee1b385202ede36389634e12076d7). |
| `minimatch@9.0.9` | Bound level-2 pattern splitting at 100,000 patterns and skip quadratic deduplication above 1,000, in both module formats. Addresses `GHSA-g3hv-75wg-m249`. Version 9 lacks `braceExpandMax`, so this backport explicitly uses upstream's default bound. | [Upstream fix 77a0f5c](https://github.com/isaacs/minimatch/commit/77a0f5c4b80bdc641c3c930a1d9edf5ad2db57f8). |
| `minimatch@10.2.6` | The same preprocessing bounds, preserving its existing `braceExpandMax` option, in both module formats. Addresses the same advisory. | Same upstream fix. |
| `get-uri@6.0.5` | Remove the FTP import, protocol/type entry and three FTP distribution files. Also retain the opened `FileHandle` in file streams and close it on setup/cache errors. | Corpi-authored capability restriction and lifecycle correction against the locked release. The original file reader passed only a numeric descriptor to its stream, allowing the owning handle to close it again during garbage collection. |

The upstream changes are adapted to the packages' shipped JavaScript. Original
package manifests, declared versions and licenses remain intact. Source-map links
are removed from edited JavaScript because the original maps describe unpatched
code; other release files are retained. The file-handle correction is checked on
the repository's pinned Node version, not every historical Node version supported
by upstream `get-uri`.

Patch SHA-256 values, using pnpm's LF-normalized text hashing:

| Patch | SHA-256 |
| --- | --- |
| `glob@10.5.0.patch` | `284d552f69e5ce10648ae2756f28f1f8d9c36029e782d9ff33724b92cb5aef7e` |
| `minimatch@9.0.9.patch` | `19a106313a475850c9229590e60010ac7b3b3be9676384218db9beab38ca5e19` |
| `minimatch@10.2.6.patch` | `ffa28c1c6da748991a5b1ee9d0c83cac670dda5cf4317da7cd001f6dd76002ed` |
| `get-uri@6.0.5.patch` | `abfbca4552e3edf7735cbd98600bd840facae275f4e1edca085e4e04b43fdf98` |

## Intentionally unavailable download paths

- `get-uri@6.0.5>basic-ftp` is removed. FTP and `pac+ftp` configuration URLs reject
  before a connection, including when PAC direct fallback is enabled. HTTP,
  HTTPS, file and data retrieval remain; ordinary HTTP(S)/SOCKS proxy support is
  not removed. The regression checks use real synthetic file/data reads and
  stub HTTP(S) dispatch, not live proxy servers or PAC JavaScript evaluation.
- `@puppeteer/browsers@2.13.2>extract-zip` is removed. That tool's ZIP branch fails
  on its missing lazy import before extraction or output-directory creation.
  Other archive branches are unchanged. This is **not** a repaired ZIP extractor
  or a guarantee that a preceding download/cache write could not occur. Corpi's
  native runner uses its existing installed-driver paths.

The eight newly unreachable package records are pruned from the lockfile. Old
unlinked files may still exist in a developer's package cache; they are not active
dependency edges and need not be deleted. No general-purpose downloader or custom
archive parser replaces the disabled capabilities.

## Remaining advisory applicability

Version-only scanners still match locally patched versions. Keep those results
and this source/patch evidence; do not rename packages, invent upstream fixed
versions or globally suppress their advisories.

As reviewed on November 8, 2026:

- `serialize-javascript@6.0.2` is unchanged. Its three matches
  (`GHSA-5c6j-r48x-rmvq`, `GHSA-m5mm-p8h7-m29v`, `GHSA-qj8w-gfj5-8c6v`) do not
  trigger in the inspected Mocha setup: parallel mode is disabled, the buffered
  worker serialization path is inactive, function bodies are ignored, and output
  is not embedded in HTML. Recheck if Mocha, serializer versions, their callers or
  `desktop/wdio.conf.ts` parallel settings change. This is a current-configuration
  finding, not a patch to the serializer.
- `rand@0.7.3` (`RUSTSEC-2026-0097` / `GHSA-cq8v-f236-94qc`) is absent from the
  default app graph. The WebDriver test feature brings it through build-time
  `phf_generator@0.8.0`, which uses fixed-seed `SmallRng`; the `log` feature and
  affected `ThreadRng`/custom-logger prerequisites are absent. Recheck consumers,
  features and targets after Rust/tooling changes.
- Linux GTK/Tauri does include `glib@0.18.5`, whose `VariantStrIter` defect remains
  (`RUSTSEC-2024-0429` / `GHSA-wrw7-89jp-8q8g`). No call to the affected public
  entry `Variant::array_iter_str()` or iterator type was found in Corpi or the
  inspected active dependency sources. Generic string/vector conversions use
  different paths. This source-scoped finding does not fix GLib, prove optimized
  Linux binary behavior or establish that hostile PDFs are sandboxed. Recheck all
  affected active sources after source, target, feature or dependency changes;
  missing source coverage is incomplete evidence, not a clean result.
- Seven Rust maintenance notices (`fxhash`, `proc-macro-error` and five `unic-*`
  crates) remain maintenance debt, separate from demonstrated vulnerabilities.

See the root [security policy](../../SECURITY.md) for application boundaries.

## Verification and maintenance

The standalone Node suite follows actual consumer dependency edges and checks
both module formats, bounded pattern preprocessing, exact native-spec discovery,
file-handle cleanup, FTP/PAC/ZIP rejection and tooling imports with socket/process
startup denied. Safe unpatched controls reproduced the relevant failures; the
unbounded amplification case is intentionally not run against vulnerable code.

After changing a patch or parent dependency, recheck upstream applicability,
installed bytes and lock hashes, preserved surviving versions/edges, public types,
the Node suite, UI/build/schema/E2E typing and production WebDriver isolation.
Rescan the resulting graph and reconsider the applicability findings above. Remove
a patch only when the replacement supplies the fix and compatible behavior is
verified. Native-platform compatibility remains a separate required check before
claiming the native test workflow or a release works on that platform.
