# Corpi

Corpi is a single-user, local-first desktop app for asking questions about imported sources and checking answers against cited evidence.

Notebooks and saved research stay on the user's device. Answer generation uses the user's configured external model provider.

## Repository layout

- `core/` — Python local application core (FastAPI over loopback HTTP), managed with uv.
- `desktop/` — Tauri 2 desktop shell: React/TypeScript UI (`src/`), Rust shell (`src-tauri/`).
- `docs/` — product requirements and architecture.

## Prerequisites

- [mise](https://mise.jdx.dev/installing-mise.html) — `mise.toml` pins the project tools (just, Node.js LTS, pnpm, uv, Rust) and `mise install` installs them
- Windows builds additionally need:
  - MSVC build tools (Visual Studio Build Tools with the "Desktop development with C++" workload)
  - WebView2 runtime (Windows ships it with Edge)
- macOS builds additionally need Homebrew Python 3.13 with SQLite 3.51.3 or newer; see the macOS setup below. `uv` still manages the core's packages.
- Native E2E tooling depends on the host:
  - Windows: [tauri-driver](https://v2.tauri.app/develop/tests/webdriver/) and Microsoft Edge WebDriver (winget: `Microsoft.EdgeDriver`).
  - Ubuntu: the Tauri build libraries, `webkit2gtk-driver`, `xvfb`, `xauth`, and `dbus-x11`; install `tauri-driver` with `cargo install tauri-driver --version 2.1.0 --locked`.
  - macOS: the test runner uses the embedded WebDriver plugin; it needs no external driver or paid subscription.

`uv` manages the core's Python environment and packages. It downloads Python 3.13 by default; on macOS, select the Homebrew interpreter below so Python loads a supported SQLite version.

## Setup

```powershell
mise trust          # trust this project's mise.toml (once per clone)
mise install        # installs just, Node.js, pnpm, uv, and Rust at the pinned versions
mise activate pwsh | Out-String | Invoke-Expression   # add to your $PROFILE to make it permanent
uv sync --directory core
pnpm --dir desktop install
just dev
```

With mise activated, `node`, `pnpm`, `just`, `uv`, and `cargo` resolve to the pinned versions inside this repository; outside it, your global mise tools are untouched.

## Notebooks and PDF originals

Create or open a notebook, then add one PDF at a time. Corpi copies the bytes into the user's
app-data directory; moving the selected file later doesn't change the saved original. Replacing a
PDF keeps the previous current file available until the new version is ready. The Files section
opens the exact saved import, including older history. Identical current/pending PDFs in one
notebook reuse their source entry, and healthy physical originals can be shared across notebooks.

The core enforces a configurable **25,000,000-byte** starting upload limit, a **120-second** total
transfer deadline, and a **30-second** no-bytes deadline. Only one incoming PDF transfer runs per
core. PDF.js downloads the complete verified file and renders selected pages under a configurable
**20-megapixel** starting canvas limit; the upload cap doesn't bound decoded memory. PDF.js and its
matching worker/support assets are copied from the pinned `pdfjs-dist` package during `just dev`
and desktop builds. Those generated assets are ignored by Git.

Environment overrides are read by the core at startup:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `CORPI_PDF_MAX_BYTES` | `25000000` | Maximum received PDF bytes. |
| `CORPI_PDF_UPLOAD_TIMEOUT_SECONDS` | `120` | Total time from transfer request acceptance through finalization. |
| `CORPI_PDF_UPLOAD_IDLE_SECONDS` | `30` | Maximum wait for the next non-empty body chunk. |
| `CORPI_PDF_RENDERED_PAGE_PIXELS` | `20000000` | Maximum canvas pixels for a rendered page. |
| `CORPI_PDF_STAGING_AGE_SECONDS` | `86400` | Minimum idle age before unowned staging can be cleaned. |
| `CORPI_MAINTENANCE_INTERVAL_SECONDS` | `3600` | Existing local maintenance interval. |

Original-PDF import and viewing are implemented before extraction. The source remains available
to inspect while `pdf-extract` waits for M1.5's parser and readability steps; it doesn't become
searchable until the required extraction and indexing work publishes readiness.

### macOS Python and SQLite

The uv-selected Python on macOS CI loaded SQLite 3.50.4, below the version Corpi requires for safe WAL use. Set up the project tools and use Homebrew's Python 3.13 as the interpreter for the `uv` environment. From the repository root in zsh:

```sh
mise trust
mise install
eval "$(mise activate zsh)"
brew install python@3.13
export UV_PYTHON="$(brew --prefix python@3.13)/bin/python3.13"
uv sync --directory core
pnpm --dir desktop install
uv run --directory core python -c 'import sqlite3; print(sqlite3.sqlite_version)'
export CORPI_CORE_DIR="$(pwd)/core"
pnpm --dir desktop tauri dev
```

Confirm that the printed SQLite version is at least 3.51.3. Keep `UV_PYTHON` set when launching the desktop app so its child core uses the same interpreter. The root `just` recipes currently require `pwsh`; the direct macOS launch command above does not. This setup does not change the Python-core packaging decision.

## Commands

Run from the repository root:

| Command | What it does |
| --- | --- |
| `just dev` | Launch the desktop app in dev mode; the shell spawns, monitors, and shuts down the local core. |
| `just test` | Core tests (pytest), lint (Ruff), types (Pyright), UI unit tests (Vitest), and the OpenAPI schema drift check. |
| `just core` | Core tests + Ruff + Pyright. |
| `just ui` | UI unit tests. |
| `just e2e` | Windows WebdriverIO + tauri-driver checks against the real window, including PDF import/viewer/worker, then an orphaned-core shutdown check. |
| `pnpm --dir desktop run test:e2e` | Build and test the native app with the external driver on Windows/Linux or the embedded provider on macOS. |
| `pnpm --dir desktop run test:e2e:embedded` | Test the embedded provider explicitly, including on a Windows development host. |
| `pnpm --dir desktop run check:e2e` | Type-check the native E2E tests and WebdriverIO configuration. |
| `pnpm --dir desktop run check:webdriver-isolation` | Check that default app dependencies and capabilities exclude the test automation plugin. |
| `just schema-export` | Re-export `core/openapi.json` after changing core routes. |
| `just schema-check` | Verify the committed OpenAPI schema still matches the live core routes. |

Core setup is `uv sync` inside `core/`; desktop setup is `pnpm install` inside `desktop/`. On macOS, set `UV_PYTHON` as shown above before syncing or launching the core.

## Native viewer tests in CI

The existing GitHub Actions matrix runs the shared viewer suite on Ubuntu/WebKitGTK and
macOS/WKWebView, after the core, UI, build, and launcher checks. Ubuntu launches the app under
Xvfb and D-Bus:

```sh
xvfb-run -a --server-args="-screen 0 1280x1024x24" dbus-run-session -- pnpm --dir desktop run test:e2e
```

Start Xvfb before D-Bus so activated desktop services inherit `DISPLAY`. The CI dependency
step uses bounded APT download timeouts/retries and a ten-minute step timeout, so an unavailable
package mirror cannot consume the whole test job.

The macOS job runs `pnpm --dir desktop run test:e2e` with the Homebrew Python selection above.
It uses WebdriverIO's Tauri launcher and the optional `webdriver` Cargo feature to control the
actual app. Both jobs embed the built frontend, local PDF.js worker, and support assets; they
use the real core and the app's CSP. The test build uses `app.corpi.desktop.e2e` as its Tauri
identifier so its saved notebooks stay separate from ordinary Corpi data.

The suite checks painted content on distinct pages, navigation and zoom, a 24,999,190-byte
image-only PDF rendered under the 20-MP canvas cap, worker/canvas release after close and file
switching, and a visible renderer error that leaves the original saved. Blank-canvas and
stale-page controls check that the rendering assertions reject those failures.

Each run writes screenshots, logs, fixture hashes, OS/webview/PDF.js versions, test results,
the test binary hash, and a shutdown result under `desktop/e2e-results/<provider>/`. CI uploads
these as `native-viewer-<runner>-<commit>` artifacts, including failed runs, for 14 days. A build
or unit-test pass alone doesn't establish native viewer compatibility; use the native run's
results and screenshots for that evidence.

Ordinary builds omit the optional WebDriver dependency and test capabilities. Only the E2E
runner passes `--features webdriver` and `tauri.webdriver.conf.json`; the default capability
file stays unchanged. Production builds must run without that feature or test configuration.

For test-only edits, `pnpm --dir desktop run test:e2e --reuse-build` reruns the external checks
against the previous test binary after verifying its hash. Add `--embedded` for that provider.
This option doesn't rebuild app changes; CI always builds a fresh binary.

## API contract

The core publishes its OpenAPI schema in `core/openapi.json`. The desktop build generates
TypeScript types from it (`pnpm generate:api`), and `just schema-check` fails when the
committed schema no longer matches the live core routes. Re-export and regenerate after
changing core endpoints.

## Communication security

The core binds to 127.0.0.1 on an OS-assigned port and requires a bearer token issued fresh
per launch. Requests must pass `Origin` and `Host` validation. The token is a credential:
whoever holds it can spend the user's model-API budget under BYOK, so it is never persisted.
