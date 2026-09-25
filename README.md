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
- End-to-end tests only: [tauri-driver](https://tauri.app/develop/tests/webdriver/) (`cargo install tauri-driver`) and Microsoft Edge WebDriver (winget: `Microsoft.EdgeDriver`)

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
| `just e2e` | WebdriverIO + tauri-driver smoke test against the real window, then an orphaned-core shutdown check. |
| `just schema-export` | Re-export `core/openapi.json` after changing core routes. |
| `just schema-check` | Verify the committed OpenAPI schema still matches the live core routes. |

Core setup is `uv sync` inside `core/`; desktop setup is `pnpm install` inside `desktop/`. On macOS, set `UV_PYTHON` as shown above before syncing or launching the core.

## API contract

The core publishes its OpenAPI schema in `core/openapi.json`. The desktop build generates
TypeScript types from it (`pnpm generate:api`), and `just schema-check` fails when the
committed schema no longer matches the live core routes. Re-export and regenerate after
changing core endpoints.

## Communication security

The core binds to 127.0.0.1 on an OS-assigned port and requires a bearer token issued fresh
per launch. Requests must pass `Origin` and `Host` validation. The token is a credential:
whoever holds it can spend the user's model-API budget under BYOK, so it is never persisted.
