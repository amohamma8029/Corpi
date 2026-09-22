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
- End-to-end tests only: [tauri-driver](https://tauri.app/develop/tests/webdriver/) (`cargo install tauri-driver`) and Microsoft Edge WebDriver (winget: `Microsoft.EdgeDriver`)

Python is managed by uv through `core/.python-version`; uv downloads its own Python 3.13.

## Setup

```powershell
mise install
uv sync --directory core
pnpm --dir desktop install
just dev
```

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

Core setup is `uv sync` inside `core/` (uv installs Python 3.13 itself); desktop setup is `pnpm install` inside `desktop/`.

## API contract

The core publishes its OpenAPI schema in `core/openapi.json`. The desktop build generates
TypeScript types from it (`pnpm generate:api`), and `just schema-check` fails when the
committed schema no longer matches the live core routes. Re-export and regenerate after
changing core endpoints.

## Communication security

The core binds to 127.0.0.1 on an OS-assigned port and requires a bearer token issued fresh
per launch. Requests must pass `Origin` and `Host` validation. The token is a credential:
whoever holds it can spend the user's model-API budget under BYOK, so it is never persisted.
