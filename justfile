set shell := ["pwsh", "-NoLogo", "-NoProfile", "-Command"]

default:
    @just --list

# Launch the desktop app in dev mode; the shell spawns and shuts down the core.
dev:
    Set-Location desktop; $env:CORPI_CORE_DIR = (Resolve-Path ..\core).Path; pnpm tauri dev

# Run every fast suite: core checks, tooling patches, UI tests, schema drift.
test: core security-patches ui schema-check

# Core: pytest + Ruff (lint and format) + Pyright.
[working-directory: 'core']
core:
    uv run pytest
    uv run ruff check .
    uv run ruff format --check .
    uv run pyright

# Verify the installed tooling security patches before running stress tests.
[working-directory: 'desktop']
security-patches:
    pnpm run check:security

# UI unit tests (Vitest).
[working-directory: 'desktop']
ui:
    pnpm test

# Re-export the core's OpenAPI schema after changing core routes.
[working-directory: 'core']
schema-export:
    uv run python scripts/export_openapi.py openapi.json

# Verify committed core/openapi.json still matches the live core routes.
[working-directory: 'desktop']
schema-check:
    pnpm run check:api

# End-to-end smoke test against the real window (WebdriverIO + tauri-driver),
# followed by the orphaned-core shutdown check.
e2e:
    pwsh -File scripts\Run-E2E.ps1
