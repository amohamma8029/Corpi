# Native E2E builds/runs the real app and checks core shutdown on every platform.

$ErrorActionPreference = "Stop"

$desktop = Join-Path (Split-Path -Parent $PSScriptRoot) "desktop"
pnpm --dir $desktop run test:e2e
exit $LASTEXITCODE
