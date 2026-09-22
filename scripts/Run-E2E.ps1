# End-to-end runner for the packaged-window smoke test.
# Starts tauri-driver, runs the WebdriverIO suite against the real window,
# then verifies the core did not outlive the app exit.

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$desktop = Join-Path $repoRoot "desktop"

# The shell resolves the core through CORPI_CORE_DIR (same contract as `just dev`).
$env:CORPI_CORE_DIR = Join-Path $repoRoot "core"

Write-Host "== Building frontend (tsc + vite) =="
Push-Location $desktop
pnpm build
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }

Write-Host "== Building release shell binary via tauri CLI (embeds dist assets) =="
pnpm tauri build --no-bundle
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }

# tauri-driver on Windows proxies msedgedriver; resolve it explicitly so the run
# works in any shell (winget adds it to PATH only for fresh terminals).
$edgeDriver = (Get-Command msedgedriver.exe -ErrorAction SilentlyContinue).Source
if (-not $edgeDriver) {
    $edgeDriver = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter msedgedriver.exe |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $edgeDriver) {
    Write-Error "msedgedriver.exe not found; install the Microsoft.EdgeDriver winget package."
    exit 1
}

$driver = Start-Process -FilePath "tauri-driver" -ArgumentList @("--port", "4444", "--native-driver", $edgeDriver) -PassThru -NoNewWindow
try {
    Write-Host "== Running WebdriverIO suite =="
    pnpm exec wdio run wdio.conf.ts
    $wdioExit = $LASTEXITCODE
} finally {
    if ($driver -and -not $driver.HasExited) {
        Stop-Process -Id $driver.Id -Force -ErrorAction SilentlyContinue
    }
    Pop-Location
}

if ($wdioExit -ne 0) { exit $wdioExit }

Write-Host "== Checking for orphaned core processes =="
& (Join-Path $repoRoot "scripts\Assert-NoOrphanCore.ps1")
exit $LASTEXITCODE
