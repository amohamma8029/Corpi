# Verifies the shell/core shutdown contract after a WebdriverIO run closes the app:
# no core process may survive the desktop app exiting.

$orphans = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like '*core.api.server*' }

if ($orphans) {
    $orphans | ForEach-Object {
        Write-Host "Orphaned core process: pid=$($_.ProcessId) cmd=$($_.CommandLine)"
    }
    Write-Error "Core process(es) survived the app exit."
    exit 1
}

Write-Host "No orphaned core process after app exit."
