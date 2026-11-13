# CI-only provisioning. The ordinary native runner never downloads or installs a driver.
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-CorpiWebView2Version {
    param([Parameter(Mandatory)][string] $Loader)

    if (-not ('CorpiCi.WebView2Runtime' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
namespace CorpiCi {
    public static class WebView2Runtime {
        [UnmanagedFunctionPointer(CallingConvention.StdCall, CharSet = CharSet.Unicode)]
        private delegate int GetVersion(string folder, out IntPtr version);
        public static string Read(string loader) {
            IntPtr library = NativeLibrary.Load(loader);
            IntPtr version = IntPtr.Zero;
            try {
                var query = Marshal.GetDelegateForFunctionPointer<GetVersion>(
                    NativeLibrary.GetExport(library, "GetAvailableCoreWebView2BrowserVersionString"));
                int result = query(null, out version);
                Marshal.ThrowExceptionForHR(result);
                if (version == IntPtr.Zero) throw new InvalidOperationException("No WebView2 runtime found");
                return Marshal.PtrToStringUni(version);
            } finally {
                if (version != IntPtr.Zero) Marshal.FreeCoTaskMem(version);
                NativeLibrary.Free(library);
            }
        }
    }
}
'@
    }
    $version = [CorpiCi.WebView2Runtime]::Read((Resolve-Path -LiteralPath $Loader).Path)
    if ($version -notmatch '^\d+\.\d+\.\d+\.\d+$') {
        throw "Expected a stable WebView2 runtime version, got '$version'."
    }
    return $version
}

function Expand-CorpiEdgeDriverArchive {
    param(
        [Parameter(Mandatory)][string] $Archive,
        [Parameter(Mandatory)][string] $Destination
    )

    if (Test-Path -LiteralPath $Destination) { throw "Driver destination already exists: $Destination" }
    if ((Get-Item -LiteralPath $Archive).Length -gt 64MB) { throw 'Driver archive exceeds 64 MiB.' }
    $zip = [System.IO.Compression.ZipFile]::OpenRead($Archive)
    try {
        $allowed = @('Driver_Notes/', 'Driver_Notes/credits.html', 'Driver_Notes/EULA',
            'Driver_Notes/LICENSE', 'msedgedriver.exe')
        $seen = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
        [long] $total = 0
        foreach ($entry in $zip.Entries) {
            if ($entry.FullName -cnotin $allowed -or -not $seen.Add($entry.FullName)) {
                throw "Unexpected or duplicate driver archive entry: $($entry.FullName)"
            }
            if (($entry.ExternalAttributes -band 0xF0000000L) -eq 0xA0000000L) {
                throw "Driver archive contains a symbolic link: $($entry.FullName)"
            }
            $total += $entry.Length
            if ($total -gt 64MB) { throw 'Expanded driver archive exceeds 64 MiB.' }
        }
        foreach ($required in $allowed | Where-Object { -not $_.EndsWith('/') }) {
            if (-not $seen.Contains($required)) { throw "Missing driver archive entry: $required" }
        }
        # Validate every entry before writing. Only these literal filenames are extracted.
        [void][System.IO.Directory]::CreateDirectory((Join-Path $Destination 'Driver_Notes'))
        foreach ($entry in $zip.Entries) {
            if (-not $entry.FullName.EndsWith('/')) {
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile(
                    $entry, (Join-Path $Destination $entry.FullName), $false)
            }
        }
    } finally { $zip.Dispose() }
    return Join-Path $Destination 'msedgedriver.exe'
}

function Assert-CorpiEdgeDriver {
    param(
        [Parameter(Mandatory)][string] $Driver,
        [Parameter(Mandatory)][string] $RuntimeVersion
    )

    if ($RuntimeVersion -notmatch '^\d+\.\d+\.\d+\.\d+$') { throw 'Invalid WebView2 version.' }
    $signature = Get-AuthenticodeSignature -LiteralPath $Driver
    if ($signature.Status -ne 'Valid' -or $null -eq $signature.SignerCertificate -or
        $signature.SignerCertificate.Subject -notmatch '(^|,\s*)O=Microsoft Corporation(,|$)') {
        throw "Driver must have a valid Microsoft signature: $($signature.Status)"
    }
    $version = (Get-Item -LiteralPath $Driver).VersionInfo.FileVersion
    if ($version -notmatch '^\d+\.\d+\.\d+\.\d+$' -or
        ($version.Split('.')[0..2] -join '.') -ne ($RuntimeVersion.Split('.')[0..2] -join '.')) {
        throw "EdgeDriver $version does not match WebView2 $RuntimeVersion (first three components)."
    }
    return [ordered]@{
        version = $version
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Driver).Hash.ToLowerInvariant()
        signature = $signature.Status.ToString()
        signer = $signature.SignerCertificate.Subject
    }
}

function Install-CorpiCiEdgeDriver {
    if (-not $IsWindows -or $env:GITHUB_ACTIONS -ne 'true' -or $env:RUNNER_OS -ne 'Windows' -or
        -not $env:RUNNER_TEMP -or -not $env:GITHUB_ENV) {
        throw 'Driver acquisition is restricted to Windows GitHub Actions.'
    }
    if ($env:WEBVIEW2_BROWSER_EXECUTABLE_FOLDER -or $env:WEBVIEW2_RELEASE_CHANNEL_PREFERENCE) {
        throw 'CI expects the default stable WebView2 runtime, without selection overrides.'
    }
    if ([Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -ne 'X64') {
        throw 'The Windows CI driver setup requires an x64 runner.'
    }
    $repo = Split-Path (Split-Path $PSScriptRoot)
    $lock = Get-Content -LiteralPath (Join-Path $repo 'desktop/src-tauri/Cargo.lock') -Raw
    $versions = [regex]::Matches($lock, '(?m)^name = "webview2-com-sys"\r?\nversion = "([^"]+)"')
    if ($versions.Count -ne 1) { throw 'Expected one locked webview2-com-sys version.' }
    $cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $HOME '.cargo' }
    $crate = 'webview2-com-sys-' + $versions[0].Groups[1].Value
    $loaders = @(Get-ChildItem -Path (Join-Path $cargoHome "registry/src/*/$crate/x64/WebView2Loader.dll") -File)
    if ($loaders.Count -ne 1) { throw 'Build the shell first; expected one locked x64 WebView2 loader.' }
    $runtime = Get-CorpiWebView2Version -Loader $loaders[0].FullName
    $root = Join-Path $env:RUNNER_TEMP 'corpi-webdriver'
    if (Test-Path -LiteralPath $root) { throw "Driver evidence directory already exists: $root" }
    [void][System.IO.Directory]::CreateDirectory($root)
    $url = "https://msedgedriver.microsoft.com/$runtime/edgedriver_win64.zip"
    $archive = Join-Path $root 'edgedriver_win64.zip'
    # Exact runtime version, fixed Microsoft origin, no redirects or latest-browser fallback.
    Invoke-WebRequest -Uri $url -OutFile $archive -TimeoutSec 90 -MaximumRedirection 0
    $driver = Expand-CorpiEdgeDriverArchive -Archive $archive -Destination (Join-Path $root 'unpacked')
    $verified = Assert-CorpiEdgeDriver -Driver $driver -RuntimeVersion $runtime
    $evidence = [ordered]@{
        runtimeVersion = $runtime
        loaderSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $loaders[0].FullName).Hash.ToLowerInvariant()
        url = $url
        archiveSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
        driver = $verified
    }
    $evidenceRoot = Join-Path $env:RUNNER_TEMP 'corpi-search-evidence'
    [void][System.IO.Directory]::CreateDirectory($evidenceRoot)
    $evidence | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $evidenceRoot 'windows-webdriver.json')
    # Only subsequent CI steps see this path; no installation or PATH/browser update.
    "CORPI_EDGE_DRIVER=$driver" >> $env:GITHUB_ENV
    Write-Host "Verified Microsoft EdgeDriver $($verified.version) for WebView2 $runtime."
}

if ($MyInvocation.InvocationName -ne '.') { Install-CorpiCiEdgeDriver }
