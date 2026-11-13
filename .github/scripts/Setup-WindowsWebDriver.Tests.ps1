# Download-free provisioning controls. An optional already acquired archive checks the real signature.
[CmdletBinding()]
param([string] $Archive, [string] $RuntimeVersion)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/Setup-WindowsWebDriver.ps1"
$root = Join-Path ([IO.Path]::GetTempPath()) ("corpi-driver-controls-" + [guid]::NewGuid())
[void][IO.Directory]::CreateDirectory($root)
$script:checks = 0

function Assert-Rejected {
    param([scriptblock] $Operation, [string] $Message)
    try { & $Operation } catch {
        if ($_.Exception.Message -notlike "*$Message*") { throw }
        $script:checks++
        return
    }
    throw "Expected rejection: $Message"
}

function New-ArchiveFixture {
    param([string] $Name, [string[]] $Entries)
    $file = Join-Path $root "$Name.zip"
    $zip = [IO.Compression.ZipFile]::Open($file, [IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($name in $Entries) {
            $stream = [IO.StreamWriter]::new($zip.CreateEntry($name).Open())
            try { $stream.Write('unsigned test data') } finally { $stream.Dispose() }
        }
    } finally { $zip.Dispose() }
    return $file
}

$files = @('Driver_Notes/credits.html', 'Driver_Notes/EULA', 'Driver_Notes/LICENSE', 'msedgedriver.exe')
foreach ($case in @(
    @{ name = 'traversal'; extra = '../escape.exe' },
    @{ name = 'absolute'; extra = 'C:/escape.exe' },
    @{ name = 'script'; extra = 'install.ps1' },
    @{ name = 'duplicate'; extra = 'msedgedriver.exe' },
    @{ name = 'case-alias'; extra = 'MSEDGEDRIVER.EXE' }
)) {
    $zip = New-ArchiveFixture -Name $case.name -Entries ($files + $case.extra)
    $destination = Join-Path $root $case.name
    Assert-Rejected { Expand-CorpiEdgeDriverArchive $zip $destination } 'Unexpected or duplicate'
    if (Test-Path -LiteralPath $destination) { throw 'Unsafe archive wrote before validation.' }
}
$missing = New-ArchiveFixture -Name 'missing-license' -Entries @('msedgedriver.exe')
Assert-Rejected { Expand-CorpiEdgeDriverArchive $missing (Join-Path $root 'missing') } 'Missing driver archive entry'
$link = New-ArchiveFixture -Name 'symbolic-link' -Entries $files
$zip = [IO.Compression.ZipFile]::Open($link, [IO.Compression.ZipArchiveMode]::Update)
try { $zip.GetEntry('msedgedriver.exe').ExternalAttributes = -1610612736 } finally { $zip.Dispose() }
Assert-Rejected { Expand-CorpiEdgeDriverArchive $link (Join-Path $root 'link') } 'symbolic link'
$large = New-ArchiveFixture -Name 'expanded-oversize' -Entries $files
$zip = [IO.Compression.ZipFile]::Open($large, [IO.Compression.ZipArchiveMode]::Update)
try {
    $zip.GetEntry('msedgedriver.exe').Delete()
    $stream = $zip.CreateEntry('msedgedriver.exe').Open()
    try {
        $buffer = [byte[]]::new(1MB)
        for ($i = 0; $i -lt 65; $i++) { $stream.Write($buffer, 0, $buffer.Length) }
    } finally { $stream.Dispose() }
} finally { $zip.Dispose() }
Assert-Rejected { Expand-CorpiEdgeDriverArchive $large (Join-Path $root 'large') } 'exceeds 64 MiB'
$valid = New-ArchiveFixture -Name 'expected-layout' -Entries $files
$destination = Join-Path $root 'expected-layout'
$driver = Expand-CorpiEdgeDriverArchive $valid $destination
if (-not (Test-Path -LiteralPath $driver)) { throw 'Expected driver layout did not extract.' }
$script:checks++
Assert-Rejected { Expand-CorpiEdgeDriverArchive $valid $destination } 'already exists'
Assert-Rejected { Assert-CorpiEdgeDriver $driver '156.0.4314.58' } 'valid Microsoft signature'
Assert-Rejected { Assert-CorpiEdgeDriver $driver '../156' } 'Invalid WebView2 version'

if ($Archive) {
    if (-not $RuntimeVersion) { throw 'RuntimeVersion is required with Archive.' }
    $driver = Expand-CorpiEdgeDriverArchive $Archive (Join-Path $root 'approved-archive')
    $verified = Assert-CorpiEdgeDriver $driver $RuntimeVersion
    if ($verified.signature -ne 'Valid') { throw 'Real signed driver was not verified.' }
    $script:checks++
    Assert-Rejected { Assert-CorpiEdgeDriver $driver '1.0.0.0' } 'does not match WebView2'
}
Write-Host "$script:checks Windows driver provisioning controls passed; no downloads or driver execution."
Write-Host "Retained control inputs: $root"
