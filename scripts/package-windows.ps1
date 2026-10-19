param([Parameter(Mandatory=$true)][string]$Version, [string]$Exe = 'proxichorizo.exe', [string]$Cuda = '12.9')
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$name = "proxichorizo-$Version-windows-x86_64"
$stage = Join-Path ($env:RUNNER_TEMP ?? $env:TEMP) $name
Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $stage | Out-Null
if (!(Test-Path $Exe)) { throw "executable missing: $Exe" }
Copy-Item $Exe "$stage/proxichorizo.exe"
Copy-Item windows/* $stage
Copy-Item README.md, USAGE.md, LICENSE, NOTICE, PLATFORMS.md, vectors.json $stage
if (Test-Path "RELEASE-v$Version.md") { Copy-Item "RELEASE-v$Version.md" $stage }
(Get-Content "$stage/README.txt") -replace 'ProxiChorizo 0\.[0-9.]+ -', "ProxiChorizo $Version -" | Set-Content "$stage/README.txt"
$info = @{
    version = $Version; source_commit = (& git rev-parse HEAD).Trim()
    ci_run = "https://github.com/$env:GITHUB_REPOSITORY/actions/runs/$env:GITHUB_RUN_ID"
    platform = 'native-windows-x86_64'; cuda = $Cuda; architectures = @(75,80,86,89,120)
    windows_gpu_hardware_test = $false
}
$info | ConvertTo-Json | Set-Content "$stage/BUILD-INFO.json"
# cudart is linked statically; a clean --help proves the executable starts (no GPU on the runner).
Push-Location $stage
try { & ./proxichorizo.exe --help 2>&1 | Select-Object -First 2 } finally { Pop-Location }
Get-ChildItem $stage -File -Recurse | Sort-Object FullName | ForEach-Object {
    $relative = $_.FullName.Substring($stage.Length + 1).Replace('\','/')
    "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $relative
} | Set-Content "$stage/SHA256SUMS.txt"
New-Item -ItemType Directory -Force dist | Out-Null
$zip = Join-Path $root "dist/$name.zip"
Compress-Archive -Path "$stage/*" -DestinationPath $zip -Force
"{0}  {1}" -f (Get-FileHash $zip -Algorithm SHA256).Hash.ToLowerInvariant(), (Split-Path $zip -Leaf) | Set-Content "$zip.sha256"
Write-Host "Packaged $zip"
