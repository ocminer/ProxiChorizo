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
# stratum+ssl:// loads OpenSSL at run time: ship OpenSSL 3's two DLLs (from the runner's OpenSSL
# installation, or $env:OPENSSL_DIR) and Mozilla's CA bundle as cacert.pem next to the miner
$ossl = @($env:OPENSSL_DIR, 'C:\Program Files\OpenSSL\bin', 'C:\Program Files\OpenSSL-Win64\bin') | Where-Object { $_ -and (Test-Path (Join-Path $_ 'libssl-3-x64.dll')) } | Select-Object -First 1
if (!$ossl) { throw 'OpenSSL 3 DLLs (libssl-3-x64.dll, libcrypto-3-x64.dll) not found; set OPENSSL_DIR' }
Copy-Item (Join-Path $ossl 'libssl-3-x64.dll'), (Join-Path $ossl 'libcrypto-3-x64.dll') $stage
$osslLicense = @((Join-Path $ossl '..\license.txt'), (Join-Path $ossl '..\LICENSE.txt')) | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($osslLicense) { Copy-Item $osslLicense "$stage/LICENSE-OpenSSL.txt" }
Invoke-WebRequest -UseBasicParsing -Uri 'https://curl.se/ca/cacert.pem' -OutFile "$stage/cacert.pem"
if ((Get-Item "$stage/cacert.pem").Length -lt 100000) { throw 'cacert.pem download looks wrong' }
(Get-Content "$stage/README.txt") -replace 'ProxiChorizo 0\.[0-9.]+ -', "ProxiChorizo $Version -" | Set-Content "$stage/README.txt"
$info = @{
    version = $Version; source_commit = (& git rev-parse HEAD).Trim()
    ci_run = "https://github.com/$env:GITHUB_REPOSITORY/actions/runs/$env:GITHUB_RUN_ID"
    platform = 'native-windows-x86_64'; cuda = $Cuda; architectures = @(75,80,86,89,90,120)
    windows_gpu_hardware_test = $false
}
$info | ConvertTo-Json | Set-Content "$stage/BUILD-INFO.json"
# cudart is linked statically; a clean --help proves the executable starts (no GPU on the runner).
Push-Location $stage
try {
    $helpText = & ./proxichorizo.exe --help 2>&1
    if ($LASTEXITCODE -ne 0) { throw "executable startup failed: $LASTEXITCODE" }
    if (!(($helpText -join "`n").Contains("ProxiChorizo $Version"))) { throw 'executable version mismatch' }
    $helpText | Select-Object -First 2
    # the packaged OpenSSL DLLs and cacert.pem against the live pool, plain and TLS (no GPU needed)
    foreach ($url in 'stratum+tcp://proxi.suprnova.cc:3355', 'stratum+ssl://proxi.suprnova.cc:3356') {
        & ./proxichorizo.exe --check-pool $url
        if ($LASTEXITCODE -ne 0) { throw "check-pool $url failed" }
    }
} finally { Pop-Location }
Get-ChildItem $stage -File -Recurse | Sort-Object FullName | ForEach-Object {
    $relative = $_.FullName.Substring($stage.Length + 1).Replace('\','/')
    "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $relative
} | Set-Content "$stage/SHA256SUMS.txt"
New-Item -ItemType Directory -Force dist | Out-Null
$zip = Join-Path $root "dist/$name.zip"
Compress-Archive -Path "$stage/*" -DestinationPath $zip -Force
"{0}  {1}" -f (Get-FileHash $zip -Algorithm SHA256).Hash.ToLowerInvariant(), (Split-Path $zip -Leaf) | Set-Content "$zip.sha256"
Write-Host "Packaged $zip"
