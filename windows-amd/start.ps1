param(
    [Parameter(Mandatory=$true)][string]$Token,
    [string]$Worker = 'windows',
    [string]$Pool = 'stratum+tcp://proxi.suprnova.cc:3355',
    [string]$KeyFile = "$PSScriptRoot/proxima.key",
    [string]$Seed = '',
    [string]$Devices = '',
    [string]$Variant = '',
    [string]$ApiBind = '',
    [string[]]$ExtraArgs = @()
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not $Pool.Contains('://')) { $Pool = "http://$Pool" }
$minerArgs = @('--proxi', $Pool, '--token', $Token, '--name', $Worker)
if ($Seed) { $minerArgs += @('--seed-hex', $Seed.Substring(0, [Math]::Min(64, $Seed.Length))) }
else {
    # current jobs (txid-blake2b) need only the token; a proxima.key is still used if present (legacy VRF only)
    if (Test-Path $KeyFile) { $minerArgs += @('--key-file', $KeyFile) }
}
if ($Devices) { $minerArgs += @('--gpus', $Devices) }
if ($Variant) { $minerArgs += @('--variant', $Variant) }
if ($ApiBind) { $minerArgs += @('--api-bind', $ApiBind) }
$minerArgs += $ExtraArgs
& "$PSScriptRoot/proxichorizo-cl.exe" @minerArgs
exit $LASTEXITCODE
