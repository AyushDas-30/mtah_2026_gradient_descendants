# Phase 10 experiments: one simulator variant, full selection + prediction in its own cache.
# Run from the mtah-possession-phase9 folder with the venv active and cache9 already built.
#   .\tools\phase10_run.ps1 -Name hold -Overrides "HANDLING_RATE=0.15,HANDLING_SPEED=18"
param([Parameter(Mandatory=$true)][string]$Name, [string]$Overrides = "")
$ErrorActionPreference = "Stop"
$root = (Get-Location).Path
$cache = Join-Path $root "cache10_$Name"
$out = Join-Path $root "results10\$Name"
New-Item -ItemType Directory -Force -Path $out | Out-Null
if (-not (Test-Path (Join-Path $cache "idsse"))) {
    New-Item -ItemType Directory -Force -Path $cache | Out-Null
    foreach ($f in "BAR-ATH.npz", "BAR-ATH_tagging.json", "noise_profile.json", "stage1_labels.csv", "state_profile.json") {
        Copy-Item (Join-Path $root "cache9\$f") $cache
    }
    robocopy (Join-Path $root "cache9\idsse") (Join-Path $cache "idsse") /E /NFL /NDL /NJH /NJS | Out-Null
}
$env:MTAH_CACHE_DIR = $cache
$env:MTAH_STATE_OVERRIDES = $Overrides
$ErrorActionPreference = "Continue"
"variant $Name  overrides '$Overrides'  started $(Get-Date)" | Tee-Object -FilePath "$out\log.txt"
python -m possession simulate 2>&1 | Tee-Object -FilePath "$out\log.txt" -Append | Select-Object -Last 3
python -m possession phase1-select 2>&1 | Tee-Object -FilePath "$out\log.txt" -Append | Select-Object -Last 22
python -m possession predict --out $out 2>&1 | Tee-Object -FilePath "$out\log.txt" -Append | Select-Object -Last 4
Copy-Item (Join-Path $cache "phase1\selection.json") $out
"finished $(Get-Date)" | Tee-Object -FilePath "$out\log.txt" -Append
$env:MTAH_STATE_OVERRIDES = ""
$env:MTAH_CACHE_DIR = Join-Path $root "cache9"
