# All phase 10 variants back to back (about 45 min each). Run from mtah-possession-phase9.
.\tools\phase10_run.ps1 -Name hold     -Overrides "HANDLING_RATE=0.15,HANDLING_SPEED=18"
.\tools\phase10_run.ps1 -Name overhalf -Overrides "OVERHANG_SCALE=0.5"
.\tools\phase10_run.ps1 -Name live92   -Overrides "ALIVE_COVERAGE=0.92"
python tools\phase10_report.py
