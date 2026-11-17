# mtah-possession

Frame-level ball possession for **More than a Hack 2027, Phase 1** (FC Barcelona vs Athletic Club).
Built from scratch on numpy; no kloppy, no pandas at runtime.

**Status: Phase 9 — simulated games that lose the ball where the real tracker does (see
`docs/specs/phase9.md`).** Phase 0 built parsing, caching, an exact re-implementation of the
scorer and a labelled validation set (IDSSE games degraded to look like this match's tracking).
Phase 1 adds per-frame evidence models and two whole-sequence decoders, chosen on that validation
set, and writes the first real submission. Phases 2–7 refine the simulator, the features, the
models and the decoder; phase 8 tried short-possession features (not adopted); phase 9 makes the simulator game-state-aware. Each phase is specified with
its results in `docs/specs/`.

## Rules this repository is built around

- **The submitted labels come from code, never from a person.** Real frames of the match were
  looked at (permitted by the organisers) only to check specific decisions — whether a filled
  ball position is the ball, whether play is stopped — never to produce labels or train a model.
- **The public 5 minutes are a pipeline check, not a tuning set.** Every design choice is made
  on the simulated IDSSE games (`evaluate --on sim`).
- **The match data is never committed.** `.gitignore` excludes `data/`, `cache/` and every CSV.
- **Frame-level data stays on the participant's machine.** Progress is reported as PASS/FAIL
  verdicts and aggregate numbers.

## Setup

### 1. Environment & Dependencies
```bash
python -m venv .venv

# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt          # numpy
pip install -r requirements-dev.txt      # pytest + pandas (pandas only for check-scorer)
```

### 2. Environment Variables
Point the environment variables to your local paths:
```powershell
# Windows PowerShell example:
$env:MTAH_DATA_DIR  = "C:\path\to\dataset"         # folder containing tactical/ and broadcast/
$env:MTAH_REPO_DIR  = "C:\path\to\more-than-a-hack-2027"  # official repo clone (for checks)
$env:MTAH_CACHE_DIR = ".\cache"                     # local cache directory
```
Or in bash:
```bash
export MTAH_DATA_DIR="/path/to/dataset"
export MTAH_REPO_DIR="/path/to/more-than-a-hack-2027"
export MTAH_CACHE_DIR="./cache"
```

## Quick Reproduction: Generate & Validate Final Submission CSV

To regenerate the official submission file directly from cached features and models:
```powershell
# Generate submission CSV (takes ~6 min)
python -m possession predict --team "Gradient Descendants"

# Validate format, row count, value ranges, dtypes, and file size
python validate_submission.py "submissions\Gradient Descendants.csv"
```
Expected output hash: SHA-256 `28f7229f96045c50` (identical on Linux and Windows).

## Phase 0, in order

| step | command | what PASS means |
|---|---|---|
| 0 | `python -m pytest -q tests` | parsers, metrics and simulator correct on synthetic files (with `MTAH_REPO_DIR` set, also checks the README baseline scores) |
| 1 | `python -m possession build-cache` | ATD parsed: 199,319 records, 46 slots, home = Barcelona, published ball-coverage and object-count figures matched, 638 tags, both kickoffs inside their tags |
| 2 | `python -m possession check-baseline` | our parse + nearest-player baseline reproduce the four committed baseline CSVs **frame for frame**, and their README scores |
| 3 | `python -m possession check-scorer` | our metrics equal the official `possession_metrics.py` on the baselines + 200 random perturbations |
| 4 | `python -m possession idsse-download` | 7 IDSSE matches fetched from figshare (~2.4 GB) into `cache/idsse_raw` |
| 5 | `python -m possession idsse-build` | each match parsed into a clean labelled Match |
| 6 | `python -m possession noise-profile` | BAR-ATH's tracking degradation measured (unsupervised) |
| 7 | `python -m possession simulate` | every IDSSE match degraded with that profile, 2 seeds; simulated ball coverage and objects/team within tolerance of BAR-ATH |
| 8 | `python -m possession evaluate` | baseline scored on simulated games (with a 90% bootstrap interval) and on the public window |

Steps 5–8 in one go, after the download: `python -m possession phase0`.

## Regenerating the submission (current phase)

Two stages. Stage 1 is the Phase 6 pipeline; its labels only tell stage 2 where play is stopped,
so the simulator can measure how BAR-ATH's tracker sees the ball in live play and in stoppages.

```powershell
# stage 1 (Phase 6, state-blind simulator)
python -m possession noise-profile
python -m possession simulate --world phase6
python -m possession phase1-select
python -m possession predict --out stage1
copy "stage1\Gradient Descendants.csv" cache\stage1_labels.csv

# stage 2 (Phase 9)
python -m possession state-profile --labels cache\stage1_labels.csv
python -m possession simulate               # --world state is the default
python -m possession phase1-select          # or split: --folds J03WMX J03WN1 ... then --merge
python -m possession predict                # writes "submissions/Gradient Descendants.csv"
```

Stage 1 must reproduce SHA-256 `edde29db13a56e2d` (the Phase 6 upload); stage 2's hash is in
`docs/specs/phase9.md`.

**Reproducibility.** Decoding runs in integer fixed-point arithmetic and fitted parameters are
rounded far above floating-point noise, so the same data should give the same file on every
platform. Verified: Linux / numpy 2.2.6 and Windows / numpy 2.5.4 give the identical SHA-256.

## Vision (experimental): ball detection on the tactical video

Runs on your own GPU in a **separate environment** (PyTorch needs CUDA 12.8+ builds for RTX 50
cards and does not yet support every new Python). The video never leaves your machine; only
per-frame ball candidates are written to `cache/vision/`.

```powershell
pip install uv
uv python install 3.12
uv venv --python 3.12 .venv-vision
.\.venv-vision\Scripts\Activate.ps1
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install ultralytics==8.4.187 opencv-python numpy pytest
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python -m possession vision-detect --mode sample --model yolo11m.pt
python -m possession vision-validate --mode sample --model yolo11m.pt
```

`vision-validate` compares detections with the tracker's ball where the tracker sees it
(recall, share within 2 m / 4 m, median error) and reports how many tracker-missing frames a
detection could fill, per confidence threshold.

## Phase 1, in order

| step | command | what PASS means |
|---|---|---|
| 1 | `python -m pytest -q tests` | relabelling, logistic fit and both decoders correct on synthetic sequences |
| 2 | `python -m possession phase1-select` | leave-one-match-out grid over Potts + semi-Markov decoders; the chosen one beats the baseline with non-overlapping 90% intervals (10–30 min) |
| 3 | `python -m possession predict --team <TEAM>` | `submissions/<TEAM>.csv` written, well-formed, deterministic; run twice and compare the sha256 |
| 4 | `python -m possession validate submissions/<TEAM>.csv` | official-style format check of the exact file to upload |
Any submission file: `python -m possession validate <file.csv>`.

Every command ends with `N PASS, N FAIL, N INFO` and lists the failures.

## Layout

```
possession/
  config.py           published constants, paths (env vars)
  schema.py           Match: the one representation every source and method shares
  metrics.py          spellF1 / macroF1 / transitions, pooled scoring, bootstrap
  harness.py          run a method over labelled matches (labels hidden from the method)
  io/atd.py           EPTS metadata (DataFormatSpecification) + raw ATD -> Match
  io/tagging.py       Smart Tagging -> frame intervals
  io/labels.py        submission CSV read / write
  io/idsse.py         IDSSE DFL XML -> labelled Match (BallStatus / BallPossession)
  noise/profile.py    measure tracking degradation from BAR-ATH (no labels)
  noise/simulate.py   impose it on clean external games
  noise/state.py      phase 9: game-state-aware ball dropout, carried-then-resting dead ball
  methods/baseline.py starter baseline, for reproduction and as the floor
  relabel.py          competition possession rule applied to external labels
  features.py         per-frame physical features (no labels)
  logistic.py         IRLS logistic regression
  emission.py         per-frame H / A / D evidence from two logistic models
  decode.py           Potts and semi-Markov decoders, boundary refinement, priors
  phase1.py           leave-one-match-out selection (resumable), shift guard, prediction
  gbdt.py             own histogram gradient boosting (deterministic, numpy only)
  kickoff.py          period-start kick-offs from Smart Tagging; pre-kick-off dead ball
docs/specs/           approved phase specs (frozen)
tests/                synthetic-format fixtures + tests
```

## External data

IDSSE: Bassek, Weber, Rein & Memmert (2025), *An integrated dataset of spatiotemporal and event
data in elite soccer*, Scientific Data 12:195, doi:10.1038/s41597-025-04505-y. Seven 2022-23
Bundesliga / 2. Bundesliga matches; per-frame ball status and possession from DFL's official
positional data. Human-curated data about **other** games, which RULES.md allows for training and
validation.
