# Phase 8 note — short-possession team calls (2026-11-01)

Frozen once delivered. Outcome: **nothing adopted; Phase 6 stays the upload** (`edde29db13a56e2d`).

## Target
Team accuracy inside possessions shorter than 2 s is ≈ 0.49 (coin-flip); perfect team calls would
lift the score to ≈ 0.84. All three candidates aim at that gap. Evaluated on the realistic
(σ 7 smoothed) simulated games, seven held-out matches, same adoption rule as before
(≥ 6 of 7 matches, then a blind real-frame check).

Phase 6 on these games: 0.5723 (team 7), 0.5718 (team 10); 0.5757 mean held-out with the decoder
re-selected leave-one-match-out on a 64-setting grid.

## Results
| Candidate | What | Pooled | Matches better | Decision |
|---|---|---|---|---|
| Receiver / co-movement features (`tools/recv.py`) | first feasible interceptor along the ball's path, lateral receiver margin, ball–player co-movement, ±1 s means; 11 inputs to the team and ball-gap models | 0.5751 (team 7, +0.003); short-spell accuracy 0.490 → 0.502 | 4 of 7 | Not adopted |
| Training augmentation | train on smoothed **and** sharp simulated games (2× data), test on smoothed | 0.5755 (team 7, +0.003) | 3 of 7 | Not adopted |
| Score ensemble | average of the Phase 6, receiver and augmentation models | +0.0046 at fixed decoder, 5 of 7; with the decoder re-selected per fold: 0.5763 vs 0.5757 | 3 of 7 | Not adopted |

Signal checks before the runs: co-movement agrees with the true team 86–88% where it fires
(≈ 30% of play); the first interceptor agrees with the team 1 s later 59–67% inside short
possessions — informative, but not enough to beat the context windows already in the models.

## What this says
- Every short-possession idea so far lands at +0.003 to +0.005 pooled, while single matches move
  ±0.01–0.02 between versions. The simulated evaluation cannot separate gains below ≈ 0.005, and
  none of these exceed it consistently.
- Two matches (J03WQQ, J03WR9 — the easiest, 0.60–0.64) get worse under every variant; gains
  come from the harder matches. A change that helps only hard games is not robust enough to ship.
- The remaining lever is information the tracking data does not hold (who touched a smoothed or
  missing ball). Phase 6's call of the stoppage / restart structure is where the score is earned.

## Tools added
`tools/recv.py` (features), `tools/experiments/` (leave-one-match-out harness: `rsx.py` fold
scores with `XPV` variants and `XPEXTRA` augmentation, `ens.py` score ensembles, `lomo.py`
decoder re-selection per fold, `pf2.py` / `cmp.py` paired comparisons). Paths in these scripts
point at the experiment work folder; they are development tools, not part of the submission path.
