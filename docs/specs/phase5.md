# Phase 5 spec — temporal context (2026-10-24)

Frozen once delivered.

## Tried and dropped first
**Restart memory** (`decode.potts_restart`): Potts with two dead states (dead after home / after
away) and separate costs for restarting with the same or the other team, from external labels
(63% of restarts change team). Full leave-one-match-out: 0.5000 vs plain Potts 0.4983 —
inside the 0.005 tie band, so the simpler decoder is kept. The per-frame team model already sees
who lines up to restart. Code and test kept; configurations removed from the grid.

## Change
**Context windows** (`features.context_windows`): for each frame, the mean over the previous and
the next 2 s and 5 s of
- ball-dependent team signals (known-ball frames only): team_margin, goalside_diff, near5_diff,
  ctrl_run
- always-defined team signals: push_diff, vel_diff, spread_lr
- dead-ball signals: speed_med_1s, ball_known
Team model 14 → 43 inputs, ball-unknown model 9 → 37, dead model 16 → 24. A turnover is a
pattern over seconds (pressing stops, the other side spreads, goal-side counts flip), which a
single frame cannot show. Future windows are legitimate: the whole match is available offline.

**Grid trimmed** to the region phases 1–4 ever chose: Potts team {3,5,7,10,14,20,40} ×
dead {10,30,100,300}, semi-Markov α {0.2,0.4} × β {0.5,2} (32 configurations).

## Results (simulated IDSSE, leave-one-match-out, mean over control thresholds)
| | objective | 90% CI (T = 1 s) |
|---|---|---|
| **Phase 5, potts(team=7, dead=100)** | **0.5855** | 0.585 – 0.608 |
| Phase 4, potts(team=7, dead=100) | 0.4983 | 0.494 – 0.516 |
| starter baseline | 0.2226 | 0.221 – 0.235 |

Probe on three folds before the full run: 0.490 → 0.593 for every decoder setting tried.

BAR-ATH: 31 PASS, 0 FAIL; all 36 context features KS < 0.15 (largest 0.139); 93.0% of graded
frames agree with phase 4; dead-ball share 0.394 (0.384); possessions per minute 2.83 (2.90).
SHA-256 d2e29ede92e1d89c (Linux). Public window, sanity only and never used for selection:
spellF1 0.600 (phase 4: 0.368).
