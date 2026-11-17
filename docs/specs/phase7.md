# Phase 7 spec — team merge and evaluation of steps 1–7 (2026-11-01)

Frozen once delivered.

## Outcome
The teammate's seven steps are merged into one tree. Every step is kept and runs, but each new
behaviour sits behind a switch that is **off by default**, because none passed the adoption rule.
With the defaults, the merged tree regenerates the Phase 6 simulated games bit for bit and
reproduces the Phase 6 submission: **SHA-256 `edde29db13a56e2d`**, 32 PASS, 0 FAIL. Phase 6
stays the upload.

## Adoption rule (unchanged)
Evaluated one change at a time, on the realistic simulated games (ball smoothed σ = 7 frames,
fitted to BAR-ATH's velocity-change distribution); adopted only if it beats the current version on
at least 6 of the 7 held-out matches and a blind check on real frames agrees.

## Results
Phase 6 on the realistic games: objective 0.5718 (team 10), 0.5723 (team 7).

| Step | What | Result | Decision | Switch |
|---|---|---|---|---|
| 1 | Simulator ball smoothing σ 7; stoppage ball capped at 1.5 m/s | Smoothing is right in principle (see phase 6 notes); the 1.5 m/s cap makes the stoppage ball slower than the real one (median 2.7 m/s), the cause of false stoppages | Not adopted | `noise.simulate.SIM_SMOOTH_SIGMA = 0.0`; `carry_dead_ball(walking_speed=None)` |
| 2 | Restart cost cap at detected restarts | 0.5717–0.5725 for caps 0–60 vs 0.5718 without; best 4 of 7 matches | Not adopted (neutral) | `lam_dead_out` absent from the grid |
| 3 | 0.4 s kinematic windows | Commented out by the teammate; our 0.5/1 s windows: +0.006 but 4 of 7 matches | Not adopted | commented toggle |
| 4–7 | Ball kink + player impulse touch detector, signed touch state (last touch team, age, decayed signal) in the team models | Touch rate 31/min simulated vs 51/min on BAR-ATH (40% gap); precision 0.88; turnover recall 0.11. Seven-match evaluation: 0.5680 vs 0.5718, better on 3 of 14 games / 2 of 7 matches, mean −0.0038 | Not adopted | `features.TOUCH_FEATURES = False` |
| 5a | Smart Tagging set-piece prior (dead bias in 35 s windows, restart bias at window end) | Prediction-only (simulated games have no tags), so never selected. Windows cover 27.6% of the match; Phase 6 calls a stoppage in only 28% of them (live play after the set piece). On BAR-ATH: 0.65% of frames change; public window 0.600 → 0.600. Blind frame check of all 18 changed stretches: 2–1 clear-cut, 3–5 uncertain, i.e. a draw | Not adopted | `phase1.TAGGING_PRIOR = False` |
| 5b | Vision ball detections as a pitch-boundary input | Models are trained with the column at zero (no detections in simulated games), so it cannot be learned; GPU output would make the file machine-dependent | Not adopted | `__main__.VISION_BOUNDARY = False` |

Tests: 59 passed, 1 skipped. The routing test now checks that touch inputs reach only the team and
ball-gap models, and only when switched on.

## What the evaluations say about the remaining gap
- The bottleneck is still the team call in short possessions (per-frame accuracy ≈ 0.50 in
  possessions under 2 s). Touch evidence helps only when the ball track resolves touches; BAR-ATH's
  smoothed ball and its tracking errors (about 6% of visible frames over 4 m off) blur them.
- The realistic simulator (smoothing, and a stoppage ball moving at real speeds) is the right
  direction but needs calibration that does not create false stoppages before it can be adopted.

## Added tool
`tools/touch_check.py` — touch rate, coverage, precision and turnover recall on BAR-ATH and the
simulated games, with PASS / FAIL targets, before any seven-match run.
