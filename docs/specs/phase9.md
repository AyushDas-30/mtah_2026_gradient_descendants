# Phase 9 spec — the simulator sees the ball the way BAR-ATH's tracker does (2026-11-02)

Frozen once delivered.

## Outcome
**Adopted.** The simulated training games now lose the ball where the real tracker loses it:
rarely in live play, mostly in stoppages, in long gaps that run past restarts. The dead ball is
carried to the restart spot and then rests there, as on the real pitch. The models and the
decoder are unchanged; they are retrained and reselected on the new games.

New submission: **SHA-256 `28f7229f96045c50`**, 32 PASS, 0 FAIL, chosen decoder
`potts_e(team=10, dead=30, gamma=64)`.

## The finding
| | live play: ball visible | stoppages: ball visible |
|---|---|---|
| BAR-ATH, Phase 6 labels (code output) | 0.87 | 0.18 |
| BAR-ATH, public window (published labels, 4 stoppages) | 0.94 | 0.66 |
| Simulated games, phases 1–7 | 0.65 | 0.50 |
| Simulated games, phase 9 | 0.88 | 0.27–0.35 |

The whole match has only 149 ball gaps (median 11 s). The old simulator placed its dropouts by
crowding × zone × ball speed but not by game state, so live play in training was far blinder
than in the real match, and a lost ball said little about whether play was on. When a stoppage
ends with the ball lost, the gap usually runs on into live play (median 5 frames, 75th
percentile 1.4 s, 90th 4 s); the old games always showed the ball at the restart.

The visible stoppage ball was also wrong. The old one was carried slowly all the way (median
0.3 m/s). The real one is fetched, then placed or held (median 1.5 m/s, often static, bursts to
9 m/s).

## What changed (code)
- `possession/noise/state.py` (new)
  - `measure(match, codes)`, from a stage-1 submission:
    - every stoppage's ball-visibility pattern (100 bins);
    - live-play gap lengths;
    - overhangs: how long a gap continues past a restart, or starts before a stop, when the
      stoppage ends or starts with the ball lost.
  - `transport_rest`: the dead ball is carried from where it went out to the restart spot
    (median 4 m/s, done by 70% of the stoppage), then rests there with short handling moves
    (0.3 per second). Fitted to BAR-ATH's visible dead-ball speed percentiles.
  - `degrade_ball`, applied in this order:
    - tracker smoothing (σ 7 frames);
    - a real visibility pattern for every stoppage;
    - overhangs into the live play around it;
    - live-play gaps at a rate fitted to 0.88 live coverage.
- `possession/noise/simulate.py`: `simulate(..., state=...)`. Players are simulated exactly as
  before (same draws).
- `possession/__main__.py`:
  - `state-profile --labels <stage-1 csv>`;
  - `simulate --world state` (default);
  - `--world phase6` regenerates the Phase 6 games bit for bit.
- `possession/emission.py`: the ball-gap model stays fittable with no ball-gap rows. This is
  only reachable in experiments and has no effect on any real run.
- `tests/test_phase9.py` (5 tests). Full suite: 64 passed, 1 skipped.

The state profile is measured from **code output** (the Phase 6 submission), never from hand
labels. It stays in the local cache with the other measurements. It is measured once from
stage 1 and not iterated: re-measuring from a phase 9 file would feed the new model's own habits
back into the simulator.

## Evidence
**1. Seven held-out matches in the realistic world.** Both models were tested on the phase 9
games, with the decoder reselected per fold on a 64-setting grid.

| held-out match | trained on Phase 6 games | trained on phase 9 games | change |
|---|---|---|---|
| J03WMX | 0.5403 | 0.5627 | +0.022 |
| J03WN1 | 0.5220 | 0.5651 | +0.043 |
| J03WOH | 0.5705 | 0.6220 | +0.052 |
| J03WOY | 0.5194 | 0.5548 | +0.035 |
| J03WPY | 0.5692 | 0.6119 | +0.043 |
| J03WQQ | 0.5669 | 0.6122 | +0.045 |
| J03WR9 | 0.6106 | 0.6493 | +0.039 |
| **mean** | **0.5570** | **0.5969** | **+0.040, 7 of 7** |

The same comparison in the intermediate worlds also came out 7 of 7:

| world | Phase 6 games | new games | change |
|---|---|---|---|
| state-aware dropout only | 0.6142 (smoothed old games) | 0.6333 | +0.019 |
| plus carried-then-resting dead ball | 0.5720 | 0.6200 | +0.048 |

**2. Feature realism, label-free.** Kolmogorov–Smirnov distance between BAR-ATH and the
simulated games over the 23 guarded features:

| | mean KS | max KS | ball speed KS | features ≥ 0.15 |
|---|---|---|---|---|
| Phase 6 games | 0.080 | 0.284 | 0.284 | 1 |
| Phase 9 games | 0.062 | 0.145 | — | 0 |

**3. Blind real-frame audits.** Frames were judged before looking at which version says what;
the public window is excluded. Each audit compares five frames (−2 s to +2 s) around a moment
where versions disagree.

| audit | how sampled | Phase 6 right | phase 9 right |
|---|---|---|---|
| 1 | disagreements with the first state-aware model | 5 / 14 | 8 / 14 |
| 2 | disagreements with the handling variant | 8 / 19 | 10 / 19 |
| 3 | frame-weighted random disagreements with the carried-ball model | 12 / 20 | 14 / 20 |
| **all** | | **25 / 53** | **32 / 53** |

Where exactly one of the two is right: phase 9 12, Phase 6 5 (one-sided sign test p = 0.07).
The gain is the same direction in every audit (4–1, 4–2, 4–2).

**4. Public window.** Sanity only, never tuned on.

| | spellF1 | macroF1 |
|---|---|---|
| Phase 6 | 0.600 | 0.727 |
| Phase 9 | 0.562 | 0.826 |

Phase 9 finds the 20 s and 5.5 s stoppages Phase 6 missed and catches half of the 2.8 s Athletic spell at 15 s.
It loses two spell matches to short dead stretches inside long possessions. With 17 possessions,
one spell moves spellF1 by about 0.03. Other decoder settings within the selection tie band score
0.62–0.63 there, but choosing among them on this window would be tuning on it, so the selected
setting stands.

**5. Plausibility.**

| | dead-ball share | possessions per minute |
|---|---|---|
| Phase 6 | 0.399 | 2.63 |
| Phase 9 | 0.451 | 2.57 |

Phase 9 sits inside the IDSSE band (0.41–0.52).

## Remaining weaknesses seen in the audits
- Long ball gaps that contain both a stoppage and the live play after it are now called dead a
  little too long (a corner and its flight, a goalkeeper's slow build-up).
- A ball held for a throw-in or placed for a free kick is sometimes still called live.

## Reproduce
`README.md`, "Regenerating the submission": stage 1 (Phase 6, `edde29db13a56e2d`), then
`state-profile`, `simulate`, `phase1-select` (31 min on 2 cores), `predict`. The container run
gave `28f7229f96045c50`; the same file is expected on Windows, as with Phase 6.
