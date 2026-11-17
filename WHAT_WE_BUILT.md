# What this project contains — team overview

Team **Gradient Descendants** · More than a Hack 2027, Phase 1 (ball possession, BAR–ATH)
Current upload: **Phase 6**, SHA-256 `edde29db13a56e2d` (reproduced on Windows and Linux).

## The problem in one line
For every video frame, output H (Barça ball), A (Athletic ball) or D (dead ball). Ranked on
spellF1 (possessions matched by overlap), tie-break macroF1.

## Core idea
We have no labels for this match (only the public 5 minutes, 17 possessions). So we **train on
7 public Bundesliga matches with possession labels (IDSSE/DFL)**, degraded by our own simulator
to look like this match's imperfect tracking, and we select every design choice by
leave-one-match-out on those simulated games.

## Everything built, by component

### Data and evaluation (Phase 0)
- Own parsers for the EPTS tracking (ATD), Smart Tagging XML and the IDSSE/DFL data; npz cache.
- **Exact re-implementation of the official scorer** (verified identical on 204 cases).
- Found that the tracker's speed field is m/s although the metadata says km/h; speed is now
  computed from positions.

### Noise simulator (Phases 1–2)
- Ball dropout in bursts, hazard measured on BAR-ATH by pitch zone × crowding × ball speed,
  calibrated by bisection to BAR-ATH's 59% ball coverage.
- Player tracklet fragmentation and identity switches at BAR-ATH's coverage.
- Dead ball: DFL freezes the ball during stoppages; we carry it from out-position to restart.
- Referees removed from the external data.

### Label rule
- Competition definition of a change of possession ("under control") implemented as a
  minimum-control rule; training at 1 s, every version scored at 0 / 0.5 / 1 / 2 s because the
  exact rule is unknown.

### Features (Phases 3, 5)
- Ball geometry: nearest player per team, margin, goal-side counts, control runs.
- Team shape: push, spread, width/length, velocity differences, attack direction from the
  goalkeepers.
- Dead-ball signals: player speeds, slow fraction, ball near/over a line, ball visibility gaps.
- **Context windows**: means over the past / next 2 s and 5 s (biggest single gain, +0.09).

### Models (Phase 4)
- **Own histogram gradient-boosted trees**, numpy only, fully deterministic.
- Three models: dead ball; team when the ball is visible; team when the ball is missing.

### Decoder (Phases 1, 6)
- Potts / Viterbi smoothing in **integer fixed-point arithmetic** so the file is bit-identical
  across machines. The team-switch penalty acts as an adaptive minimum-hold rule.
- **Phase 6: stoppage-edge detectors** — two extra models flag the moment play stops / restarts;
  the decoder makes stopping / restarting cheap there.
- Kick-off frames from Smart Tagging; pre-kick-off forced to dead ball.

### Selection and safety
- Leave-one-match-out over 7 matches, 50 decoder settings, objective = mean spellF1 over the 4
  control thresholds; tie band 0.005 (simpler wins) plus a paired check (≥ 6 of 7 matches).
- Shift guard: KS test between simulated and real feature distributions; strongly shifted
  features are excluded automatically.
- Plausibility checks on the output (dead-ball share, possessions per minute, format).

## Results (simulated, leave-one-match-out, spellF1)
| Version | Score |
|---|---|
| Starter baseline | 0.22 |
| Phase 1 | 0.33 |
| Phase 3 team features | 0.47 |
| Phase 4 boosted trees | 0.50 |
| Phase 5 context windows | 0.586 |
| **Phase 6 stoppage edges (upload)** | **0.592** |

Public 5 minutes (sanity only, never tuned on): spellF1 0.600, macroF1 0.727 (best official
baseline 0.42).

## Checked against the real video
The organisers allowed AI review of frames; used only to verify decisions, never to make labels.
- Stoppage calls: right in all 28 clear-cut sampled frames.
- Phase 6 edge changes: right in 9 of 10 judgeable stretches (corners, throw-ins, goal kicks).

## Tried and not adopted (documented on purpose)
| Idea | Result |
|---|---|
| Restart memory (who restarts after a stoppage) | +0.002, inside noise |
| Turnover change detector | +0.005, inside noise |
| 10 s windows, bigger trees | no gain |
| Ball detection from video (YOLO, own tracker) | 97% accurate fills, but no effect on labels; GPU reproducibility risk |
| Short 0.5 / 1 s windows | +0.006, only 4 of 7 matches |
| Touch / impulse features | +0.033 on old sims, +0.004 on realistic sims |
| Realistic stoppage ball in simulator | public macroF1 0.73 → 0.86 but false stoppages across the match |

## Key findings
- The real tracker **smooths the ball** (~σ 7 frames); our old simulation did not, so it was
  optimistic (Phase 6 is ≈ 0.57 on realistic sims). Smoothing is now measured and implemented
  experimentally.
- The remaining gap is **short possessions**: half of all possessions last < 5 s, and team
  accuracy in those < 2 s is ~50%. Perfect team calls would lift the score to ~0.84.

## Next ideas
- Player impulses (receiver decelerates / turns) paired with ball direction changes.
- Relative vectors to the nearest players (direction, closing speed), not only distances.
- Calibrate the stoppage ball more conservatively so the realistic simulator can be adopted.

## Reproduce
See `README.md`: build cache → IDSSE download/build → noise profile → simulate →
`phase1-select` → `predict`. Specs with full results per phase: `docs/specs/phase1.md` … `phase6.md`.
Match data and derived frame-level files are never in this repository (licence).
