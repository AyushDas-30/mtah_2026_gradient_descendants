# Phase 2 spec — dead ball and domain match (2026-10-24)

Frozen once delivered. Written after inspecting the tactical tracking, metadata and Smart Tagging
directly (organisers confirmed that only the video files are restricted from LLM use).

## Findings that drove this phase
- The tracker's speed field is **m/s** although the metadata declares `speed_kph` (raw field /
  position-derived speed = 0.994). Phase 1.0 divided by 3.6 and labelled almost the whole match
  dead. Since 1.1 every speed is recomputed from positions, identically for all sources.
- DFL (IDSSE) freezes the ball for the whole of every stoppage (step speed exactly 0 on 100% of
  dead frames, 0% of live ones) and parks it beyond the line (38% of ball frames near/outside a
  line, 89% of them dead). BAR-ATH's tracker sees the ball carried at walking pace; only 9% of its
  ball frames are near a line. Models trained on raw IDSSE learn the artefact.
- BAR-ATH's tracker loses the ball 4x more often within 2 m of a line (hazard 0.0050 vs 0.0012).
- Tracker noise is negligible (~3 mm per frame): adding jitter is not the fix.
- The second half is graded from 110,808 but kicks off later; Phase 1.1 called 73% of those
  2,934 dead frames live.

## Changes
1. **Carried dead ball** (`noise/simulate.py: carry_dead_ball`): in simulated games the ball moves
   smoothly (smoothstep) from the last live position to the next one through every stoppage.
2. **Zone-aware dropout** (`noise/profile.py`): hazard table zone (inside / within 2 m of a line or
   outside) × crowding × ball speed, cells shrunk towards the zone mean (prior weight 200 frames).
3. **Pre-kick-off dead ball** (`kickoff.py`): kick-off = Smart Tagging "Kick Off" SET PIECES start
   + 3.5 s (the tag's documented lead). Graded frames of a period before its first kick-off are
   constrained to D inside the decoder. A formation detector was tried and dropped: warm-up balls
   and a metres-off ball at the kick make it unreliable on this tracking.
4. **Shift guard** (`phase1.shift_guard`): features with two-sample KS ≥ 0.30 between simulated
   games and BAR-ATH (graded frames, unlabelled) are removed from both models.
5. **Features**: ball gaps ≤ 0.6 s interpolated (both sources); new dead-ball features
   `ball_line_dist`, `ball_centre_dist`, `speed_med_10s`, `slow_frac`, `speed_trend`.
6. **Resumable selection**: `phase1-select --folds ...` saves each fold; `--merge` combines.

## Results (simulated IDSSE, leave-one-match-out, mean over control thresholds)
| | objective | 90% CI (T = 1 s) |
|---|---|---|
| Phase 2, chosen potts(team=20, dead=100) | 0.3234 | 0.320 – 0.340 |
| Phase 1 features, same new simulated games | 0.3225 | 0.319 – 0.338 |
| starter baseline | 0.2226 | 0.221 – 0.235 |

The new features are neutral on simulation; the phase's value is in transfer: every feature but
`ball_speed` (KS 0.28, moderate) now passes the shift check, none is excluded, and the
pre-kick-off stretch is 100% dead. BAR-ATH predict: 21 PASS, 0 FAIL.

## Error analysis carried into Phase 3
Dead ball is largely solved (95% of dead frames). The team call is the bottleneck: 81% correct
with the ball known, 64% without it; possessions under 5 s (half of all) are recovered 6–10%.
