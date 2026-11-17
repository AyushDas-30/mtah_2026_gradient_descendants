# Phase 3 spec — the team signal (2026-10-24)

Frozen once delivered.

## Why
Phase 2 error analysis: dead ball was largely solved (95% of dead frames), but the team call was
the bottleneck — 81% correct with the ball known, 64% without it (30% of live frames). Possessions
under 5 s, half of all possessions, were recovered 6–10% of the time.

## Reproducibility (2.1, shipped before this phase)
Decoding runs in integer fixed point (1/1000 nat); fitted logistic parameters are rounded to 1e-9
and per-frame scores to 1e-6; IRLS converges to 1e-12. Verified: Linux / numpy 2.2.6 and Windows /
numpy 2.5.4 produce the identical file (SHA-256 4135d596… for phase 2.1).

## Changes
1. **Attack directions without metadata** (`features.attack_directions`): per period, the team that
   owns the deepest player near x = 0 more often defends that end. Agrees with the metadata on
   BAR-ATH (93% / 86% of sampled frames in the two halves); 95–98% clear on simulated games.
2. **Team-context features** (`features.team_context`), identical for all sources:
   push_diff, spread_lr, width_diff, length_diff, vel_diff (slot-wise forward velocity, robust to
   tracking dropouts), goalside_diff, near5_diff, ball_att, ctrl_run (control persistence),
   margin_past / margin_next, last_margin / next_margin / gap_pos (bridging ball gaps).
3. **Gap model** (`emission.py`): a third logistic model gives P(home) on live frames where the ball
   is unknown, from team shape / movement and the last / next ball sighting; previously those
   frames had no team evidence.
4. **Grid**: Potts team penalty extended down to 5 (the optimum sat at the old edge, 10). The
   objective is flat over 5–10 (0.4681–0.4684), so the edge hides nothing.

## Results (simulated IDSSE, leave-one-match-out, mean over control thresholds)
| | objective | 90% CI (T = 1 s) |
|---|---|---|
| Phase 3, chosen potts(team=5, dead=100) | **0.4684** | 0.464 – 0.487 |
| Phase 2, potts(team=20, dead=100) | 0.3232 | 0.319 – 0.339 |
| starter baseline | 0.2226 | 0.221 – 0.235 |

Held-out diagnostics, same decoder settings as phase 2 (J03WMX, J03WR9):
team accuracy 0.809 → 0.883 (ball known), 0.644 → 0.829 (ball unknown); recall of possessions
10–20 s 0.50 → 0.74, > 20 s 0.66 → 0.85.

BAR-ATH: all 24 shift checks pass except `ball_speed` (moderate, kept); nothing excluded;
31 PASS, 0 FAIL; SHA-256 5eaf2eb3d5545690.

## Not done here (Phase 4 candidates)
Smart Tagging team phases as evidence (needs reliability estimated without labels — simulated
games carry no tags); event-based boundary snapping; short possessions (< 5 s) remain the
weakest bucket.
