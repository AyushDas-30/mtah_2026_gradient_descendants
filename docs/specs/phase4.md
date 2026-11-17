# Phase 4 spec — gradient-boosted evidence (2026-10-24)

Frozen once delivered.

## Error analysis that drove it (phase 3 model, held-out J03WMX / J03WR9)
| transition | found within 3 s | median timing error |
|---|---|---|
| turnover (team → team) | 47% | 27 frames (~1.1 s) |
| restart (dead → team) | 67% | 1 frame |
| stoppage (team → dead) | 66% | 1 frame |

Restarts go to the other team 63% of the time; the restart team was right 69% of the time.

## Tried and dropped
**Turnover snapping** — moving each predicted turnover to the nearest frame where the closest
team to the ball switches (windows 25–75 frames, control radius 2–3 m): objective 0.4618 →
0.4632 at best on three held-out matches. Late timing is not the main loss; missed turnovers are.

## Change
**Own histogram gradient boosting** (`gbdt.py`) replaces the three logistic evidence models
(dead / team / gap). Same inputs, same decoder. Probe first: scikit-learn
HistGradientBoosting on two held-out matches lifted the objective 0.465 → 0.503 — used only as a
yardstick; the pipeline uses its own implementation (no extra dependency, built for
platform-identical results).

- 32 quantile bins per feature, 150 trees, depth 5, ≥ 200 samples per leaf, learning rate 0.1,
  L2 1.0; trained on every 10th frame (consecutive frames are near-duplicates)
- Level-wise growth with one `bincount` per level for all nodes and features
- Determinism: no randomness or threads; bin edges, probabilities, gains and leaf values rounded
  far above floating-point noise; ties resolve to the lowest (feature, bin)

## Results (simulated IDSSE, leave-one-match-out, mean over control thresholds)
| | objective | 90% CI (T = 1 s) |
|---|---|---|
| **Phase 4, potts(team=7, dead=100)** | **0.4983** | 0.494 – 0.516 |
| Phase 3, potts(team=5, dead=100) | 0.4684 | 0.464 – 0.487 |
| starter baseline | 0.2226 | 0.221 – 0.235 |

BAR-ATH: 31 PASS, 0 FAIL; 93.7% of graded frames agree with phase 3; dead-ball share 0.384
(phase 3: 0.386); possessions per minute 2.90 (2.59). SHA-256 1626426abe63344d (Linux).

## Next candidates
Restart team (63% of restarts change team); Smart Tagging team phases with label-free
reliability; short possessions (< 5 s).
