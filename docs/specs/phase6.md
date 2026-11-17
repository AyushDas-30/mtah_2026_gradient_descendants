# Phase 6 spec — stoppage-edge detectors (2026-10-25)

Frozen once delivered.

## Tried first and not adopted: a ball track from the video
The organisers allow LLMs to look at the video frames. A pretrained detector (YOLO11m, COCO
"sports ball", 640 px tiles, run locally on the GPU) was run on every graded frame and its
candidates were linked over time (static spots and resting objects masked; tracklets scored by a
boosted model trained against the tracker's ball). Checked on real frames:
- gap fills: 31 of 32 sampled fills show the real ball; they cover 9.3% of the tracker's gaps
  (ball known on 59.3% → 63.1% of graded frames)
- where the detector and the tracker disagree by > 4 m, the frames mostly show the ball at the
  detection; the homography is correct (projected players land on the players)
- effect on the possession labels: 1.3% (fills) / 1.9% (fills + corrections) of graded frames
  change; on the 24 longest changed stretches judged from the frames the two versions tie
  (3–3 clear-cut cases, 6–3 uncertain ones); public window 0.600 → 0.600 / 0.538 (one spell)
Not adopted: no measurable gain, and a GPU step would put exact reproducibility at risk. The code
is kept as an optional tool (`vision-detect`, `vision-validate`); the pipeline does not use it.

Hand-labelling random clips from still frames was also stopped after calibration: the moments
that decide spellF1 (aerial duels, the instant a throw-in is taken) cannot be read reliably at
this resolution. Real frames are used instead for targeted checks where the answer is visible.

## Diagnosis
On the simulated games (leave-one-match-out, Phase 5 scores reproduced exactly: 0.5855):
- an oracle that knows where the true stoppage edges are (decoder costs drop there) reaches
  0.6076; perfect dead / alive frames 0.6358
- long stoppages are found (10–20 s: 91%, > 20 s: 98%), short ones are not (< 3 s: 17%,
  3–6 s: 49%, 6–10 s: 65%); found edges are tight (median offset 0.04 s)
Real frames (80 random frames, model blind): every one of 28 clear-cut cases correct; the errors
sit at the edges of stoppages.

## Change
**Edge detectors** (`EmissionModel(edges=True)`): two more boosted models on the dead-model
features — P(a dead → alive edge within ± 5 frames) and P(an alive → dead edge within ± 5
frames). Training rows: every 10th graded frame plus every positive frame. Held-out AUC 0.94
(restart) and 0.93 (stop).

**Decoder `potts_e`**: Potts where the cost of entering a stoppage at block t is
lam_dead − γ · (stop log-odds), of leaving one lam_dead − γ · (restart log-odds), clipped to
[0, 3 lam_dead]; integer arithmetic as before. Grid adds team {5,7,10} × dead {30,100} ×
γ {16,32,64} (50 configurations in total).

**Selection rule**: within the tie band the simplest family wins (Potts, Potts with edges,
restart memory, semi-Markov); a more complex family also wins when its best configuration beats
the simple choice on ≥ 6 of the 7 held-out matches (paired sign test). Not needed this time: the
gain exceeds the band on its own.

## Results (simulated IDSSE, leave-one-match-out, mean over control thresholds)
| | objective | 90% CI (T = 1 s) |
|---|---|---|
| **Phase 6, potts_e(team=10, dead=100, γ=32)** | **0.5916** | 0.591 – 0.614 |
| Phase 5, potts(team=7, dead=100) | 0.5855 | 0.585 – 0.608 |
| starter baseline | 0.2226 | 0.221 – 0.235 |

Probe before the full run (team 7, dead 100, γ 32 vs Phase 5): better on 13 of 14 simulated
games, mean +0.0053.

Real frames: 24 random stretches where the edge decoder changes BAR-ATH, judged blind. In the 10
that could be judged it was right 9 times — every clear-cut case (a corner being set up, two
throw-ins with the ball held overhead, a keeper about to take a goal kick) was called possession
by Phase 5 and stoppage by Phase 6.

BAR-ATH: 32 PASS, 0 FAIL; 98.5% of graded frames agree with Phase 5; dead-ball share 0.399
(0.394); possessions per minute 2.63 (2.83). Public window, sanity only and never used for
selection: spellF1 0.600 (0.600), macroF1 0.727 (0.726). SHA-256 edde29db13a56e2d (Linux,
numpy 2.2.6).
