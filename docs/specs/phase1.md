# Phase 1 spec — decoder (approved 2026-10-24)

Frozen once approved. Changes go in a later phase's spec, not here.

## Goal
Replace majority-vote smoothing with a whole-sequence decoder; upload the first real submission
as the safety file.

## 1. Label harmonisation (decision: option B)
External (IDSSE / DFL) labels flip possession on looser contact than the competition's
"controlled possession" rule. Relabel: inside each alive stretch, a run of the other team counts
as a change of possession only if it lasts ≥ T seconds; shorter runs stay with the previous team.
The first run after a restart is kept. Dead ball untouched. (`possession/relabel.py`)

T is not fixed by the rules, so every configuration is scored against four label variants,
T ∈ {0, 0.5, 1, 2} s, and ranked by the **mean** over them. Models are always trained on
T_TRAIN = 1 s labels — the deployed system cannot know the true T either, so evaluation mimics
that uncertainty. T is never fitted to the public window.

## 2. Per-frame evidence (`features.py`, `emission.py`, `logistic.py`)
- Team: margin between nearest away and nearest home player to the visible ball (tanh at
  1 / 3 / 8 m scales). No team evidence when the ball is not visible.
- Dead ball: distance of the ball outside the pitch, median player speed (1 s / 3 s), ball
  visibility, current ball-gap length, ball speed, nearest-player distance.
- Two L2 logistic models fitted by IRLS (own implementation), trained on simulated IDSSE only.
- Scores = log posterior − log class prior, per frame, for H / A / D.

## 3. Decoders (`decode.py`)
Both run on 5-frame blocks, then every boundary is moved to the best frame within ±5 frames.
- **Potts**: Viterbi with switch penalties λ_team (H↔A) and λ_dead (alive↔dead).
  Grid λ_team ∈ {10…640}, λ_dead ∈ {10…1000}. (The emission scale is redundant with the
  penalties for Potts, so it is fixed at 1.)
- **Semi-Markov**: segment score = α·Σ evidence + β·log-normal duration prior + log transition
  probability; consecutive segments always differ in state. Priors fitted on external labels;
  segments up to 400 s. Grid α ∈ {0.05, 0.1, 0.2, 0.4}, β ∈ {0.5, 1, 2, 4}.

## 4. Selection protocol (`phase1.py`, `python -m possession phase1-select`)
Leave-one-match-out over the 7 IDSSE matches (× 2 simulation seeds). Pooled spellF1 per
configuration and label variant; objective = mean over variants. The best objective wins; a
Potts configuration within 0.005 of the best is preferred (simpler). The starter baseline is
scored identically. 90% bootstrap intervals over possessions at T = 1 s.

## 5. Prediction (`python -m possession predict --team <name>`)
Final models fitted on all simulated games; chosen configuration decodes BAR-ATH's two graded
halves; X elsewhere. Writes `submissions/<team>.csv` and `<team>_model.json` (configuration and
fitted parameters — no match data).

## 6. Done when
1. Decoder beats the baseline on simulated games with non-overlapping 90% intervals.
2. All tests pass.
3. `predict` takes under 10 minutes; two runs give the same SHA-256.
4. The submission validates and is uploaded as the safety file.

## Out of scope (later phases)
Smart Tagging, ball reconstruction, learned multi-feature evidence models, event-based boundary
snapping.
