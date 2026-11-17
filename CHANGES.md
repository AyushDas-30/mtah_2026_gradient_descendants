# Code Changes Log

This file records actual code changes applied to this repository by the agent as improvements are implemented. Each entry logs the target file, exact lines added/modified, the role of the change, and test verification results.

---

## Baseline State
- **Status:** Phase 6 (Baseline Upload)
- **Baseline SHA-256:** `edde29db13a56e2d`
- **Scores:** `spellF1` 0.592 (simulated IDSSE games), 0.600 (public 5-minute slice)
- **State:** No code modifications applied yet. All existing baseline files, models, and tests remain intact.

---

## Change Log

*(Entries will be appended here by the agent as each step from `PROPOSED_IMPROVEMENTS.md` is implemented.)*

### Log Entry Template
```markdown
### [YYYY-MM-DD] Step X: <Step Title>
- **File Modified:** `<path/to/file.py>`
- **Lines Added/Modified:** `L<start>-L<end>`
- **Role / Purpose:** <Brief explanation of what the code does>
- **Backwards Compatibility:** <Confirmation that existing tests and defaults remain unbroken>
- **Verification:** <Command run and result, e.g. pytest output>
```

### [2026-10-28] Step 1: Simulator Smoothing Calibration & Conservative Stoppage Ball
- **Files Modified:** `possession/noise/simulate.py`; `possession/__main__.py`
- **Lines Added/Modified:** `possession/noise/simulate.py` L32, L45-L46, L74, L80-L111, L114, L124-L125, L158-L178; `possession/__main__.py` L232
- **Role / Purpose:** Added optional `smooth_sigma=0.0` to simulation and ball degradation, plus NaN-aware Gaussian smoothing within visible runs and periods. Calibrated CLI simulations use sigma 7 frames. Dead-ball carry now moves smoothly toward the restart with peak speed capped at 1.5 m/s and stops at the reachable point when the restart is farther away.
- **Backwards Compatibility:** `simulate()` and `_degrade_ball()` retain no-smoothing behavior by default; new metadata records the selected sigma. Existing function call positions and output formats are unchanged.
- **Verification:** `python -m compileall -q possession` passed. Manual synthetic smoke checks for bounded dead-ball movement and smoothing passed. `python -m pytest -q tests` could not run because `pytest` is not installed in the active Python environment (`No module named pytest`). Full simulated-data KS calibration was not run because no local IDSSE cache/profile is present.

### [2026-10-28] Step 2: Asymmetric Decoder Restart Penalties
- **Files Modified:** `possession/decode.py`; `tests/test_phase6.py`
- **Lines Added/Modified:** `possession/decode.py` L102-L108, L112-L116, L263-L275; `tests/test_phase6.py` L49-L57
- **Role / Purpose:** Added opt-in `lam_dead_out` for the edge-aware Potts decoder. At blocks where restart log-odds are positive, the D-to-H/A transition cost is capped at this value; H-to-A/A-to-H and alive-to-dead costs are unchanged.
- **Backwards Compatibility:** `lam_dead_out=None` is the default and leaves the existing Phase 6 decoding path unchanged. Existing positional arguments and output format remain unchanged.
- **Verification:** `python -m compileall -q possession tests` passed. A synthetic fixed-point decode confirmed that `None` matches the original path and the enabled cap admits a restart possession at a flagged boundary. `python -m pytest -q tests` could not run because `pytest` is not installed in the active Python environment (`No module named pytest`). Full leave-one-match-out evaluation and the 6-of-7 paired check were not run because no local IDSSE cache/simulations are available.

### [2026-10-29] Step 3: Targeted Micro-Context Windows for Kinematic Signals
- **Files Modified:** `possession/features.py`; `tests/test_phase3.py`
- **Lines Added/Modified:** `possession/features.py` L313, L335-L345, L384-L386, L392-L408, L445, L450-L451, L524-L531; `tests/test_phase3.py` L63-L77
- **Role / Purpose:** Added ball acceleration and heading-change signals from adjacent known ball positions, then computed period-local past/future means over 10 frames for those signals only. Appended four scaled micro-window columns to team and ball-gap designs; existing macro context windows and all earlier feature columns retain their order.
- **Backwards Compatibility:** Existing feature definitions, 2 s/5 s windows, and prior design-column ordering are unchanged; new columns are appended to the team and ball-gap designs only.
- **Verification:** `python -m compileall -q possession tests` passed. Direct execution of the new period-boundary test and existing feature-computation test passed. `python -m pytest -q tests` could not run because `pytest` is not installed in the active Python environment (`No module named pytest`). Leave-one-match-out evaluation and the 6-of-7 paired check were not run because no local IDSSE cache/simulations are available.

### [2026-10-29] Step 4: Kinematic Impulse Features & Relative Closing Velocity
- **Files Modified:** `possession/features.py`; `tests/test_phase3.py`
- **Lines Added/Modified:** `possession/features.py` L43, L122-L199, L409-L438, L448; `tests/test_phase3.py` L80-L113
- **Role / Purpose:** Added `KINK_SPAN=6`, a ball kink detector using inbound/outbound vectors at ±6 frames, player deceleration and turn measurements, and signed player-to-ball closing speed. `ball_kink_6f` is set only for a >60° kink near an opponent track (≤2 m) with deceleration below −2 m/s², a >30° receiver turn, and positive closing speed. Missing samples, sequence boundaries, and period edges suppress the flag or pad helper outputs with zero.
- **Backwards Compatibility:** Added only the isolated `ball_kink_6f` feature key. Existing Phase 0–6 design matrices and feature keys are unchanged; `SPEED_SPAN` remains 5.
- **Verification:** `python -m compileall -q possession tests` passed. Direct execution of synthetic kink, missing-gap, impulse, closing-speed, and isolated-feature checks passed; prior feature and micro-window checks also passed. `python -m pytest -q tests` could not run because `pytest` is not installed in the active Python environment (`No module named pytest`). Team-accuracy evaluation on sub-2-second possessions was not run because the local IDSSE cache/simulations are unavailable.

### [2026-10-29] Step 5: Soft Pitch Boundary Features & Weak Tactical Set-Piece Priors
- **Files Modified:** `possession/features.py`; `possession/emission.py`; `possession/phase1.py`; `possession/__main__.py`; `tests/test_phase5.py`
- **Lines Added/Modified:** `possession/features.py` L185-L196, L364-L387, L464, L530; `possession/emission.py` L110-L119, L128-L142, L153-L176; `possession/phase1.py` L269-L298; `possession/__main__.py` L334-L355; `tests/test_phase5.py` L1-L45
- **Role / Purpose:** Added the continuous normalized `max(0, -x, x-1, -y, y-1)` boundary margin to the dead-ball design. Prediction projects the best available ball candidate through the per-frame homography when both homography and detector caches exist; missing projections and missing files produce zero boundary margin. Smart Tagging `SET PIECES` intervals apply a +0.65-nat dead-ball score bias, and a linearly tapered +0.65-nat maximum restart-edge bias around tagged interval ends (±2 seconds). These are soft additions; no tagged frame is forced by the new prior. Added tests for the margin, neutral defaults, taper, score bias, and invalid shapes.
- **Backwards Compatibility:** `features.compute(match)` still works and returns zero boundary margin when projection coordinates are absent. New `EmissionModel` arguments are optional; omitted priors preserve existing scores and edge logits. Detector/homography caches are optional at prediction time. Existing pre-kickoff behavior is unchanged.
- **Verification:** `python -m compileall -q possession tests` passed. Direct smoke checks passed for the exact boundary formula, zero/missing defaults, tagging window and restart taper, bounded prior, and emission-model serialization method availability. `python -m pytest -q tests` could not run because `pytest` is not installed (`No module named pytest`). Full stoppage-transition macroF1 and aerial-cross evaluation could not be run because local IDSSE simulation data and the optional homography/detection caches are unavailable.

### [2026-10-29] Step 3 Instruction Update: Micro-Window Experimental Toggle
- **Files Modified:** `possession/features.py`; `tests/test_phase3.py`
- **Lines Added/Modified:** `possession/features.py` L326, L348-L372, L438-L439, L483-L485, L557-L565; `tests/test_phase3.py` L63-L71
- **Role / Purpose:** Applied the updated Step 3 instruction. The established 2s/5s context remains active; the 10-frame micro-window helper, call sites, and team/gap design columns are now a commented experimental toggle. Added a default-mode test that confirms no 10-frame feature keys enter model inputs while 2s/5s windows remain present.
- **Backwards Compatibility:** Default feature design matrices no longer include the Step 3 micro-window columns, restoring the standard macro-window configuration. Step 4 kink/impulse signals and Step 5 boundary/tagging changes remain separate.
- **Verification:** `python -m compileall -q possession tests` passed. Seven direct Phase 3/4 feature checks passed, including the new default-toggle assertion. `python -m pytest -q tests` could not run because `pytest` is not installed (`No module named pytest`). Full IDSSE leave-one-match-out evaluation remains unavailable because local simulation data is absent.

### [2026-10-30] Step 6: Signed Touch State & Possession Persistence Features
- **Files Modified:** `possession/features.py`; `tests/test_phase3.py`
- **Lines Added/Modified:** `possession/features.py` L46-L49, L157-L218, L534-L550, L628-L635; `tests/test_phase3.py` L110-L148
- **Role / Purpose:** Added signed touch assignment using the nearest team's clear distance margin, requiring the same nearest player to remain within 2m for three consecutive frames. Ambiguous contests below 0.5m separation are suppressed. Added forward-filled last-touch team, elapsed time, and exponentially decayed touch signal. Touch memory resets at period boundaries, after more than 2 seconds without a visible ball, and after the 5-second carry cap. Appended the three persistence inputs to team and ball-gap designs only; existing dead-ball/change designs and the Step 4 kink key are untouched.
- **Backwards Compatibility:** Existing `compute(match)` call signature and prior keys are retained; four new keys are additive. Model design change is limited to the team and ball-gap models where touch ownership is relevant.
- **Verification:** `python -m compileall -q possession tests` passed. Direct synthetic checks passed for signed owner, three-frame proximity, contested suppression, decay, long-gap/period resets, carry cap, finite new keys, and design routing. `python -m pytest -q tests` could not run because `pytest` is not installed. Full IDSSE accuracy/paired evaluation was not run because local simulation data is unavailable.
- **Step 6 review follow-up (2026-10-30):** Removed the unused `fps` argument from `_signed_touch()` and updated its call and unit checks. Rechecked the touch-state path; feature arrays are computed once and then reused in the model design specs, with no duplicate computation found.

### [2026-10-31] Step 7: Touch Detector Recall & Team Ownership Correction
- **Files Modified:** `possession/features.py`; `tests/test_phase3.py`
- **Lines Added/Modified:** `possession/features.py` L50-L52, L129-L156, L527-L538; `tests/test_phase3.py` L152-L168
- **Role / Purpose:** Removed the opposite-team-only eligibility gate so either team can register a signed touch. Kinematic corroboration now accepts deceleration, turn, or closing-speed evidence alongside proximity instead of requiring every cue. Added configurable kink, deceleration, and turn thresholds while retaining sustained same-player proximity and the minimum team-separation guard.
- **Backwards Compatibility:** Existing helper call patterns remain valid; `_ball_angle_kink()` defaults to its original 60-degree threshold. New thresholds affect only touch detection in `compute()`.
- **Verification:** `python -m compileall -q possession tests` passed. Focused Step 6/7 synthetic checks passed for both signed teams, proximity and contested guards, resets, carry cap, and configurable kink angle. Full precision/recall and paired evaluation were not run because local IDSSE data is unavailable.

### [2026-11-01] Phase 7: Merge and evaluation of Steps 1-7
- **Files Modified:** `possession/features.py` (`TOUCH_FEATURES` switch); `possession/phase1.py` (`TAGGING_PRIOR` switch); `possession/__main__.py` (`VISION_BOUNDARY` switch, `SIM_SMOOTH_SIGMA` from the simulator); `possession/noise/simulate.py` (`SIM_SMOOTH_SIGMA = 0.0`, `carry_dead_ball(walking_speed=None)` restores the Phase 6 carry exactly); `tests/test_phase3.py` (touch routing follows the switch); `tools/touch_check.py` (new); `docs/specs/phase7.md` (new).
- **Role / Purpose:** Every step is kept, but each new behaviour is behind an off-by-default switch because none passed the adoption rule (6 of 7 held-out matches on the smoothed simulated games, plus a blind real-frame check). Results per step are in `docs/specs/phase7.md`.
- **Backwards Compatibility:** Defaults reproduce Phase 6 exactly: regenerated simulated game identical to the Phase 6 one; `predict` gives SHA-256 `edde29db13a56e2d`, 32 PASS, 0 FAIL.
- **Verification:** `python -m pytest -q tests`: 59 passed, 1 skipped. Seven-match evaluations run on the realistic simulated games (touch features: 0.5680 vs 0.5718, 2 of 7 matches; restart cap: neutral, at most 4 of 7). Tagging prior: blind frame check of all 18 changed stretches, a draw.

### [2026-11-01] Phase 8: Short-possession features (note, not adopted)
- **Files Added:** `docs/specs/phase8.md`; `tools/recv.py`; `tools/experiments/`. Receiver / co-movement features, training augmentation and a score ensemble: none passed 6 of 7 matches.

### [2026-11-01] Phase 9: Game-state-aware simulator (adopted)
- **Files Added:** `possession/noise/state.py`; `tests/test_phase9.py`; `docs/specs/phase9.md`.
- **Files Modified:** `possession/noise/simulate.py` (`simulate(..., state=...)`); `possession/__main__.py` (`state-profile` command, `simulate --world {phase6,state}`, default `state`); `possession/emission.py` (ball-gap model stays fittable with no gap rows; unreachable in real runs); `README.md` (two-stage pipeline).
- **Role / Purpose:** On BAR-ATH the tracker keeps the ball in ~87% of live play and loses it mostly in stoppages, in long gaps that run past restarts; the old simulator dropped it ~35% of live play and ~50% of stoppages. The simulator now uses stoppage visibility patterns, live-play gaps and overhangs measured from the stage-1 (Phase 6) submission, a tracker-smoothed ball, and a dead ball carried to the restart spot then resting. Models and decoder are unchanged, retrained and reselected.
- **Backwards Compatibility:** `simulate --world phase6` regenerates the Phase 6 simulated games bit for bit; the Phase 6 file (`edde29db13a56e2d`) is reproduced from them (verified in the container).
- **Verification:** `python -m pytest -q tests` 64 passed, 1 skipped. Seven-match test in the realistic world: better on 7 of 7 matches (mean +0.040). Blind real-frame audits of the final file: right on 32 vs 25 of 53 firm judgements (12 vs 5 where they differ). Final SHA-256 `28f7229f96045c50`. Details and hashes in `docs/specs/phase9.md`.
