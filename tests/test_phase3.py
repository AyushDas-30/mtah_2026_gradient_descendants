"""Phase 3 tests: attack directions, run lengths, gap filling, team-context features."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from possession import features  # noqa: E402
from possession.schema import Match  # noqa: E402


def _two_team_match(T=500, home_left_first=True):
    """Home GK near x=2 in period 1 and near x=103 in period 2; outfield spread around."""
    S = 6
    xy = np.zeros((T, S, 2))
    team = np.array([0, 0, 0, 1, 1, 1], np.int8)
    half = T // 2
    for t0, t1, home_left in ((0, half, home_left_first), (half, T, not home_left_first)):
        gk_home, gk_away = (2.0, 103.0) if home_left else (103.0, 2.0)
        xy[t0:t1, 0, 0], xy[t0:t1, 3, 0] = gk_home, gk_away
        xy[t0:t1, [1, 2, 4, 5], 0] = [40, 60, 45, 65]
    xy[:, :, 1] = 34.0
    ball = np.full((T, 2), np.nan)
    ball[:, 0], ball[:, 1] = 50.0, 34.0
    return Match("t", 25, (105.0, 68.0), xy, np.zeros((T, S), np.float32), team, ball,
                 [(1, 0, half), (2, half, T)])


def test_attack_directions_follow_the_goalkeepers():
    m = _two_team_match()
    d = features.attack_directions(m)
    assert np.all(d[:250] == 1) and np.all(d[250:] == -1)


def test_signed_runs():
    code = np.array([0, 1, 1, 1, -1, -1, 0, 1])
    out = features._signed_runs(code, cap=2)
    assert out.tolist() == [0, 0.5, 1, 1, -0.5, -1, 0, 0.5]


def test_ffill_bfill_respect_periods():
    v = np.array([1., 2, 3, 4, 5, 6, 7, 8])
    valid = np.array([1, 0, 0, 1, 0, 1, 0, 0], bool)
    periods = [(1, 0, 4), (2, 4, 8)]
    f = features._ffill(v, valid, periods)
    b = features._bfill(v, valid, periods)
    assert np.allclose(f[:4], [1, 1, 1, 4]) and np.isnan(f[4]) and np.allclose(f[5:], [6, 6, 6])
    assert np.allclose(b[:4], [1, 4, 4, 4]) and np.allclose(b[4:6], [6, 6]) and np.isnan(b[6:]).all()


def test_compute_has_team_context_and_no_nans():
    m = _two_team_match()
    f = features.compute(m)
    for k in ("push_diff", "spread_lr", "vel_diff", "goalside_diff", "ctrl_run", "last_margin",
              "next_margin", "gap_pos", "margin_past", "margin_next", "ball_att", "near5_diff"):
        assert k in f and f[k].shape == (m.n_frames,) and np.isfinite(f[k]).all(), k
    for X in (features.team_design(f), features.gap_design(f), features.dead_design(f)):
        assert np.isfinite(X).all()


def test_micro_kinematic_windows_are_disabled_by_default():
    m = _two_team_match()
    f = features.compute(m)
    assert not any(name.endswith(("_p10", "_f10")) for name in f)
    for name in ("ctx_speed_med_1s_p2", "ctx_speed_med_1s_f2",
                 "ctx_speed_med_1s_p5", "ctx_speed_med_1s_f5"):
        assert name in f and f[name].shape == (m.n_frames,)
    assert features.team_design(f).shape[1] == len(features.TEAM_SPEC)
    assert features.gap_design(f).shape[1] == len(features.GAP_SPEC)


def test_step4_kink_impulse_and_closing_helpers():
    T = 25
    ball = np.zeros((T, 2), dtype=float)
    ball[:7, 0] = np.linspace(-6.0, 0.0, 7)
    ball[7:13, 1] = np.arange(1.0, 7.0)
    active = np.zeros(T, dtype=bool)
    active[6] = True
    kink = features._ball_angle_kink(ball, active)
    assert kink[6] == 1.0 and not kink[:6].any() and not kink[19:].any()
    assert features._ball_angle_kink(ball, np.zeros(T, dtype=bool))[6] == 0.0
    ball_gap = ball.copy()
    ball_gap[2] = np.nan
    assert features._ball_angle_kink(ball_gap, active)[6] == 0.0

    player = np.zeros((T, 1, 2), dtype=float)
    player[:7, 0, 0] = np.arange(7) * 0.2       # 5 m/s into the touch
    player[7:, 0, 0] = 1.2
    player[7:, 0, 1] = np.arange(1, T - 6) * 0.04  # turns and slows to 1 m/s
    decel, turn = features._player_impulses(player, fps=25)
    assert decel[6, 0] < -2.0 and turn[6, 0] > np.deg2rad(30.0)
    assert not decel[:6].any() and not turn[-6:].any()

    approach = np.zeros((T, 1, 2), dtype=float)
    approach[:, 0, 0] = 4.0 - np.arange(T) * 0.1
    closing = features._closing_velocity(np.zeros((T, 2)), approach, fps=25)
    assert closing[6, 0] > 0.0 and closing[0, 0] == 0.0 and closing[-1, 0] == 0.0


def test_ball_kink_is_an_isolated_finite_feature():
    f = features.compute(_two_team_match())
    assert "ball_kink_6f" in f
    assert np.isfinite(f["ball_kink_6f"]).all()
    for spec in (features.TEAM_SPEC, features.GAP_SPEC, features.DEAD_SPEC):
        assert all("ball_kink_6f" not in sources for sources, _ in spec)


def test_step6_signed_touch_persistence_and_resets():
    T, fps = 160, 25
    slot_team = np.array([0, 1], dtype=np.int8)
    present = np.ones(T, dtype=bool)
    distances = np.full((T, 2), 10.0)
    distances[7:10, 0] = 1.0
    distances[7:10, 1] = 2.0
    kink = np.zeros(T)
    kink[9] = 1.0
    touch = features._signed_touch(kink, distances, slot_team, present)
    assert touch[9] == 1.0

    # Too brief or too contested proximity is suppressed.
    distances[7:9, 0] = 10.0
    assert features._signed_touch(kink, distances, slot_team, present)[9] == 0.0
    distances[7:10, 0], distances[7:10, 1] = 1.0, 1.2
    assert features._signed_touch(kink, distances, slot_team, present)[9] == 0.0

    touch = np.zeros(T)
    touch[10], touch[120] = 1.0, -1.0
    present[40:92] = False  # >2s lost-ball run resets touch memory
    last, age, signal = features._last_touch_state(
        touch, [(1, 0, 100), (2, 100, T)], fps, present)
    assert last[10] == 1.0 and age[10] == 0.0 and signal[10] == 1.0
    assert last[39] == 1.0 and last[40] == 1.0
    assert last[92] == 0.0  # lost-ball run exceeded 2 seconds
    assert last[120] == -1.0 and signal[120] == -1.0
    assert not last[100:120].any()  # period boundary clears the previous state
    no_new_touch, _, _ = features._last_touch_state(
        np.eye(1, T, 10, dtype=float)[0], [(1, 0, T)], fps, np.ones(T, dtype=bool))
    assert no_new_touch[136] == 0.0  # carry is capped at 5 seconds

    f = features.compute(_two_team_match())
    for key in ("touch_team", "last_touch_team", "time_since_touch", "touch_signal"):
        assert key in f and f[key].shape == (len(f["ball_known"]),)
        assert np.isfinite(f[key]).all()
    # touch inputs are routed to the team / ball-gap models only, and only when switched on
    assert {s for sources, _ in features.TOUCH_SPEC for s in sources} <= {
        "touch_signal", "last_touch_team", "time_since_touch"}
    in_team = any("touch_signal" in sources for sources, _ in features.TEAM_SPEC)
    in_gap = any("touch_signal" in sources for sources, _ in features.GAP_SPEC)
    assert in_team == in_gap == features.TOUCH_FEATURES
    assert all(all("touch_signal" not in sources for sources, _ in spec)
               for spec in (features.DEAD_SPEC, features.CHANGE_SPEC))


def test_step7_touch_gate_accepts_either_team_and_tunable_kink():
    T = 12
    teams = np.array([0, 1], dtype=np.int8)
    present = np.ones(T, dtype=bool)
    distances = np.full((T, 2), 8.0)
    distances[2:5, 0] = 1.0
    distances[2:5, 1] = 3.0
    kink = np.zeros(T)
    kink[4] = 1.0
    assert features._signed_touch(kink, distances, teams, present)[4] == 1.0
    distances[2:5, 0], distances[2:5, 1] = 3.0, 1.0
    assert features._signed_touch(kink, distances, teams, present)[4] == -1.0

    ball = np.column_stack([np.arange(30, dtype=float), np.zeros(30)])
    ball[15:, 0] = 15.0
    ball[15:, 1] = np.arange(15, dtype=float)
    assert features._ball_angle_kink(ball, angle_deg=45.0)[15] == 1.0
