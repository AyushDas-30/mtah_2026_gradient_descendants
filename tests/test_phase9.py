"""Phase 9: game-state-aware ball dropout and carried-then-resting dead ball in the simulator."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from possession.noise import simulate as noise_sim  # noqa: E402
from possession.noise import state  # noqa: E402
from possession.schema import A, D, H, X, Match  # noqa: E402


def _match(T=6000, seg=500, seed=0, labels=True):
    """Alternating live / dead stretches; the clean ball walks with player 1 and is frozen when dead
    (as in the external data)."""
    rng = np.random.default_rng(seed)
    S = 6
    y = np.empty(T, np.int8)
    for k, s in enumerate(range(0, T, seg)):
        y[s:s + seg] = (H, D, A, D)[k % 4]
    xy = np.cumsum(0.2 * rng.standard_normal((T, S, 2)), axis=0) + np.array(
        [[[20, 20], [40, 30], [60, 40], [30, 25], [50, 35], [80, 45]]])
    xy = np.clip(xy, 1, 60)
    ball = xy[:, 1].copy()
    for s in range(seg, T, 2 * seg):                       # frozen where it went out
        ball[s:s + seg] = ball[s - 1]
    team = np.array([0, 0, 0, 1, 1, 1], np.int8)
    return Match("t", 25, (105.0, 68.0), xy, np.zeros((T, S), np.float32), team, ball,
                 [(1, 0, T)], labels=y if labels else None)


def _profile():
    pats = [[0] * 100, [1] * 100, [1] * 20 + [0] * 70 + [1] * 10]
    return {"dead_patterns": pats, "alive_gaps": [30, 60, 120]}


def test_measure_reads_stoppage_patterns_and_live_gaps():
    m = _match()
    codes = m.labels.copy()
    ball = m.ball.copy()
    ball[500:1000] = np.nan                                 # first stoppage fully invisible
    ball[1500:1600] = np.nan                                # second: visible only after 100 frames
    ball[200:230] = np.nan                                  # one live gap strictly inside live play
    m.ball = ball
    p = state.measure(m, codes)
    assert p["n_stoppages"] == 6
    assert p["dead_patterns"][0] == [0] * 100
    assert p["dead_patterns"][1][:20] == [0] * 20 and p["dead_patterns"][1][20:] == [1] * 80
    assert p["alive_gaps"] == [30]
    assert codes is not None and np.array_equal(codes, m.labels)


def test_transport_rest_reaches_the_restart_spot_and_rests():
    m = _match()
    ball = state.transport_rest(m, np.random.default_rng(0))
    a, b = 500, 1000                                        # first stoppage
    assert np.allclose(ball[b - 1], m.ball[b], atol=1e-9) or np.linalg.norm(ball[b - 1] - m.ball[b]) < 1.0
    speed = np.hypot(*np.diff(ball[a:b], axis=0).T) * 25
    assert np.median(speed) < 1.0                           # mostly at rest
    assert np.array_equal(ball[:a], m.ball[:a])             # live play untouched
    assert np.array_equal(state.transport_rest(m, np.random.default_rng(0)), ball)


def test_degrade_ball_hits_the_live_coverage_and_uses_patterns():
    m = _match(T=20000)
    prof = _profile()
    out = state.degrade_ball(m.ball.copy(), m.labels, m.periods, np.random.default_rng(3), prof,
                             lambda b, per: b)
    vis = ~np.isnan(out[:, 0])
    assert abs(vis[m.labels < D].mean() - state.ALIVE_COVERAGE) < 0.03
    assert vis[m.labels == D].mean() < 0.7                  # patterns drop most of the stoppages
    again = state.degrade_ball(m.ball.copy(), m.labels, m.periods, np.random.default_rng(3), prof,
                               lambda b, per: b)
    assert np.array_equal(out, again, equal_nan=True)


def test_simulate_with_state_changes_only_the_ball():
    from possession.noise import profile as noise_profile
    m = _match()
    prof = noise_profile.measure(_match(labels=False))
    old = noise_sim.simulate(m, prof, seed=2, calibrate=False)
    new = noise_sim.simulate(m, prof, seed=2, state=_profile())
    assert np.array_equal(old.xy, new.xy, equal_nan=True)
    assert np.array_equal(old.labels, new.labels)
    assert new.meta["world"] == "phase9-state"
    assert not np.array_equal(old.ball, new.ball, equal_nan=True)


def test_cli_has_the_state_profile_command_and_world_switch():
    from possession import __main__ as cli
    src = open(cli.__file__).read()
    assert '"state-profile": cmd_state_profile' in src
    assert 'choices=("phase6", "state")' in src
