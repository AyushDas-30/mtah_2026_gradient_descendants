"""Phase 2 tests: kick-off rule, ball interpolation, carried dead ball, zone-aware hazard."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from possession import features, kickoff  # noqa: E402
from possession.noise import profile, simulate  # noqa: E402
from possession.schema import A, D, H, X, Match  # noqa: E402


def _match(T=1000, ball=None, labels=None, periods=None):
    xy = np.full((T, 4, 2), 50.0)
    xy[:, :, 0] += np.arange(4)
    return Match("t", 25, (105.0, 68.0), xy, np.zeros((T, 4), np.float32),
                 np.array([0, 0, 1, 1], np.int8),
                 ball if ball is not None else np.full((T, 2), 30.0),
                 periods or [(1, 0, T)], labels)


def test_kickoff_rule_from_tags():
    ev = [{"code": "SET PIECES", "tags": {"Type": "Kick Off (Start)"}, "start_frame": 11097},
          {"code": "SET PIECES", "tags": {"Type": "Kick Off (Start)"}, "start_frame": 113660},
          {"code": "SET PIECES", "tags": {"Type": "Throw-in"}, "start_frame": 20000},
          {"code": "SET PIECES", "tags": {"Type": "Kick Off (After Goal)"}, "start_frame": 49631}]
    ko = kickoff.kickoff_frames(ev)
    assert ko == [11185, 49719, 113748]
    spans = kickoff.pre_kickoff_spans([(1, 11185, 90652), (2, 110808, 190198)], ko)
    assert spans == [(110808, 113748)]           # half 1 is graded from its kick-off: nothing forced
    s = kickoff.constrain(np.zeros((200000, 3)), spans)
    assert np.all(np.argmax(s[110808:113748], axis=1) == 2) and np.all(s[113748] == 0)


def test_interpolate_ball_fills_short_gaps_only():
    T = 200
    ball = np.tile(np.arange(T, dtype=float)[:, None], (1, 2))
    ball[50:60] = np.nan            # 10 frames: filled
    ball[100:140] = np.nan          # 40 frames: left
    ball[0:5] = np.nan              # leading: left
    out = features.interpolate_ball(_match(T, ball))
    np.testing.assert_allclose(out[50:60, 0], np.arange(50, 60))
    assert np.isnan(out[100:140]).all() and np.isnan(out[0:5]).all()


def test_carry_dead_ball_moves_frozen_ball():
    T = 300
    ball = np.full((T, 2), 40.0)
    ball[:100, 0] = np.linspace(40, 104, 100)           # alive, runs to the line
    ball[100:200] = [106.0, 30.0]                         # dead: frozen beyond the goal line
    ball[200:, 0] = 5.0                                   # alive again at the other end (goal kick)
    ball[200:, 1] = 34.0
    labels = np.array([H] * 100 + [D] * 100 + [A] * 100, np.int8)
    carried = simulate.carry_dead_ball(_match(T, ball, labels))
    step = np.sqrt((np.diff(carried[100:200], axis=0) ** 2).sum(1))
    assert step.min() > 0                                  # never frozen
    assert np.allclose(carried[:100], ball[:100]) and np.allclose(carried[200:], ball[200:])


def test_zone_hazard_shape_and_smoothing():
    T = 4000
    rng = np.random.default_rng(0)
    ball = np.column_stack([rng.uniform(-2, 107, T), rng.uniform(5, 60, T)])
    edge = (ball[:, 0] < 2) | (ball[:, 0] > 103)
    ball[np.flatnonzero(edge) + 1 if edge[-1] == 0 else np.flatnonzero(edge[:-1]) + 1] = np.nan
    prof = profile.measure(_match(T, ball))
    h = np.array(prof["ball_hazard"])
    assert h.shape == (2, 4, 4)
    assert h[1].mean() > h[0].mean()                       # losses concentrate at the edge
