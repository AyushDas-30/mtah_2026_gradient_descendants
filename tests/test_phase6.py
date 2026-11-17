"""Phase 6: stoppage-edge detectors, the edge-aware Potts decoder and the paired selection rule."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from possession import decode, features, phase1  # noqa: E402
from possession.emission import EmissionModel, edge_near  # noqa: E402
from possession.schema import A, D, H, X, Match  # noqa: E402


def _stop_start_match(T=3000, seg=300, seed=0):
    """Players run while the ball is alive and stand still while it is dead."""
    rng = np.random.default_rng(seed)
    S = 6
    labels = np.empty(T, np.int8)
    for k, s in enumerate(range(0, T, seg)):
        labels[s:s + seg] = (H, D, A, D)[k % 4]
    step = np.where(labels[:, None, None] == D, 0.002, 0.25) * rng.standard_normal((T, S, 2))
    xy = np.cumsum(step, axis=0) + np.array([[[20, 20], [40, 30], [60, 40], [30, 25], [50, 35], [80, 45]]])
    xy[:, 0, 0], xy[:, 3, 0] = 2.0, 103.0            # goalkeepers fix the attack directions
    ball = xy[:, 1].copy()
    ball[labels == D] = np.nan
    ball[np.arange(T) % 100 < 20] = np.nan                # the tracker also loses a live ball now and then
    team = np.array([0, 0, 0, 1, 1, 1], np.int8)
    m = Match("t", 25, (105.0, 68.0), xy, np.zeros((T, S), np.float32), team, ball, [(1, 0, T)])
    return m, labels


def test_edge_near_marks_restarts_and_stops_separately():
    y = np.array([H, H, D, D, D, A, A, X, X, D, H], np.int8)
    r = edge_near(y, "restart", 1)
    s = edge_near(y, "stop", 1)
    assert list(np.flatnonzero(s)) == [1, 2, 3]            # H -> D at 2
    assert list(np.flatnonzero(r)) == [4, 5, 6, 9, 10]     # D -> A at 5, D -> H at 10 (X -> D ignored)


def test_potts_edges_with_flat_costs_is_potts():
    rng = np.random.default_rng(1)
    b = decode.to_fixed(rng.normal(size=(400, 3)))
    lam_t, lam_d = 7000, 100000
    flat = np.full(400, lam_d, np.int64)
    assert np.array_equal(decode.potts_edges(b, lam_t, flat, flat), decode.potts(b, lam_t, lam_d))


def test_potts_edges_optional_restart_penalty_only_applies_at_restart():
    b = np.array([[20000, 0, 0], [0, 0, 20000], [9000, 0, 0]], dtype=np.int64)
    flat = np.full(3, 10000, dtype=np.int64)
    restart = np.array([False, False, True])
    legacy = decode.potts_edges(b, 10000, flat, flat)
    default = decode.potts_edges(b, 10000, flat, flat, lam_dead_out=None)
    cheaper = decode.potts_edges(b, 10000, flat, flat, lam_dead_out=0, restart_mask=restart)
    assert np.array_equal(default, legacy)
    assert legacy[2] == D and cheaper[2] == H


def test_edge_costs_drop_where_the_detector_fires_and_stay_bounded():
    logit = np.zeros(50)
    logit[20:25] = 5.0
    logit[:5] = -40.0
    c = decode.edge_costs(logit, 0, 50, 10, lam_dead=100, gamma=32)
    assert c[4] == 0                                       # 100 - 32 * 5 < 0 -> clipped to 0
    assert c[0] == 300 * decode.SCALE                 # clipped at 3 * lam_dead
    assert c[1] == 100 * decode.SCALE


def test_potts_e_decodes_a_short_flagged_stoppage_plain_potts_misses():
    n = 200
    s = np.zeros((n, 3))
    s[:, H] = 0.5
    s[80:100, D] = 1.0                                     # weak dead evidence over 20 frames
    s[80:100, H] = 0.0
    restart = np.full(n, -10.0)
    stop = np.full(n, -10.0)
    stop[78:83] = 6.0
    restart[97:103] = 6.0
    periods = [(1, 0, n)]
    plain = decode.decode_match(s, periods, n, {"decoder": "potts", "alpha": 1.0, "lam_team": 7, "lam_dead": 30})
    edged = decode.decode_match(s, periods, n, {"decoder": "potts_e", "alpha": 1.0, "lam_team": 7,
                                               "lam_dead": 30, "edge_gamma": 16}, edges=(restart, stop))
    assert not np.any(plain == D)
    assert np.all(edged[82:98] == D) and edged[10] == H and edged[150] == H


def test_emission_edge_detectors_fire_near_edges_and_are_deterministic():
    m, y = _stop_start_match()
    f = features.compute(m)
    model = EmissionModel(kind="logistic", edges=True).fit([f], [y])
    r1, s1 = model.edge_logits(f)
    r2, s2 = model.edge_logits(f)
    assert np.array_equal(r1, r2) and np.array_equal(s1, s2)
    assert r1.shape == (m.n_frames,) and np.isfinite(r1).all() and np.isfinite(s1).all()
    near_r = edge_near(y, "restart", 5)
    near_s = edge_near(y, "stop", 5)
    assert r1[near_r].mean() > r1[~near_r].mean()
    assert s1[near_s].mean() > s1[~near_s].mean()
    assert EmissionModel(kind="logistic").fit([f], [y]).edge_logits(f) is None


def _row(dec, obj, per_match, **cfg):
    c = dict({"decoder": dec, "alpha": 1.0}, **cfg)
    return {"key": phase1.config_key(c), "config": c, "objective": obj, "per_match": per_match}


def test_choose_prefers_simple_within_band_unless_paired_wins():
    base = [0.50, 0.52, 0.55, 0.53, 0.51, 0.56, 0.54]
    simple = _row("potts", float(np.mean(base)), base, lam_team=7, lam_dead=100)
    consistent = [v + 0.004 for v in base[:6]] + [base[6] - 0.001]
    edge6 = _row("potts_e", float(np.mean(consistent)), consistent, lam_team=7, lam_dead=100, edge_gamma=32)
    chosen, best, paired = phase1.choose([simple, edge6])
    assert chosen["config"]["decoder"] == "potts_e" and paired[0]["wins"] == 6
    mixed = [v + 0.007 if i < 4 else v - 0.002 for i, v in enumerate(base)]
    edge4 = _row("potts_e", float(np.mean(mixed)), mixed, lam_team=7, lam_dead=100, edge_gamma=32)
    chosen, best, paired = phase1.choose([simple, edge4])
    assert chosen["config"]["decoder"] == "potts" and paired[0]["wins"] == 4


def test_grid_contains_edge_configs_with_unique_keys():
    g = phase1.grid()
    keys = [phase1.config_key(c) for c in g]
    assert len(keys) == len(set(keys))
    assert sum(c["decoder"] == "potts_e" for c in g) == len(phase1.EDGE_TEAM) * len(phase1.EDGE_DEAD) * len(phase1.EDGE_GAMMA)
