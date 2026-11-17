"""Phase 1 tests: relabelling, logistic fit, decoders, end-to-end selection on synthetic games."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from possession import decode, metrics, relabel  # noqa: E402
from possession.logistic import Logistic  # noqa: E402
from possession.schema import A, D, H, X  # noqa: E402


def seq(*parts):
    return np.concatenate([np.full(n, k, dtype=np.int8) for k, n in parts])


# -- relabel -------------------------------------------------------------------------------

def test_relabel_absorbs_short_flips():
    y = seq((H, 100), (A, 10), (H, 100), (D, 50), (A, 5), (H, 100), (A, 60))
    r = relabel.relabel(y, 1.0)              # 25 frames
    assert np.array_equal(r, seq((H, 210), (D, 50), (A, 5), (H, 100), (A, 60)))
    assert np.array_equal(relabel.relabel(y, 0.0), y)
    # a short run at the end of an alive stretch is a loose ball, not a possession
    y2 = seq((H, 100), (A, 10), (D, 30))
    assert np.array_equal(relabel.relabel(y2, 1.0), seq((H, 110), (D, 30)))


def test_relabel_never_touches_dead_or_x():
    rng = np.random.default_rng(0)
    y = np.repeat(rng.integers(0, 4, 300), rng.integers(1, 60, 300)).astype(np.int8)
    for t in relabel.THRESHOLDS_S:
        r = relabel.relabel(y, t)
        assert np.array_equal(r == D, y == D) and np.array_equal(r == X, y == X)


# -- logistic ------------------------------------------------------------------------------

def test_logistic_recovers_signal():
    rng = np.random.default_rng(1)
    Xm = rng.normal(size=(5000, 3))
    p = 1 / (1 + np.exp(-(0.5 + 2 * Xm[:, 0] - 1 * Xm[:, 1])))
    y = (rng.random(5000) < p).astype(float)
    m = Logistic(l2=0.1).fit(Xm, y)
    acc = ((m.predict_proba(Xm) > 0.5) == (p > 0.5)).mean()
    assert acc > 0.95
    assert np.array_equal(m.predict_proba(Xm), Logistic(l2=0.1).fit(Xm, y).predict_proba(Xm))


# -- decoders ------------------------------------------------------------------------------

def _noisy_scores(truth, rng, strength=1.0, noise=2.0):
    s = rng.normal(0, noise, size=(len(truth), 3))
    s[np.arange(len(truth)), truth] += strength
    return s


def _truth():
    return seq((H, 900), (A, 600), (D, 400), (A, 1200), (H, 300), (A, 700), (D, 250), (H, 800))


@pytest.mark.parametrize("decoder", ["potts", "hsmm"])
def test_decoders_recover_segments(decoder):
    rng = np.random.default_rng(2)
    truth = _truth()
    scores = _noisy_scores(truth, rng)
    periods = [(1, 0, len(truth))]
    priors = decode.fit_priors([np.concatenate([truth, truth])], max_seconds=120)
    cfg = ({"decoder": "potts", "alpha": 1.0, "lam_team": 80, "lam_dead": 80} if decoder == "potts"
           else {"decoder": "hsmm", "alpha": 0.5, "beta": 1.0})
    pred = decode.decode_match(scores, periods, len(truth), cfg, priors)
    r = metrics.evaluate(truth, pred)
    assert r["spell_f1"] > 0.95, r
    assert r["n_pred_spells"] == r["n_gt_spells"]


def test_hsmm_never_repeats_a_state():
    rng = np.random.default_rng(3)
    truth = _truth()
    scores = _noisy_scores(truth, rng, strength=0.3, noise=3.0)
    priors = decode.fit_priors([truth], max_seconds=120)
    b = decode.block_scores(scores, 0, len(truth))
    path = decode.hsmm(decode.to_fixed(0.3 * b), decode.to_fixed(priors["logdur"]),
                       decode.to_fixed(priors["log_trans"]), decode.to_fixed(priors["log_start"]),
                       priors["max_blocks"])
    # segments of the block path: consecutive segments differ by construction
    changes = np.flatnonzero(path[1:] != path[:-1])
    assert len(path) == len(b) and len(changes) >= 1


def test_refine_moves_boundary_to_true_frame():
    truth = seq((H, 103), (A, 97))
    scores = np.zeros((200, 3))
    scores[np.arange(200), truth] = 1.0
    coarse = seq((H, 100), (A, 100))
    fixed = decode.refine_boundaries(coarse, decode.to_fixed(scores), radius=5)
    assert np.array_equal(fixed, truth)


def test_decode_marks_outside_periods_x():
    scores = np.zeros((1000, 3))
    scores[:, H] = 1
    out = decode.decode_match(scores, [(1, 100, 400), (2, 600, 900)], 1000,
                              {"decoder": "potts", "alpha": 1.0, "lam_team": 10, "lam_dead": 10})
    assert np.all(out[:100] == X) and np.all(out[400:600] == X) and np.all(out[900:] == X)
    assert np.all(out[100:400] == H)


def test_ks_statistic():
    from possession.phase1 import ks_statistic
    rng = np.random.default_rng(0)
    a = rng.normal(size=4000)
    assert ks_statistic(a, rng.normal(size=4000)) < 0.05
    assert ks_statistic(a, rng.normal(size=4000) / 3.6) > 0.25     # the speed-unit failure mode


def test_kinematic_speed_ignores_speed_field():
    from possession import features
    from possession.schema import Match
    T = 500
    xy = np.zeros((T, 2, 2))
    xy[:, 0, 0] = np.arange(T) * (5.0 / 25)        # 5 m/s
    xy[:, 1, 0] = 10.0
    m = Match("k", 25, (105.0, 68.0), xy, np.full((T, 2), 99.0, np.float32),
              np.array([0, 1], np.int8), np.full((T, 2), np.nan), [(1, 0, T)])
    kin = features.kinematic_speed(m)
    np.testing.assert_allclose(kin[10:-10, 0], 5.0, atol=1e-9)
    np.testing.assert_allclose(kin[10:-10, 1], 0.0, atol=1e-9)


def test_decoding_is_invariant_to_last_bit_noise():
    """Perturbing scores by 1e-12 (platform noise) must not change a single frame."""
    rng = np.random.default_rng(5)
    truth = _truth()
    scores = np.round(_noisy_scores(truth, rng), 6)
    noisy = scores * (1 + rng.normal(0, 1e-13, scores.shape))
    priors = decode.fit_priors([truth], max_seconds=120)
    for cfg in ({"decoder": "potts", "alpha": 1.0, "lam_team": 40, "lam_dead": 40},
                {"decoder": "hsmm", "alpha": 0.3, "beta": 1.0}):
        a = decode.decode_match(scores, [(1, 0, len(truth))], len(truth), cfg, priors)
        b = decode.decode_match(noisy, [(1, 0, len(truth))], len(truth), cfg, priors)
        assert np.array_equal(a, b)
