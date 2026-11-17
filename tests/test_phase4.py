"""Phase 4 tests: own gradient boosting."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from possession.gbdt import GBDT  # noqa: E402
from possession.logistic import Logistic  # noqa: E402


def _xor(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 3))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)
    flip = rng.random(n) < 0.05
    return X, np.where(flip, 1 - y, y)


def test_gbdt_learns_an_interaction_logistic_cannot():
    X, y = _xor()
    Xt, yt = _xor(seed=1)
    gb = GBDT(n_trees=60, max_depth=3, min_leaf=50).fit(X, y)
    lr = Logistic().fit(X, y)
    acc_gb = ((gb.predict_proba(Xt) > 0.5) == yt).mean()
    acc_lr = ((lr.predict_proba(Xt) > 0.5) == yt).mean()
    assert acc_gb > 0.9 and acc_lr < 0.65


def test_gbdt_is_deterministic():
    X, y = _xor()
    a = GBDT(n_trees=20, max_depth=3, min_leaf=50).fit(X, y).decision_function(X)
    b = GBDT(n_trees=20, max_depth=3, min_leaf=50).fit(X, y).decision_function(X)
    assert np.array_equal(a, b)


def test_gbdt_handles_constant_feature_and_small_nodes():
    X, y = _xor(n=500)
    X[:, 2] = 1.0
    gb = GBDT(n_trees=10, max_depth=4, min_leaf=400).fit(X, y)   # min_leaf too big to split much
    p = gb.predict_proba(X)
    assert np.isfinite(p).all() and p.shape == (500,)


def test_restart_memory_prefers_the_other_team_after_a_stoppage():
    from possession import decode
    from possession.schema import H, A, D
    # H, a stoppage, then ambiguous team evidence: with a strong 'other team restarts' prior the
    # decoder should resume with A; with no prior memory it follows the slight evidence for H.
    n = 300
    s = np.zeros((n, 3))
    s[:100, H] = 2.0
    s[100:150, D] = 3.0
    s[150:, D] = -2.0             # clearly live again ...
    s[150:, H] = 0.02             # ... with only a whisker of evidence for H over A
    b = decode.to_fixed(decode.block_scores(s, 0, n))
    plain = decode.potts_restart(b, 5000, 5000, 0, 0)
    biased = decode.potts_restart(b, 5000, 5000, int(-20 * np.log(0.37) * 1000), int(-20 * np.log(0.63) * 1000))
    assert plain[-1] == H and biased[-1] == A
    assert set(np.unique(biased)) <= {H, A, D}


def test_turnover_near_marks_a_window():
    from possession.emission import turnover_near
    from possession.schema import H, A, D
    y = np.array([H] * 10 + [A] * 10 + [D] * 5 + [H] * 5)
    m = turnover_near(y, 2)
    assert np.flatnonzero(m).tolist() == [8, 9, 10, 11, 12]     # only the H->A turnover, not D->H


def test_time_varying_switch_cost_allows_a_flagged_turnover():
    from possession import decode
    from possession.schema import H, A
    n = 400
    s = np.zeros((n, 3))
    s[:200, H], s[200:, A] = 0.4, 0.4
    s[:, 2] = -5
    b = decode.to_fixed(decode.block_scores(s, 0, n))
    high = np.full(len(b), 10 ** 6, dtype=np.int64)        # switch dearer than all the evidence
    assert np.all(decode.potts_tv(b, high, 10 ** 6) == decode.potts_tv(b, high, 10 ** 6)[0])
    low = high.copy()
    low[40] = 0                                             # detector fires at the true change
    path = decode.potts_tv(b, low, 10 ** 6)
    assert path[0] == H and path[-1] == A and len(np.flatnonzero(np.diff(path))) == 1
