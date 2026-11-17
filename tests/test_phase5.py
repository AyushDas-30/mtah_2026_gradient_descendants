"""Step 5: continuous pitch-margin features and weak Smart Tagging priors."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from possession import emission, features  # noqa: E402
from possession.schema import D  # noqa: E402


def test_boundary_margin_is_continuous_and_defaults_to_zero():
    xy = np.array([[0.5, 0.5], [-0.1, 0.4], [1.2, 0.4], [0.3, 1.1], [np.nan, 0.2]])
    margin = features.boundary_margin(xy, len(xy))
    np.testing.assert_allclose(margin, [0.0, 0.1, 0.2, 0.1, 0.0])
    assert np.all(np.isfinite(margin))
    assert not features.boundary_margin(None, len(xy)).any()
    with pytest.raises(ValueError):
        features.boundary_margin(np.zeros((2, 2)), len(xy))


def test_tagging_priors_are_bounded_soft_and_tolerate_missing_events():
    dead, restart = emission.tagging_priors(
        [{"code": "SET PIECES", "start_frame": 10, "end_frame": 20}], 40, fps=10)
    assert np.all(dead[10:20] == 0.65)
    assert dead[9] == 0.0 and dead[20] == 0.0
    assert restart[20] == 0.65
    assert 0.0 < restart[0] < 0.65
    assert np.max(restart) <= 0.8
    d0, r0 = emission.tagging_priors(None, 40)
    assert not d0.any() and not r0.any()


def test_dead_score_prior_is_log_odds_only_and_shape_checked():
    model = emission.EmissionModel()
    scores = np.zeros((3, 3))
    adjusted = model.scores_with_tagging_prior(scores, np.array([0.0, 0.65, 0.0]))
    assert adjusted[1, D] == pytest.approx(0.65)
    assert np.array_equal(adjusted[[0, 2]], scores[[0, 2]])
    with pytest.raises(ValueError):
        model.scores_with_tagging_prior(scores, np.zeros(2))

