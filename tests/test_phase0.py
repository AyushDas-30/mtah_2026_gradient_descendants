"""Phase 0 tests. Synthetic fixtures only; the competition data is never needed to run these.

    python -m pytest -q tests

Set MTAH_REPO_DIR to an official repo clone to also run the published-number checks.
"""

import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synth  # noqa: E402
from possession import metrics  # noqa: E402
from possession.io import atd, idsse, labels as label_io, tagging  # noqa: E402
from possession.methods import baseline  # noqa: E402
from possession.noise import profile, simulate  # noqa: E402
from possession.schema import A, D, H, X, Match  # noqa: E402

PITCH = (105.3, 68.0)


# -- ATD parser ----------------------------------------------------------------------------

@pytest.mark.parametrize("shuffle,empty", [(False, False), (True, False), (False, True)])
def test_atd_parse_roundtrip(tmp_path, shuffle, empty):
    s = synth.write_epts(tmp_path, n_frames=600, shuffle_groups=shuffle, empty_fields=empty, seed=3)
    md = atd.read_metadata(s["meta"])
    layout = atd.build_layout(md)
    assert layout["source"] == "DataFormatSpecification"
    assert layout["player_fields"] == ["x", "y", "speed"]
    assert layout["ball_fields"][:2] == ["x", "y"]
    assert np.array_equal(layout["slot_team"], s["slot_team"])
    assert layout["agrees_with_player_order"] == (not shuffle)
    xy, speed, ball, stats = atd.read_raw(s["raw"], layout, 600, PITCH)
    assert stats["records"] == 600 and stats["contiguous_from_1"] and stats["malformed"] == 0
    np.testing.assert_allclose(xy[..., 0], np.round(s["x"], 5) * PITCH[0], rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(xy[..., 1], np.round(s["y"], 5) * PITCH[1], rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(ball[:, 0], np.round(s["bx"], 5) * PITCH[0], atol=1e-9, equal_nan=True)


def test_nearest_team_matches_starter_line_parser(tmp_path):
    s = synth.write_epts(tmp_path, n_frames=1500, seed=7)
    layout = atd.build_layout(atd.read_metadata(s["meta"]))
    xy, speed, ball, _ = atd.read_raw(s["raw"], layout, 1500, PITCH)
    m = Match("t", 25, PITCH, xy, speed, layout["slot_team"], ball, [(1, 0, 1500)]).validate()
    ours = baseline.nearest_team(m)
    ref = synth.starter_reference(s["raw"], s["meta"], 1500, PITCH)
    assert np.array_equal(ours, ref)
    for w in (0, 25, 50):
        assert baseline.majority(ours, w).shape == (1500,)


def test_match_save_load(tmp_path):
    s = synth.write_epts(tmp_path, n_frames=200, seed=1)
    layout = atd.build_layout(atd.read_metadata(s["meta"]))
    xy, speed, ball, _ = atd.read_raw(s["raw"], layout, 200, PITCH)
    m = Match("t", 25, PITCH, xy, speed, layout["slot_team"], ball, [(1, 10, 190)],
              labels=np.full(200, X, np.int8), meta={"a": [1, 2]})
    m2 = Match.load(m.save(tmp_path / "m.npz"))
    assert np.array_equal(m.xy, m2.xy, equal_nan=True) and m2.periods == [(1, 10, 190)]
    assert m2.meta == {"a": [1, 2]} and np.array_equal(m2.labels, m.labels)


# -- tagging -------------------------------------------------------------------------------

def test_tagging(tmp_path):
    p = synth.write_tagging(tmp_path / "t.xml", [
        ("SET PIECES", 444.6, 449.7, {"Team": "Atletic Club", "Type": "Kick Off (Start)", "Half": "1st Half"}),
        ("BUILD UP", 500.02, 512.5, {"Team": "Barcelona", "Half": "1st Half"}),
        ("SET PIECES", 600, 603, {"Team": "N/A", "Type": "Throw-in"}),
    ])
    ev = tagging.load_tagging(p)
    assert [e["start_frame"] for e in ev] == [11115, 12500, 15000]
    assert ev[0]["end_frame"] == round(449.7 * 25)
    assert tagging.team_side("Barcelona", "FC Barcelona", "Athletic Club") == "H"
    assert tagging.team_side("Atletic Club", "FC Barcelona", "Athletic Club") == "A"
    assert tagging.team_side("N/A", "FC Barcelona", "Athletic Club") is None


# -- metrics -------------------------------------------------------------------------------

def _brute_overlaps(g, p):
    out = []
    for i, (a, b, k) in enumerate(g):
        for j, (c, d, l) in enumerate(p):
            if l == k and d > a and b > c:
                out.append(((min(b, d) - max(a, c)) / float(max(b, d) - min(a, c)), i, j))
    out.sort(key=lambda z: (-z[0], z[1], z[2]))
    return out


def test_overlap_sweep_equals_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(50):
        gt = np.repeat(rng.integers(0, 3, 60), rng.integers(1, 40, 60)).astype(np.int8)
        pred = np.repeat(rng.integers(0, 3, 80), rng.integers(1, 30, 80)).astype(np.int8)[:len(gt)]
        pred = np.concatenate([pred, np.full(len(gt) - len(pred), H, np.int8)])
        runs = metrics.valid_runs(gt)
        g, p = metrics.spells(gt, runs), metrics.spells(pred, runs)
        assert metrics.spell_overlaps(g, p) == _brute_overlaps(g, p)


def test_metric_basics():
    gt = np.array([X] * 5 + [H] * 50 + [D] * 20 + [A] * 40 + [X] * 5, dtype=np.int8)
    assert metrics.evaluate(gt, gt)["spell_f1"] == 1.0
    assert metrics.evaluate(gt, np.full_like(gt, D))["spell_f1"] == 0.0
    shifted = gt.copy()
    shifted[55:56] = H  # D starts one frame late
    assert metrics.evaluate(gt, shifted)["spell_f1"] == 1.0
    # X inside a possession is forward-filled for spell F1
    holed = gt.copy()
    holed[20:30] = X
    r = metrics.evaluate(gt, holed)
    assert r["spell_f1"] == 1.0 and r["macro_f1"] < 1.0
    pooled = metrics.evaluate_pooled([(gt, gt), (gt, shifted)], bootstrap_n=50)
    assert pooled["spell_f1"] == 1.0 and pooled["n_gt_spells"] == 4


REPO = os.environ.get("MTAH_REPO_DIR")


@pytest.mark.skipif(not REPO, reason="MTAH_REPO_DIR not set")
def test_published_public_scores():
    from possession import config
    repo = Path(REPO)
    gt = label_io.read_labels(repo / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv")
    for name, (macro, spell) in config.PUBLISHED_BASELINE_PUBLIC.items():
        pred = label_io.read_labels(repo / "evaluation" / "results" / (name + ".csv"))
        r = metrics.evaluate(gt, pred)
        assert round(r["macro_f1"], 4) == macro, name
        assert round(r["spell_f1"], 4) == spell, name


def test_labels_roundtrip(tmp_path):
    codes = np.array([H, A, D, X] * 10, dtype=np.int8)
    p = label_io.write_labels(tmp_path / "s.csv", codes, n_expected=40)
    assert np.array_equal(label_io.read_labels(p), codes)
    with pytest.raises(ValueError):
        label_io.write_labels(tmp_path / "bad.csv", codes, n_expected=41)


# -- IDSSE ---------------------------------------------------------------------------------

def test_idsse_parse(tmp_path):
    expected = synth.write_dfl(tmp_path, "J03WMX")
    m = idsse.load_match("J03WMX", tmp_path)
    (_, a1, b1), (_, a2, b2) = m.periods
    assert (b1 - a1, b2 - a2) == (3000, 2800) and a2 - b1 == idsse.HALF_GAP
    assert np.array_equal(m.labels[a1:b1], expected["firstHalf"])
    assert np.array_equal(m.labels[a2:b2], expected["secondHalf"])
    assert np.all(m.labels[b1:a2] == X)
    assert m.n_slots == 22 and (m.slot_team == 0).sum() == 11      # referee excluded
    assert m.meta["n_skipped_non_players"] == 1
    assert np.nanmin(m.ball[:, 0]) >= 0 and np.nanmax(m.ball[:, 0]) <= 105.0
    assert np.all(np.isnan(m.xy[b1:a2]))


def test_idsse_labels_carry_team():
    st = np.array([1, 1, 0, 0, 1, 1], dtype=np.int8)
    po = np.array([2, 0, 2, 1, 0, 1], dtype=np.int8)
    assert idsse.labels_from_codes(st, po).tolist() == [A, A, D, D, H, H]


# -- noise ---------------------------------------------------------------------------------

def _atd_like(tmp_path, n=6000):
    s = synth.write_epts(tmp_path / "atd", n_frames=n, seed=11, ball_missing=0.0, player_missing=0.0)
    layout = atd.build_layout(atd.read_metadata(s["meta"]))
    xy, speed, ball, _ = atd.read_raw(s["raw"], layout, n, PITCH)
    # burst ball dropout + slot gaps, as a real tracker would produce
    rng = np.random.default_rng(5)
    t = 0
    while t < n:
        t += int(rng.integers(20, 120))
        ball[t:t + int(rng.integers(5, 80))] = np.nan
        t += 80
    team = layout["slot_team"]
    for t in (0, 1):                       # 11 on the pitch per team, the rest empty slots
        xy[:, np.flatnonzero(team == t)[11:]] = np.nan
    for s_ in range(xy.shape[1]):
        a = int(rng.integers(0, n))
        xy[a:a + int(rng.integers(50, 600)), s_] = np.nan
    return Match("atd", 25, PITCH, xy, speed, layout["slot_team"], ball, [(1, 0, n)]).validate()


def test_profile_and_simulate(tmp_path):
    target = profile.measure(_atd_like(tmp_path))
    assert 0.2 < target["ball_present_fraction"] < 0.9
    expected = synth.write_dfl(tmp_path / "dfl", "J03WN1", n_half=(4000, 4000))
    clean = idsse.load_match("J03WN1", tmp_path / "dfl")
    sim1 = simulate.simulate(clean, target, seed=0)
    sim2 = simulate.simulate(clean, target, seed=0)
    sim3 = simulate.simulate(clean, target, seed=1)
    assert np.array_equal(sim1.ball, sim2.ball, equal_nan=True)
    assert np.array_equal(sim1.xy, sim2.xy, equal_nan=True)
    assert not np.array_equal(sim1.ball, sim3.ball, equal_nan=True)
    assert np.array_equal(sim1.labels, clean.labels)
    got = profile.summary(profile.measure(sim1))
    want = profile.summary(target)
    assert abs(got["ball_present_fraction"] - want["ball_present_fraction"]) < 0.02   # calibrated
    assert abs(got["objects_home_mean"] - want["objects_home_mean"]) < 0.7
    # every true player position that survives sits in a slot of the right team
    assert set(np.unique(sim1.slot_team)) == {0, 1}
    pred = baseline.baseline(sim1, 50)
    assert metrics.evaluate(sim1.labels, pred)["n_graded"] == 8000


def test_speed_unit_conversion(tmp_path, monkeypatch):
    from possession import config
    s = synth.write_epts(tmp_path / "tactical_src", n_frames=50, seed=2, speed_unit="kph")
    tact = tmp_path / "bundle" / "tactical"
    tact.mkdir(parents=True)
    (tact / "BAR-ATH_tactical_atd.xml").write_text(s["meta"].read_text())
    (tact / "BAR-ATH_tactical_atd.txt").write_text(s["raw"].read_text())
    m = atd.load_tactical(tmp_path / "bundle", n_frames=50, periods=[(1, 0, 50)])
    assert m.meta["speed_unit_in_file"] == "kph"
    np.testing.assert_allclose(m.speed, (np.round(s["speed"], 5) / 3.6).astype(np.float32),
                               rtol=1e-5, equal_nan=True)
