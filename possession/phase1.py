"""Phase 1: choose a decoder on simulated external games, then label BAR-ATH with it.

Selection (phase1-select)
  - every simulated IDSSE game is featurised once
  - leave-one-match-out: for each held-out match, the emission model and the semi-Markov priors
    are fitted on the OTHER matches' simulated games, with labels relabelled at T_TRAIN seconds
  - each held-out game is decoded with every configuration in the grid and scored against all
    four label variants (control threshold 0 / 0.5 / 1 / 2 s), because the competition's exact
    threshold is unknown. Objective = mean over the variants of pooled spellF1
  - the starter baseline is scored the same way, with bootstrap intervals for the comparison
  - choice: the best objective; within TIE_BAND of the best the simplest decoder family wins
    (Potts, then Potts with stoppage-edge detectors, then restart memory, then semi-Markov).
    Phase 6 adds a paired check: a more complex family still wins if its best configuration
    beats the simple choice on at least PAIRED_MIN of the held-out matches (a sign test across
    matches is more sensitive than the pooled band when the gain is small but consistent)

Prediction (predict)
  - final models fitted on ALL simulated games, the chosen configuration decodes BAR-ATH
  - X outside the graded halves, H / A / D everywhere inside them
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import config, decode, emission, features, metrics, relabel
from .emission import EmissionModel
from .methods import baseline as baseline_mod
from .schema import Match, X

CODE_VERSION = "8.0"       # bump whenever features, models or decoding change: stale folds rerun
CHANGE_MODEL = False       # turnover detector: +0.0046 on three held-out folds (inside the tie band), off
EMISSION_KIND = "gbdt"     # phase 4; "logistic" reproduces phase 3
EDGE_MODEL = True
# Phase 7 evaluation: the set-piece prior is applied at prediction only (the simulated games have no
# tags, so it is never selected), and the blind frame check of its changes was a draw. Off.
TAGGING_PRIOR = False          # phase 6: restart / stop detectors for the potts_e decoder
T_TRAIN = 1.0
TIE_BAND = 0.005
# Phase 5 grid: phases 1-4 never chose a team penalty above 20 or a semi-Markov decoder, so the
# grid keeps the region that matters (runtime: three boosted models are trained per fold).
POTTS_TEAM = (3, 5, 7, 10, 14, 20, 40)
POTTS_DEAD = (10, 30, 100, 300)
HSMM_ALPHA = (0.2, 0.4)
HSMM_BETA = (0.5, 2.0)
HSMM_MAX_SECONDS = 400
RESTART_TEAM = (5, 7, 10)
RESTART_DEAD = (30, 100)
RESTART_W = ()             # phase 5: restart memory tested (+0.0017, inside the tie band), off
EDGE_TEAM = (5, 7, 10)
EDGE_DEAD = (30, 100)
EDGE_GAMMA = (16, 32, 64)
PAIRED_MIN = 6             # of the held-out matches (7 IDSSE matches)
FAMILY_RANK = {"potts": 0, "potts_e": 1, "potts_r": 2, "hsmm": 3}


def grid():
    out = [{"decoder": "potts", "alpha": 1.0, "lam_team": lt, "lam_dead": ld}
           for lt in POTTS_TEAM for ld in POTTS_DEAD]
    out += [{"decoder": "hsmm", "alpha": a, "beta": b} for a in HSMM_ALPHA for b in HSMM_BETA]
    out += [{"decoder": "potts_r", "alpha": 1.0, "lam_team": lt, "lam_dead": ld, "restart_w": w}
            for lt in RESTART_TEAM for ld in RESTART_DEAD for w in RESTART_W]
    if EDGE_MODEL:
        out += [{"decoder": "potts_e", "alpha": 1.0, "lam_team": lt, "lam_dead": ld, "edge_gamma": g}
                for lt in EDGE_TEAM for ld in EDGE_DEAD for g in EDGE_GAMMA]
    return out


def config_key(c):
    if c["decoder"] == "potts" and c.get("gamma"):
        return "potts(team={},dead={},gamma={})".format(c["lam_team"], c["lam_dead"], c["gamma"])
    if c["decoder"] == "potts":
        return "potts(team={},dead={})".format(c["lam_team"], c["lam_dead"])
    if c["decoder"] == "potts_e":
        return "potts_e(team={},dead={},gamma={})".format(c["lam_team"], c["lam_dead"], c["edge_gamma"])
    if c["decoder"] == "potts_r":
        return "potts_r(team={},dead={},w={})".format(c["lam_team"], c["lam_dead"], c["restart_w"])
    return "hsmm(alpha={},beta={})".format(c["alpha"], c["beta"])


class Accumulator:
    """Pooled spell counts + confusion for one (configuration, label variant)."""

    def __init__(self):
        self.tp = np.zeros(len(metrics.SPELL_IOUS), dtype=np.int64)
        self.n_pred = self.n_gt = 0
        self.cm3 = np.zeros((3, 4), dtype=np.int64)
        self.flags = []

    def add(self, gt, pred):
        sc = metrics.spell_counts(gt, pred)
        self.tp += sc["tp"]
        self.n_pred += sc["n_pred"]
        self.n_gt += sc["n_gt"]
        self.cm3 += metrics.macro_f1_counts(gt, pred)
        self.flags.append(sc["flags"])

    def spell_f1(self):
        return float(np.mean([metrics._f1(int(t), self.n_pred, self.n_gt) for t in self.tp]))

    def macro_f1(self):
        return metrics.macro_f1_from_cm(self.cm3)

    def bootstrap(self, n=1000, seed=0):
        flags = np.hstack(self.flags)
        rng = np.random.RandomState(seed)
        draws = np.empty(n)
        for i in range(n):
            pick = rng.randint(0, self.n_gt, self.n_gt)
            f = []
            for row in flags:
                t = int(row[pick].sum())
                fp = max(0, self.n_pred - t)
                f.append(2.0 * t / (2 * t + fp + (self.n_gt - t)) if t else 0.0)
            draws[i] = np.mean(f)
        return tuple(float(v) for v in np.percentile(draws, (5, 95)))


def _group_sims(cache):
    groups = {}
    for p in sorted((Path(cache) / "sim").glob("*.npz")):
        groups.setdefault(p.stem.rsplit("_s", 1)[0], []).append(p)
    return groups


def _load_featurised(paths, log):
    """Featurise simulated games, keeping only what selection needs (not the positions)."""
    items = []
    for p in paths:
        m = Match.load(p)
        t0 = time.time()
        items.append({"name": m.name, "periods": list(m.periods), "n_frames": m.n_frames,
                      "graded": m.graded_mask(), "features": features.compute(m),
                      "variants": relabel.variants(m.labels, m.fps),
                      "baseline": baseline_mod.baseline(m, 50)})
        del m
        log("    featurised {} ({:.0f}s)".format(items[-1]["name"], time.time() - t0))
    return items


def fit_models(items, exclude=()):
    labels = [it["variants"][T_TRAIN] for it in items]
    model = EmissionModel(exclude=exclude, kind=EMISSION_KIND, change=CHANGE_MODEL,
                          edges=EDGE_MODEL).fit(
        [it["features"] for it in items], labels)
    priors = decode.fit_priors(labels, max_seconds=HSMM_MAX_SECONDS)
    return model, priors


def shift_guard(items, bar_ath):
    """Features whose BAR-ATH distribution differs strongly (KS >= KS_FAIL) from the simulated
    games are excluded from the models. Unsupervised: reads BAR-ATH positions, never labels."""
    fb = features.compute(bar_ath)
    ks = shift_report(items, fb, bar_ath.graded_mask())
    return ks, sorted(name for name, v in ks if v >= KS_FAIL)


def select(cache, bar_ath=None, log=print, grid_override=None, folds=None, merge_only=False):
    """Leave-one-match-out selection. Each finished fold is saved under cache/phase1/folds, so a
    run can be split (folds=[...]) or resumed; merge_only combines saved folds without decoding."""
    import pickle
    groups = _group_sims(cache)
    if len(groups) < 2:
        raise SystemExit("need simulated games from at least 2 IDSSE matches")
    configs = grid_override or grid()
    keys = [config_key(c) for c in configs]
    fold_dir = Path(cache) / "phase1" / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    signature = {"keys": keys, "matches": sorted(groups), "t_train": T_TRAIN,
                 "code_version": CODE_VERSION}
    todo = [g for g in sorted(groups) if folds is None or g in folds]
    ks, exclude = ([], [])
    data = None
    if not merge_only:
        log("featurising {} simulated games ...".format(sum(len(v) for v in groups.values())))
        data = {g: _load_featurised(paths, log) for g, paths in groups.items()}
        if bar_ath is not None:
            ks, exclude = shift_guard([it for its in data.values() for it in its], bar_ath)
            log("  shift guard excludes: {}".format(", ".join(exclude) or "nothing"))
        for held in todo:
            t0 = time.time()
            acc = {(k, t): Accumulator() for k in keys + ["baseline_w50"] for t in relabel.THRESHOLDS_S}
            train = [it for g, its in data.items() if g != held for it in its]
            model, priors = fit_models(train, exclude)
            for it in data[held]:
                scores = model.scores(it["features"])
                change = model.change_logit(it["features"])
                edges = model.edge_logits(it["features"])
                for t, gt in it["variants"].items():
                    acc[("baseline_w50", t)].add(gt, it["baseline"])
                for c, k in zip(configs, keys):
                    pred = decode.decode_match(scores, it["periods"], it["n_frames"], c, priors, change,
                                               edges)
                    for t, gt in it["variants"].items():
                        acc[(k, t)].add(gt, pred)
            with open(fold_dir / (held + ".pkl"), "wb") as handle:
                pickle.dump({"signature": signature, "acc": acc, "excluded": exclude, "ks": ks}, handle)
            log("  fold {} done in {:.0f}s".format(held, time.time() - t0))

    acc = {(k, t): Accumulator() for k in keys + ["baseline_w50"] for t in relabel.THRESHOLDS_S}
    per_match = {k: [] for k in keys}
    missing = []
    for g in sorted(groups):
        path = fold_dir / (g + ".pkl")
        if not path.is_file():
            missing.append(g)
            continue
        with open(path, "rb") as handle:
            saved = pickle.load(handle)
        if saved["signature"] != signature:
            missing.append(g + " (stale)")
            continue
        exclude, ks = saved["excluded"], saved["ks"]
        for k in keys:
            per_match[k].append(float(np.mean([saved["acc"][(k, t)].spell_f1()
                                               for t in relabel.THRESHOLDS_S])))
        for key, a in saved["acc"].items():
            acc[key].tp += a.tp
            acc[key].n_pred += a.n_pred
            acc[key].n_gt += a.n_gt
            acc[key].cm3 += a.cm3
            acc[key].flags.extend(a.flags)
    if missing:
        return {"incomplete": missing}

    table = []
    for c, k in zip(configs + [{"decoder": "baseline"}], keys + ["baseline_w50"]):
        per_t = {t: acc[(k, t)].spell_f1() for t in relabel.THRESHOLDS_S}
        table.append({"key": k, "config": c, "objective": float(np.mean(list(per_t.values()))),
                      "spell_f1_by_threshold": {str(t): v for t, v in per_t.items()},
                      "macro_f1_t1": acc[(k, T_TRAIN)].macro_f1(),
                      "n_pred_t1": acc[(k, T_TRAIN)].n_pred, "n_gt_t1": acc[(k, T_TRAIN)].n_gt})
    decoders = [r for r in table if r["key"] != "baseline_w50"]
    for r in decoders:
        r["per_match"] = per_match[r["key"]]
    chosen, best, paired = choose(decoders)
    base = next(r for r in table if r["key"] == "baseline_w50")
    return {"chosen": chosen, "best": best, "baseline": base,
            "paired": paired,
            "chosen_ci_t1": acc[(chosen["key"], T_TRAIN)].bootstrap(),
            "baseline_ci_t1": acc[("baseline_w50", T_TRAIN)].bootstrap(),
            "t_train": T_TRAIN, "tie_band": TIE_BAND, "excluded": exclude,
            "shift_ks": ks, "table": sorted(table, key=lambda r: -r["objective"]),
            "matches": sorted(groups)}


def choose(decoders):
    """Selection rule. decoders: rows with key, config, objective, per_match (one objective per
    held-out match, same match order in every row). Returns (chosen, best, paired_report)."""
    ranked = sorted(decoders, key=lambda r: -r["objective"])
    best = ranked[0]
    close = sorted([r for r in ranked if r["objective"] >= best["objective"] - TIE_BAND],
                   key=lambda r: (FAMILY_RANK[r["config"]["decoder"]], -r["objective"]))
    chosen = close[0]
    paired = []
    families = sorted({r["config"]["decoder"] for r in ranked}, key=lambda f: FAMILY_RANK[f])
    for fam in families:
        if FAMILY_RANK[fam] <= FAMILY_RANK[chosen["config"]["decoder"]]:
            continue
        top = next(r for r in ranked if r["config"]["decoder"] == fam)
        wins = int(sum(a > b for a, b in zip(top["per_match"], chosen["per_match"])))
        paired.append({"family": fam, "key": top["key"], "wins": wins, "of": len(top["per_match"]),
                       "gain": top["objective"] - chosen["objective"]})
        if top["objective"] > chosen["objective"] and wins >= PAIRED_MIN:
            chosen = top
    return chosen, best, paired


def predict(cache, bar_ath, selection, events=None, log=print, boundary_xy=None):
    from . import kickoff
    groups = _group_sims(cache)
    items = [it for paths in groups.values() for it in _load_featurised(paths, log)]
    exclude = selection.get("excluded", [])
    model, priors = fit_models(items, exclude)
    f = features.compute(bar_ath, boundary_xy=boundary_xy)
    scores = model.scores(f)
    dead_prior, restart_prior = emission.tagging_priors(
        events if TAGGING_PRIOR else None, bar_ath.n_frames, bar_ath.fps)
    scores = model.scores_with_tagging_prior(scores, dead_prior if TAGGING_PRIOR else None)
    spans = []
    if events is not None:
        spans = kickoff.pre_kickoff_spans(bar_ath.periods, kickoff.kickoff_frames(events, bar_ath.fps),
                                          bar_ath.fps)
        scores = kickoff.constrain(scores, spans)
    c = selection["chosen"]["config"]
    change = model.change_logit(f)
    edges = model.edge_logits(f, restart_prior=restart_prior if TAGGING_PRIOR else None)
    labels = decode.decode_match(scores, bar_ath.periods, bar_ath.n_frames, c, priors, change, edges)
    again = decode.decode_match(scores, bar_ath.periods, bar_ath.n_frames, c, priors, change, edges)
    info = {"config": c, "deterministic_in_process": bool(np.array_equal(labels, again)),
            "shift_ks": shift_report(items, f, bar_ath.graded_mask()), "excluded": exclude,
            "context_ks_max": context_shift(items, f, bar_ath.graded_mask()),
            "pre_kickoff_spans": spans, "speed_field_unit": speed_field_unit(bar_ath),
            "tagging_prior_bias_nats": float(dead_prior.max(initial=0.0)),
            "boundary_margin_frames": int(np.count_nonzero(f["boundary_margin"])),
            "emission": model.to_dict(),
            "priors": {"params": priors["params"], "log_trans": priors["log_trans"].tolist()},
            "trained_on": sorted(groups), "t_train": T_TRAIN}
    return labels, info


# -- domain-shift and plausibility checks (verdicts only; no values need to leave the machine) ---

SHIFT_FEATURES = (("speed_med_1s", None), ("speed_med_3s", None), ("speed_med_10s", None),
                  ("slow_frac", None), ("speed_trend", None), ("gap_log", None),
                  ("ball_out", "known"), ("ball_line_dist", "known"), ("ball_centre_dist", "known"),
                  ("ball_speed", "known"), ("d_home", "known"), ("d_away", "known"),
                  ("team_margin", "known"), ("goalside_diff", "known"), ("near5_diff", "known"),
                  ("ball_att", "known"), ("ctrl_run", "known"), ("push_diff", None),
                  ("spread_lr", None), ("width_diff", None), ("length_diff", None),
                  ("vel_diff", None), ("last_margin", None))
KS_PASS, KS_FAIL = 0.15, 0.30
# Bands from the external games (IDSSE dead-ball share 0.41-0.52; 370-511 raw DFL possessions per
# match, fewer under the competition rule) -- wide on purpose, they catch breakage, not tune.
DEAD_SHARE_BAND = (0.20, 0.70)
POSS_PER_MIN_BAND = (1.0, 8.0)


def ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (max CDF gap)."""
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / float(len(a))
    cdf_b = np.searchsorted(b, grid, side="right") / float(len(b))
    return float(np.max(np.abs(cdf_a - cdf_b)))


def shift_report(sim_items, target_features, target_graded, stride=10):
    out = []
    for name, cond in SHIFT_FEATURES:
        sim_vals = []
        for it in sim_items:
            g = it["graded"]
            if cond == "known":
                g = g & (it["features"]["ball_known"] > 0)
            sim_vals.append(it["features"][name][g][::stride])
        tg = target_graded & (target_features["ball_known"] > 0) if cond == "known" else target_graded
        out.append((name, ks_statistic(np.concatenate(sim_vals), target_features[name][tg][::stride])))
    return out


def context_shift(sim_items, target_features, target_graded, stride=10):
    """Largest KS over the phase-5 context-window features (name, value)."""
    worst = (0.0, "")
    for k in sorted(x for x in target_features if x.startswith("ctx_")):
        sim = np.concatenate([it["features"][k][it["graded"]][::stride] for it in sim_items])
        v = ks_statistic(sim, target_features[k][target_graded][::stride])
        worst = max(worst, (v, k))
    return worst


def speed_field_unit(match):
    """Which unit the tracker's speed field is really in, judged against position-derived speed.
    The parser divided by 3.6 when the metadata said kph: ratio ~1 then means the field truly was
    km/h; ratio ~0.28 means it was already m/s."""
    kin = features.kinematic_speed(match)
    stored = match.speed.astype(np.float64)
    g = match.graded_mask()
    both = ~np.isnan(kin[g]) & ~np.isnan(stored[g]) & (kin[g] > 1.0)
    if not both.any():
        return "unknown"
    ratio = float(np.median(stored[g][both] / kin[g][both]))
    if 0.7 < ratio < 1.4:
        return "as declared in the metadata"
    if 0.2 < ratio < 0.4:
        return "m/s despite the metadata (the 3.6 division was wrong; speed field now unused)"
    return "unclear"


def plausibility(labels, periods, fps=25):
    graded = np.zeros(len(labels), dtype=bool)
    for _, a, b in periods:
        graded[a:b] = True
    from .schema import D
    dead_share = float((labels[graded] == D).mean())
    runs = metrics.valid_runs(np.where(graded, 0, X).astype(np.int8))
    n_spells = len(metrics.spells(labels, runs))
    per_min = n_spells / (graded.sum() / float(fps) / 60.0)
    return dead_share, per_min


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist()
                                     if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    return path
