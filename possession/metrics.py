"""Our own implementation of the competition metrics, written from evaluation/possession_metrics.py.

It must agree exactly with the official scorer; `python -m possession check-scorer` proves that on
the committed baselines and on random perturbations, using the official module as the reference.

Additions over the official scorer, all for development and none affecting the numbers:
- pooled scoring over several matches (sum tp / fp / fn per IoU threshold before the F1), which is
  how one long match would be scored -- the right way to aggregate simulated validation games;
- a possession-level bootstrap over the pooled set;
- per-threshold counts, kept so a failure mode (too many / too few spells) is visible.
"""

import numpy as np

from .schema import A, D, H, X

SPELL_IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
TAUS = (12, 25)


def valid_runs(gt):
    graded = (gt != X).astype(np.int8)
    edges = np.diff(np.concatenate(([0], graded, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def fill_abstentions(pred, runs):
    """X -> last non-X label in the run, back-filled at the run head (official behaviour)."""
    filled = pred.copy()
    for start, stop in runs:
        window = filled[start:stop]
        known = np.flatnonzero(window != X)
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(window != X, np.arange(len(window)), -1))
        carry[carry < 0] = known[0]
        filled[start:stop] = window[carry]
    return filled


def segments(codes, runs):
    out = []
    for start, stop in runs:
        window = codes[start:stop]
        if not len(window):
            continue
        bounds = np.concatenate(([0], np.flatnonzero(window[1:] != window[:-1]) + 1, [len(window)]))
        for i in range(len(bounds) - 1):
            out.append((start + int(bounds[i]), start + int(bounds[i + 1]), int(window[bounds[i]])))
    return out


def spells(codes, runs):
    return [s for s in segments(codes, runs) if s[2] in (H, A)]


def spell_overlaps(gt_spells, pred_spells):
    """Same-team overlapping pairs as (iou, gt_index, pred_index), sorted for greedy matching.

    Two-pointer sweep over the time-sorted spell lists instead of the official all-pairs loop;
    identical output, linear rather than quadratic in the number of spells.
    """
    out = []
    j0 = 0
    for i, (gs, ge, gk) in enumerate(gt_spells):
        while j0 < len(pred_spells) and pred_spells[j0][1] <= gs:
            j0 += 1
        j = j0
        while j < len(pred_spells) and pred_spells[j][0] < ge:
            ps, pe, pk = pred_spells[j]
            if pk == gk and pe > gs:
                inter = min(ge, pe) - max(gs, ps)
                union = max(ge, pe) - min(gs, ps)
                out.append((inter / float(union), i, j))
            j += 1
    out.sort(key=lambda item: (-item[0], item[1], item[2]))
    return out


def matched_flags(overlaps, n_gt, threshold):
    flags = np.zeros(n_gt, dtype=bool)
    used = set()
    for iou, i, j in overlaps:
        if iou < threshold:
            break
        if flags[i] or j in used:
            continue
        flags[i] = True
        used.add(j)
    return flags


def _f1(tp, n_pred, n_gt):
    fp, fn = n_pred - tp, n_gt - tp
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def spell_counts(gt, pred):
    """Per-threshold (tp, n_pred, n_gt) plus the per-possession matched flags, for one match."""
    runs = valid_runs(gt)
    filled = fill_abstentions(pred, runs)
    g, p = spells(gt, runs), spells(filled, runs)
    overlaps = spell_overlaps(g, p)
    flags = np.vstack([matched_flags(overlaps, len(g), t) for t in SPELL_IOUS]) \
        if len(g) else np.zeros((len(SPELL_IOUS), 0), dtype=bool)
    return {"tp": flags.sum(axis=1), "n_pred": len(p), "n_gt": len(g), "flags": flags,
            "gt_spells": g, "pred_spells": p}


def macro_f1_counts(gt, pred):
    """3x4 confusion slice (GT H/A/D rows x prediction H/A/D/X columns)."""
    cm = np.bincount(gt.astype(np.int64) * 4 + pred.astype(np.int64), minlength=16).reshape(4, 4)
    return cm[:3, :]


def macro_f1_from_cm(cm3):
    f1s = []
    for k in range(3):
        tp = float(cm3[k, k])
        fp = float(cm3[:, k].sum()) - tp
        fn = float(cm3[k, :].sum()) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def transitions(codes, runs):
    out = []
    for start, stop in runs:
        window = codes[start:stop]
        for i in (np.flatnonzero(window[1:] != window[:-1]) + 1).tolist():
            out.append((start + i, int(window[i - 1]), int(window[i])))
    return out


def transition_f1(gt, pred, tol):
    """Transitions matched one-to-one on destination label within +/- tol (official typed='to')."""
    runs = valid_runs(gt)
    gtt, prt = transitions(gt, runs), transitions(fill_abstentions(pred, runs), runs)
    tp = 0
    lat = []
    for key in set(t[2] for t in gtt) | set(t[2] for t in prt):
        gs = [t[0] for t in gtt if t[2] == key]
        ps = [t[0] for t in prt if t[2] == key]
        i = j = 0
        while i < len(gs) and j < len(ps):
            delta = ps[j] - gs[i]
            if delta < -tol:
                j += 1
            elif delta > tol:
                i += 1
            else:
                tp += 1
                lat.append(delta)
                i += 1
                j += 1
    return {"f1": _f1(tp, len(prt), len(gtt)), "n_gt": len(gtt), "n_pred": len(prt),
            "tp": tp, "mean_abs_latency": float(np.mean(np.abs(lat))) if lat else None}


def evaluate(gt, pred):
    """Headline numbers for one match, mirroring the official evaluate()."""
    gt = np.asarray(gt, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.int8)
    if gt.shape != pred.shape:
        raise ValueError("length mismatch {} vs {}".format(gt.shape, pred.shape))
    sc = spell_counts(gt, pred)
    per_iou = {t: _f1(int(tp), sc["n_pred"], sc["n_gt"]) for t, tp in zip(SPELL_IOUS, sc["tp"])}
    cm3 = macro_f1_counts(gt, pred)
    runs = valid_runs(gt)
    filled = fill_abstentions(pred, runs)
    durations = [b - a for a, b, _ in sc["pred_spells"]]
    gt_durations = [b - a for a, b, _ in sc["gt_spells"]]
    return {
        "spell_f1": float(np.mean(list(per_iou.values()))),
        "spell_f1_per_iou": per_iou,
        "macro_f1": macro_f1_from_cm(cm3),
        "n_gt_spells": sc["n_gt"], "n_pred_spells": sc["n_pred"],
        "pred_mean_spell_s": float(np.mean(durations)) / 25 if durations else 0.0,
        "gt_mean_spell_s": float(np.mean(gt_durations)) / 25 if gt_durations else 0.0,
        "n_graded": int(cm3.sum()), "n_abstain_graded": int(cm3[:, X].sum()),
        "alive_dead_f1": _alive_dead_f1(cm3),
        "team_acc_both_alive": _team_acc(cm3),
        "transitions": {tol: transition_f1(gt, pred, tol) for tol in TAUS},
        "n_pred_segments": len(segments(filled, runs)),
    }


def _alive_dead_f1(cm3):
    tp = float(cm3[:D, :D].sum())
    fp = float(cm3[D, :D].sum())
    fn = float(cm3[:D, D:].sum())
    return _f1(tp, tp + fp, tp + fn) if tp else 0.0


def _team_acc(cm3):
    both = float(cm3[:D, :D].sum())
    return float(cm3[H, H] + cm3[A, A]) / both if both else 0.0


def evaluate_pooled(pairs, bootstrap_n=0, seed=0):
    """Pool several (gt, pred) matches into one score, as if they were one long match.

    spell F1 is computed from summed tp / n_pred / n_gt per threshold; macro-F1 from the summed
    confusion. The bootstrap resamples ground-truth possessions across the pooled set (the
    official bootstrap_spell_f1 scheme), holding n_pred fixed.
    """
    tp = np.zeros(len(SPELL_IOUS), dtype=np.int64)
    n_pred = n_gt = 0
    cm3 = np.zeros((3, 4), dtype=np.int64)
    all_flags = []
    for gt, pred in pairs:
        gt = np.asarray(gt, dtype=np.int8)
        pred = np.asarray(pred, dtype=np.int8)
        sc = spell_counts(gt, pred)
        tp += sc["tp"]
        n_pred += sc["n_pred"]
        n_gt += sc["n_gt"]
        cm3 += macro_f1_counts(gt, pred)
        all_flags.append(sc["flags"])
    spell = float(np.mean([_f1(int(t), n_pred, n_gt) for t in tp]))
    out = {"spell_f1": spell, "macro_f1": macro_f1_from_cm(cm3),
           "n_gt_spells": n_gt, "n_pred_spells": n_pred, "n_matches": len(pairs)}
    if bootstrap_n and n_gt:
        flags = np.hstack(all_flags)
        rng = np.random.RandomState(seed)
        draws = np.empty(bootstrap_n)
        for d in range(bootstrap_n):
            pick = rng.randint(0, n_gt, n_gt)
            f = []
            for row in flags:
                t = int(row[pick].sum())
                fp = max(0, n_pred - t)
                f.append(2.0 * t / (2 * t + fp + (n_gt - t)) if t else 0.0)
            draws[d] = np.mean(f)
        out["spell_f1_p5"], out["spell_f1_p95"] = (float(v) for v in np.percentile(draws, (5, 95)))
    return out
