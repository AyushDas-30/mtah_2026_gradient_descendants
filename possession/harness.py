"""Run a method over a set of labelled matches and score it the way the competition would.

A method is any callable Match -> int8 label codes (one per frame). Methods never see
Match.labels: the harness hands them a label-free copy, so leakage is impossible by construction.

Two evaluation sets:
  sim     -- simulated external games (IDSSE + noise profile). This is where choices are made.
  public  -- BAR-ATH with the public 5-minute ground truth. Pipeline sanity only, never selection.
"""

import dataclasses
from pathlib import Path

import numpy as np

from . import metrics
from .schema import Match


def _strip_labels(match):
    return dataclasses.replace(match, labels=None)


def run(method, matches, bootstrap_n=0, seed=0):
    pairs, per_match = [], []
    for match in matches:
        if match.labels is None:
            raise ValueError("{} has no labels to score against".format(match.name))
        pred = np.asarray(method(_strip_labels(match)), dtype=np.int8)
        if pred.shape != (match.n_frames,):
            raise ValueError("{}: method returned shape {}".format(match.name, pred.shape))
        result = metrics.evaluate(match.labels, pred)
        per_match.append((match.name, result))
        pairs.append((match.labels, pred))
    pooled = metrics.evaluate_pooled(pairs, bootstrap_n=bootstrap_n, seed=seed)
    return pooled, per_match


def sim_matches(cache):
    paths = sorted((Path(cache) / "sim").glob("*.npz"))
    return [Match.load(p) for p in paths]


def format_result(name, pooled, per_match):
    lines = ["{:<32} spellF1 {:.4f}  macroF1 {:.4f}  spells pred/gt {}/{}".format(
        name, pooled["spell_f1"], pooled["macro_f1"], pooled["n_pred_spells"], pooled["n_gt_spells"])]
    if "spell_f1_p5" in pooled:
        lines.append("{:<32} spellF1 90% interval [{:.4f}, {:.4f}]".format(
            "", pooled["spell_f1_p5"], pooled["spell_f1_p95"]))
    for match_name, r in per_match:
        lines.append("    {:<28} spellF1 {:.4f}  macroF1 {:.4f}  spells {}/{}  mean {:.1f}s/{:.1f}s".format(
            match_name, r["spell_f1"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
            r["pred_mean_spell_s"], r["gt_mean_spell_s"]))
    return "\n".join(lines)
