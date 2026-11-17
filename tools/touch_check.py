"""Match-level check of the touch detector (Steps 6-7), before any leave-one-match-out run.

usage:  python tools/touch_check.py [--cache CACHE_DIR] [--labels SUBMISSION_CSV] [--sims N]

Measures, with the features exactly as the models see them:
  rate       detected touches per minute of play (BAR-ATH and simulated games)
  coverage   share of in-play frames whose last_touch_team is set
  precision  simulated games: last-touch team == true possessing team, touches < 1 s old
  recall     simulated games: share of true H<->A changes with a touch by the new team
             within +/- 0.5 s

In-play on BAR-ATH comes from a submission file (default: submissions/Gradient Descendants.csv),
because BAR-ATH has no labels beyond the public window; on simulated games from their labels.
Targets (PASS / FAIL) are printed per line; only aggregate numbers are printed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from possession import config, features  # noqa: E402
from possession.io import labels as label_io  # noqa: E402
from possession.schema import A, D, H, Match  # noqa: E402

RATE_RANGE = (15.0, 25.0)        # touches per minute of play
RATE_GAP_MAX = 0.25              # |sim - real| / real
COVERAGE_MIN = 0.25
PRECISION_MIN = 0.85
RECALL_MIN = 0.40
WINDOW = 12                      # frames: +/- 0.5 s for turnover recall


def touch_stats(f, alive, fps):
    touch = f["touch_team"]
    minutes = alive.sum() / (fps * 60.0)
    rate = float(((touch != 0) & alive).sum() / max(minutes, 1e-9))
    coverage = float((f["last_touch_team"][alive] != 0).mean()) if alive.any() else 0.0
    return rate, coverage


def precision_recall(f, y):
    alive = y < D
    lt, age = f["last_touch_team"], f["time_since_touch"]
    s = alive & (lt != 0) & (age < 1.0)
    precision = float(np.mean((lt[s] > 0) == (y[s] == H))) if s.any() else float("nan")
    changes = np.flatnonzero((y[1:] != y[:-1]) & (y[1:] < D) & (y[:-1] < D)) + 1
    touch = f["touch_team"]
    hit = 0
    for t in changes:
        want = 1.0 if y[t] == H else -1.0
        seg = touch[max(0, t - WINDOW):t + WINDOW + 1]
        hit += bool(np.any(seg == want))
    recall = hit / max(len(changes), 1)
    return precision, recall, int(s.sum()), len(changes)


def verdict(ok, text):
    print("  [{}] {}".format("PASS" if ok else "FAIL", text))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--labels", default="submissions/Gradient Descendants.csv")
    ap.add_argument("--sims", type=int, default=4, help="how many simulated games to check")
    args = ap.parse_args()
    cache = config.cache_dir(args.cache)
    results = []

    bar_rate = None
    bar_path = cache / (config.GAME + ".npz")
    if bar_path.is_file() and Path(args.labels).is_file():
        m = Match.load(bar_path)
        lab = label_io.read_labels(args.labels)
        alive = (lab < D) & m.graded_mask()
        bar_rate, bar_cov = touch_stats(features.compute(m), alive, m.fps)
        print("BAR-ATH")
        results.append(verdict(RATE_RANGE[0] <= bar_rate <= RATE_RANGE[1],
                               "touch rate {:.2f}/min (target {}-{})".format(bar_rate, *RATE_RANGE)))
        results.append(verdict(bar_cov >= COVERAGE_MIN,
                               "coverage {:.1%} of in-play frames (target >= {:.0%})".format(bar_cov, COVERAGE_MIN)))
    else:
        print("BAR-ATH skipped: need {} and {}".format(bar_path, args.labels))

    sims = sorted((cache / "sim").glob("*.npz"))[:args.sims]
    rates, covs, precs, recs = [], [], [], []
    for p in sims:
        s = Match.load(p)
        if s.labels is None:
            continue
        f = features.compute(s)
        y = s.labels
        rate, cov = touch_stats(f, (y < D) & (y >= H), s.fps)
        prec, rec, n_s, n_c = precision_recall(f, y)
        rates.append(rate), covs.append(cov), precs.append(prec), recs.append(rec)
        print("  {:24s} rate {:5.2f}/min  coverage {:5.1%}  precision {:.2f} ({} frames)  "
              "recall {:.2f} ({} changes)".format(s.name, rate, cov, prec, n_s, rec, n_c))
    if rates:
        r, c = float(np.mean(rates)), float(np.mean(covs))
        p, q = float(np.nanmean(precs)), float(np.mean(recs))
        print("SIMULATED GAMES (mean of {})".format(len(rates)))
        results.append(verdict(RATE_RANGE[0] <= r <= RATE_RANGE[1],
                               "touch rate {:.2f}/min (target {}-{})".format(r, *RATE_RANGE)))
        results.append(verdict(c >= COVERAGE_MIN, "coverage {:.1%} (target >= {:.0%})".format(c, COVERAGE_MIN)))
        results.append(verdict(p >= PRECISION_MIN, "precision {:.2f} (target >= {})".format(p, PRECISION_MIN)))
        results.append(verdict(q >= RECALL_MIN, "turnover recall {:.2f} (target >= {})".format(q, RECALL_MIN)))
        if bar_rate:
            gap = abs(r - bar_rate) / bar_rate
            results.append(verdict(gap <= RATE_GAP_MAX,
                                   "sim vs BAR-ATH rate gap {:.0%} (target <= {:.0%})".format(gap, RATE_GAP_MAX)))
    else:
        print("no simulated games found in {}".format(cache / "sim"))
    print("\n{} PASS, {} FAIL".format(sum(results), len(results) - sum(results)))
    print("All PASS -> run the leave-one-match-out evaluation on the smoothed simulated games.")


if __name__ == "__main__":
    main()
