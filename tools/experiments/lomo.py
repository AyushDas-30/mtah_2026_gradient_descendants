"""LOMO decoder selection for a score ensemble. usage: lomo.py LABEL members [weights]; resumable."""
import os, sys, time, pickle, itertools, numpy as np
import rseval
from rseval import *
import rsgrid_lib as G
from possession import metrics, relabel
from pathlib import Path
T0 = time.time()
S = os.environ.get("XPSET", "")
GRID = list(itertools.product((5, 7, 10, 14), (16, 32, 64, 128), (50, 70, 100, 140)))
def load(v):
    rseval.FOLDS = Path(os.path.expanduser("~/work/rs/folds" + S + ("_" + v if v else ""))); return {it["name"]: it for it in load_folds(with_edges=True)}
label = sys.argv[1]; members = sys.argv[2].split(",")
w = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1.0] * len(members)
cache = Path("lomo_%s%s.pkl" % (label, S))
res = pickle.load(open(cache, "rb")) if cache.exists() else {}
if len(res) < len(GRID):
    mem = [load(v) for v in members]
    items = {}
    for k, it in mem[0].items():
        e = dict(it); e["scores"] = sum(wi * m[k]["scores"] for wi, m in zip(w, mem)) / sum(w); items[k] = e
    for cfg in GRID:
        if cfg in res: continue
        if time.time() - T0 > 150: break
        team, gamma, ld = cfg; dec = G.mk(gamma, 0, ld, lam_team=team); out = {}
        for k, it in items.items():
            pred = dec(it); out[k] = {}
            for t, gt in it["variants"].items():
                sc = metrics.spell_counts(gt, pred); out[k][t] = (sc["tp"].copy(), sc["n_pred"], sc["n_gt"])
        res[cfg] = out; pickle.dump(res, open(cache, "wb"))
print(label, "configs done %d/%d" % (len(res), len(GRID)))
if len(res) == len(GRID):
    games = sorted(next(iter(res.values())))
    matches = sorted({g.split("~")[0] for g in games})
    def obj(cfg, gs):
        vals = []
        for t in relabel.THRESHOLDS_S:
            tp = sum(res[cfg][g][t][0] for g in gs); npd = sum(res[cfg][g][t][1] for g in gs); ngt = sum(res[cfg][g][t][2] for g in gs)
            vals.append(np.mean([metrics._f1(int(x), npd, ngt) for x in tp]))
        return float(np.mean(vals))
    held = {}
    for m in matches:
        tr = [g for g in games if not g.startswith(m)]; te = [g for g in games if g.startswith(m)]
        best = max(GRID, key=lambda c: obj(c, tr)); held[m] = (obj(best, te), best)
    allg = max(GRID, key=lambda c: obj(c, games))
    pickle.dump({"held": held, "best_all": (allg, obj(allg, games))}, open("lomo_%s%s_sel.pkl" % (label, S), "wb"))
    for m in matches: print("  %s %.4f %s" % (m, held[m][0], held[m][1]))
    print("  mean held %.4f | best overall %s %.4f" % (np.mean([v[0] for v in held.values()]), allg, obj(allg, games)))
