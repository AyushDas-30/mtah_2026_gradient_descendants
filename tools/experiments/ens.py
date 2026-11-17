import os, sys, numpy as np
import rseval
from rseval import *
import rsgrid_lib as G
from pathlib import Path
def load(folder):
    rseval.FOLDS = Path(os.path.expanduser(folder)); return {it["name"]: it for it in load_folds(with_edges=True)}
S = os.environ.get("XPSET", "")
base = load("~/work/rs/folds" + S)
mem = [load("~/work/rs/folds" + S + "_" + v) if v else base for v in sys.argv[1].split(",")]
w = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1.0] * len(mem)
ens = {}
for k, it in base.items():
    e = dict(it); e["scores"] = sum(wi * m[k]["scores"] for wi, m in zip(w, mem)) / sum(w); ens[k] = e
for team in (7, 10):
    dec = G.mk(32, 0, 100, lam_team=team)
    a = {k: objective([it], dec)[0] for k, it in base.items()}
    b = {k: objective([it], dec)[0] for k, it in ens.items()}
    pa = objective(list(base.values()), dec)[0]; pb = objective(list(ens.values()), dec)[0]
    m = {}
    for k in a: m.setdefault(k.split("~")[0], []).append(b[k] - a[k])
    d = np.array([b[k] - a[k] for k in a])
    print("team %2d pooled %.4f -> %.4f | games %d/14 matches %d/7 mean %+.4f | per-match %s" % (team, pa, pb, (d > 0).sum(),
          sum(np.mean(v) > 0 for v in m.values()), d.mean(), " ".join("%+.4f" % np.mean(m[x]) for x in sorted(m))))
