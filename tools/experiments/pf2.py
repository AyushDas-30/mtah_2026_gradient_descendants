import os, numpy as np
import rseval
from rseval import *
import rsgrid_lib as G
from pathlib import Path
def per_game(folder):
    rseval.FOLDS = Path(os.path.expanduser(folder))
    its = load_folds(with_edges=True)
    return {it["name"]: objective([it], G.mk(32, 0, 100, lam_team=10))[0] for it in its}
S = os.environ.get("XPSET", ""); a = per_game("~/work/rs/folds" + S); b = per_game("~/work/rs/folds" + S + "_" + os.environ["XPV"])
d = np.array([b[k] - a[k] for k in sorted(a)])
for k in sorted(a): print(f"  {k:22s} {a[k]:.4f} -> {b[k]:.4f} ({b[k]-a[k]:+.4f})")
m = {}
for k in a: m.setdefault(k.split("~")[0], []).append(b[k] - a[k])
print("games improved %d/14, matches improved %d/7, mean %+.4f" % ((d > 0).sum(), sum(np.mean(v) > 0 for v in m.values()), d.mean()))
