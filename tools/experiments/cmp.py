import os, sys
from rseval import *
import rsgrid_lib as G
items = load_folds(with_edges=True)
print("variant", os.environ.get("XPV", "base"), "items", len(items))
for lt in (5, 7, 10, 14):
    print("  potts_e team %2d dead 100 g32: obj %.4f macro %.4f pred/gt %d/%d" % ((lt,) + objective(items, G.mk(32, 0, 100, lam_team=lt))), flush=True)
# per-frame team accuracy near changes
C, A, Dd, Ln = [], [], [], []
for it in items:
    y = it["variants"][1.0]; sc = it["scores"]; n = len(y)
    pred = np.where(sc[:, 0] > sc[:, 1], 0, 1)
    L = np.zeros(n); i = 0
    while i < n:
        j = i
        while j < n and y[j] == y[i]: j += 1
        L[i:j] = (j - i) / 25; i = j
    C.append(pred == y); A.append(y < 2); Ln.append(L)
C, A, Ln = map(np.concatenate, (C, A, Ln))
print("  frame team acc: all %.3f | in 0-2s spells %.3f | 2-5s %.3f | 5-10s %.3f" % (C[A].mean(), C[A & (Ln < 2)].mean(), C[A & (Ln >= 2) & (Ln < 5)].mean(), C[A & (Ln >= 5) & (Ln < 10)].mean()))
