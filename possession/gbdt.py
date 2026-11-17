"""Histogram gradient-boosted trees for binary classification. numpy only, deterministic.

Why our own: the evidence models need feature interactions a logistic model cannot express
(ball close to an away player AND ball slow AND away players goal-side ...). A scikit-learn
HistGradientBoosting probe showed the size of the gain; this implementation reproduces the idea
with the properties the competition needs: no randomness, no threads, and arithmetic arranged
so two machines produce the same trees.

Algorithm (logistic loss, Newton boosting):
  - each feature is cut into N_BINS quantile bins once, on the training data
  - every tree is grown level by level to MAX_DEPTH; at each level one np.bincount builds the
    gradient / hessian / count histograms of all nodes and features at once
  - split gain  G_L^2/(H_L+l2) + G_R^2/(H_R+l2) - G^2/(H+l2), children need >= MIN_LEAF samples;
    a node with no positive gain becomes a pass-through (all samples go left)
  - leaf value  -lr * G/(H+l2)

Platform independence: bin edges, probabilities, gains and leaf values are rounded well above
floating-point noise, sums run in a fixed order (bincount is sequential), and ties resolve to
the lowest (feature, bin) index.
"""

import numpy as np

N_BINS = 32
N_TREES = 150
MAX_DEPTH = 5
MIN_LEAF = 200
LEARNING_RATE = 0.1
L2 = 1.0


class GBDT:
    def __init__(self, n_trees=N_TREES, max_depth=MAX_DEPTH, min_leaf=MIN_LEAF,
                 learning_rate=LEARNING_RATE, l2=L2, n_bins=N_BINS):
        self.n_trees, self.max_depth, self.min_leaf = n_trees, max_depth, min_leaf
        self.lr, self.l2, self.n_bins = learning_rate, l2, n_bins
        self.edges = self.trees = self.base = None

    # -- binning --------------------------------------------------------------------------
    def _fit_bins(self, X):
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges = []
        for j in range(X.shape[1]):
            e = np.unique(np.round(np.quantile(X[:, j], qs), 9))
            self.edges.append(e)

    def _bin(self, X):
        Xb = np.empty(X.shape, dtype=np.int64)
        for j, e in enumerate(self.edges):
            Xb[:, j] = np.searchsorted(e, np.round(X[:, j], 9), side="right")
        return Xb

    # -- training -------------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, F = X.shape
        self._fit_bins(X)
        Xb = self._bin(X)
        NB = self.n_bins
        p0 = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        self.base = round(float(np.log(p0 / (1 - p0))), 9)
        raw = np.full(n, self.base)
        # (sample, feature) -> feature * NB + bin, built once; per level only the node offset changes
        base_idx = (np.arange(F, dtype=np.int32)[None, :] * NB + Xb.astype(np.int32)).ravel()
        node_stride = np.int32(F * NB)
        self.trees = []
        for _ in range(self.n_trees):
            p = np.round(1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30))), 12)
            g = p - y
            h = p * (1.0 - p)
            gF, hF = np.repeat(g, F), np.repeat(h, F)
            node = np.zeros(n, dtype=np.int64)
            levels = []
            for depth in range(self.max_depth):
                n_nodes = 1 << depth
                size = n_nodes * F * NB
                idx = base_idx if depth == 0 else base_idx + np.repeat(node.astype(np.int32) * node_stride, F)
                Gh = np.bincount(idx, weights=gF, minlength=size).reshape(n_nodes, F, NB)
                Hh = np.bincount(idx, weights=hF, minlength=size).reshape(n_nodes, F, NB)
                Ch = np.bincount(idx, minlength=size).reshape(n_nodes, F, NB)
                GL = np.cumsum(Gh, axis=2)[:, :, :-1]
                HL = np.cumsum(Hh, axis=2)[:, :, :-1]
                CL = np.cumsum(Ch, axis=2)[:, :, :-1]
                G = Gh.sum(axis=2, keepdims=True)
                H = Hh.sum(axis=2, keepdims=True)
                C = Ch.sum(axis=2, keepdims=True)
                GR, HR, CR = G - GL, H - HL, C - CL
                gain = GL ** 2 / (HL + self.l2) + GR ** 2 / (HR + self.l2) - G ** 2 / (H + self.l2)
                gain = np.round(gain, 9)
                gain[(CL < self.min_leaf) | (CR < self.min_leaf)] = -1.0
                flat = gain.reshape(n_nodes, -1)
                best = np.argmax(flat, axis=1)
                ok = flat[np.arange(n_nodes), best] > 0
                feat = np.where(ok, best // (NB - 1), 0)
                thr = np.where(ok, best % (NB - 1), NB)          # NB: everything goes left
                levels.append((feat.astype(np.int64), thr.astype(np.int64)))
                go_right = Xb[np.arange(n), feat[node]] > thr[node]
                node = node * 2 + go_right
            n_leaves = 1 << self.max_depth
            Gl = np.bincount(node, weights=g, minlength=n_leaves)
            Hl = np.bincount(node, weights=h, minlength=n_leaves)
            value = np.round(-self.lr * Gl / (Hl + self.l2), 9)
            self.trees.append((levels, value))
            raw = raw + value[node]
        return self

    # -- inference ------------------------------------------------------------------------
    def decision_function(self, X):
        Xb = self._bin(np.asarray(X, dtype=np.float64))
        n = len(Xb)
        rows = np.arange(n)
        raw = np.full(n, self.base)
        for levels, value in self.trees:
            node = np.zeros(n, dtype=np.int64)
            for feat, thr in levels:
                node = node * 2 + (Xb[rows, feat[node]] > thr[node])
            raw = raw + value[node]
        return raw

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(self.decision_function(X), -30, 30)))

    def to_dict(self):
        return {"kind": "gbdt", "n_trees": self.n_trees, "max_depth": self.max_depth,
                "min_leaf": self.min_leaf, "learning_rate": self.lr, "l2": self.l2,
                "n_bins": self.n_bins, "base": self.base,
                "edges": [e.tolist() for e in self.edges]}
