"""L2-regularised logistic regression fitted by Newton / IRLS. numpy only, deterministic.

Features are standardised with the training mean / std, an intercept is added, and the intercept
is not penalised.
"""

import numpy as np


class Logistic:
    def __init__(self, l2=1.0, max_iter=100, tol=1e-12):
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.mean_ = self.std_ = self.coef_ = None

    def _design(self, X):
        Z = (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_
        return np.column_stack([np.ones(len(Z)), Z])

    def fit(self, X, y, weight=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        w_obs = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        Z = self._design(X)
        k = Z.shape[1]
        beta = np.zeros(k)
        penalty = np.full(k, self.l2)
        penalty[0] = 0.0
        for _ in range(self.max_iter):
            eta = np.clip(Z @ beta, -30, 30)
            p = 1.0 / (1.0 + np.exp(-eta))
            r = w_obs * p * (1 - p)
            grad = Z.T @ (w_obs * (y - p)) - penalty * beta
            H = (Z * r[:, None]).T @ Z + np.diag(penalty) + 1e-9 * np.eye(k)
            step = np.linalg.solve(H, grad)
            beta = beta + step
            if np.max(np.abs(step)) < self.tol:
                break
        # Round the fitted parameters: BLAS / summation order make the last bits differ between
        # machines; rounding far above that noise makes the model itself platform-independent.
        self.mean_ = np.round(self.mean_, 9)
        self.std_ = np.round(self.std_, 9)
        self.coef_ = np.round(beta, 9)
        return self

    def predict_proba(self, X):
        eta = np.clip(self._design(X) @ self.coef_, -30, 30)
        return 1.0 / (1.0 + np.exp(-eta))

    def to_dict(self):
        return {"l2": self.l2, "mean": self.mean_.tolist(), "std": self.std_.tolist(),
                "coef": self.coef_.tolist()}
