"""Per-frame evidence for H / A / D, as log-likelihood ratios a decoder can sum over time.

Three models (logistic, or own gradient boosting since phase 4), trained on simulated external
games only:
  dead model  P(D | x)                        on every graded frame
  team model  P(H | x, alive, ball known)     ball geometry + team shape and movement
  gap model   P(H | x, alive, ball unknown)   team shape and movement + last / next sighting

Combined posterior: P(D) = pD; P(H) = (1 - pD) * pH; P(A) = (1 - pD) * (1 - pH), pH from the
team model where the ball is known and from the gap model where it is not.
Scores = log posterior - log class prior.

Edge detectors (phase 6, optional): two more boosted models on the dead-model features,
  restart  P(a dead -> alive edge lies within +/- EDGE_RADIUS frames | x)
  stop     P(an alive -> dead edge lies within +/- EDGE_RADIUS frames | x)
trained on every 10th graded frame plus every positive frame. Their log-odds lower the decoder's
cost of entering / leaving a stoppage where an edge is likely (decode.potts_edges).

`exclude` drops every design column built from a named feature -- the shift guard uses it to
keep features whose distribution on BAR-ATH differs strongly from the simulated games out of
both models.
"""

import numpy as np

from . import features as feat
from .logistic import Logistic
from .schema import A, D, H, X

CHANGE_RADIUS = 12      # frames: a turnover within +/- 0.5 s makes a frame positive
EDGE_RADIUS = 5         # frames: a stoppage edge within +/- 0.2 s makes a frame positive
EDGE_STRIDE = 10
TRAIN_STRIDE = 5
GBDT_STRIDE = 10      # trees: every 10th frame (consecutive frames are near-duplicates)
EPS = 1e-4


def make_learner(kind, l2=1.0):
    if kind == "logistic":
        return Logistic(l2=l2)
    if kind == "gbdt":
        from .gbdt import GBDT
        return GBDT()
    raise ValueError(kind)


class EmissionModel:
    """kind: 'logistic' (phases 1-3) or 'gbdt' (phase 4, own histogram boosting)."""

    def __init__(self, l2=1.0, exclude=(), kind="logistic", change=False, edges=False):
        self.kind = kind
        self.change = make_learner(kind, l2) if change else None
        self.restart = make_learner(kind, l2) if edges else None
        self.stop = make_learner(kind, l2) if edges else None
        self.dead = make_learner(kind, l2)
        self.team = make_learner(kind, l2)
        self.gap = make_learner(kind, l2)
        self.prior_dead = None
        self.exclude = tuple(sorted(exclude))

    def fit(self, feature_list, label_list):
        Xd, yd, Xt, yt, Xg, yg, Xc, yc = [], [], [], [], [], [], [], []
        for f, y in zip(feature_list, label_list):
            rows = np.flatnonzero(y != X)[::(GBDT_STRIDE if self.kind == "gbdt" else TRAIN_STRIDE)]
            Xd.append(feat.dead_design(f, rows, self.exclude))
            yd.append((y[rows] == D).astype(np.float64))
            alive = rows[(y[rows] != D) & (f["ball_known"][rows] > 0)]
            Xt.append(feat.team_design(f, alive, self.exclude))
            yt.append((y[alive] == H).astype(np.float64))
            blind = rows[(y[rows] != D) & (f["ball_known"][rows] == 0)]
            if not len(blind):          # full ball coverage: no ball-gap rows; keep the model fittable
                blind = alive[::50]
            Xg.append(feat.gap_design(f, blind, self.exclude))
            yg.append((y[blind] == H).astype(np.float64))
            if self.change is not None:
                live = rows[y[rows] != D]
                Xc.append(feat.change_design(f, live, self.exclude))
                yc.append(turnover_near(y, CHANGE_RADIUS)[live].astype(np.float64))
        Xd, yd = np.vstack(Xd), np.concatenate(yd)
        Xt, yt = np.vstack(Xt), np.concatenate(yt)
        Xg, yg = np.vstack(Xg), np.concatenate(yg)
        self.dead.fit(Xd, yd)
        self.team.fit(Xt, yt)
        self.gap.fit(Xg, yg)
        if self.change is not None:
            self.change.fit(np.vstack(Xc), np.concatenate(yc))
        if self.restart is not None:
            for learner, kind in ((self.restart, "restart"), (self.stop, "stop")):
                Xe, ye = [], []
                for f, y in zip(feature_list, label_list):
                    pos = edge_near(y, kind, EDGE_RADIUS) & (y != X)
                    rows = np.union1d(np.flatnonzero(y != X)[::EDGE_STRIDE], np.flatnonzero(pos))
                    Xe.append(feat.dead_design(f, rows, self.exclude))
                    ye.append(pos[rows].astype(np.float64))
                learner.fit(np.vstack(Xe), np.concatenate(ye))
        self.prior_dead = round(float(yd.mean()), 9)
        return self

    def posteriors(self, f):
        pd = np.clip(self.dead.predict_proba(feat.dead_design(f, None, self.exclude)), EPS, 1 - EPS)
        ph = np.clip(self.team.predict_proba(feat.team_design(f, None, self.exclude)), EPS, 1 - EPS)
        pg = np.clip(self.gap.predict_proba(feat.gap_design(f, None, self.exclude)), EPS, 1 - EPS)
        ph = np.where(f["ball_known"] > 0, ph, pg)
        return np.column_stack([(1 - pd) * ph, (1 - pd) * (1 - ph), pd])

    def scores(self, f):
        post = self.posteriors(f)
        prior = np.array([(1 - self.prior_dead) / 2, (1 - self.prior_dead) / 2, self.prior_dead])
        # rounded to 1e-6: anything finer is platform noise (libm exp/log differ by an ulp)
        return np.round(np.log(np.clip(post, EPS, 1.0)) - np.log(prior), 6)

    def scores_with_tagging_prior(self, scores, tagging_prior=None):
        """Add a weak dead-ball log-odds bias; a zero/omitted prior is a no-op."""
        if tagging_prior is None:
            return scores
        prior = np.asarray(tagging_prior, dtype=np.float64)
        if prior.shape != (len(scores),):
            raise ValueError("tagging_prior must have one value per frame")
        out = np.array(scores, copy=True)
        out[:, D] += prior
        return out

    def change_logit(self, f):
        """Per-frame log-odds that a turnover happens within CHANGE_RADIUS (None without model)."""
        if self.change is None:
            return None
        p = np.clip(self.change.predict_proba(feat.change_design(f, None, self.exclude)), EPS, 1 - EPS)
        return np.round(np.log(p / (1 - p)), 6)

    def edge_logits(self, f, restart_prior=None):
        """(restart, stop) per-frame log-odds of a nearby stoppage edge (None without models)."""
        if self.restart is None:
            return None
        Xd = feat.dead_design(f, None, self.exclude)
        out = []
        for learner in (self.restart, self.stop):
            p = np.clip(learner.predict_proba(Xd), EPS, 1 - EPS)
            out.append(np.round(np.log(p / (1 - p)), 6))
        if restart_prior is not None:
            prior = np.asarray(restart_prior, dtype=np.float64)
            if prior.shape != (len(out[0]),):
                raise ValueError("restart_prior must have one value per frame")
            out[0] = out[0] + prior
        return tuple(out)

    def to_dict(self):
        d = {"kind": self.kind, "dead": self.dead.to_dict(), "team": self.team.to_dict(),
             "gap": self.gap.to_dict(),
             "prior_dead": self.prior_dead, "exclude": list(self.exclude)}
        if self.restart is not None:
            d["restart"], d["stop"] = self.restart.to_dict(), self.stop.to_dict()
        return d


def tagging_priors(events, n_frames, fps=25, bias=0.65, jitter_s=2.0):
    """Return weak (dead-ball, restart-edge) log-odds priors from Smart Tagging set pieces.

    The dead-ball bias is applied only over tagged intervals. Restart-edge confidence tapers
    linearly around each interval end to tolerate automated-tag timing jitter. No frame is forced.
    """
    dead = np.zeros(n_frames, dtype=np.float64)
    restart = np.zeros(n_frames, dtype=np.float64)
    if events is None:
        return dead, restart
    radius = max(1, int(round(jitter_s * fps)))
    for event in events:
        if event.get("code", "").upper() != "SET PIECES":
            continue
        start = int(np.clip(event.get("start_frame", 0), 0, n_frames))
        end = int(np.clip(event.get("end_frame", start), 0, n_frames))
        if end > start:
            dead[start:end] = np.maximum(dead[start:end], bias)
        centre = int(np.clip(end, 0, max(0, n_frames - 1)))
        lo, hi = max(0, centre - radius), min(n_frames, centre + radius + 1)
        if hi > lo:
            taper = 1.0 - np.abs(np.arange(lo, hi) - centre) / float(radius + 1)
            restart[lo:hi] = np.maximum(restart[lo:hi], bias * taper)
    return dead, restart

def turnover_near(y, radius):
    """Boolean per frame: an H<->A transition lies within +/- radius frames."""
    y = np.asarray(y)
    t = np.flatnonzero((y[1:] != y[:-1]) & (y[1:] < D) & (y[:-1] < D)) + 1
    mark = np.zeros(len(y) + 1, dtype=np.int64)
    np.add.at(mark, np.clip(t - radius, 0, len(y)), 1)
    np.add.at(mark, np.clip(t + radius + 1, 0, len(y)), -1)
    return np.cumsum(mark)[:-1] > 0


def edge_near(y, kind, radius):
    """Boolean per frame: a dead -> alive ('restart') or alive -> dead ('stop') edge lies within
    +/- radius frames. Edges into or out of X (ungraded) do not count."""
    y = np.asarray(y)
    a, b = y[:-1], y[1:]
    if kind == "restart":
        t = np.flatnonzero((a == D) & (b < D)) + 1
    elif kind == "stop":
        t = np.flatnonzero((a < D) & (b == D)) + 1
    else:
        raise ValueError(kind)
    mark = np.zeros(len(y) + 1, dtype=np.int64)
    np.add.at(mark, np.clip(t - radius, 0, len(y)), 1)
    np.add.at(mark, np.clip(t + radius + 1, 0, len(y)), -1)
    return np.cumsum(mark)[:-1] > 0
