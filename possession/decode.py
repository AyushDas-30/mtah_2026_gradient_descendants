"""Turn per-frame scores into possession segments. Every period is decoded on its own.

Both decoders work on blocks of BLOCK frames (5 frames = 0.2 s) for speed and then place every
boundary at the exact frame with a local search (refine_boundaries).

All decoding arithmetic is integer: scores, penalties and priors are converted to fixed point
(SCALE units per nat) before any sum or comparison. Floating-point results differ in the last
bits between machines and numpy builds (BLAS, SIMD summation order, libm); in floating point
those differences occasionally flip a near-tie and change a frame. Integer sums and comparisons
are exact everywhere, so the same scores give the same file on every platform.

potts  Viterbi over H / A / D where staying costs nothing and switching costs a penalty
       (lam_team for H<->A, lam_dead for alive<->dead). The penalties are, in effect, the price
       of one more possession -- the knob that decides how many possessions are reported.

potts_edges  Potts whose alive->dead and dead->alive costs vary over time (phase 6): at block t
       the cost is lam_dead - gamma * logit, clipped to [0, 3 lam_dead], where logit is the
       edge detector's largest log-odds in the block (stop detector for entering a stoppage,
       restart detector for leaving one). Without edge evidence it is plain Potts.

hsmm   Semi-Markov Viterbi: a path is a sequence of whole segments, each scored by its summed
       evidence + beta * log duration prior(state, length) + log transition prob(prev -> state).
       Two consecutive segments never share a state, so a possession cannot be split in two.
       Duration priors are log-normal per state family (possession / dead ball), fitted on
       external labels only.
"""

import numpy as np

from .schema import A, D, H, X

BLOCK = 5
NEG = -1e18                 # float sentinel (priors)
NEG_INT = -(1 << 59)        # integer sentinel (decoding)
SCALE = 1000                # fixed-point units per nat


def to_fixed(x):
    """Float array -> int64 fixed point; NEG-like values map to NEG_INT."""
    x = np.asarray(x, dtype=np.float64)
    out = np.rint(np.clip(x, -1e9, 1e9) * SCALE).astype(np.int64)
    out[x <= NEG / 2] = NEG_INT
    return out


# -- block helpers -------------------------------------------------------------------------

def block_scores(scores, start, stop, block=BLOCK):
    """Sum frame scores into blocks over [start, stop). Last block may be short."""
    seg = scores[start:stop]
    n = int(np.ceil(len(seg) / block))
    pad = n * block - len(seg)
    if pad:
        seg = np.vstack([seg, np.zeros((pad, seg.shape[1]), dtype=seg.dtype)])
    return seg.reshape(n, block, -1).sum(axis=1)


def blocks_to_frames(states, start, stop, block=BLOCK):
    return np.repeat(states, block)[:stop - start]


# -- Potts ---------------------------------------------------------------------------------

def potts(bscores, lam_team, lam_dead):
    """bscores: int64 fixed-point block scores; penalties in the same units."""
    n, K = bscores.shape
    cost = np.array([[0, lam_team, lam_dead],
                     [lam_team, 0, lam_dead],
                     [lam_dead, lam_dead, 0]], dtype=np.int64)
    e = bscores
    delta = e[0].copy()
    back = np.zeros((n, K), dtype=np.int8)
    for t in range(1, n):
        cand = delta[:, None] - cost             # cand[j, k]: from j to k
        back[t] = np.argmax(cand, axis=0)
        delta = cand[back[t], np.arange(K)] + e[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(delta))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def potts_tv(bscores, team_cost, lam_dead):
    """Potts with a per-block H<->A switch cost (int64 array, >= 0), constant dead cost."""
    n, K = bscores.shape
    delta = bscores[0].copy()
    back = np.zeros((n, K), dtype=np.int8)
    for t in range(1, n):
        c = team_cost[t]
        cost = np.array([[0, c, lam_dead], [c, 0, lam_dead], [lam_dead, lam_dead, 0]], dtype=np.int64)
        cand = delta[:, None] - cost
        back[t] = np.argmax(cand, axis=0)
        delta = cand[back[t], np.arange(K)] + bscores[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(delta))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def potts_edges(bscores, lam_team, dead_in, dead_out, lam_dead_out=None, restart_mask=None):
    """Potts with per-block alive/dead costs and an optional dead-exit cost cap.

    ``dead_out`` and ``lam_dead_out`` are int64 fixed-point costs. With no cap, behavior is
    exactly the legacy decoder. When enabled, the cap lowers D -> H/A cost at detected restart
    blocks while preserving the team-switch penalty.
    """
    n, K = bscores.shape
    delta = bscores[0].copy()
    back = np.zeros((n, K), dtype=np.int8)
    for t in range(1, n):
        i, o = dead_in[t], dead_out[t]
        if lam_dead_out is not None and restart_mask is not None and restart_mask[t]:
            o = min(o, lam_dead_out)
        cost = np.array([[0, lam_team, i], [lam_team, 0, i], [o, o, 0]], dtype=np.int64)
        cand = delta[:, None] - cost
        back[t] = np.argmax(cand, axis=0)
        delta = cand[back[t], np.arange(K)] + bscores[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(delta))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def edge_costs(logit, start, stop, n_blocks, lam_dead, gamma, block=BLOCK):
    """Per-block fixed-point cost lam_dead - gamma * (max log-odds in the block), in [0, 3 lam_dead]."""
    seg = np.asarray(logit[start:stop], dtype=np.float64)
    pad = n_blocks * block - len(seg)
    if pad:
        seg = np.concatenate([seg, np.full(pad, seg[-1])])
    b = seg.reshape(n_blocks, block).max(axis=1)
    return to_fixed(np.clip(lam_dead - gamma * b, 0.0, 3.0 * lam_dead))


def potts_restart(bscores, lam_team, lam_dead, cost_same, cost_switch):
    """Potts with memory of who had the ball before a stoppage.

    Four states: H, A, D-after-H, D-after-A (both dead states emit the D score). Entering dead
    from a team leads to that team's dead state; leaving it back to the same team costs
    lam_dead + cost_same, to the other team lam_dead + cost_switch. All int64 fixed point.
    """
    n = len(bscores)
    INF = np.int64(1) << 50
    e = bscores[:, [0, 1, 2, 2]]
    cost = np.array([
        [0, lam_team, lam_dead, INF],
        [lam_team, 0, INF, lam_dead],
        [lam_dead + cost_same, lam_dead + cost_switch, 0, INF],
        [lam_dead + cost_switch, lam_dead + cost_same, INF, 0]], dtype=np.int64)
    K = 4
    delta = e[0].copy()
    back = np.zeros((n, K), dtype=np.int8)
    for t in range(1, n):
        cand = delta[:, None] - cost
        back[t] = np.argmax(cand, axis=0)
        delta = cand[back[t], np.arange(K)] + e[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(delta))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return np.where(path >= 2, 2, path).astype(np.int8)


# -- semi-Markov ---------------------------------------------------------------------------

def lognormal_logpdf_blocks(mu, sigma, max_blocks, block=BLOCK, fps=25):
    """log density of a segment lasting d blocks, d = 1..max_blocks (index 0 unused)."""
    d = np.arange(1, max_blocks + 1) * block / float(fps)
    logpdf = -np.log(d * sigma * np.sqrt(2 * np.pi)) - (np.log(d) - mu) ** 2 / (2 * sigma ** 2)
    return np.concatenate(([NEG], logpdf + np.log(block / float(fps))))


def hsmm(bscores, bl, log_trans, log_start, max_blocks):
    """All int64 fixed point: bscores (n, 3) already scaled by alpha; bl (3, max_blocks + 1) the
    beta-weighted log duration prior; log_trans (3, 3) with NEG_INT diagonal; log_start (3,)."""
    n, K = bscores.shape
    C = np.vstack([np.zeros((1, K), dtype=np.int64), np.cumsum(bscores, axis=0)])   # (n+1, K)
    delta = np.full((n + 1, K), NEG_INT, dtype=np.int64)
    M = np.full((n + 1, K), NEG_INT, dtype=np.int64)   # best score entering state k at boundary t
    Mprev = np.zeros((n + 1, K), dtype=np.int8)
    back_d = np.zeros((n + 1, K), dtype=np.int32)
    M[0] = log_start
    Mprev[0] = -1
    for t in range(1, n + 1):
        lo = max(0, t - max_blocks)
        prev = np.arange(lo, t)                  # segment starts
        d = t - prev                              # durations
        for k in range(K):
            cand = M[prev, k] + (C[t, k] - C[prev, k]) + bl[k, d]
            i = int(np.argmax(cand))
            delta[t, k] = cand[i]
            back_d[t, k] = d[i]
        # entering each state at boundary t from the best other state ending at t
        trans = delta[t][:, None] + log_trans     # [j, k]
        j = np.argmax(trans, axis=0)
        M[t] = trans[j, np.arange(K)]
        Mprev[t] = j
    # backtrack
    path = np.empty(n, dtype=np.int8)
    k = int(np.argmax(delta[n]))
    t = n
    while t > 0:
        d = int(back_d[t, k])
        path[t - d:t] = k
        t -= d
        if t > 0:
            k = int(Mprev[t, k])
    return path


# -- boundary refinement -------------------------------------------------------------------

def refine_boundaries(frame_states, frame_scores, radius=BLOCK):
    """Move each boundary to the frame within +/- radius that best separates the two states."""
    s = frame_states.copy()
    n = len(s)
    cs = np.vstack([np.zeros((1, frame_scores.shape[1]), dtype=frame_scores.dtype),
                    np.cumsum(frame_scores, axis=0)])
    changes = np.flatnonzero(s[1:] != s[:-1]) + 1
    prev_b = 0
    for idx, b in enumerate(changes):
        k1, k2 = int(s[b - 1]), int(s[b])
        nxt = changes[idx + 1] if idx + 1 < len(changes) else n
        lo = max(prev_b + 1, b - radius)
        hi = min(nxt - 1, b + radius)
        if lo > hi:
            prev_b = b
            continue
        cand = np.arange(lo, hi + 1)
        # boundary at f: frames < f are k1, >= f are k2
        gain = cs[cand, k1] - cs[cand, k2]
        f = int(cand[np.argmax(gain)])
        s[min(b, f):max(b, f)] = k1 if f > b else k2
        prev_b = f
    return s


# -- whole-match decoding ------------------------------------------------------------------

def decode_match(scores, periods, n_frames, config, priors=None, change=None, edges=None):
    """Labels for every frame: decoded inside periods, X elsewhere."""
    out = np.full(n_frames, X, dtype=np.int8)
    q = to_fixed(config["alpha"] * np.asarray(scores, dtype=np.float64))
    if config["decoder"] == "hsmm":
        bl = to_fixed(config["beta"] * priors["logdur"])
        lt = to_fixed(priors["log_trans"])
        ls = to_fixed(priors["log_start"])
    for _, start, stop in periods:
        b = block_scores(q, start, stop)
        if config["decoder"] == "potts" and config.get("gamma", 0) and change is not None:
            # switch cost lowered where the change detector fires: lam_team - gamma * logit, >= 0
            lc = np.clip(np.asarray(change[start:stop], dtype=np.float64), -6.0, 6.0)
            pad = len(b) * BLOCK - len(lc)
            lc = np.concatenate([lc, np.full(pad, lc[-1])]) if pad else lc
            block_logit = lc.reshape(len(b), BLOCK).max(axis=1)
            tc = to_fixed(np.maximum(0.0, config["lam_team"] - config["gamma"] * block_logit))
            path = potts_tv(b, tc, int(round(config["lam_dead"] * SCALE)))
        elif config["decoder"] == "potts_e":
            if edges is None:
                raise ValueError("potts_e needs edge log-odds")
            restart_logit, stop_logit = edges
            d_in = edge_costs(stop_logit, start, stop, len(b), config["lam_dead"], config["edge_gamma"])
            d_out = edge_costs(restart_logit, start, stop, len(b), config["lam_dead"], config["edge_gamma"])
            dead_out_cap = config.get("lam_dead_out")
            restart_mask = None
            if dead_out_cap is not None:
                dead_out_cap = int(round(dead_out_cap * SCALE))
                r = np.asarray(restart_logit[start:stop], dtype=np.float64)
                pad = len(b) * BLOCK - len(r)
                r = np.concatenate([r, np.full(pad, r[-1])]) if pad else r
                restart_mask = r.reshape(len(b), BLOCK).max(axis=1) > 0.0
            path = potts_edges(b, int(round(config["lam_team"] * SCALE)), d_in, d_out,
                               lam_dead_out=dead_out_cap, restart_mask=restart_mask)
        elif config["decoder"] == "potts":
            path = potts(b, int(round(config["lam_team"] * SCALE)), int(round(config["lam_dead"] * SCALE)))
        elif config["decoder"] == "potts_r":
            # restart costs: w * -log P(same / other team restarts), from external labels
            w, ps = config["restart_w"], priors["p_restart_same"]
            c_same = int(round(-w * np.log(ps) * SCALE))
            c_switch = int(round(-w * np.log(1.0 - ps) * SCALE))
            path = potts_restart(b, int(round(config["lam_team"] * SCALE)),
                                 int(round(config["lam_dead"] * SCALE)), c_same, c_switch)
        elif config["decoder"] == "hsmm":
            path = hsmm(b, bl, lt, ls, priors["max_blocks"])
        else:
            raise ValueError(config["decoder"])
        frames = blocks_to_frames(path, start, stop)
        frames = refine_boundaries(frames, q[start:stop])
        out[start:stop] = frames
    return out


# -- priors from labels (external data only) -----------------------------------------------

def fit_priors(label_list, max_seconds=300, fps=25, block=BLOCK):
    """Log-normal duration per family (possession: H/A pooled; dead: D), transition and start
    probabilities, all counted on (relabelled) external labels."""
    from .metrics import segments, valid_runs
    durations = {"poss": [], "dead": []}
    restarts = [1, 1]                               # [same team, other team], add-one smoothing
    trans = np.ones((3, 3))                         # add-one smoothing
    starts = np.ones(3)
    for y in label_list:
        runs = valid_runs(y)
        for s, e in runs:
            segs = segments(y, [(s, e)])
            starts[segs[0][2]] += 1
            for i, (a, b, k) in enumerate(segs):
                if 0 < i < len(segs) - 1:            # truncated edge segments say little
                    durations["dead" if k == D else "poss"].append((b - a) / float(fps))
                if i > 0:
                    trans[segs[i - 1][2], k] += 1
                if 0 < i < len(segs) - 1 and k == D and segs[i - 1][2] != D and segs[i + 1][2] != D:
                    restarts[0 if segs[i - 1][2] == segs[i + 1][2] else 1] += 1
    np.fill_diagonal(trans, 0.0)
    trans = trans / trans.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        log_trans = np.log(trans)
    np.fill_diagonal(log_trans, NEG)
    params = {}
    max_blocks = int(max_seconds * fps / block)
    logdur = np.zeros((3, max_blocks + 1))
    for fam, states in (("poss", (H, A)), ("dead", (D,))):
        x = np.log(np.maximum(np.array(durations[fam]), 0.2))
        mu, sigma = float(x.mean()), float(max(x.std(), 0.3))
        params[fam] = (mu, sigma, len(durations[fam]))
        for k in states:
            logdur[k] = lognormal_logpdf_blocks(mu, sigma, max_blocks, block, fps)
    return {"logdur": logdur, "log_trans": log_trans, "log_start": np.log(starts / starts.sum()),
            "max_blocks": max_blocks, "params": params,
            "p_restart_same": round(restarts[0] / float(sum(restarts)), 9)}
