"""Game-state-aware ball dropout and dead-ball motion for the simulator (phase 9).

Why: on BAR-ATH the tracker keeps the ball through live play (~0.87-0.94 of live frames) and
loses it mostly during stoppages (~0.2 of dead frames are visible), in few, long gaps. The
phase 1-6 simulator dropped the ball at a rate that ignores the game state (~0.65 live / ~0.5
dead), so the models learned from live play that is far more blind than the real one, and from
stoppages where the ball is visible more often than it really is.

Two measurements on BAR-ATH, both from a submission file (the game state comes from a previous
version of this pipeline -- code output, never hand labels):
  dead_patterns  for every stoppage of >= 2 s, its ball-visibility mask resampled to NB bins
  alive_gaps     lengths (frames) of the ball gaps strictly inside live play
They are aggregate shapes and lengths, kept in the local cache only (state_profile.json).

Simulation, for a clean external match with labels:
  1. dead-ball motion (transport_rest): inside each stoppage the ball is carried from where it
     went out to the restart spot at a walking/throwing pace, then rests there (placed / held)
     with occasional short handling moves; real stoppage balls are often static
  2. Gaussian smoothing (the tracker's, sigma 7 frames)
  3. every stoppage of >= DEAD_MIN_FRAMES gets a real visibility pattern; live play gets gaps
     from the measured live-gap lengths, at a rate fitted so live coverage is ALIVE_COVERAGE
Deterministic given the seed. Labels only locate stoppages; they are never changed.
"""

import json
from pathlib import Path

import numpy as np

from ..schema import D, Match

NB = 100                       # bins of a stoppage visibility pattern
DEAD_MEASURE_MIN = 50          # frames: stoppages measured on BAR-ATH (>= 2 s)
DEAD_MIN_FRAMES = 12           # frames: stoppages that get a pattern in the simulation
ALIVE_COVERAGE = 0.88          # live-play ball coverage target (BAR-ATH: 0.87 by our labels, 0.94 public)
SMOOTH_SIGMA = 7.0             # tracker smoothing (phase 7 measurement)
OVERHANG = True                # ball gaps run on past restarts / start before stops (measured)
# transport_rest parameters, fitted to BAR-ATH's visible dead-ball speed percentiles (25/50/75/90:
# 0.41 / 1.46 / 3.42 / 8.88 m/s); simulated 0.81 / 1.59 / 3.38 / 6.88
CARRY_SPEED = 4.0              # m/s, median transport pace (lognormal, sd 0.4)
CARRY_MAX_FRACTION = 0.7       # transport ends by this share of the stoppage
HANDLING_RATE = 0.3            # short moves per second while the ball rests
HANDLING_SPEED = 10.0          # m/s (lognormal, sd 0.5)
HANDLING_DURATION = 1.0        # s (exponential)
HANDLING_JITTER = 0.05         # m/s per frame of rest jitter
OVERHANG_SCALE = 1.0           # multiplies the measured overhang lengths (sensitivity tests)

# Experiments (phase 10): MTAH_STATE_OVERRIDES="HANDLING_RATE=0.15,ALIVE_COVERAGE=0.92" changes the
# constants above for one run. Unset (the default), nothing changes.
import os as _os
for _item in filter(None, _os.environ.get("MTAH_STATE_OVERRIDES", "").split(",")):
    _k, _v = _item.split("=")
    _k = _k.strip().upper()
    if _k not in ("ALIVE_COVERAGE", "OVERHANG_SCALE", "HANDLING_RATE", "HANDLING_SPEED",
                  "HANDLING_DURATION", "HANDLING_JITTER", "CARRY_SPEED", "CARRY_MAX_FRACTION"):
        raise ValueError("MTAH_STATE_OVERRIDES: unknown setting " + _k)
    globals()[_k] = float(_v)


def runs(mask):
    e = np.diff(np.concatenate([[0], np.asarray(mask).astype(int), [0]]))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


# -- measurement on BAR-ATH ----------------------------------------------------------------

def measure(match, codes):
    """match: BAR-ATH; codes: per-frame H/A/D/X labels from a submission (code output)."""
    graded = match.graded_mask()
    seen = ~np.isnan(match.ball[:, 0])
    dead = (codes == D) & graded
    alive = (codes < D) & graded
    patterns = []
    for a, b in runs(dead):
        if b - a < DEAD_MEASURE_MIN:
            continue
        s = seen[a:b].astype(float)
        idx = np.minimum((np.arange(NB) + 0.5) * (b - a) / NB, b - a - 1).astype(int)
        patterns.append((s[idx] > 0.5).astype(int).tolist())
    gaps = []
    for a, b in runs(alive):
        for c, d in runs(~seen[a:b]):
            if c > 0 and d < b - a:
                gaps.append(int(d - c))
    # overhangs: a stoppage that ends (starts) with the ball lost often keeps it lost into the
    # live play after (before) it -- the tracker re-acquires the ball late / loses it early
    post, pre = [], []
    for a, b in runs(dead):
        if b - a < DEAD_MEASURE_MIN:
            continue
        if not seen[b - 1]:
            k = b
            while k < len(seen) and not seen[k] and graded[k]:
                k += 1
            post.append(int(k - b))
        if not seen[a]:
            k = a - 1
            while k >= 0 and not seen[k] and graded[k]:
                k -= 1
            pre.append(int(a - 1 - k))
    return {"dead_patterns": patterns, "alive_gaps": gaps, "post_overhang": post, "pre_overhang": pre,
            "live_coverage": float(seen[alive].mean()), "dead_coverage": float(seen[dead].mean()),
            "n_stoppages": len(patterns), "n_alive_gaps": len(gaps)}


def save(profile, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# -- dead-ball motion ----------------------------------------------------------------------

def handling_offsets(n, rng, fps, rate, speed, duration, jitter):
    """Displacement (n, 2) m of bursty handling, bridged to zero at both ends."""
    v = np.zeros((n, 2))
    t = 0
    while t < n:
        t += int(rng.exponential(fps / max(rate, 1e-6)))
        if t >= n:
            break
        dur = max(3, int(rng.exponential(duration * fps)))
        sp = speed * np.exp(0.5 * rng.standard_normal())
        ang = rng.uniform(0, 2 * np.pi)
        v[t:t + dur] = sp * np.array([np.cos(ang), np.sin(ang)])
        t += dur
    v += jitter * rng.standard_normal((n, 2))
    off = np.cumsum(v, axis=0) / fps
    return off - np.linspace(0, 1, n)[:, None] * off[-1:]


def transport_rest(match, rng):
    """Clean ball with every stoppage replaced by carry-then-rest motion (labels locate them)."""
    ball = match.ball.copy()
    fps = match.fps
    if match.labels is None:
        return ball
    dead = match.labels == D
    for _, start, stop in match.periods:
        for a, b in runs(dead[start:stop]):
            a, b = a + start, b + start
            p0 = ball[a - 1] if a > start and not np.isnan(ball[a - 1, 0]) else ball[a]
            p1 = ball[b] if b < stop and not np.isnan(ball[b, 0]) else ball[b - 1]
            if np.isnan(p0).any() or np.isnan(p1).any():
                continue
            n = b - a
            dist = float(np.linalg.norm(p1 - p0))
            v = CARRY_SPEED * np.exp(0.4 * rng.standard_normal())
            tau = int(min(max(1, dist / v * fps * 1.5), CARRY_MAX_FRACTION * n)) if dist > 0 else 1
            tau = max(1, min(tau, n))
            u = (np.arange(tau) + 1) / float(tau)
            u = u * u * (3 - 2 * u)
            ball[a:a + tau] = p0[None] + u[:, None] * (p1 - p0)[None]
            ball[a + tau:b] = p1[None]
            r = b - (a + tau)
            if r > 10:
                ball[a + tau:b] += handling_offsets(r, rng, fps, HANDLING_RATE, HANDLING_SPEED,
                                                    HANDLING_DURATION, HANDLING_JITTER)
    return ball


# -- state-aware dropout -------------------------------------------------------------------

def degrade_ball(ball, labels, periods, rng, state, smooth, alive_coverage=ALIVE_COVERAGE):
    """ball: clean (already moved) ball; smooth(ball, periods) applies the tracker smoothing."""
    pats = np.array(state["dead_patterns"], dtype=bool)
    pool = np.array(state["alive_gaps"], dtype=np.int64)
    out = smooth(ball, periods)
    vis = ~np.isnan(out[:, 0])
    drop = np.zeros(len(out), dtype=bool)
    dead = labels == D
    alive = labels < D
    post = np.array(state.get("post_overhang", []), dtype=np.int64)
    pre = np.array(state.get("pre_overhang", []), dtype=np.int64)
    alive_run = np.full(len(out), -1)
    alive_runs = runs(alive)
    for k, (a, b) in enumerate(alive_runs):
        alive_run[a:b] = k
    for a, b in runs(dead):
        if b - a < DEAD_MIN_FRAMES:
            continue
        p = pats[rng.integers(len(pats))]
        idx = np.minimum((np.arange(b - a) * NB) // (b - a), NB - 1)
        drop[a:b] |= ~p[idx]
        if OVERHANG and len(post) and not p[-1] and b < len(out) and alive_run[b] >= 0:
            e = min(b + int(round(OVERHANG_SCALE * post[rng.integers(len(post))])), alive_runs[alive_run[b]][1])
            drop[b:e] = True
        if OVERHANG and len(pre) and not p[0] and a > 0 and alive_run[a - 1] >= 0:
            s0 = max(a - int(round(OVERHANG_SCALE * pre[rng.integers(len(pre))])), alive_runs[alive_run[a - 1]][0])
            drop[s0:a] = True
    u = rng.random(len(out))
    lengths = pool[rng.integers(len(pool), size=len(out))]

    def place(h):
        d = np.zeros(len(out), dtype=bool)
        for a, b in alive_runs:
            t = a
            while t < b:
                if u[t] < h:
                    e = min(b, t + 1 + lengths[t])
                    d[t + 1:e] = True
                    t = e
                else:
                    t += 1
        return d

    # bisection (geometric) on the per-frame gap-start rate; the closest try is kept
    lo, hi = 1e-5, 0.05
    best = None
    for _ in range(25):
        h = np.sqrt(lo * hi)
        d = place(h)
        cov = 1 - (d | ~vis | (drop if OVERHANG else False))[alive].mean()
        if best is None or abs(cov - alive_coverage) < best[0]:
            best = (abs(cov - alive_coverage), d)
        if cov > alive_coverage:
            lo = h
        else:
            hi = h
    drop |= best[1]
    out[drop] = np.nan
    return out


def simulate_ball(match, seed, state, smooth):
    """The phase 9 simulated ball for a clean labelled match (players are simulated as before)."""
    carried = transport_rest(match, np.random.default_rng([seed, 13]))
    return degrade_ball(carried, match.labels, match.periods, np.random.default_rng([seed, 7]),
                        state, smooth)
