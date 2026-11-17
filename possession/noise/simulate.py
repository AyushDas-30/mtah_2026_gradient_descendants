"""Degrade a clean, labelled external Match so it looks like BAR-ATH's automatic tracking.

Inputs: a clean Match (identities, ball nearly always present) and a profile from
noise/profile.py. Output: a new Match with the same labels and timeline, but with

  ball     -- dropped in bursts: at each visible frame a gap starts with the hazard measured for
              that crowding x ball-speed bin, and its length is drawn from the gaps measured in
              that crowding bin; small Gaussian position noise
  players  -- each true trajectory cut into tracklets (lengths drawn from the measured tracklet
              lengths) separated by off-gaps sized to hit the measured objects-per-team coverage;
              each tracklet dropped into a random free slot of its team, so identities switch the
              way a re-identifying tracker's do; occasional duplicate tracklets; position jitter

Deterministic given the seed. Labels are never touched.
"""

import numpy as np

from ..io.idsse import speed_from_xy
from ..schema import Match
from .profile import ball_speed, density_bin, speed_bin, zone_bin

# No per-frame jitter: automatic tracker output is temporally smooth, and i.i.d. noise of even
# 0.1 m adds ~1 m/s of fake speed after differentiation -- a feature shift the real tracking
# does not have. Degradation is carried by dropouts, fragmentation and identity switches.
BALL_JITTER_M = 0.0
PLAYER_JITTER_M = 0.0
DUP_OFFSET_M = 0.30
TEAM_SIZE = 11
# Ball smoothing used by the CLI. 0 reproduces the Phase 6 simulated games; 7 matches BAR-ATH's
# velocity-change percentiles (phase 7 measurement) but is not adopted yet -- see phase7.md.
SIM_SMOOTH_SIGMA = 0.0


def simulate(match, profile, seed=0, duplicates=False, calibrate=True, smooth_sigma=0.0, state=None):
    """Players are degraded first; the ball dropout then reads crowding off the *degraded* players,
    which is how the hazard was measured on BAR-ATH. With calibrate, a single scalar on the hazard
    is fitted (a few deterministic passes) so overall ball coverage matches the profile.

    With state (a phase 9 state profile, noise/state.py) the ball is simulated game-state-aware
    instead: carried-then-resting dead ball, tracker smoothing, real stoppage visibility patterns
    and live-play gaps. Players are unchanged (same seed, same draws)."""
    rng = np.random.default_rng(seed)
    xy, speed, slot_team = _degrade_players(match, profile, rng, duplicates=duplicates)
    if state is not None:
        from . import state as state_mod
        sigma = state_mod.SMOOTH_SIGMA
        ball = state_mod.simulate_ball(match, seed, state, lambda b, per: _smooth_ball(b, per, sigma))
        return Match(
            name="{}~sim{}".format(match.name, seed), fps=match.fps, pitch=match.pitch,
            xy=xy, speed=speed, slot_team=slot_team, ball=ball, periods=list(match.periods),
            labels=None if match.labels is None else match.labels.copy(),
            meta={"source": "simulated", "clean": match.name, "seed": seed,
                  "profile_source": profile.get("source"), "hazard_scale": float("nan"),
                  "duplicates": duplicates, "smooth_sigma": sigma, "world": "phase9-state"},
        ).validate()
    carried = carry_dead_ball(match)
    degraded = Match(match.name, match.fps, match.pitch, xy, speed, slot_team, carried,
                     list(match.periods))
    target = float(profile["ball_present_fraction"])
    graded = match.graded_mask()

    def coverage(s):
        b = _degrade_ball(degraded, profile, np.random.default_rng([seed, 1]), s,
                          smooth_sigma=smooth_sigma)
        return float((~np.isnan(b[graded, 0])).mean()), b

    scale = 1.0
    got, ball = coverage(scale)
    if calibrate and abs(got - target) >= 0.003:
        # Bisection on log(scale): coverage falls as the hazard rises. Random draws are fixed
        # across tries, so this is deterministic; the closest try is kept even if not monotone.
        lo, hi = np.log(0.05), np.log(20.0)
        best = (abs(got - target), scale, ball)
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            g, b = coverage(float(np.exp(mid)))
            if abs(g - target) < best[0]:
                best = (abs(g - target), float(np.exp(mid)), b)
            if abs(g - target) < 0.003:
                break
            if g > target:
                lo = mid        # too much ball visible -> raise the hazard
            else:
                hi = mid
        _, scale, ball = best
    return Match(
        name="{}~sim{}".format(match.name, seed), fps=match.fps, pitch=match.pitch,
        xy=xy, speed=speed, slot_team=slot_team, ball=ball, periods=list(match.periods),
        labels=None if match.labels is None else match.labels.copy(),
        meta={"source": "simulated", "clean": match.name, "seed": seed,
              "profile_source": profile.get("source"), "hazard_scale": scale,
              "duplicates": duplicates, "smooth_sigma": float(smooth_sigma)},
    ).validate()


# -- ball ----------------------------------------------------------------------------------

def carry_dead_ball(match, walking_speed=None):
    """Replace the frozen dead-ball position of the external data with a carried ball.

    DFL freezes the ball where it went out for the whole stoppage (speed exactly 0, often parked
    beyond a line). BAR-ATH's tracker instead sees the ball moving while it is fetched and placed.
    Inside every dead stretch the ball moves smoothly (smoothstep) from the last live position to
    the next live position (Phase 6 behaviour, walking_speed=None). With walking_speed (m/s,
    teammate step 1, not adopted) the move is capped and stops at the reachable point. Labels only
    locate the stretches; only the simulated input changes -- never the labels.
    """
    ball = match.ball.copy()
    if match.labels is None:
        return ball
    dead = match.labels == 2
    for _, start, stop in match.periods:
        d = dead[start:stop].astype(np.int8)
        edges = np.diff(np.concatenate(([0], d, [0])))
        for a, b in zip(np.flatnonzero(edges == 1) + start, np.flatnonzero(edges == -1) + start):
            p0 = ball[a - 1] if a > start and not np.isnan(ball[a - 1, 0]) else ball[a]
            p1 = ball[b] if b < stop and not np.isnan(ball[b, 0]) else ball[b - 1]
            if np.isnan(p0).any() or np.isnan(p1).any():
                continue
            n = b - a
            if walking_speed is None:
                u = (np.arange(n) + 1) / float(n + 1)
                u = u * u * (3 - 2 * u)
                ball[a:b] = p0[None, :] + u[:, None] * (p1 - p0)[None, :]
                continue
            distance = float(np.linalg.norm(p1 - p0))
            # smoothstep's peak derivative is 1.5 times its average, so budget for it
            reach = min(distance, max(0.0, float(walking_speed)) * n / (1.5 * match.fps))
            u = np.arange(1, n + 1, dtype=np.float64) / float(n)
            u = u * u * (3 - 2 * u)
            direction = (p1 - p0) / distance if distance > 0 else np.zeros(2)
            ball[a:b] = p0[None, :] + (reach * u)[:, None] * direction[None, :]
    return ball


def _degrade_ball(match, profile, rng, scale=1.0, smooth_sigma=0.0):
    hazard = np.clip(np.array(profile["ball_hazard"]) * scale, 0.0, 1.0)
    if hazard.ndim == 2:                       # profile written before zones existed
        hazard = np.stack([hazard, hazard])
    gaps = {int(k): np.array(v, dtype=np.int64)
            for k, v in profile["ball_gap_lengths_by_density"].items()}
    pooled = np.concatenate([g for g in gaps.values() if len(g)] or [np.array([25])])
    out = match.ball.copy()
    present = ~np.isnan(out[:, 0])
    out[present] += rng.normal(0.0, BALL_JITTER_M, size=(int(present.sum()), 2))
    if smooth_sigma > 0:
        out = _smooth_ball(out, match.periods, float(smooth_sigma))

    bspeed = ball_speed(match.ball, match.fps)
    drop = np.zeros(match.n_frames, dtype=bool)
    for _, start, stop in match.periods:
        frames = np.arange(start, stop)
        frames = frames[present[frames]]
        if not len(frames):
            continue
        dens = density_bin(match, frames)
        spd = speed_bin(bspeed[frames])
        h = hazard[zone_bin(match, frames), dens, spd]
        u = rng.random(len(frames))
        t = start
        k = 0
        # walk forward; a gap swallows the frames it covers
        while k < len(frames):
            f = frames[k]
            if f < t:
                k += 1
                continue
            if u[k] < h[k]:
                pool = gaps.get(int(dens[k]))
                pool = pool if pool is not None and len(pool) >= 20 else pooled
                length = int(pool[rng.integers(len(pool))])
                a, b = f + 1, min(stop, f + 1 + length)
                drop[a:b] = True
                t = b
            k += 1
    out[drop] = np.nan
    return out


def _smooth_ball(ball, periods, sigma):
    """Gaussian-smooth each contiguous visible run without filling dropouts or crossing halves."""
    out = ball.copy()
    radius = max(1, int(np.ceil(4 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    visible = np.isfinite(ball).all(axis=1)
    for _, start, stop in periods:
        idx = np.flatnonzero(visible[start:stop])
        if not len(idx):
            continue
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[breaks + 1]] + start
        stops = np.r_[idx[breaks], idx[-1]] + start + 1
        for a, b in zip(starts, stops):
            # Edge-pad each visible run to avoid attenuating its endpoints.
            for axis in range(2):
                padded = np.pad(ball[a:b, axis], (radius, radius), mode="edge")
                out[a:b, axis] = np.convolve(padded, kernel, mode="valid")
    return out


# -- players -------------------------------------------------------------------------------

def _degrade_players(match, profile, rng, duplicates=False):
    T = match.n_frames
    tl = np.array(profile["tracklet_lengths"], dtype=np.int64)
    tl = tl[tl >= 2] if len(tl) else np.array([250])
    slots_per_team = profile.get("slots_per_team", [23, 23])
    coverage = [min(0.995, max(0.3, m / TEAM_SIZE)) for m in profile["objects_per_team_mean"]]

    # tracklets: (team, start, stop, person_slot)
    tracklets = []
    for s in range(match.n_slots):
        team = int(match.slot_team[s])
        seen = ~np.isnan(match.xy[:, s, 0])
        mean_on = float(tl.mean())
        # off-gaps are geometric on {0, 1, ...}: a 0 gap is an identity switch with no break.
        # Mean (1-p)/p == mean_off gives coverage mean_on / (mean_on + mean_off).
        mean_off = mean_on * (1 - coverage[team]) / coverage[team]
        p_off = 1.0 / (1.0 + mean_off)
        for _, start, stop in match.periods:
            idx = np.flatnonzero(seen[start:stop])
            if not len(idx):
                continue
            # contiguous true presence runs inside the period
            breaks = np.flatnonzero(np.diff(idx) > 1)
            run_starts = np.concatenate(([idx[0]], idx[breaks + 1])) + start
            run_stops = np.concatenate((idx[breaks], [idx[-1]])) + start + 1
            for a, b in zip(run_starts, run_stops):
                t = a + int(rng.geometric(p_off)) - 1 if rng.random() > coverage[team] else a
                while t < b:
                    length = int(tl[rng.integers(len(tl))])
                    e = min(b, t + length)
                    tracklets.append((team, t, e, s))
                    t = e + int(rng.geometric(p_off)) - 1

    # duplicates: a short copy of an existing tracklet, offset a little
    dup_frac = float(profile.get("duplicate_frame_fraction", 0.0))
    dup_lengths = np.array(profile.get("duplicate_run_lengths") or [25], dtype=np.int64)
    # Off by default: the measured tracklet lengths already contain BAR-ATH's short duplicate
    # fragments, so injecting more would count them twice.
    n_dup = 0 if not duplicates else int(round(
        dup_frac * sum(stop - start for _, start, stop in match.periods)
        / max(1.0, float(dup_lengths.mean()))))
    dups = []
    if tracklets and n_dup:
        for i in rng.integers(len(tracklets), size=n_dup):
            team, a, b, s = tracklets[i]
            length = int(dup_lengths[rng.integers(len(dup_lengths))])
            if b - a <= 1:
                continue
            a2 = int(rng.integers(a, b))
            dups.append((team, a2, min(b, a2 + length), s))

    n_slots = [max(int(slots_per_team[0]), TEAM_SIZE + 2), max(int(slots_per_team[1]), TEAM_SIZE + 2)]
    S = n_slots[0] + n_slots[1]
    slot_team = np.array([0] * n_slots[0] + [1] * n_slots[1], dtype=np.int8)
    xy = np.full((T, S, 2), np.nan)
    free_at = np.zeros(S, dtype=np.int64)
    team_slots = [np.arange(n_slots[0]), np.arange(n_slots[0], S)]

    items = [(t, a, b, s, False) for t, a, b, s in tracklets] + \
            [(t, a, b, s, True) for t, a, b, s in dups]
    items.sort(key=lambda z: (z[1], z[2], z[3]))
    dropped = 0
    for team, a, b, s, is_dup in items:
        candidates = team_slots[team][free_at[team_slots[team]] <= a]
        if not len(candidates):
            dropped += 1
            continue
        slot = int(candidates[rng.integers(len(candidates))])
        seg = match.xy[a:b, s].copy()
        seg += rng.normal(0.0, PLAYER_JITTER_M, size=seg.shape)
        if is_dup:
            seg += rng.normal(0.0, DUP_OFFSET_M, size=(1, 2))
        xy[a:b, slot] = seg
        free_at[slot] = b

    # speed: true speed of whoever occupies the slot (the ATD supplies a tracker speed too)
    speed = speed_from_xy(xy, match.periods, fps=match.fps)
    return xy, speed, slot_team
