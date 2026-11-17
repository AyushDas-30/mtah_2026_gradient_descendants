"""Measure how the automatic tracking is degraded, from the tracking alone (no labels).

This is unsupervised: it reads positions and gaps, never a ground truth, so it is allowed on the
competition match. The resulting profile is what the simulator imposes on clean external games,
so that methods are validated on data that fails the way BAR-ATH's tracking fails.

Measured on graded frames only:
  ball    -- per-frame dropout hazard (visible at t, missing at t+1) by local crowding and ball
             speed; gap lengths by crowding bin
  players -- tracked objects per team per frame; tracklet (slot-run) lengths; duplicate rate
"""

import json
from pathlib import Path

import numpy as np

DENSITY_RADIUS = 3.0                       # metres around the ball
DENSITY_BINS = (0, 1, 2, 3)                # 0, 1, 2, 3+ players within the radius
SPEED_EDGES = (2.0, 8.0, 15.0)             # m/s -> 4 bins
DUPLICATE_RADIUS = 0.5                     # same-team objects closer than this = duplicate


def run_lengths(mask):
    """Lengths of True runs and False runs of a boolean array, as two int arrays."""
    if not len(mask):
        return np.zeros(0, int), np.zeros(0, int)
    m = mask.astype(np.int8)
    edges = np.flatnonzero(np.diff(m)) + 1
    bounds = np.concatenate(([0], edges, [len(m)]))
    lengths = np.diff(bounds)
    values = m[bounds[:-1]].astype(bool)
    return lengths[values], lengths[~values]


def ball_speed(ball, fps):
    """Frame-to-frame ball speed (m/s), NaN where either frame is missing."""
    d = np.full(len(ball), np.nan)
    step = np.sqrt(((ball[1:] - ball[:-1]) ** 2).sum(axis=1)) * fps
    d[1:] = step
    return d


def density_bin(match, frames):
    """Players within DENSITY_RADIUS of the ball at the given frames, capped at 3."""
    dx = match.xy[frames, :, 0] - match.ball[frames, None, 0]
    dy = match.xy[frames, :, 1] - match.ball[frames, None, 1]
    near = np.nansum((dx * dx + dy * dy) < DENSITY_RADIUS ** 2, axis=1)
    return np.minimum(near, 3).astype(np.int64)


EDGE_M = 2.0                               # "edge" zone: within 2 m of a line, or outside


def zone_bin(match, frames):
    """0 inside the pitch (>= EDGE_M from every line), 1 near a line or outside it."""
    L, W = match.pitch
    b = match.ball[frames]
    d = np.minimum.reduce([b[:, 0], L - b[:, 0], b[:, 1], W - b[:, 1]])
    return (d < EDGE_M).astype(np.int64)


def speed_bin(speeds):
    s = np.where(np.isnan(speeds), 0.0, speeds)
    return np.digitize(s, SPEED_EDGES).astype(np.int64)


def measure(match):
    fps = match.fps
    present = match.ball_present()
    hazard_events = np.zeros((2, 4, 4), dtype=np.int64)       # zone x crowding x ball speed
    hazard_total = np.zeros((2, 4, 4), dtype=np.int64)
    gaps_by_density = {b: [] for b in DENSITY_BINS}
    vis_runs, gap_runs = [], []
    objects = {0: [], 1: []}
    tracklets, dup_flags = [], []

    bspeed = ball_speed(match.ball, fps)
    for _, start, stop in match.periods:
        p = present[start:stop]
        v, g = run_lengths(p)
        vis_runs.append(v)
        gap_runs.append(g)
        # hazard: visible frame t (not the last frame), event if t+1 missing
        t = np.flatnonzero(p[:-1]) + start
        if len(t):
            dens = density_bin(match, t)
            spd = speed_bin(bspeed[t])
            zone = zone_bin(match, t)
            event = ~present[t + 1]
            np.add.at(hazard_total, (zone, dens, spd), 1)
            np.add.at(hazard_events, (zone[event], dens[event], spd[event]), 1)
            # gap lengths attributed to the crowding bin at the frame before the gap
            for frame, db in zip(t[event], dens[event]):
                j = frame + 1
                end = j
                while end < stop and not present[end]:
                    end += 1
                if end < stop:                      # gaps truncated by the period end are dropped
                    gaps_by_density[int(db)].append(end - j)

        seen = ~np.isnan(match.xy[start:stop, :, 0])
        for team in (0, 1):
            objects[team].append(seen[:, match.slot_team == team].sum(axis=1))
        for s in range(match.n_slots):
            on, _ = run_lengths(seen[:, s])
            tracklets.append(on)
        dup_flags.append(_duplicate_flags(match, start, stop))

    vis_runs, gap_runs = np.concatenate(vis_runs), np.concatenate(gap_runs)
    dup = np.concatenate(dup_flags)
    dup_on, _ = run_lengths(dup)
    obj = {t: np.concatenate(objects[t]) for t in (0, 1)}
    tl = np.concatenate(tracklets)
    graded = int(sum(stop - start for _, start, stop in match.periods))
    profile = {
        "source": match.name,
        "n_graded_frames": graded,
        "ball_present_fraction": float(present[match.graded_mask()].mean()),
        "ball_hazard": _smoothed_hazard(hazard_events, hazard_total).tolist(),
        "ball_hazard_axes": ["zone(inside, edge/outside)", "crowding 0..3+", "ball speed bin"],
        "ball_hazard_support": hazard_total.tolist(),
        "ball_gap_lengths_by_density": {str(b): sorted(int(x) for x in gaps_by_density[b])
                                        for b in DENSITY_BINS},
        "ball_visible_runs": sorted(int(x) for x in vis_runs),
        "ball_gap_runs": sorted(int(x) for x in gap_runs),
        "objects_per_team_mean": [float(obj[0].mean()), float(obj[1].mean())],
        "objects_per_team_hist": [np.bincount(obj[t], minlength=31)[:31].tolist() for t in (0, 1)],
        "tracklet_lengths": sorted(int(x) for x in tl),
        "slots_per_team": [int((match.slot_team == t).sum()) for t in (0, 1)],
        "duplicate_frame_fraction": float(dup.mean()) if len(dup) else 0.0,
        "duplicate_run_lengths": sorted(int(x) for x in dup_on),
        "params": {"density_radius": DENSITY_RADIUS, "speed_edges": SPEED_EDGES,
                   "duplicate_radius": DUPLICATE_RADIUS},
    }
    return profile


def _smoothed_hazard(events, total, prior_n=200.0):
    """Cell hazards shrunk towards their zone's pooled hazard, so sparse cells (few visible
    frames) do not get extreme rates from a handful of events."""
    out = np.zeros(events.shape)
    for z in range(events.shape[0]):
        pooled = events[z].sum() / max(1.0, float(total[z].sum()))
        out[z] = (events[z] + prior_n * pooled) / (total[z] + prior_n)
    return out


def _duplicate_flags(match, start, stop, chunk=5000):
    """Per frame: does any same-team pair sit closer than DUPLICATE_RADIUS?"""
    out = np.zeros(stop - start, dtype=bool)
    for team in (0, 1):
        cols = np.flatnonzero(match.slot_team == team)
        for a in range(start, stop, chunk):
            b = min(stop, a + chunk)
            p = match.xy[a:b][:, cols]                          # (n, k, 2)
            d = np.sqrt(((p[:, :, None, :] - p[:, None, :, :]) ** 2).sum(-1))
            iu = np.triu_indices(len(cols), 1)
            close = d[:, iu[0], iu[1]] < DUPLICATE_RADIUS
            out[a - start:b - start] |= np.any(close, axis=1)
    return out


def summary(profile):
    """The handful of numbers worth comparing between a profile and a simulated match."""
    tl = np.array(profile["tracklet_lengths"]) if profile["tracklet_lengths"] else np.zeros(1)
    gaps = np.array(profile["ball_gap_runs"]) if profile["ball_gap_runs"] else np.zeros(1)
    vis = np.array(profile["ball_visible_runs"]) if profile["ball_visible_runs"] else np.zeros(1)
    return {
        "ball_present_fraction": round(profile["ball_present_fraction"], 4),
        "ball_gap_median_frames": float(np.median(gaps)),
        "ball_gap_p90_frames": float(np.percentile(gaps, 90)),
        "ball_visible_run_median_frames": float(np.median(vis)),
        "objects_home_mean": round(profile["objects_per_team_mean"][0], 2),
        "objects_away_mean": round(profile["objects_per_team_mean"][1], 2),
        "tracklet_median_frames": float(np.median(tl)),
        "duplicate_frame_fraction": round(profile["duplicate_frame_fraction"], 4),
    }


def save(profile, path):
    Path(path).write_text(json.dumps(profile), encoding="utf-8")
    return path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
