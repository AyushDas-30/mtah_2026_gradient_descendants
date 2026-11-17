"""The starter baseline, re-implemented on a Match: nearest tracked player to the ball, then a
centred majority vote. Exists for two reasons -- reproducing the committed baseline CSVs frame for
frame proves our ATD parsing and alignment, and it is the reference every later method must beat
on the validation harness.
"""

import numpy as np

from ..schema import A, AWAY, D, H, HOME

NOBODY = -1


def nearest_team(match):
    """Per-frame team of the nearest tracked object to the ball; NOBODY if no ball / no player.

    Distances in metres, ties to the lowest slot index (np.argmin == the starter's strict '<').
    """
    ball = match.ball
    present = ~np.isnan(ball[:, 0])
    out = np.full(match.n_frames, NOBODY, dtype=np.int8)
    idx = np.flatnonzero(present)
    for chunk in np.array_split(idx, max(1, len(idx) // 20000)):
        if not len(chunk):
            continue
        dx = match.xy[chunk, :, 0] - ball[chunk, None, 0]
        dy = match.xy[chunk, :, 1] - ball[chunk, None, 1]
        dist = dx * dx + dy * dy
        has = ~np.all(np.isnan(dist), axis=1)
        dist = np.where(np.isnan(dist), np.inf, dist)
        slot = np.argmin(dist, axis=1)
        team = match.slot_team[slot].astype(np.int8)
        out[chunk[has]] = team[has]
    return out


def majority(codes, window):
    """Centred majority vote over +/- window frames; ties go home; D where nobody is seen."""
    if window <= 0:
        return np.where(codes == HOME, H, np.where(codes == AWAY, A, D)).astype(np.int8)
    home_c = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_c = np.concatenate(([0], np.cumsum(codes == AWAY)))
    i = np.arange(len(codes))
    lo = np.maximum(0, i - window)
    hi = np.minimum(len(codes), i + window + 1)
    home = home_c[hi] - home_c[lo]
    away = away_c[hi] - away_c[lo]
    out = np.full(len(codes), D, dtype=np.int8)
    seen = (home + away) > 0
    out[seen] = np.where(home[seen] >= away[seen], H, A)
    return out


def baseline(match, window=50):
    return majority(nearest_team(match), window)
