"""Bring external possession labels to the competition's definition of a change of possession.

CHALLENGE.md: "A change of possession occurs when a player from the opposing team gains
*controlled* possession ... In phases where the ball is contested or loose, possession stays with
the last team that had it." A simple touch, a deflection or an aerial duel without subsequent
control does not change possession.

DFL's per-frame BallPossession flips on looser contact than that. The rule implemented here:
inside each alive stretch (bounded by dead ball or a period edge), a run of the other team only
becomes a change of possession if it lasts at least `min_control_s`; a shorter run is a loose /
contested ball and stays with the team that had it. The first run after a restart is kept as is
(a restart always has a clear possessor). Dead-ball frames are never touched.

The rules do not fix the threshold, so callers evaluate several (0, 0.5, 1, 2 s) and choose
settings that hold up across all of them. The threshold is never fitted to the public window.
"""

import numpy as np

from .schema import A, D, H, X

THRESHOLDS_S = (0.0, 0.5, 1.0, 2.0)


def relabel(labels, min_control_s, fps=25):
    labels = np.asarray(labels, dtype=np.int8)
    out = labels.copy()
    min_frames = int(round(min_control_s * fps))
    if min_frames <= 1:
        return out
    alive = (labels == H) | (labels == A)
    edges = np.diff(np.concatenate(([0], alive.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        window = labels[start:stop]
        bounds = np.concatenate(([0], np.flatnonzero(window[1:] != window[:-1]) + 1, [len(window)]))
        current = int(window[0])
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            team = int(window[a])
            if i > 0 and team != current and (b - a) >= min_frames:
                current = team
            out[start + a:start + b] = current
    return out


def variants(labels, fps=25, thresholds=THRESHOLDS_S):
    return {t: relabel(labels, t, fps) for t in thresholds}
