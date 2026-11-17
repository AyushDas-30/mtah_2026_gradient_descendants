"""Measure detections against the tracker's ball, where the tracker sees it. numpy only.

For each processed frame the candidates are projected to the pitch (ground plane) and compared
with the ATD ball. Reported per confidence threshold:
  recall      share of tracker-visible frames with a candidate at or above the threshold
  within 2 m  share of those whose best candidate lies within 2 m of the tracker's ball (precision)
  within 4 m  same at 4 m (a ball in the air projects long)
  fill rate   share of tracker-missing frames with a candidate at or above the threshold
These aggregate numbers are what choose the threshold used to fill the gaps.
"""

import numpy as np

from . import homography

THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6)


def project(dets, frames, H, pitch):
    """dets (n, K, 3) pixels+conf -> (n, K, 2) metres on the pitch (NaN where none / no H)."""
    L, W = pitch
    out = np.full(dets.shape[:2] + (2,), np.nan)
    for i, f in enumerate(frames):
        if np.isnan(H[f]).any():
            continue
        ok = ~np.isnan(dets[i, :, 2])
        if not ok.any():
            continue
        xy = homography.pixel_to_pitch(H[f], dets[i, ok, :2])
        out[i, ok, 0] = xy[:, 0] * L
        out[i, ok, 1] = xy[:, 1] * W
    return out


def report(dets, frames, ball, H, pitch):
    pos = project(dets, frames, H, pitch)
    best_conf = dets[:, 0, 2]
    best_xy = pos[:, 0]
    seen = ~np.isnan(ball[frames, 0])
    dist = np.hypot(*(best_xy - ball[frames]).T)
    rows = []
    for t in THRESHOLDS:
        has = ~np.isnan(best_conf) & (best_conf >= t)
        s = seen & has
        rows.append({
            "threshold": t,
            "recall": float(s.sum() / max(1, seen.sum())),
            "within_2m": float(np.mean(dist[s] < 2.0)) if s.any() else float("nan"),
            "within_4m": float(np.mean(dist[s] < 4.0)) if s.any() else float("nan"),
            "median_err_m": float(np.median(dist[s])) if s.any() else float("nan"),
            "fill_rate": float(np.mean(has[~seen])) if (~seen).any() else float("nan"),
        })
    return rows, int(seen.sum()), int((~seen).sum())
