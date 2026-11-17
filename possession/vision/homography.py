"""Tactical homography: normalised pitch (0-1) <-> image pixels, one 3x3 matrix per frame.

CSV columns: frame, h00..h22, reprojection_error, valid. The matrix maps (x, y, 1) in normalised
pitch coordinates -- the same convention as the ATD -- to homogeneous pixels. Pixels go back to
the pitch with the inverse, on the ground plane (a ball in the air projects too far away).
"""

import numpy as np


def load(path, n_frames):
    H = np.full((n_frames, 3, 3), np.nan)
    err = np.full(n_frames, np.nan)
    valid = np.zeros(n_frames, dtype=bool)
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        col = {name: i for i, name in enumerate(header)}
        keys = ["h%d%d" % (r, c) for r in range(3) for c in range(3)]
        for line in handle:
            parts = line.rstrip("\r\n").split(",")
            f = int(parts[col["frame"]])
            if not 0 <= f < n_frames:
                continue
            ok = parts[col["valid"]].strip() in ("1", "1.0", "True", "true")
            valid[f] = ok
            if ok:
                H[f] = np.array([float(parts[col[k]]) for k in keys]).reshape(3, 3)
                err[f] = float(parts[col["reprojection_error"]])
    return H, err, valid


def pitch_to_pixel(H, xy_norm):
    """H (3,3), xy (n,2) normalised -> (n,2) pixels."""
    p = np.column_stack([xy_norm, np.ones(len(xy_norm))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def pixel_to_pitch(H, uv):
    """H (3,3), uv (n,2) pixels -> (n,2) normalised pitch coordinates (ground plane)."""
    p = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(H).T
    return p[:, :2] / p[:, 2:3]
