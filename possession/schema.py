"""The one in-memory representation every part of the pipeline speaks.

A Match is a recording on a single frame timeline, whether it came from the competition ATD or
from an external dataset passed through the noise simulator. Methods take a Match and return one
label code per frame; the harness scores those codes against Match.labels. Keeping both sources
in the same shape is what lets a method be validated on external games and then run unchanged
on BAR-ATH.

Conventions
-----------
- Coordinates are metres, origin at a pitch corner: x in [0, L] along the length, y in [0, W].
- Missing observations are NaN (positions, speeds, ball).
- slot_team[s] is 0 for home, 1 for away. A slot is a tracklet holder, not a person.
- labels use the scorer's codes: H=0, A=1, D=2, X=3 (X = not graded / no answer).
- periods are half-open [start, stop) graded frame ranges, one per half. Nothing temporal is
  ever computed across a period boundary.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

H, A, D, X = 0, 1, 2, 3
LABEL_CHARS = np.array(["H", "A", "D", "X"])
CHAR_TO_CODE = {"H": H, "A": A, "D": D, "X": X}
HOME, AWAY = 0, 1


@dataclass
class Match:
    name: str
    fps: int
    pitch: tuple                      # (length_m, width_m)
    xy: np.ndarray                    # (T, S, 2) float64 metres
    speed: np.ndarray                 # (T, S) float32 m/s
    slot_team: np.ndarray             # (S,) int8
    ball: np.ndarray                  # (T, 2) float64 metres
    periods: list                     # [(period_id, start, stop)] half-open
    labels: np.ndarray = None         # (T,) int8 or None
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return self.xy.shape[0]

    @property
    def n_slots(self):
        return self.xy.shape[1]

    def ball_present(self):
        return ~np.isnan(self.ball[:, 0])

    def graded_mask(self):
        mask = np.zeros(self.n_frames, dtype=bool)
        for _, start, stop in self.periods:
            mask[start:stop] = True
        return mask

    def validate(self):
        T, S = self.xy.shape[:2]
        assert self.xy.shape == (T, S, 2), self.xy.shape
        assert self.speed.shape == (T, S), self.speed.shape
        assert self.slot_team.shape == (S,), self.slot_team.shape
        assert self.ball.shape == (T, 2), self.ball.shape
        assert set(np.unique(self.slot_team).tolist()) <= {HOME, AWAY}
        for _, start, stop in self.periods:
            assert 0 <= start < stop <= T, (start, stop, T)
        if self.labels is not None:
            assert self.labels.shape == (T,)
            assert set(np.unique(self.labels).tolist()) <= {H, A, D, X}
        return self

    # -- persistence: one .npz per match, uncompressed for load speed ------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            name=np.array(self.name), fps=np.array(self.fps), pitch=np.array(self.pitch),
            xy=self.xy, speed=self.speed, slot_team=self.slot_team, ball=self.ball,
            periods=np.array(self.periods, dtype=np.int64).reshape(-1, 3),
            labels=self.labels if self.labels is not None else np.zeros(0, dtype=np.int8),
            has_labels=np.array(self.labels is not None),
            meta_json=np.array(_json_dumps(self.meta)),
        )
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            labels = z["labels"] if bool(z["has_labels"]) else None
            return cls(
                name=str(z["name"]), fps=int(z["fps"]), pitch=tuple(z["pitch"].tolist()),
                xy=z["xy"], speed=z["speed"], slot_team=z["slot_team"], ball=z["ball"],
                periods=[tuple(int(v) for v in row) for row in z["periods"]],
                labels=labels, meta=_json_loads(str(z["meta_json"])),
            ).validate()


def _json_dumps(obj):
    import json
    return json.dumps(obj, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def _json_loads(text):
    import json
    return json.loads(text) if text else {}


def periods_from_labels(labels):
    """Half-open runs where labels != X, i.e. the scorer's valid_runs()."""
    graded = (labels != X).astype(np.int8)
    edges = np.diff(np.concatenate(([0], graded, [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(i + 1, int(a), int(b)) for i, (a, b) in enumerate(zip(starts, stops))]
