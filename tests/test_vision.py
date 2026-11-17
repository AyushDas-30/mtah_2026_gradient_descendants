"""Vision plumbing tests: synthetic video + stand-in detector (no torch needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = pytest.importorskip("cv2")

from possession.schema import Match  # noqa: E402
from possession.vision import detect, homography, validate  # noqa: E402

W, H_PX, L, WD = 1920, 1080, 105.3, 68.0
HMAT = np.array([[1700.0, 0, 110], [0, 900.0, 90], [0, 0, 1]])   # normalised pitch -> pixels


class _Boxes:
    def __init__(self, xywh, conf):
        self.xywh = _T(xywh)
        self.conf = _T(conf)

    def __len__(self):
        return len(self.conf.a)


class _T:
    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _R:
    def __init__(self, boxes):
        self.boxes = boxes


class BrightBlobModel:
    """Stand-in for YOLO: reports the brightest blob in each tile as a ball."""

    def predict(self, tiles, **kw):
        out = []
        for t in tiles:
            g = t[:, :, 1].astype(float)
            if g.max() < 200:
                out.append(_R(None))
                continue
            ys, xs = np.nonzero(g >= 200)
            out.append(_R(_Boxes([[xs.mean(), ys.mean(), 8, 8]], [0.9])))
        return out


def _video(path, n, ball_px):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H_PX))
    for i in range(n):
        img = np.zeros((H_PX, W, 3), np.uint8)
        img[:] = (40, 120, 40)
        if not np.isnan(ball_px[i, 0]):
            cv2.circle(img, (int(ball_px[i, 0]), int(ball_px[i, 1])), 6, (255, 255, 255), -1)
        vw.write(img)
    vw.release()


def test_tile_origins_cover_frame():
    o = detect.tile_origins(1920, 1080)
    assert (0, 0) in o and (1280, 440) in o and len(o) == 6


def test_detect_and_validate_end_to_end(tmp_path):
    n = 60
    t = np.arange(n)
    ball_norm = np.column_stack([0.2 + 0.01 * t, 0.3 + 0.004 * t])
    ball_px = homography.pitch_to_pixel(HMAT, ball_norm)
    _video(tmp_path / "v.mp4", n, ball_px)
    atd_ball = np.column_stack([ball_norm[:, 0] * L, ball_norm[:, 1] * WD])
    atd_ball[30:40] = np.nan                                   # tracker loses the ball here
    m = Match("t", 25, (L, WD), np.zeros((n, 2, 2)), np.zeros((n, 2), np.float32),
              np.array([0, 1], np.int8), atd_ball, [(1, 0, n)])
    frames = np.arange(n)
    r = detect.run(tmp_path / "v.mp4", frames, tmp_path / "d.npz", model=BrightBlobModel(),
                   batch_frames=7, device="cpu", half=False, log=lambda s: None, expected_frames=n)
    assert r["frames"] == n and r["with_candidate"] == n and r["decoded_to_end"] is True
    z = np.load(tmp_path / "d.npz")
    Hs = np.repeat(HMAT[None], n, axis=0)
    rows, n_seen, n_miss = validate.report(z["dets"], z["frames"], atd_ball, Hs, (L, WD))
    assert n_seen == 50 and n_miss == 10
    top = rows[0]
    assert top["recall"] == 1.0 and top["within_2m"] == 1.0 and top["fill_rate"] == 1.0
    assert top["median_err_m"] < 0.3


def test_homography_roundtrip():
    xy = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.95]])
    back = homography.pixel_to_pitch(HMAT, homography.pitch_to_pixel(HMAT, xy))
    np.testing.assert_allclose(back, xy, atol=1e-12)


class _Interrupt(Exception):
    pass


class FlakyModel(BrightBlobModel):
    """Raises after a number of predict calls, to simulate an interrupted run."""

    def __init__(self, fail_after):
        self.calls, self.fail_after = 0, fail_after

    def predict(self, tiles, **kw):
        self.calls += 1
        if self.calls > self.fail_after:
            raise _Interrupt()
        return super().predict(tiles, **kw)


def test_detect_resumes_from_checkpoint(tmp_path, monkeypatch):
    n = 60
    t = np.arange(n)
    ball_px = homography.pitch_to_pixel(HMAT, np.column_stack([0.2 + 0.01 * t, 0.3 + 0.004 * t]))
    _video(tmp_path / "v.mp4", n, ball_px)
    frames = np.arange(5, n)
    monkeypatch.setattr(detect, "CHECKPOINT", 10)
    out = tmp_path / "d.npz"
    with pytest.raises(_Interrupt):
        detect.run(tmp_path / "v.mp4", frames, out, model=FlakyModel(4), batch_frames=5,
                   device="cpu", half=False, log=lambda s: None)
    assert Path(str(out) + ".partial.npz").is_file() and not out.is_file()
    model = BrightBlobModel()
    r = detect.run(tmp_path / "v.mp4", frames, out, model=model, batch_frames=5,
                   device="cpu", half=False, log=lambda s: None)
    assert r["processed"] < len(frames)                      # earlier work was reused
    ref = detect.run(tmp_path / "v.mp4", frames, tmp_path / "ref.npz", model=BrightBlobModel(),
                     batch_frames=5, device="cpu", half=False, log=lambda s: None)
    assert ref["processed"] == len(frames)
    np.testing.assert_array_equal(np.load(out)["dets"], np.load(tmp_path / "ref.npz")["dets"])
    assert not Path(str(out) + ".partial.npz").is_file()


def test_frame_plan_all_is_every_graded_frame():
    n = 100
    ball = np.zeros((n, 2)); ball[::3] = np.nan
    m = Match("t", 25, (L, WD), np.zeros((n, 2, 2)), np.zeros((n, 2), np.float32),
              np.array([0, 1], np.int8), ball, [(1, 10, 40), (2, 60, 90)])
    allf = detect.frame_plan(m, "all")
    assert np.array_equal(allf, np.r_[10:40, 60:90])
    assert set(detect.frame_plan(m, "missing")) <= set(allf)
