"""Ball detection on the tactical video with a pretrained YOLO detector, run locally.

The video is decoded and processed on the participant's own machine; nothing from it leaves it.
Only per-frame candidate positions (pixels + confidence) are written to the local cache.

Detector: an Ultralytics YOLO model pretrained on COCO, class 32 "sports ball". The tactical view
is wide and the ball is a few pixels across, so every frame is cut into overlapping 640 x 640
tiles at full resolution (3 x 2 tiles for 1920 x 1080) and the detections are mapped back.

Frames to process:
  sample   N graded frames where the tracker sees the ball + N where it does not, spread over the
           match -- to validate agreement and estimate the fill rate quickly
  missing  every graded frame where the tracker has lost the ball (the frames worth filling)
  all      every graded frame, contiguous -- what a temporal filter needs (validated on the frames
           the tracker sees, applied to the ones it does not)
Long runs checkpoint to <out>.partial.npz every CHECKPOINT frames and resume from it after an
interruption (decoding restarts at frame 0, but skipping is fast; only inference is repeated never).
Frames are decoded sequentially (grab() skips unneeded ones), so frame i is exactly row i of a
submission; the decoded count is checked against 199,319 when the run reaches the end.
"""

import time
from pathlib import Path

import numpy as np

COCO_SPORTS_BALL = 32
TILE = 640
TOP_K = 5
CHECKPOINT = 4000


def tile_origins(width, height, tile=TILE):
    xs = list(range(0, max(1, width - tile) + 1, tile))
    if xs[-1] + tile < width:
        xs.append(width - tile)
    ys = list(range(0, max(1, height - tile) + 1, tile))
    if ys[-1] + tile < height:
        ys.append(height - tile)
    return [(x, y) for y in ys for x in xs]


def frame_plan(match, mode, sample_n=3000, seed=0):
    graded = match.graded_mask()
    present = match.ball_present()
    seen = np.flatnonzero(graded & present)
    missing = np.flatnonzero(graded & ~present)
    if mode == "missing":
        return missing
    if mode == "all":
        return np.flatnonzero(graded)
    if mode == "sample":
        rng = np.random.default_rng(seed)
        a = np.sort(rng.choice(seen, size=min(sample_n, len(seen)), replace=False))
        b = np.sort(rng.choice(missing, size=min(sample_n, len(missing)), replace=False))
        return np.union1d(a, b)
    raise ValueError(mode)


def run(video, frames, out, model_name="yolo11s.pt", conf=0.05, batch_frames=8, device=0,
        half=True, log=print, expected_frames=None, model=None):
    """model: an object with Ultralytics' predict() interface; loaded from model_name if None."""
    import cv2

    frames = np.asarray(sorted(set(int(f) for f in frames)), dtype=np.int64)
    if model is None:
        from ultralytics import YOLO
        model = YOLO(model_name)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError("cannot open video {}".format(video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    origins = tile_origins(width, height)
    log("video {}x{}, {} frames declared; {} frames to process, {} tiles each".format(
        width, height, declared, len(frames), len(origins)))

    dets = np.full((len(frames), TOP_K, 3), np.nan)
    finished = np.zeros(len(frames), dtype=bool)
    partial = Path(str(out) + ".partial.npz")
    if partial.is_file():
        z = np.load(partial, allow_pickle=False)
        if np.array_equal(z["frames"], frames) and str(z["model"]) == str(model_name) \
                and float(z["conf"]) == float(conf) and z["dets"].shape == dets.shape:
            dets[:] = z["dets"]
            finished[:] = z["finished"]
            log("resuming: {} of {} frames already done".format(int(finished.sum()), len(frames)))
    need = {int(f): i for i, f in enumerate(frames) if not finished[i]}
    last_save = int(finished.sum())
    batch, batch_rows = [], []
    t0 = time.time()
    done = 0

    def flush():
        nonlocal done
        if not batch:
            return
        tiles = [img[y:y + TILE, x:x + TILE] for img in batch for (x, y) in origins]
        results = model.predict(tiles, imgsz=TILE, conf=conf, classes=[COCO_SPORTS_BALL],
                                device=device, quantize=16 if half else None, verbose=False)
        for k, row in enumerate(batch_rows):
            cands = []
            for j, (x0, y0) in enumerate(origins):
                r = results[k * len(origins) + j]
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xywh = r.boxes.xywh.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                for (cx, cy, _, _), c in zip(xywh, cf):
                    cands.append((cx + x0, cy + y0, float(c)))
            cands = _dedupe(cands)
            for m, c in enumerate(sorted(cands, key=lambda z: -z[2])[:TOP_K]):
                dets[row, m] = c
            finished[row] = True
        done += len(batch_rows)
        batch.clear()
        batch_rows.clear()

    if not need:
        last = -1
    else:
        last = max(need)
    index = 0
    while index <= last:
        if index in need:
            ok, img = cap.read()
            if not ok:
                raise IOError("decode stopped at frame {}".format(index))
            batch.append(img)
            batch_rows.append(need[index])
            if len(batch) >= batch_frames:
                flush()
                if int(finished.sum()) - last_save >= CHECKPOINT:
                    last_save = int(finished.sum())
                    partial.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(partial, frames=frames, dets=dets, finished=finished,
                             model=model_name, conf=conf)
                if done % (batch_frames * 250) == 0:
                    rate = done / max(1e-9, time.time() - t0)
                    left = len(need) - done
                    log("  {}/{} frames, {:.1f} frames/s, ~{:.0f} min left".format(
                        int(finished.sum()), len(frames), rate, left / max(rate, 1e-9) / 60))
        else:
            if not cap.grab():
                raise IOError("decode stopped at frame {}".format(index))
        index += 1
    flush()
    decoded_to_end = None
    if expected_frames is not None and last == expected_frames - 1:
        decoded_to_end = not cap.grab()            # nothing left after the last expected frame
    cap.release()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if not finished.all():
        raise RuntimeError("{} frames were not processed".format(int((~finished).sum())))
    np.savez(out, frames=frames, dets=dets, width=width, height=height, model=model_name,
             conf=conf, declared=declared)
    if partial.is_file():
        partial.unlink()
    return {"frames": len(frames), "processed": done, "seconds": time.time() - t0, "declared": declared,
            "with_candidate": int(np.sum(~np.isnan(dets[:, 0, 2]))), "decoded_to_end": decoded_to_end}


def _dedupe(cands, radius=8.0):
    """Merge candidates closer than radius pixels (overlapping tiles see the same ball)."""
    out = []
    for c in sorted(cands, key=lambda z: -z[2]):
        if all((c[0] - o[0]) ** 2 + (c[1] - o[1]) ** 2 > radius ** 2 for o in out):
            out.append(c)
    return out
