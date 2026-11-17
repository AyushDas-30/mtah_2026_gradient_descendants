"""Paths and published constants. Nothing in here is learned from the data.

Every constant below is published in the competition repository (DATA.md, FAQ.md, games.py).
They are boundaries and conventions, not answers.

Locations are resolved from environment variables, overridable on the command line:

    MTAH_DATA_DIR   the unpacked data bundle (folder containing tactical/ and broadcast/)
    MTAH_REPO_DIR   a clone of metrica-sports/more-than-a-hack-2027 (public GT, baselines, scorer)
    MTAH_CACHE_DIR  where this project writes its numpy caches (default ./cache)
    MTAH_IDSSE_DIR  the downloaded IDSSE XML files (default <cache>/idsse_raw)
"""

import os
from pathlib import Path

GAME = "BAR-ATH"
FPS = 25
N_VIDEO_FRAMES = 199319
# Real pitch in metres, as carried by games.py (measured from the corrected tracking). The
# baseline converts normalised coordinates with exactly these numbers, so we do too.
PITCH = (105.3, 68.0)
N_TRACK_SLOTS = 46

# Graded video-frame blocks, inclusive, from DATA.md / games.half_video_blocks("BAR-ATH").
GRADED_HALVES = ((1, 11185, 90651), (2, 110808, 190197))
# Public ground-truth window (DATA.md). For pipeline checks only -- never for tuning.
PUBLIC_WINDOW = (11185, 18684)

# Published aggregate figures, used only as self-checks on our own parsing.
PUBLISHED = {
    "ball_present_all": 0.569,      # 113,488 / 199,319
    "ball_present_graded": 0.593,   # 94,214 / 158,857
    "mean_objects_graded": 20.5,
    "n_graded": 158857,
    "n_smart_tagging": 638,
}

# Public-window scores of the four committed baselines, as printed by run_evaluation.py in the
# official README. Our scorer must reproduce these exactly.
PUBLISHED_BASELINE_PUBLIC = {
    "baseline_raw": (0.5972, 0.0378),
    "baseline_smooth25": (0.6200, 0.2868),
    "baseline_smooth50": (0.6207, 0.3256),
    "baseline_smooth125": (0.5812, 0.4242),
}
BASELINE_WINDOWS = {"baseline_raw": 0, "baseline_smooth25": 25,
                    "baseline_smooth50": 50, "baseline_smooth125": 125}


def _env_path(name, default):
    value = os.environ.get(name)
    return Path(value) if value else Path(default)


def data_dir(override=None):
    return Path(override) if override else _env_path("MTAH_DATA_DIR", "data")


def repo_dir(override=None):
    return Path(override) if override else _env_path("MTAH_REPO_DIR", "more-than-a-hack-2027")


def cache_dir(override=None):
    path = Path(override) if override else _env_path("MTAH_CACHE_DIR", "cache")
    path.mkdir(parents=True, exist_ok=True)
    return path


def idsse_dir(override=None, cache=None):
    if override:
        return Path(override)
    value = os.environ.get("MTAH_IDSSE_DIR")
    return Path(value) if value else cache_dir(cache) / "idsse_raw"


def tactical_paths(root=None):
    folder = data_dir(root) / "tactical"
    stem = GAME + "_tactical"
    return {
        "atd_meta": folder / (stem + "_atd.xml"),
        "atd_raw": folder / (stem + "_atd.txt"),
        "homography": folder / (stem + "_homography.csv"),
        "smart_tagging": folder / (stem + "_smart_tagging.xml"),
        "video": folder / (stem + ".mp4"),
    }
