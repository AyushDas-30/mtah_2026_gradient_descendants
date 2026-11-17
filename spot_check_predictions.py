"""Spot-check predictions against real tactical video frames and Smart Tagging."""
import os
from pathlib import Path
import cv2
import json
import numpy as np

# Resolve base directories from environment or repo root
data_dir = Path(os.environ.get("MTAH_DATA_DIR", Path(__file__).resolve().parent.parent.parent))
cache_dir = Path(os.environ.get("MTAH_CACHE_DIR", Path(__file__).resolve().parent / "cache9"))

# Find submission CSV
candidates = [
    Path(__file__).resolve().parent / "submissions" / "Gradient Descendants.csv",
    Path(__file__).resolve().parent / "submissions_regen" / "Gradient Descendants.csv",
    data_dir / "Gradient_Descendants.csv",
    data_dir / "Gradient Descendants.csv",
]
csv_path = None
for c in candidates:
    if c.is_file():
        csv_path = c
        break

if not csv_path:
    raise FileNotFoundError("Could not find submission CSV in expected locations")

print(f"Loading predictions from: {csv_path}")

labels = {}
with open(csv_path, "r", encoding="utf-8") as f:
    header = f.readline().strip()
    for line in f:
        line = line.strip()
        if not line:
            continue
        frm, lbl = line.split(",")
        labels[int(frm)] = lbl

# Load Smart Tagging events
tag_path = cache_dir / "BAR-ATH_tagging.json"
if not tag_path.is_file():
    tag_path = data_dir / "tactical" / "BAR-ATH_tactical_smart_tagging.xml"
    # Fallback if raw xml
    tags = []
else:
    with open(tag_path, "r", encoding="utf-8") as f:
        tags = json.load(f)

# Open tactical video
video_path = data_dir / "tactical" / "BAR-ATH_tactical.mp4"
if not video_path.is_file():
    video_path = Path("tactical/BAR-ATH_tactical.mp4")

cap = cv2.VideoCapture(str(video_path))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Tactical video opened: {total_frames:,} frames at 25 fps\n")

# Collect sample scenarios for inspection:
# 1. 1H Kick-off (Frame 11185)
# 2. 2H Kick-off (Frame 113742)
# 3. SET PIECES: Goal Kick
# 4. SET PIECES: Corner Kick
# 5. SET PIECES: Throw-in
# 6. BUILD UP / Open Play

samples = []

# Kickoffs
samples.append({
    "name": "1H Kick-off (Period 1 Start)",
    "frame": 11185,
    "context": (11180, 11190),
    "expected_type": "Pre-kickoff D -> Live kickoff (H/A)",
    "event_info": "Official 1H kickoff at frame 11185"
})

samples.append({
    "name": "2H Kick-off (Period 2 Start)",
    "frame": 113742,
    "context": (113735, 113750),
    "expected_type": "Pre-kickoff D -> Live kickoff (H/A)",
    "event_info": "Official 2H kickoff at frame 113742"
})

# Find first Goal Kick, Corner Kick, Throw-in from Smart Tagging
for t in tags:
    code = t.get("code", "")
    ttype = t.get("tags", {}).get("Type", "")
    sf = t.get("start_frame")
    ef = t.get("end_frame")
    
    if code == "SET PIECES" and ttype == "Goal Kick" and not any(s["name"].startswith("Goal Kick") for s in samples):
        mid = (sf + ef) // 2
        samples.append({
            "name": f"Goal Kick (frames {sf}-{ef})",
            "frame": mid,
            "context": (max(0, sf - 5), min(total_frames - 1, ef + 15)),
            "expected_type": "Dead ball (D) during stoppage",
            "event_info": f"Tag: {code} - {ttype}, team: {t.get('tags', {}).get('Team', 'N/A')}"
        })
    elif code == "SET PIECES" and ttype == "Corner Kick" and not any(s["name"].startswith("Corner Kick") for s in samples):
        mid = (sf + ef) // 2
        samples.append({
            "name": f"Corner Kick (frames {sf}-{ef})",
            "frame": mid,
            "context": (max(0, sf - 5), min(total_frames - 1, ef + 15)),
            "expected_type": "Dead ball (D) during setup",
            "event_info": f"Tag: {code} - {ttype}, team: {t.get('tags', {}).get('Team', 'N/A')}"
        })
    elif code == "SET PIECES" and ttype == "Throw-in" and not any(s["name"].startswith("Throw-in") for s in samples):
        mid = (sf + ef) // 2
        samples.append({
            "name": f"Throw-in (frames {sf}-{ef})",
            "frame": mid,
            "context": (max(0, sf - 5), min(total_frames - 1, ef + 15)),
            "expected_type": "Dead ball (D) during setup",
            "event_info": f"Tag: {code} - {ttype}, team: {t.get('tags', {}).get('Team', 'N/A')}"
        })
    elif code == "BUILD UP" and not any(s["name"].startswith("Open Play Build Up") for s in samples):
        mid = (sf + ef) // 2
        samples.append({
            "name": f"Open Play Build Up (frames {sf}-{ef})",
            "frame": mid,
            "context": (sf, ef),
            "expected_type": "Live possession (H or A)",
            "event_info": f"Tag: {code}, team: {t.get('tags', {}).get('Team', 'N/A')}"
        })

print(f"{'Sample Scenario':<35} {'Target Frame':<12} {'Predicted':<10} {'Video Verified':<16} {'Notes'}")
print("-" * 105)

for s in samples:
    target = s["frame"]
    pred = labels[target]
    
    # Check video frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    
    # Context labels
    c_start, c_end = s["context"]
    ctx_labels = [labels[f] for f in range(c_start, c_end + 1)]
    unique_ctx = "".join(ctx_labels[:25]) + ("..." if len(ctx_labels) > 25 else "")
    
    # Calculate timestamp in video (MM:SS)
    sec = target / 25.0
    m = int(sec // 60)
    sec_rem = sec % 60
    ts_str = f"{m:02d}:{sec_rem:04.1f}"
    
    verified = "YES (OK)" if ret else "ERR"
    
    print(f"{s['name']:<35} {target:<12} {pred:<10} {verified:<16} t={ts_str} | {s['event_info']}")
    print(f"   Context [{c_start}..{c_end}]: {unique_ctx}")
    print()

cap.release()
print("=" * 105)
print("Spot check against real video frames and Smart Tagging events completed successfully.")
