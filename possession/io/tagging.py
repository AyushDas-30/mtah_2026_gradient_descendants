"""Smart Tagging (SportsCode XML) -> list of intervals on the video-frame timeline.

<start>/<end> are seconds from the first frame of the tactical video (verified by the organisers
against both kickoffs), so frame = round(seconds * fps). Intervals are half-open: end_frame is
the first frame after the phase. Codes are upper-cased and tag values kept verbatim; every
comparison downstream should be case-insensitive.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .. import config


def load_tagging(path, fps=config.FPS):
    out = []
    for instance in ET.parse(str(path)).getroot().iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        tags = {}
        for label in instance.findall("label"):
            group = (label.findtext("group") or "").strip()
            text = (label.findtext("text") or "").strip()
            if group:
                tags[group] = text
        out.append({
            "id": (instance.findtext("ID") or "").strip(),
            "code": (instance.findtext("code") or "").strip().upper(),
            "start_s": start_s, "end_s": end_s,
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
            "tags": tags,
        })
    out.sort(key=lambda e: (e["start_frame"], e["end_frame"], e["code"]))
    return out


def team_side(tag_value, home_name, away_name):
    """'H' / 'A' / None for a Team tag value. Matching is loose on purpose: the file spells the
    away club 'Atletic Club', not 'Athletic Club'."""
    if not tag_value:
        return None
    value = tag_value.strip().lower()
    if value in ("n/a", "na", "none", ""):
        return None

    def norm(name):
        return "".join(ch for ch in (name or "").lower() if ch.isalnum())

    v = norm(value)
    h, a = norm(home_name), norm(away_name)
    if v and (v in h or h in v):
        return "H"
    if v and (v in a or a in v or v.replace("atletic", "athletic") in a):
        return "A"
    return None


def save_tagging(events, path):
    Path(path).write_text(json.dumps(events, indent=1), encoding="utf-8")
    return path


def read_saved_tagging(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
