"""Period-start kick-offs from Smart Tagging, and the dead ball before them.

Every graded frame of a period before its kick-off is dead ball by definition (for BAR-ATH the
second half is graded from 110,808 but kicks off later). The kick-off is located from Smart
Tagging, which tags each kick-off as a SET PIECES window that opens KICKOFF_LEAD_S before the
kick (the organisers' notebook: "they open about 3.5 seconds early").

A formation detector (ball on the centre mark, both teams in their own halves) was tried first
and dropped: in this automatic tracking the pre-kick-off minutes contain warm-up balls and the
ball position at the kick is metres off the mark, so the formation never cleanly appears.

Applied as a hard constraint on the decoder's evidence, not as an overwrite afterwards, so the
first possession after the kick-off is still decoded consistently.
"""

import numpy as np

KICKOFF_LEAD_S = 3.5
SEARCH_S = 600            # a period-start kick-off must fall within its first 10 minutes
FORCE = 1e3


def kickoff_frames(events, fps=25):
    out = []
    for e in events:
        if e["code"] == "SET PIECES" and "kick off" in e["tags"].get("Type", "").lower():
            out.append(int(e["start_frame"] + round(KICKOFF_LEAD_S * fps)))
    return sorted(out)


def pre_kickoff_spans(periods, kickoffs, fps=25):
    """[(start, kickoff)] for each period whose first SEARCH_S seconds contain a kick-off."""
    spans = []
    for _, start, stop in periods:
        inside = [k for k in kickoffs if start <= k < min(stop, start + SEARCH_S * fps)]
        if inside and inside[0] > start:
            spans.append((start, inside[0]))
    return spans


def constrain(scores, spans, dead_code=2):
    out = scores.copy()
    for a, b in spans:
        out[a:b, :] = -FORCE
        out[a:b, dead_code] = FORCE
    return out
