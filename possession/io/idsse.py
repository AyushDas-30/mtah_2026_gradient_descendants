"""IDSSE (Bassek et al., Scientific Data 2025) DFL XML -> clean labelled Match.

Seven Bundesliga / 2. Bundesliga 2022-23 matches, 25 Hz positions for every player and the ball.
Every ball frame carries BallStatus (1 alive, 0 dead) and BallPossession (1 home, 2 away) -- the
same kind of source the competition ground truth is derived from ("D where the corrected tracking
marks the ball dead, otherwise the team it names as owning the ball"). Labels here are
therefore: D where BallStatus == 0, else H / A from BallPossession.

This is human-curated data about OTHER games, which RULES.md explicitly allows for training and
validation. Licence: see the dataset's figshare page (cite the paper in the technical document).

Files per match, as published on figshare (names as floodlight expects them):
    DFL_02_01_matchinformation_<COMP>_DFL-MAT-<ID>.xml
    DFL_04_03_positions_raw_observed_<COMP>_DFL-MAT-<ID>.xml

Coordinates: DFL is metres with the origin at the centre spot. We shift to a corner origin so a
Match from here and from the ATD share one convention. Speeds are recomputed from positions.
Halves are laid end to end with a gap of ungraded (X) frames, so nothing temporal crosses them.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from ..schema import A, D, H, X, AWAY, HOME, Match

MATCHES = {
    # match id: (competition, info file id, events file id, positions file id) on figshare
    "J03WMX": ("DFL-COM-000001", "51643475", "51643493", "51643514"),
    "J03WN1": ("DFL-COM-000001", "51643472", "51643496", "51643517"),
    "J03WPY": ("DFL-COM-000002", "51643487", "51643505", "51643526"),
    "J03WOH": ("DFL-COM-000002", "51643478", "51643499", "51643520"),
    "J03WQQ": ("DFL-COM-000002", "51643484", "51643508", "51643529"),
    "J03WOY": ("DFL-COM-000002", "51643481", "51643502", "51643523"),
    "J03WR9": ("DFL-COM-000002", "51643490", "51643511", "51643532"),
}
FIGSHARE = "https://ndownloader.figshare.com/files/{}"
HALF_GAP = 250   # ungraded frames inserted between halves
SECTIONS = ("firstHalf", "secondHalf")


def file_names(match_id):
    comp = MATCHES[match_id][0]
    return {
        "info": "DFL_02_01_matchinformation_{}_DFL-MAT-{}.xml".format(comp, match_id),
        "positions": "DFL_04_03_positions_raw_observed_{}_DFL-MAT-{}.xml".format(comp, match_id),
    }


def download(match_id, folder, log=print):
    """Fetch info + positions XML for one match (skips files already present)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    comp, info_id, _, pos_id = MATCHES[match_id]
    names = file_names(match_id)
    for key, file_id in (("info", info_id), ("positions", pos_id)):
        target = folder / names[key]
        if target.is_file() and target.stat().st_size > 0:
            log("  have {}".format(target.name))
            continue
        log("  downloading {} ...".format(target.name))
        tmp = target.with_suffix(".part")
        request = Request(FIGSHARE.format(file_id), headers={"User-Agent": "mtah-possession/0.1"})
        with urlopen(request, timeout=60) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done, shown = 0, -1
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
                done += len(block)
                mb = done >> 20
                if mb // 10 != shown:
                    shown = mb // 10
                    log("    {} MB{}".format(mb, " of {}".format(total >> 20) if total else ""))
        tmp.replace(target)
    return folder


def read_match_info(path):
    root = ET.parse(str(path)).getroot()
    info = root.find("MatchInformation")
    general = info.find("General")
    home_id = general.get("HomeTeamId")
    away_id = general.get("GuestTeamId") or general.get("AwayTeamId")
    env = info.find("Environment")
    pitch = (float(env.get("PitchX")), float(env.get("PitchY")))
    person_team = {}
    for team in info.find("Teams").iter("Team"):
        tid = team.get("TeamId")
        for player in team.iter("Player"):
            person_team[player.get("PersonId")] = tid
    return {"home_id": home_id, "away_id": away_id, "pitch": pitch, "person_team": person_team}


def _code(value, default):
    if value is None or not value.strip():
        return default
    return int(float(value))


def _iter_framesets(path):
    """Yield (attrib, n, x, y, status, poss) per FrameSet as numpy arrays, streaming the XML.

    status is -1 where BallStatus is absent, poss 0 where BallPossession is absent.
    """
    context = ET.iterparse(str(path), events=("start", "end"))
    _, root = next(context)
    current = None
    n, xs, ys, st, po = [], [], [], [], []
    for event, elem in context:
        if event == "start" and elem.tag == "FrameSet":
            current = dict(elem.attrib)
            n, xs, ys, st, po = [], [], [], [], []
        elif event == "end" and elem.tag == "Frame" and current is not None:
            a = elem.attrib
            n.append(int(a["N"]))
            xs.append(float(a["X"]))
            ys.append(float(a["Y"]))
            st.append(_code(a.get("BallStatus"), -1))
            po.append(_code(a.get("BallPossession"), 0))
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            yield (current, np.array(n, dtype=np.int64), np.array(xs), np.array(ys),
                   np.array(st, dtype=np.int8), np.array(po, dtype=np.int8))
            current = None
            elem.clear()
            root.clear()


def labels_from_codes(status, poss):
    """D where dead; else the possessing team, carrying the last named team over gaps."""
    team = np.where(poss == 1, H, np.where(poss == 2, A, -1)).astype(np.int8)
    known = np.flatnonzero(team >= 0)
    carried = np.full(len(team), H, dtype=np.int8)
    if len(known):
        idx = np.maximum.accumulate(np.where(team >= 0, np.arange(len(team)), -1))
        idx[idx < 0] = known[0]
        carried = team[idx]
    return np.where(status == 0, D, carried).astype(np.int8)


def load_match(match_id, folder):
    names = file_names(match_id)
    folder = Path(folder)
    info = read_match_info(folder / names["info"])
    length, width = info["pitch"]

    ball_sets, player_sets = {}, []
    for attrib, n, xs, ys, st, po in _iter_framesets(folder / names["positions"]):
        section = attrib.get("GameSection")
        if section not in SECTIONS or not len(n):
            continue
        if (attrib.get("TeamId") or "").lower() == "ball":
            ball_sets[section] = (n, xs, ys, st, po)
        else:
            player_sets.append((section, attrib.get("PersonId"), attrib.get("TeamId"), n, xs, ys))
    if set(ball_sets) != set(SECTIONS):
        raise ValueError("{}: ball frame sets for {}".format(match_id, sorted(ball_sets)))

    # Timeline: half 1, gap, half 2, each half spanning its ball FrameSet's N range.
    offsets, periods, cursor = {}, [], 0
    for pid, section in enumerate(SECTIONS, start=1):
        n0, n1 = int(ball_sets[section][0][0]), int(ball_sets[section][0][-1])
        offsets[section] = cursor - n0
        periods.append((pid, cursor, cursor + n1 - n0 + 1))
        cursor += n1 - n0 + 1 + (HALF_GAP if pid == 1 else 0)
    T = cursor

    # Only the two teams' players. DFL positions also carry the referees (and can carry other
    # tracked persons); they are not players and must not land in a team's slots.
    teams = {info["home_id"], info["away_id"]}
    skipped = sorted({p[1] for p in player_sets if p[2] not in teams})
    player_sets = [p for p in player_sets if p[2] in teams]
    persons = sorted({(p[2], p[1]) for p in player_sets})       # (team id, person id)
    slot_of = {person: s for s, (_, person) in enumerate(persons)}
    slot_team = np.array([HOME if team == info["home_id"] else AWAY for team, _ in persons],
                         dtype=np.int8)
    S = len(persons)
    xy = np.full((T, S, 2), np.nan)
    ball = np.full((T, 2), np.nan)
    labels = np.full(T, X, dtype=np.int8)

    for section, (n, xs, ys, st, po) in ball_sets.items():
        idx = n + offsets[section]
        ball[idx, 0] = xs + length / 2.0
        ball[idx, 1] = ys + width / 2.0
        labels[idx] = labels_from_codes(st, po)
        # frames of a half absent from the ball set stay X (ungraded)

    for section, person, team, n, xs, ys in player_sets:
        idx = n + offsets[section]
        keep = (idx >= 0) & (idx < T)
        s = slot_of[person]
        xy[idx[keep], s, 0] = xs[keep] + length / 2.0
        xy[idx[keep], s, 1] = ys[keep] + width / 2.0

    speed = speed_from_xy(xy, periods, fps=25)
    return Match(
        name="IDSSE-" + match_id, fps=25, pitch=(length, width), xy=xy, speed=speed,
        slot_team=slot_team, ball=ball, periods=periods, labels=labels,
        meta={"source": "idsse", "match_id": match_id, "n_persons": S,
              "n_home_persons": int((slot_team == HOME).sum()),
              "n_away_persons": int((slot_team == AWAY).sum()),
              "n_skipped_non_players": len(skipped)},
    ).validate()


def speed_from_xy(xy, periods, fps=25, span=2):
    """Central-difference speed (m/s) over +/- span frames, never across a period boundary."""
    speed = np.full(xy.shape[:2], np.nan, dtype=np.float32)
    for _, start, stop in periods:
        seg = xy[start:stop]
        if len(seg) <= 2 * span:
            continue
        d = seg[2 * span:] - seg[:-2 * span]
        v = np.sqrt((d ** 2).sum(axis=-1)) * fps / (2.0 * span)
        speed[start + span:stop - span] = v
    return speed
