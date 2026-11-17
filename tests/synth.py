"""Synthetic files in the exact formats the pipeline reads -- no competition data involved.

The EPTS writer produces a metadata XML with a DataFormatSpecification and a raw file of
"<frame>:<x,y,speed;...>:<x,y,z>" lines; the DFL writer produces matchinformation + positions
XML with BallStatus / BallPossession; the SportsCode writer a Smart-Tagging-like file.
"""

from pathlib import Path

import numpy as np


def random_walk(rng, n, start, step, lo, hi):
    out = np.empty(n)
    v = start
    for i in range(n):
        v = min(hi, max(lo, v + rng.normal(0, step)))
        out[i] = v
    return out


def write_epts(folder, n_frames=2000, n_slots=46, shuffle_groups=False, seed=0,
               ball_missing=0.4, player_missing=0.1, empty_fields=False, home_name="FC Barcelona",
               speed_unit="meters per second"):
    """Returns dict with paths and the normalised truth arrays (x, y, speed, ball, slot_team)."""
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    teams = [("T1", home_name), ("T2", "Atletic Club")]
    player_ids = ["P{}".format(i + 1) for i in range(n_slots)]
    player_team = ["T1" if i < n_slots // 2 else "T2" for i in range(n_slots)]
    group_order = list(range(n_slots))
    if shuffle_groups:
        rng.shuffle(group_order)

    players_xml = "\n".join(
        '<Player id="{}" teamId="{}"><Name>Track {}</Name><ShirtNumber>{}</ShirtNumber></Player>'.format(
            pid, team, i + 1, i + 1) for i, (pid, team) in enumerate(zip(player_ids, player_team)))
    pc_xml, groups_xml = [], []
    for i, pid in enumerate(player_ids):
        for ch in ("x", "y", "speed"):
            pc_xml.append('<PlayerChannel id="{}_{}" playerId="{}" channelId="{}"/>'.format(pid, ch, pid, ch))
    for slot in group_order:
        pid = player_ids[slot]
        groups_xml.append('<SplitRegister separator=",">' + "".join(
            '<PlayerChannelRef playerChannelId="{}_{}"/>'.format(pid, ch) for ch in ("x", "y", "speed"))
            + "</SplitRegister>")
    meta = """<?xml version="1.0" encoding="UTF-8"?>
<main>
<Metadata>
<GlobalConfig><FrameRate>25</FrameRate><ProviderName>Metrica Sports</ProviderName></GlobalConfig>
<Sessions><Session id="1"><FieldSize><Width>105</Width><Height>68</Height></FieldSize></Session></Sessions>
<Teams>{teams}</Teams>
<Players>{players}</Players>
<Devices><Device id="d"><Sensors>
<Sensor id="position"><Name>position</Name><Channels>
<Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>
<Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel>
</Channels></Sensor>
<Sensor id="speed"><Name>speed</Name><Channels>
<Channel id="speed"><Name>speed</Name><Unit>{speed_unit}</Unit></Channel>
</Channels></Sensor>
</Sensors></Device></Devices>
<PlayerChannels>{pcs}</PlayerChannels>
</Metadata>
<DataFormatSpecifications>
<DataFormatSpecification startFrame="1" endFrame="{n}" separator=":">
<StringRegister name="frameCount"/>
<SplitRegister separator=";">{groups}</SplitRegister>
<SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/><BallChannelRef channelId="z"/></SplitRegister>
</DataFormatSpecification>
</DataFormatSpecifications>
</main>""".format(teams="".join('<Team id="{}"><Name>{}</Name></Team>'.format(t, n) for t, n in teams),
                  players=players_xml, pcs="".join(pc_xml), n=n_frames, groups="".join(groups_xml),
                  speed_unit=speed_unit)
    meta_path = folder / "meta.xml"
    meta_path.write_text(meta, encoding="utf-8")

    x = np.stack([random_walk(rng, n_frames, rng.uniform(0.1, 0.9), 0.003, 0, 1) for _ in range(n_slots)], 1)
    y = np.stack([random_walk(rng, n_frames, rng.uniform(0.1, 0.9), 0.003, 0, 1) for _ in range(n_slots)], 1)
    sp = rng.uniform(0, 8, size=(n_frames, n_slots))
    miss = rng.random((n_frames, n_slots)) < player_missing
    x[miss] = np.nan
    y[miss] = np.nan
    sp[miss] = np.nan
    bx = random_walk(rng, n_frames, 0.5, 0.01, -0.05, 1.05)
    by = random_walk(rng, n_frames, 0.5, 0.01, -0.05, 1.05)
    bmiss = rng.random(n_frames) < ball_missing
    bx[bmiss] = np.nan
    by[bmiss] = np.nan

    def fmt(v):
        return "NaN" if np.isnan(v) else "{:.5f}".format(v)

    raw_path = folder / "raw.txt"
    with open(raw_path, "w", newline="\n") as handle:
        for t in range(n_frames):
            groups = []
            for slot in group_order:
                if empty_fields and np.isnan(x[t, slot]) and t % 2:
                    groups.append(",,")
                else:
                    groups.append("{},{},{}".format(fmt(x[t, slot]), fmt(y[t, slot]), fmt(sp[t, slot])))
            handle.write("{}:{}:{},{},{}\n".format(t + 1, ";".join(groups), fmt(bx[t]), fmt(by[t]),
                                                   "NaN" if np.isnan(bx[t]) else "0.0"))
    slot_team = np.array([0 if player_team[s] == "T1" else 1 for s in group_order], dtype=np.int8)
    order = np.array(group_order)
    return {"meta": meta_path, "raw": raw_path, "x": x[:, order], "y": y[:, order],
            "speed": sp[:, order], "bx": bx, "by": by, "slot_team": slot_team}


def starter_reference(raw_path, meta_path, n_frames, pitch):
    """Line-by-line mirror of starter/baseline.py's nearest_player_labels (team by <Player> order)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(str(meta_path)).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    team_of_track = [0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")]
    length, width = pitch
    out = np.full(n_frames, -1, dtype=np.int8)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            ball = parts[2]
            if ball[0] == "N":
                continue
            c = ball.index(",")
            bxm = float(ball[:c]) * length
            bym = float(ball[c + 1:ball.index(",", c + 1)]) * width
            best, best_team = None, -1
            for slot, group in enumerate(parts[1].split(";")):
                if group[0] == "N":
                    continue
                c = group.index(",")
                dx = float(group[:c]) * length - bxm
                dy = float(group[c + 1:group.index(",", c + 1)]) * width - bym
                d = dx * dx + dy * dy
                if best is None or d < best:
                    best, best_team = d, team_of_track[slot]
            out[int(parts[0]) - 1] = best_team
    return out


def write_tagging(path, events):
    """events: [(code, start_s, end_s, {group: text})]"""
    body = []
    for i, (code, a, b, tags) in enumerate(events):
        labels = "".join("<label><group>{}</group><text>{}</text></label>".format(g, t) for g, t in tags.items())
        body.append("<instance><ID>{}</ID><start>{}</start><end>{}</end><code>{}</code>{}</instance>".format(
            i + 1, a, b, code, labels))
    Path(path).write_text("<file><ALL_INSTANCES>{}</ALL_INSTANCES></file>".format("".join(body)), encoding="utf-8")
    return path


def write_dfl(folder, match_id, comp="DFL-COM-000001", n_half=(3000, 2800), n_per_team=12, seed=0,
              informative=False):
    """Minimal DFL matchinformation + positions XML; returns the expected labels per half."""
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    info = """<PutDataRequest><MatchInformation>
<General MatchId="DFL-MAT-{m}" HomeTeamId="HOME1" GuestTeamId="AWAY1"/>
<Environment PitchX="105.0" PitchY="68.0"/>
<Teams>
<Team TeamId="HOME1" TeamName="Home FC" Role="home"><Players>{hp}</Players></Team>
<Team TeamId="AWAY1" TeamName="Away FC" Role="guest"><Players>{ap}</Players></Team>
</Teams></MatchInformation></PutDataRequest>""".format(
        m=match_id,
        hp="".join('<Player PersonId="H{}" ShirtNumber="{}" PlayingPosition="X"/>'.format(i, i + 1) for i in range(n_per_team)),
        ap="".join('<Player PersonId="G{}" ShirtNumber="{}" PlayingPosition="X"/>'.format(i, i + 1) for i in range(n_per_team)))
    (folder / "DFL_02_01_matchinformation_{}_DFL-MAT-{}.xml".format(comp, match_id)).write_text(info)

    sets, expected = [], {}
    n_start = {"firstHalf": 10000, "secondHalf": 100000}
    for section, n in zip(("firstHalf", "secondHalf"), n_half):
        n0 = n_start[section]
        status = np.ones(n, dtype=int)
        poss = np.ones(n, dtype=int)
        t = 0
        team = 1
        while t < n:                                   # alternate possessions with dead balls
            length = int(rng.integers(100, 400))
            poss[t:t + length] = team
            if rng.random() < 0.3:
                d = int(rng.integers(50, 200))
                status[t + length:t + length + d] = 0
                poss[t + length:t + length + d] = team
                length += d
            team = 3 - team
            t += length
        expected[section] = np.where(status == 0, 2, np.where(poss == 1, 0, 1))
        bx = random_walk(rng, n, 0, 0.3, -52, 52)
        by = random_walk(rng, n, 0, 0.3, -34, 34)
        frames = "".join('<Frame N="{}" T="x" X="{:.2f}" Y="{:.2f}" Z="0.0" S="0" BallPossession="{}" BallStatus="{}"/>'.format(
            n0 + i, bx[i], by[i], poss[i], status[i]) for i in range(n))
        sets.append('<FrameSet GameSection="{}" TeamId="BALL" PersonId="DFL-OBJ-0000XT">{}</FrameSet>'.format(section, frames))
        ref = "".join('<Frame N="{}" T="x" X="0.00" Y="0.00" D="0" S="0" A="0" M="1"/>'.format(n0 + j)
                      for j in range(n))
        sets.append('<FrameSet GameSection="{}" TeamId="referee" PersonId="REF1">{}</FrameSet>'.format(
            section, ref))
        if informative:                     # dead ball: ball beyond the touchline
            dead = status == 0
            by = np.where(dead, -35.5, by)
        for team_id, prefix in (("HOME1", "H"), ("AWAY1", "G")):
            for i in range(11):
                px = random_walk(rng, n, rng.uniform(-40, 40), 0.1, -52, 52)
                py = random_walk(rng, n, rng.uniform(-30, 30), 0.1, -34, 34)
                if informative and i == 0:       # the possessing team's player 0 is on the ball
                    mine = (poss == (1 if team_id == "HOME1" else 2)) & (status == 1)
                    px = np.where(mine, bx + rng.normal(0, 0.7, n), px)
                    py = np.where(mine, by + rng.normal(0, 0.7, n), py)
                frames = "".join('<Frame N="{}" T="x" X="{:.2f}" Y="{:.2f}" D="0" S="0" A="0" M="1"/>'.format(
                    n0 + j, px[j], py[j]) for j in range(n))
                sets.append('<FrameSet GameSection="{}" TeamId="{}" PersonId="{}{}">{}</FrameSet>'.format(
                    section, team_id, prefix, i, frames))
    (folder / "DFL_04_03_positions_raw_observed_{}_DFL-MAT-{}.xml".format(comp, match_id)).write_text(
        "<PutDataRequest><Positions>{}</Positions></PutDataRequest>".format("".join(sets)))
    return expected
