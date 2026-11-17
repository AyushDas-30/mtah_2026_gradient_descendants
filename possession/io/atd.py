"""Metrica FIFA/EPTS ATD -> Match. Standard library + numpy, no kloppy.

The metadata XML says, in its DataFormatSpecification, exactly how a raw line is laid out:

    <frameCount> : <player group> ; <player group> ; ... : <ball group>

Each player group is a comma-separated list of PlayerChannelRefs (e.g. x, y, speed for one
track slot), and the ball group a list of BallChannelRefs. We compile that specification once
into a layout -- which slot is which team, which field is x / y / speed -- and then parse every
line with plain string splits, converting a chunk of lines to floats in one numpy call.

The starter baseline assumes instead that groups follow the order of <Player> elements and that
the fields are x, y, speed. build_layout() reports whether the specification agrees with that
assumption, so a disagreement is visible rather than silently inherited.
"""

import xml.etree.ElementTree as ET

import numpy as np

from .. import config
from ..schema import AWAY, HOME, Match

CHUNK = 4096
# Everything downstream speaks m/s. The tactical ATD declares its speed channel in kph.
SPEED_TO_MS = {"kph": 1 / 3.6, "km/h": 1 / 3.6, "kmh": 1 / 3.6, "kilometers per hour": 1 / 3.6,
               "m/s": 1.0, "meters per second": 1.0, "metres per second": 1.0, "": 1.0}


def _speed_unit(md):
    units = {c["unit"].strip().lower() for c in md["channels"].values() if c["role"] == "speed"}
    if len(units) > 1:
        raise ValueError("several speed units: {}".format(sorted(units)))
    return units.pop() if units else ""


# -- metadata ------------------------------------------------------------------------------

def _text(elm, tag, default=None):
    child = elm.find(tag) if elm is not None else None
    return child.text.strip() if child is not None and child.text else default


def _channel_role(channel_id, name):
    key = "{} {}".format(channel_id or "", name or "").lower()
    if "speed" in key or "velocity" in key:
        return "speed"
    tokens = key.replace("_", " ").replace("-", " ").split()
    if "x" in tokens or "position x" in key or key.strip().endswith("x"):
        return "x"
    if "y" in tokens or "position y" in key or key.strip().endswith("y"):
        return "y"
    if "z" in tokens:
        return "z"
    return "other"


def read_metadata(meta_path):
    """Everything we use from the metadata XML, as plain Python."""
    root = ET.parse(str(meta_path)).getroot()
    meta = root.find("Metadata") if root.tag != "Metadata" else root
    if meta is None:
        meta = root

    teams = [(t.get("id"), _text(t, "Name", "")) for t in meta.iter("Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    players = [(p.get("id"), p.get("teamId")) for p in meta.iter("Player")]

    # Home side: MatchParameters/Score@idLocalTeam when present, else the first team (EPTS /
    # kloppy convention, and what the starter baseline uses). Both are recorded.
    score = meta.find(".//MatchParameters/Score")
    home_by_score = score.get("idLocalTeam") if score is not None else None
    home_by_order = teams[0][0]

    channels = {}
    for sensor in meta.iter("Sensor"):
        sensor_id = sensor.get("id")
        for channel in sensor.iter("Channel"):
            channels[channel.get("id")] = {
                "sensor": sensor_id, "name": _text(channel, "Name", ""),
                "unit": _text(channel, "Unit", ""),
                "role": _channel_role(channel.get("id"), _text(channel, "Name", "")),
            }
    player_channels = {pc.get("id"): (pc.get("playerId"), pc.get("channelId"))
                       for pc in meta.iter("PlayerChannel")}

    field_size = meta.find(".//FieldSize")
    specs = root.findall(".//DataFormatSpecification")
    return {
        "teams": teams, "players": players, "home_by_score": home_by_score,
        "home_by_order": home_by_order, "channels": channels,
        "player_channels": player_channels, "specs": specs,
        "frame_rate": _text(meta.find("GlobalConfig"), "FrameRate"),
        "field_size": (_text(field_size, "Width"), _text(field_size, "Height"))
        if field_size is not None else None,
    }


def build_layout(md):
    """Compile the DataFormatSpecification into slot/field indices.

    Returns a dict: slot_player (player id per slot), slot_team (0/1), player_fields (role per
    field inside a group), ball_fields, n_slots, and how the result compares to the starter
    baseline's Player-order assumption.
    """
    team_of_player = dict(md["players"])
    home_id = md["home_by_score"] or md["home_by_order"]
    notes = []
    if md["home_by_score"] and md["home_by_score"] != md["home_by_order"]:
        notes.append("WARNING: Score@idLocalTeam disagrees with first-team order")

    layout = None
    if md["specs"]:
        if len(md["specs"]) > 1:
            notes.append("{} DataFormatSpecifications; using the first, checking the rest".format(
                len(md["specs"])))
        layout = _layout_from_spec(md["specs"][0], md, team_of_player, home_id)
        for extra in md["specs"][1:]:
            other = _layout_from_spec(extra, md, team_of_player, home_id)
            if other["slot_player"] != layout["slot_player"]:
                raise NotImplementedError("DataFormatSpecifications differ between frame ranges")
    if layout is None:
        notes.append("no DataFormatSpecification: falling back to <Player> order, x,y,speed")
        slot_player = [pid for pid, _ in md["players"]]
        layout = {"slot_player": slot_player, "player_fields": ["x", "y", "speed"],
                  "ball_fields": ["x", "y", "other"], "source": "player-order"}

    layout["slot_team"] = np.array(
        [HOME if team_of_player[p] == home_id else AWAY for p in layout["slot_player"]],
        dtype=np.int8)
    layout["n_slots"] = len(layout["slot_player"])
    order_team = np.array([HOME if t == md["home_by_order"] else AWAY for _, t in md["players"]],
                          dtype=np.int8)
    layout["agrees_with_player_order"] = bool(
        len(order_team) == layout["n_slots"] and np.array_equal(order_team, layout["slot_team"]))
    layout["home_id"] = home_id
    layout["home_name"] = dict(md["teams"])[home_id]
    layout["notes"] = notes
    return layout


def _layout_from_spec(spec, md, team_of_player, home_id):
    # The DataFormatSpecification element is itself the top-level split register (that is how
    # the EPTS standard and kloppy read it). Tolerate one wrapping SplitRegister as well.
    top = spec
    if [c.tag for c in spec] == ["SplitRegister"]:
        top = spec[0]
    children = list(top)
    # Canonical Metrica layout: StringRegister, SplitRegister(';', player groups), SplitRegister(',', ball)
    kinds = [c.tag for c in children]
    if top.get("separator", ":") != ":" or kinds[:1] != ["StringRegister"] or len(children) != 3:
        raise NotImplementedError("non-canonical EPTS layout: separator={!r} children={}".format(
            top.get("separator"), kinds))
    players_reg, ball_reg = children[1], children[2]
    if players_reg.get("separator") != ";":
        raise NotImplementedError("player register separator {!r}".format(players_reg.get("separator")))

    slot_player, player_fields = [], None
    for group in players_reg:
        refs = [r for r in group if r.tag == "PlayerChannelRef"]
        if not refs:
            raise NotImplementedError("player group without PlayerChannelRef")
        owners = {md["player_channels"][r.get("playerChannelId")][0] for r in refs}
        if len(owners) != 1:
            raise NotImplementedError("player group mixes players: {}".format(owners))
        roles = [md["channels"][md["player_channels"][r.get("playerChannelId")][1]]["role"]
                 for r in refs]
        if player_fields is None:
            player_fields = roles
        elif roles != player_fields:
            raise NotImplementedError("player groups disagree on field roles")
        slot_player.append(owners.pop())

    ball_fields = []
    for ref in ball_reg:
        if ref.tag == "BallChannelRef":
            channel = md["channels"].get(ref.get("channelId"), {})
            ball_fields.append(channel.get("role", "other"))
        else:
            ball_fields.append("other")
    return {"slot_player": slot_player, "player_fields": player_fields,
            "ball_fields": ball_fields, "source": "DataFormatSpecification"}


# -- raw data ------------------------------------------------------------------------------

def _clean(field_string):
    """Empty fields mean missing. Only pay for the fix when there is one."""
    if ",," in field_string or field_string.startswith(",") or field_string.endswith(","):
        return ",".join(t if t else "nan" for t in field_string.split(","))
    return field_string


def read_raw(raw_path, layout, n_frames, pitch):
    """Parse the raw file into arrays on the video-frame timeline (video_frame = frame_id - 1)."""
    S = layout["n_slots"]
    k = len(layout["player_fields"])
    kb = len(layout["ball_fields"])
    ix, iy = layout["player_fields"].index("x"), layout["player_fields"].index("y")
    isp = layout["player_fields"].index("speed") if "speed" in layout["player_fields"] else None
    bx, by = layout["ball_fields"].index("x"), layout["ball_fields"].index("y")
    length, width = pitch

    xy = np.full((n_frames, S, 2), np.nan, dtype=np.float64)
    speed = np.full((n_frames, S), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float64)
    seen = np.zeros(n_frames, dtype=bool)
    stats = {"lines": 0, "malformed": 0, "out_of_range": 0, "duplicates": 0,
             "first_id": None, "last_id": None, "monotonic": True}

    rows, frames = [], []

    def flush():
        if not rows:
            return
        values = np.array(rows, dtype=np.float64)          # (n, S*k + kb)
        idx = np.array(frames, dtype=np.int64)
        players = values[:, :S * k].reshape(len(idx), S, k)
        xy[idx, :, 0] = players[:, :, ix] * length
        xy[idx, :, 1] = players[:, :, iy] * width
        if isp is not None:
            speed[idx] = players[:, :, isp]
        ball[idx, 0] = values[:, S * k + bx] * length
        ball[idx, 1] = values[:, S * k + by] * width
        rows.clear()
        frames.clear()

    previous = None
    with open(raw_path, "r", encoding="ascii", errors="replace") as handle:
        for line in handle:
            stats["lines"] += 1
            parts = line.rstrip("\r\n").split(":")
            if len(parts) == 4 and parts[3] == "":
                parts = parts[:3]
            if len(parts) != 3:
                stats["malformed"] += 1
                continue
            frame_id = int(parts[0])
            if stats["first_id"] is None:
                stats["first_id"] = frame_id
            stats["last_id"] = frame_id
            if previous is not None and frame_id <= previous:
                stats["monotonic"] = False
            previous = frame_id

            player_part = parts[1].rstrip(";")
            tokens = _clean(player_part.replace(";", ",")).split(",")
            ball_tokens = _clean(parts[2]).split(",")
            if len(ball_tokens) < kb:
                ball_tokens += ["nan"] * (kb - len(ball_tokens))
            if len(tokens) != S * k or len(ball_tokens) != kb:
                stats["malformed"] += 1
                continue
            video_frame = frame_id - 1
            if not 0 <= video_frame < n_frames:
                stats["out_of_range"] += 1
                continue
            if seen[video_frame]:
                stats["duplicates"] += 1
            seen[video_frame] = True
            tokens.extend(ball_tokens)
            rows.append(tokens)
            frames.append(video_frame)
            if len(rows) >= CHUNK:
                flush()
    flush()
    stats["records"] = int(seen.sum())
    stats["contiguous_from_1"] = bool(seen.all() and stats["first_id"] == 1)
    return xy, speed, ball, stats


def load_tactical(data_root=None, n_frames=config.N_VIDEO_FRAMES, pitch=config.PITCH, periods=None):
    paths = config.tactical_paths(data_root)
    for key in ("atd_meta", "atd_raw"):
        if not paths[key].is_file():
            raise FileNotFoundError("missing {}: {}".format(key, paths[key]))
    md = read_metadata(paths["atd_meta"])
    layout = build_layout(md)
    xy, speed, ball, stats = read_raw(paths["atd_raw"], layout, n_frames, pitch)
    speed_unit = _speed_unit(md)
    factor = SPEED_TO_MS.get(speed_unit)
    if factor is None:
        raise ValueError("unknown speed unit {!r}; add it to SPEED_TO_MS".format(speed_unit))
    speed = (speed * factor).astype(np.float32)
    if periods is None:
        periods = [(half, first, last + 1) for half, first, last in config.GRADED_HALVES]
    match = Match(
        name=config.GAME, fps=config.FPS, pitch=pitch, xy=xy, speed=speed,
        slot_team=layout["slot_team"], ball=ball, periods=periods, labels=None,
        meta={"source": "tactical-atd", "parse_stats": stats,
              "layout_source": layout["source"], "player_fields": layout["player_fields"],
              "ball_fields": layout["ball_fields"], "home_name": layout["home_name"],
              "agrees_with_player_order": layout["agrees_with_player_order"],
              "notes": layout["notes"], "frame_rate": md["frame_rate"],
              "field_size": md["field_size"],
              "units": sorted({c["unit"] for c in md["channels"].values()}),
              "speed_unit_in_file": speed_unit, "speed_unit_stored": "m/s"},
    )
    return match.validate()
