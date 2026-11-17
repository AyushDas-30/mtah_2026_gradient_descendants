"""Pass-direction and co-movement features (phase 8 candidate). numpy only, label-free.

recv_margin   while the ball travels (speed >= RECV_MIN_SPEED), for each team the smallest
              lateral distance from a player to the ball's path ahead (the player must be ahead of
              the ball and reachable within RECV_HORIZON_S at the current speed); away minus home,
              clipped: positive = a home player is better placed to receive
recv_active   1 where recv_margin is defined (ball travelling), else 0
comove_diff   for the nearest player of each team within COMOVE_RADIUS: speed of the ball relative
              to that player (|v_ball - v_player|); away minus home, clipped: positive = the ball moves
              with a home player (dribble / control)
Ball velocity is a central difference over +/- VEL_SPAN frames, which survives the tracker's
smoothing; players' velocities use the same span. Missing values are 0 (neutral).
"""

import numpy as np

VEL_SPAN = 4
RECV_MIN_SPEED = 4.0
RECV_HORIZON_S = 1.5
RECV_CLIP = 10.0
COMOVE_RADIUS = 3.0
COMOVE_CLIP = 8.0


def _velocity(pos, fps, span=VEL_SPAN):
    v = np.full(pos.shape, np.nan)
    v[span:-span] = (pos[2 * span:] - pos[:-2 * span]) * fps / (2.0 * span)
    return v


def compute(ball, xy, slot_team, periods, fps=25):
    """ball (T,2) with NaN where unknown (interpolated ball recommended); xy (T,S,2)."""
    T = len(ball)
    vb = np.full((T, 2), np.nan)
    vp = np.full(xy.shape, np.nan)
    for _, a, b in periods:                       # never difference across a period edge
        vb[a:b] = _velocity(ball[a:b], fps)
        vp[a:b] = _velocity(xy[a:b], fps)
    speed = np.hypot(vb[:, 0], vb[:, 1])
    out_recv = np.zeros(T)
    active = np.zeros(T)
    out_co = np.zeros(T)
    teams = [np.flatnonzero(slot_team == t) for t in (0, 1)]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = vb / speed[:, None]
        rel = xy - ball[:, None, :]                                     # (T,S,2)
        along = rel[:, :, 0] * u[:, None, 0] + rel[:, :, 1] * u[:, None, 1]
        lateral = np.abs(rel[:, :, 0] * u[:, None, 1] - rel[:, :, 1] * u[:, None, 0])
        reach = speed * RECV_HORIZON_S
        ok = (along > 0) & (along < reach[:, None]) & np.isfinite(lateral)
        lat = np.where(ok, lateral, np.inf)
        best = [lat[:, c].min(axis=1) if len(c) else np.full(T, np.inf) for c in teams]
        moving = np.isfinite(speed) & (speed >= RECV_MIN_SPEED)
        bh = np.minimum(best[0], RECV_CLIP)
        ba = np.minimum(best[1], RECV_CLIP)
        has = moving & (np.isfinite(best[0]) | np.isfinite(best[1]))
        out_recv = np.where(has, ba - bh, 0.0)
        active = has.astype(np.float64)
        # co-movement: nearest player of each team within the radius
        dist = np.hypot(rel[:, :, 0], rel[:, :, 1])
        dv = np.hypot(vb[:, None, 0] - vp[:, :, 0], vb[:, None, 1] - vp[:, :, 1])
        co = []
        for c in teams:
            d = np.where(np.isfinite(dist[:, c]), dist[:, c], np.inf)
            j = np.argmin(d, axis=1)
            dmin = d[np.arange(T), j]
            dvj = dv[:, c][np.arange(T), j]
            co.append(np.where((dmin <= COMOVE_RADIUS) & np.isfinite(dvj), np.minimum(dvj, COMOVE_CLIP), COMOVE_CLIP))
        out_co = co[1] - co[0]
    return {"recv_margin": np.nan_to_num(out_recv), "recv_active": active,
            "comove_diff": np.nan_to_num(out_co)}


SPEC_NAMES = ("recv_margin", "recv_active", "comove_diff")


PLAYER_VMAX = 7.0        # m/s a player can cover to meet the ball
REACT_S = 0.3            # reaction allowance


def first_interceptor(ball, xy, slot_team, periods, fps=25):
    """While the ball travels: the team of the first player (along the path) able to reach the
    ball's path before the ball does. +1 home, -1 away, 0 none / ball not travelling; also the
    distance (m) along the path to that interception point."""
    T = len(ball)
    vb = np.full((T, 2), np.nan)
    for _, a, b in periods:
        vb[a:b] = _velocity(ball[a:b], fps)
    speed = np.hypot(vb[:, 0], vb[:, 1])
    team = np.zeros(T)
    dist = np.full(T, RECV_HORIZON_S * 40.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = vb / speed[:, None]
        rel = xy - ball[:, None, :]
        along = rel[:, :, 0] * u[:, None, 0] + rel[:, :, 1] * u[:, None, 1]
        lateral = np.abs(rel[:, :, 0] * u[:, None, 1] - rel[:, :, 1] * u[:, None, 0])
        t_ball = along / speed[:, None]
        t_player = lateral / PLAYER_VMAX
        feasible = (along > -0.5) & (t_ball <= RECV_HORIZON_S) & (t_player <= t_ball + REACT_S)
        moving = np.isfinite(speed) & (speed >= RECV_MIN_SPEED)
        key = np.where(feasible & moving[:, None], along, np.inf)
        j = np.argmin(key, axis=1)
        hit = np.isfinite(key[np.arange(T), j])
        team = np.where(hit, np.where(slot_team[j] == 0, 1.0, -1.0), 0.0)
        dist = np.where(hit, key[np.arange(T), j], dist)
    return {"intercept_team": team, "intercept_dist": np.clip(np.nan_to_num(dist), -1, 60)}


def _wmean(x, lo, hi):
    cs = np.concatenate(([0.0], np.cumsum(x)))
    i = np.arange(len(x)); a = np.clip(i + lo, 0, len(x)); b = np.clip(i + hi, 0, len(x))
    return (cs[b] - cs[a]) / np.maximum(b - a, 1)


def features_for(match, feat, fps=25):
    ball = feat.interpolate_ball(match)
    o = compute(ball, match.xy, match.slot_team, match.periods, fps)
    o.update(first_interceptor(ball, match.xy, match.slot_team, match.periods, fps))
    for name in ("intercept_team", "comove_diff", "recv_margin"):
        x = o[name]
        o[name + "_p1"] = _wmean(x, -fps, 0)
        o[name + "_f1"] = _wmean(x, 1, fps + 1)
    return o


def specs():
    S = []
    for n in ("intercept_team", "intercept_team_p1", "intercept_team_f1"):
        S.append(((n,), (lambda k: (lambda g: g(k)))(n)))
    S.append((("intercept_dist",), lambda g: np.log1p(np.maximum(g("intercept_dist"), 0)) / 4.0))
    for n in ("comove_diff", "comove_diff_p1", "comove_diff_f1"):
        S.append(((n,), (lambda k: (lambda g: np.tanh(g(k) / 3.0)))(n)))
    for n in ("recv_margin", "recv_margin_f1"):
        S.append(((n,), (lambda k: (lambda g: np.tanh(g(k) / 3.0)))(n)))
    S.append((("recv_active",), lambda g: g("recv_active")))
    return S
