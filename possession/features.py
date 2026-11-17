"""Per-frame features computed from a Match (positions, ball) -- never from labels.

Computed identically for every source (simulated external games and BAR-ATH); a feature that is
not computed the same way on both learns a domain shift instead of a signal.

Ball
  ball_present      the tracker sees the ball on this frame
  ball_known        seen, or inside a gap of <= INTERP_MAX_FRAMES filled by linear interpolation
  d_home, d_away    distance (m) from the (known) ball to the nearest tracked player of each team
  team_margin       d_away - d_home, clipped: positive = home player closer
  ball_out          metres outside the pitch rectangle, 0 inside
  ball_line_dist    metres to the nearest touchline / goal line (inside), 0 outside
  ball_centre_dist  metres to the centre mark
  ball_speed        ball speed (m/s) between consecutive known frames, smoothed
  gap_log           log1p(length in seconds of the current ball gap), 0 when known
Players (speed always from positions, central difference over +/- 0.2 s; the tracker's own
speed field is not used -- its declared unit is wrong, see phase-1 notes)
  speed_med_1s/3s/10s   median player speed, centred means over 1 / 3 / 10 s
  slow_frac             share of tracked players under SLOW_MS, centred 1 s mean
  speed_trend           mean median speed over the next 2 s minus the previous 2 s
Team context (phase 3; every "_diff" is home minus away, attack directions inferred per period
from which team owns the deepest player at each end -- no metadata, identical for all sources)
  push_diff        team centroids in their own attacking direction (the side with the ball pushes up)
  spread_lr        log ratio of team spread (std x * std y): the side with the ball spreads out
  width_diff, length_diff   std y / std x differences
  vel_diff         mean forward velocity of each team's players (slot-wise, robust to dropouts)
  goalside_diff    share of each team goal-side of the (known) ball: defenders stay goal-side
  near5_diff       players within 5 m of the (known) ball
  ball_att         ball position along home's attacking direction (known ball)
  ctrl_run         signed run length of the same team within 2 m of the ball (control persistence)
  margin_past/next mean team_margin over the previous / next second (known ball)
  last_margin, next_margin   team_margin at the last / next known ball frame (bridges ball gaps)
  gap_pos          position inside the current ball gap, 0..1
"""

import warnings

import numpy as np

CLIP_DIST = 30.0
CLIP_MARGIN = 15.0
SPEED_SPAN = 5                 # frames each side for the speed difference: +/- 0.2 s
KINK_SPAN = 6                  # frames on either side of ball/player impulse checks
INTERP_MAX_FRAMES = 15         # fill ball gaps up to 0.6 s
SLOW_MS = 1.0
MIN_PROXIMITY_FRAMES = 3
MIN_SEPARATION_M = 0.5
MAX_CARRY_S = 5.0
TOUCH_DECAY_TAU = 1.5
TOUCH_KINK_ANGLE_DEG = 45.0
# Phase 7 evaluation (docs/specs/phase7.md): touch-state inputs scored -0.0038 on the smoothed
# simulated games (better on 2 of 7 matches), so they are computed but kept out of the models.
TOUCH_FEATURES = False
TOUCH_DECEL_THRESHOLD = -1.0
TOUCH_TURN_ANGLE_DEG = 20.0


def _centred_nanmean(x, half):
    valid = ~np.isnan(x)
    v = np.where(valid, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(valid)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - half)
    hi = np.minimum(len(x), i + half + 1)
    n = cn[hi] - cn[lo]
    return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def _window_mean(x, lo_off, hi_off):
    """Mean of x over [t + lo_off, t + hi_off) for every t, ignoring NaN."""
    valid = ~np.isnan(x)
    v = np.where(valid, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(valid)))
    i = np.arange(len(x))
    lo = np.clip(i + lo_off, 0, len(x))
    hi = np.clip(i + hi_off, 0, len(x))
    n = cn[hi] - cn[lo]
    return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def _run_lengths_of_false(mask):
    out = np.zeros(len(mask), dtype=np.int64)
    m = (~mask).astype(np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[a:b] = b - a
    return out


def kinematic_speed(match):
    """(T, S) player speed in m/s from positions, inside periods only. Same code for all sources."""
    from .io.idsse import speed_from_xy
    return speed_from_xy(match.xy, match.periods, fps=match.fps, span=SPEED_SPAN).astype(np.float64)


def interpolate_ball(match, max_gap=INTERP_MAX_FRAMES):
    """Ball with interior gaps of <= max_gap frames filled linearly, never across periods."""
    ball = match.ball.copy()
    present = ~np.isnan(ball[:, 0])
    for _, start, stop in match.periods:
        p = present[start:stop]
        m = (~p).astype(np.int8)
        edges = np.diff(np.concatenate(([0], m, [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if a == 0 or b == len(p) or b - a > max_gap:
                continue
            p0, p1 = ball[start + a - 1], ball[start + b]
            u = (np.arange(1, b - a + 1) / float(b - a + 1))[:, None]
            ball[start + a:start + b] = p0 + u * (p1 - p0)
    return ball


def nearest_by_team(xy, slot_team, ball, chunk=20000):
    T = len(ball)
    d = np.full((T, 2), np.nan)
    idx = np.flatnonzero(~np.isnan(ball[:, 0]))
    for team in (0, 1):
        cols = np.flatnonzero(slot_team == team)
        for a in range(0, len(idx), chunk):
            f = idx[a:a + chunk]
            dx = xy[f][:, cols, 0] - ball[f, None, 0]
            dy = xy[f][:, cols, 1] - ball[f, None, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            m = np.min(np.where(np.isnan(dist), np.inf, dist), axis=1)
            m[np.isinf(m)] = np.nan
            d[f, team] = m
    return d


def _ball_angle_kink(ball, active=None, angle_deg=60.0):
    """Flag ball direction changes above a configurable angle across +/- KINK_SPAN.

    ``active`` optionally gates frames on a nearby player touch candidate. Boundaries, zero-length
    vectors, and missing positions return 0.
    """
    ball = np.asarray(ball, dtype=np.float64)
    T = len(ball)
    out = np.zeros(T, dtype=np.float64)
    if T <= 2 * KINK_SPAN:
        return out
    past, mid, future = ball[:-2 * KINK_SPAN], ball[KINK_SPAN:-KINK_SPAN], ball[2 * KINK_SPAN:]
    v_in, v_out = mid - past, future - mid
    n_in = np.linalg.norm(v_in, axis=1)
    n_out = np.linalg.norm(v_out, axis=1)
    denom = n_in * n_out
    cos_angle = np.divide(np.sum(v_in * v_out, axis=1), denom,
                          out=np.ones_like(denom), where=denom > 1e-9)
    track_valid = np.isfinite(ball).all(axis=1)
    missing = (~track_valid).astype(np.int32)
    missing_prefix = np.concatenate(([0], np.cumsum(missing)))
    span_missing = missing_prefix[2 * KINK_SPAN + 1:] - missing_prefix[:-2 * KINK_SPAN - 1]
    valid = (span_missing == 0) & (np.isfinite(past).all(axis=1) & np.isfinite(mid).all(axis=1)
             & np.isfinite(future).all(axis=1) & (n_in > 1e-9) & (n_out > 1e-9)
             & (cos_angle < np.cos(np.deg2rad(angle_deg))))
    if active is not None:
        valid &= np.asarray(active, dtype=bool)[KINK_SPAN:-KINK_SPAN]
    out[KINK_SPAN:-KINK_SPAN] = valid.astype(np.float64)
    return out


def _signed_touch(ball_kink, distances, slot_team, present):
    """Assign +/-1 to clear, sustained touches; suppress brief and contested candidates."""
    T = len(ball_kink)
    touch = np.zeros(T, dtype=np.float64)
    home_slots = np.flatnonzero(slot_team == 0)
    away_slots = np.flatnonzero(slot_team == 1)
    if not len(home_slots) or not len(away_slots):
        return touch
    kink_frames = np.flatnonzero(ball_kink > 0)
    for frame in kink_frames:
        if not present[frame] or frame < MIN_PROXIMITY_FRAMES - 1:
            continue
        a = frame - MIN_PROXIMITY_FRAMES + 1
        if not np.all(present[a:frame + 1]):
            continue
        home_history = distances[a:frame + 1, home_slots]
        away_history = distances[a:frame + 1, away_slots]
        home_idx = np.argmin(np.where(np.isfinite(home_history), home_history, np.inf), axis=1)
        away_idx = np.argmin(np.where(np.isfinite(away_history), away_history, np.inf), axis=1)
        home_near = home_history[np.arange(MIN_PROXIMITY_FRAMES), home_idx]
        away_near = away_history[np.arange(MIN_PROXIMITY_FRAMES), away_idx]
        home_consistent = np.all(home_near <= 2.0) and np.all(home_idx == home_idx[-1])
        away_consistent = np.all(away_near <= 2.0) and np.all(away_idx == away_idx[-1])
        d_home = distances[frame, home_slots[home_idx[-1]]]
        d_away = distances[frame, away_slots[away_idx[-1]]]
        if not np.isfinite(d_home) or not np.isfinite(d_away):
            continue
        separation = d_away - d_home
        if home_consistent and separation >= MIN_SEPARATION_M:
            touch[frame] = 1.0
        elif away_consistent and separation <= -MIN_SEPARATION_M:
            touch[frame] = -1.0
    return touch


def _last_touch_state(touch_team, periods, fps, ball_present):
    """Forward-fill touch owner with decay and reset at long ball loss and period boundaries."""
    touch_team = np.asarray(touch_team, dtype=np.float64)
    T = len(touch_team)
    last = np.zeros(T, dtype=np.float64)
    since = np.zeros(T, dtype=np.float64)
    signal = np.zeros(T, dtype=np.float64)
    reset_after = int(round(2.0 * fps))
    for _, start, stop in periods:
        owner, last_frame, missing_run = 0.0, -1, 0
        for frame in range(start, stop):
            missing_run = 0 if ball_present[frame] else missing_run + 1
            if missing_run > reset_after:
                owner, last_frame = 0.0, -1
            if touch_team[frame] != 0:
                owner, last_frame = touch_team[frame], frame
            if last_frame >= 0:
                age = (frame - last_frame) / float(fps)
                if age > MAX_CARRY_S:
                    owner, last_frame = 0.0, -1
                else:
                    last[frame] = owner
                    since[frame] = age
                    signal[frame] = owner * np.exp(-age / TOUCH_DECAY_TAU)
    return last, since, signal


def _player_impulses(xy, fps):
    """Return per-player acceleration (m/s^2) and turn angle over +/- KINK_SPAN frames.

    Missing tracklets and boundary frames are padded with zero, so the outputs are NaN-free.
    """
    xy = np.asarray(xy, dtype=np.float32)
    T, S = xy.shape[:2]
    accel = np.zeros((T, S), dtype=np.float32)
    turn = np.zeros((T, S), dtype=np.float32)
    k = KINK_SPAN
    if T <= 2 * k:
        return accel, turn
    past, mid, future = xy[:-2 * k], xy[k:-k], xy[2 * k:]
    track_valid = np.isfinite(xy).all(axis=2)
    # The rolling span is only 13 frames, so uint8 wraparound preserves each local count.
    missing_prefix = np.concatenate([
        np.zeros((1, S), dtype=np.uint8),
        np.cumsum(~track_valid, axis=0, dtype=np.uint8)], axis=0)
    span_missing = missing_prefix[2 * k + 1:] - missing_prefix[:-2 * k - 1]
    valid = (span_missing == 0) & (np.isfinite(past).all(axis=2) & np.isfinite(mid).all(axis=2)
             & np.isfinite(future).all(axis=2))
    v_in = (mid - past) * (float(fps) / k)
    v_out = (future - mid) * (float(fps) / k)
    accel_mid = (np.linalg.norm(v_out, axis=2) - np.linalg.norm(v_in, axis=2)) * (float(fps) / k)
    cross = v_in[:, :, 0] * v_out[:, :, 1] - v_in[:, :, 1] * v_out[:, :, 0]
    dot = np.sum(v_in * v_out, axis=2)
    turn_mid = np.arctan2(np.abs(cross), dot)
    accel[k:-k] = np.where(valid, accel_mid, 0.0)
    turn[k:-k] = np.where(valid, turn_mid, 0.0)
    return accel, turn


def boundary_margin(boundary_xy, n_frames):
    """Continuous normalised distance outside the pitch; unavailable projections are neutral."""
    margin = np.zeros(n_frames, dtype=np.float64)
    if boundary_xy is None:
        return margin
    xy = np.asarray(boundary_xy, dtype=np.float64)
    if xy.shape != (n_frames, 2):
        raise ValueError("boundary_xy must have shape (n_frames, 2)")
    valid = np.isfinite(xy).all(axis=1)
    x, y = xy[valid].T
    margin[valid] = np.maximum.reduce([np.zeros(len(x)), -x, x - 1.0, -y, y - 1.0])
    return margin


def _closing_velocity(ball, xy, fps):
    """Signed radial player-to-ball closing speed (m/s), positive when distance is shrinking."""
    ball = np.asarray(ball, dtype=np.float32)
    xy = np.asarray(xy, dtype=np.float32)
    T, S = xy.shape[:2]
    closing = np.zeros((T, S), dtype=np.float32)
    if T < 3:
        return closing
    d_before = np.hypot(xy[:-2, :, 0] - ball[:-2, None, 0],
                        xy[:-2, :, 1] - ball[:-2, None, 1])
    d_after = np.hypot(xy[2:, :, 0] - ball[2:, None, 0],
                       xy[2:, :, 1] - ball[2:, None, 1])
    valid = (np.isfinite(d_before) & np.isfinite(d_after)
             & np.isfinite(xy[1:-1]).all(axis=2)
             & np.isfinite(ball[1:-1]).all(axis=1)[:, None])
    closing[1:-1] = np.where(valid, (d_before - d_after) * (float(fps) / 2.0), 0.0)
    return closing


def attack_directions(match, step=25):
    """(T,) +1 where home attacks towards +x, -1 where towards -x, 0 outside periods.

    Per period: the team that owns the deepest player near x = 0 more often defends that end.
    """
    d = np.zeros(match.n_frames)
    x = match.xy[:, :, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for _, a, b in match.periods:
            f = np.arange(a, b, step)
            mins = [np.nanmin(np.where(match.slot_team == t, x[f], np.nan), axis=1) for t in (0, 1)]
            home_left = np.nanmean(mins[0] < mins[1])
            d[a:b] = 1.0 if home_left >= 0.5 else -1.0
    return d


def _ffill(values, valid, periods):
    """Forward fill values over invalid frames inside each period (NaN before the first valid)."""
    out = np.full(len(values), np.nan)
    for _, a, b in periods:
        v = valid[a:b]
        idx = np.where(v, np.arange(b - a), -1)
        idx = np.maximum.accumulate(idx)
        ok = idx >= 0
        seg = np.full(b - a, np.nan)
        seg[ok] = values[a:b][idx[ok]]
        out[a:b] = seg
    return out


def _bfill(values, valid, periods):
    rev = [(p, len(values) - b, len(values) - a) for p, a, b in periods]
    return _ffill(values[::-1], valid[::-1], rev)[::-1]


def _signed_runs(code, cap):
    """For a sequence of -1/0/+1, the signed length of the current run of the same nonzero value."""
    out = np.zeros(len(code))
    run = 0
    prev = 0
    for i, c in enumerate(code):
        if c != 0 and c == prev:
            run = min(run + 1, cap)
        elif c != 0:
            run = 1
        else:
            run = 0
        prev = c
        out[i] = c * run
    return out / float(cap)


def team_context(match, ball, known, margin, d_home, d_away):
    T, fps = match.n_frames, match.fps
    L, W = match.pitch
    att = attack_directions(match)
    x, y = match.xy[:, :, 0], match.xy[:, :, 1]
    cols = [np.flatnonzero(match.slot_team == t) for t in (0, 1)]
    sign = [att, -att]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cx = [np.nanmean(x[:, c], axis=1) for c in cols]
        sx = [np.nanstd(x[:, c], axis=1) for c in cols]
        sy = [np.nanstd(y[:, c], axis=1) for c in cols]
        push = [np.nan_to_num((cx[t] - L / 2) * sign[t]) for t in (0, 1)]
        out["push_diff"] = np.clip(push[0] - push[1], -40, 40)
        area = [np.nan_to_num(sx[t] * sy[t]) for t in (0, 1)]
        out["spread_lr"] = np.clip(np.log((area[0] + 1) / (area[1] + 1)), -3, 3)
        out["width_diff"] = np.clip(np.nan_to_num(sy[0] - sy[1]), -20, 20)
        out["length_diff"] = np.clip(np.nan_to_num(sx[0] - sx[1]), -20, 20)
        # slot-wise forward velocity, so players entering / leaving tracking do not move the mean
        h = fps // 4
        dx = np.full(x.shape, np.nan)
        dx[h:-h] = (x[2 * h:] - x[:-2 * h]) * fps / (2.0 * h)
        v = [np.nanmean(dx[:, c], axis=1) * sign[t] for t, c in enumerate(cols)]
        vd = np.nan_to_num(v[0] - v[1])
        out["vel_diff"] = np.clip(np.nan_to_num(_centred_nanmean(vd, fps // 2)), -8, 8)
        bx, by = ball[:, 0], ball[:, 1]
        gs, near5 = [], []
        for t, c in enumerate(cols):
            px, py = x[:, c], y[:, c]
            seen = ~np.isnan(px)
            behind = ((px - bx[:, None]) * sign[t][:, None] < 0) & seen
            gs.append(behind.sum(1) / np.maximum(seen.sum(1), 1))
            dist = np.hypot(px - bx[:, None], py - by[:, None])
            near5.append(np.nansum(dist < 5.0, axis=1))
    out["goalside_diff"] = np.where(known, gs[0] - gs[1], 0.0)
    out["near5_diff"] = np.where(known, np.clip(near5[0] - near5[1], -5, 5), 0.0)
    out["ball_att"] = np.where(known, np.clip(np.nan_to_num((bx - L / 2) * att), -60, 60), 0.0)
    close = known & (np.minimum(d_home, d_away) < 2.0)
    code = np.where(close, np.sign(margin), 0).astype(int)
    out["ctrl_run"] = _signed_runs(code, 2 * fps)
    m = np.where(known, margin, np.nan)
    out["margin_past"] = np.nan_to_num(_window_mean(m, -fps, 0))
    out["margin_next"] = np.nan_to_num(_window_mean(m, 1, fps + 1))
    out["last_margin"] = np.nan_to_num(_ffill(margin, known, match.periods))
    out["next_margin"] = np.nan_to_num(_bfill(margin, known, match.periods))
    idx = np.arange(T, dtype=float)
    last_i = _ffill(idx, known, match.periods)
    next_i = _bfill(idx, known, match.periods)
    span = next_i - last_i
    pos = np.where(known, 0.0, (idx - last_i) / np.where(span > 0, span, 1.0))
    out["gap_pos"] = np.clip(np.nan_to_num(pos, nan=0.5), 0, 1)
    return out


CONTEXT_SECONDS = (2, 5)
CONTEXT_KNOWN = ("team_margin", "goalside_diff", "near5_diff", "ctrl_run")   # ball-dependent
CONTEXT_FREE = ("push_diff", "vel_diff", "spread_lr")                        # always defined
CONTEXT_DEAD = ("speed_med_1s", "ball_known")


def context_windows(f, fps):
    """Past / future means over 2 s and 5 s of the main team and dead-ball signals (phase 5).

    Ball-dependent signals are averaged over known-ball frames only. Names: ctx_<feature>_<p|f><s>.
    """
    out = {}
    known = f["ball_known"] > 0
    for name in CONTEXT_KNOWN + CONTEXT_FREE + CONTEXT_DEAD:
        x = np.where(known, f[name], np.nan) if name in CONTEXT_KNOWN else f[name].astype(np.float64)
        for sec in CONTEXT_SECONDS:
            w = sec * fps
            out["ctx_{}_p{}".format(name, sec)] = np.nan_to_num(_window_mean(x, -w, 0))
            out["ctx_{}_f{}".format(name, sec)] = np.nan_to_num(_window_mean(x, 1, w + 1))
    return out


# =========================================================================
# STEP 3: CONTEXT WINDOW CONFIGURATION
# =========================================================================
# MODE 1 (active by default): the established 2s and 5s windows only.
# CONTEXT_SECONDS remains (2, 5).
#
# MODE 2 (experimental): uncomment this helper, the call in compute(), and the
# feature spec below to evaluate 10-frame (0.4s) ball acceleration / heading windows.
# Keep the block commented for baseline runs so no micro-window columns reach the models.
#
# MICRO_CONTEXT_FRAMES = 10
#
# def micro_kinematic_windows(accel, heading, known, periods):
#     """Past/future means over a 10-frame window, local to each period."""
#     out = {}
#     for name, values in (("ball_accel", accel), ("ball_heading_change", heading)):
#         x = np.where(known, values, np.nan)
#         past = np.full(len(x), np.nan)
#         future = np.full(len(x), np.nan)
#         for _, start, stop in periods:
#             past[start:stop] = _window_mean(x[start:stop], -MICRO_CONTEXT_FRAMES, 0)
#             future[start:stop] = _window_mean(x[start:stop], 1, MICRO_CONTEXT_FRAMES + 1)
#         out["ctx_{}_p10".format(name)] = np.nan_to_num(past)
#         out["ctx_{}_f10".format(name)] = np.nan_to_num(future)
#     return out


def compute(match, boundary_xy=None):
    """Dict of float arrays, one value per frame."""
    T, fps = match.n_frames, match.fps
    L, W = match.pitch
    present = match.ball_present()
    ball = interpolate_ball(match)
    known = ~np.isnan(ball[:, 0])

    d = nearest_by_team(match.xy, match.slot_team, ball)
    d_home = np.clip(np.nan_to_num(d[:, 0], nan=CLIP_DIST), 0, CLIP_DIST)
    d_away = np.clip(np.nan_to_num(d[:, 1], nan=CLIP_DIST), 0, CLIP_DIST)
    margin = np.where(known, np.clip(d_away - d_home, -CLIP_MARGIN, CLIP_MARGIN), 0.0)

    bx, by = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):
        out = np.maximum.reduce([-bx, bx - L, -by, by - W, np.zeros(T)])
        line = np.minimum.reduce([bx, L - bx, by, W - by])
    out = np.where(known, np.clip(np.nan_to_num(out), 0, 10.0), 0.0)
    line = np.where(known, np.clip(np.nan_to_num(line), 0, 20.0), 20.0)
    centre = np.where(known, np.clip(np.nan_to_num(np.hypot(bx - L / 2, by - W / 2)), 0, 40.0), 40.0)

    # Detector positions projected through the frame's ground-plane homography are already
    # normalised to the pitch. Keep parallax as a continuous value; missing projection is neutral.
    boundary_feature = boundary_margin(boundary_xy, T)

    kin = kinematic_speed(match)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.clip(np.nanmedian(kin, axis=1), 0, 12)
        tracked = (~np.isnan(kin)).sum(axis=1)
        slow = np.where(tracked > 0, (kin < SLOW_MS).sum(axis=1) / np.maximum(tracked, 1), np.nan)
    s1 = np.nan_to_num(_centred_nanmean(med, fps // 2), nan=0.0)
    s3 = np.nan_to_num(_centred_nanmean(med, (3 * fps) // 2), nan=0.0)
    s10 = np.nan_to_num(_centred_nanmean(med, 5 * fps), nan=0.0)
    slow1 = np.nan_to_num(_centred_nanmean(slow, fps // 2), nan=0.5)
    trend = np.nan_to_num(_window_mean(med, 0, 2 * fps) - _window_mean(med, -2 * fps, 0), nan=0.0)

    step = np.full(T, np.nan)
    both = known[1:] & known[:-1]
    for _, start, _ in match.periods:
        if start > 0:
            both[start - 1] = False
    disp = np.sqrt(((ball[1:] - ball[:-1]) ** 2).sum(axis=1)) * fps
    step[1:][both] = disp[both]
    bspeed = np.clip(np.nan_to_num(_centred_nanmean(step, 2), nan=0.0), 0, 40)

    # Derivatives use only adjacent known positions. They are kept as separate signals so the
    # short context window below applies only to kinematics, never to the position features.
    velocity = np.full((T, 2), np.nan)
    velocity[1:][both] = (ball[1:][both] - ball[:-1][both]) * fps
    vvalid = np.isfinite(velocity).all(axis=1)
    accel = np.zeros(T, dtype=np.float64)
    heading = np.zeros(T, dtype=np.float64)
    derivative_valid = vvalid[1:] & vvalid[:-1]
    if T > 1:
        accel[1:][derivative_valid] = (
            np.linalg.norm(velocity[1:][derivative_valid], axis=1)
            - np.linalg.norm(velocity[:-1][derivative_valid], axis=1)) * fps
        v0, v1 = velocity[:-1][derivative_valid], velocity[1:][derivative_valid]
        cross = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        dot = np.sum(v0 * v1, axis=1)
        heading[1:][derivative_valid] = np.arctan2(np.abs(cross), dot)
    accel = np.clip(accel, -20.0, 20.0)
    # Experimental Step 3 toggle; activate only for the separate micro-window evaluation:
    # micro = micro_kinematic_windows(accel, heading, known, match.periods)

    # The prior closest team is a label-free proxy for the team controlling the incoming ball;
    # a nearby player from the other team can then corroborate an intentional touch.
    touch_xy = np.asarray(match.xy, dtype=np.float32)
    touch_ball = np.asarray(match.ball, dtype=np.float32)
    decel, player_turn = _player_impulses(touch_xy, fps)
    closing = _closing_velocity(touch_ball, touch_xy, fps)
    player_dist = np.hypot(touch_xy[:, :, 0] - touch_ball[:, None, 0],
                           touch_xy[:, :, 1] - touch_ball[:, None, 1])
    nearest_team = np.argmin(np.where(np.isfinite(d), d, np.inf), axis=1)
    owner_valid = np.isfinite(d).any(axis=1)
    prior_owner = np.full(T, -1, dtype=np.int8)
    prior_valid = np.zeros(T, dtype=bool)
    if T > KINK_SPAN:
        prior_owner[KINK_SPAN:] = nearest_team[:-KINK_SPAN]
        prior_valid[KINK_SPAN:] = owner_valid[:-KINK_SPAN]
    # Either side may touch; combine kinematics as a permissive corroboration, not a hard gate.
    candidate_touch = (np.isfinite(player_dist) & (player_dist <= 2.0)
                       & ((decel < TOUCH_DECEL_THRESHOLD)
                          | (player_turn > np.deg2rad(TOUCH_TURN_ANGLE_DEG))
                          | (closing > 0.0)))
    touch_near = candidate_touch.any(axis=1) & present
    active_kink = np.zeros(T, dtype=bool)
    for _, start, stop in match.periods:
        lo, hi = start + KINK_SPAN, stop - KINK_SPAN
        if lo < hi:
            active_kink[lo:hi] = touch_near[lo:hi]
    ball_kink = _ball_angle_kink(match.ball, active=active_kink, angle_deg=TOUCH_KINK_ANGLE_DEG)
    touch_team = _signed_touch(ball_kink, player_dist, match.slot_team, present)
    last_touch_team, time_since_touch, touch_signal = _last_touch_state(
        touch_team, match.periods, fps, present)
    del touch_xy, touch_ball, decel, player_turn, closing, player_dist, candidate_touch

    gap = _run_lengths_of_false(known) / float(fps)
    ctx = team_context(match, ball, known, margin, d_home, d_away)
    base = {**ctx, 
        "ball_present": present.astype(np.float64), "ball_known": known.astype(np.float64),
        "d_home": d_home, "d_away": d_away, "team_margin": margin,
        "ball_out": out, "ball_line_dist": line, "ball_centre_dist": centre,
        "boundary_margin": boundary_feature,
        "ball_speed": bspeed, "gap_log": np.log1p(gap),
        "ball_accel": accel, "ball_heading_change": heading,
        "ball_kink_6f": ball_kink,
        "touch_team": touch_team, "last_touch_team": last_touch_team,
        "time_since_touch": time_since_touch, "touch_signal": touch_signal,
        "speed_med_1s": s1, "speed_med_3s": s3, "speed_med_10s": s10,
        "slow_frac": slow1, "speed_trend": trend,
    }
    base.update(context_windows(base, fps))
    # Experimental Step 3 toggle (paired with the helper and spec below):
    # base.update(micro)
    return base


# -- design matrices: every column names the features it is built from ---------------------

def _cols(f, rows, spec):
    g = (lambda k: f[k]) if rows is None else (lambda k: f[k][rows])
    X = np.column_stack([fn(g) for _, fn in spec])
    return X, [srcs for srcs, _ in spec]


CONTEXT_SPEC = [
    (("push_diff",), lambda g: g("push_diff") / 20.0),
    (("spread_lr",), lambda g: g("spread_lr")),
    (("width_diff",), lambda g: g("width_diff") / 10.0),
    (("length_diff",), lambda g: g("length_diff") / 10.0),
    (("vel_diff",), lambda g: np.tanh(g("vel_diff") / 2.0)),
]

TEAM_SPEC = [
    (("team_margin",), lambda g: np.tanh(g("team_margin") / 1.0)),
    (("team_margin",), lambda g: np.tanh(g("team_margin") / 3.0)),
    (("team_margin",), lambda g: np.tanh(g("team_margin") / 8.0)),
    (("d_home", "d_away"), lambda g: (g("d_away") - g("d_home")) / CLIP_DIST),
    (("goalside_diff",), lambda g: g("goalside_diff")),
    (("near5_diff",), lambda g: g("near5_diff") / 5.0),
    (("ball_att",), lambda g: g("ball_att") / 50.0),
    (("ctrl_run",), lambda g: g("ctrl_run")),
    (("margin_past",), lambda g: np.tanh(g("margin_past") / 3.0)),
    (("margin_next",), lambda g: np.tanh(g("margin_next") / 3.0)),
] + CONTEXT_SPEC

# Ball unknown: no ball geometry, but the shape of both teams and the last / next sighting.
GAP_SPEC = [
    (("last_margin",), lambda g: np.tanh(g("last_margin") / 3.0)),
    (("next_margin",), lambda g: np.tanh(g("next_margin") / 3.0)),
    (("last_margin", "gap_pos"), lambda g: np.tanh(g("last_margin") / 3.0) * (1 - g("gap_pos"))),
    (("next_margin", "gap_pos"), lambda g: np.tanh(g("next_margin") / 3.0) * g("gap_pos")),
] + CONTEXT_SPEC

DEAD_SPEC = [
    (("ball_out",), lambda g: np.minimum(g("ball_out"), 3.0)),
    (("ball_out",), lambda g: (g("ball_out") > 0.5).astype(np.float64)),
    (("ball_line_dist",), lambda g: np.log1p(g("ball_line_dist"))),
    (("ball_centre_dist",), lambda g: (g("ball_centre_dist") < 3.0).astype(np.float64)),
    (("speed_med_1s",), lambda g: g("speed_med_1s")),
    (("speed_med_3s",), lambda g: g("speed_med_3s")),
    (("speed_med_3s",), lambda g: g("speed_med_3s") ** 2),
    (("speed_med_3s",), lambda g: np.log1p(g("speed_med_3s"))),
    (("speed_med_10s",), lambda g: g("speed_med_10s")),
    (("slow_frac",), lambda g: g("slow_frac")),
    (("speed_trend",), lambda g: g("speed_trend")),
    (("ball_present",), lambda g: g("ball_present")),
    (("ball_known",), lambda g: g("ball_known")),
    (("gap_log",), lambda g: g("gap_log")),
    (("ball_speed", "ball_known"), lambda g: g("ball_known") * np.log1p(g("ball_speed"))),
    (("d_home", "d_away"), lambda g: np.minimum(g("d_home"), g("d_away")) / CLIP_DIST),
    (("boundary_margin",), lambda g: g("boundary_margin")),
]


def _ctx_spec(names):
    return [((n,), (lambda key: (lambda g: g(key)))("ctx_{}_{}{}".format(n, d, s)))
            for n in names for s in CONTEXT_SECONDS for d in ("p", "f")]


TEAM_SPEC = TEAM_SPEC + _ctx_spec(CONTEXT_KNOWN + CONTEXT_FREE)
GAP_SPEC = GAP_SPEC + _ctx_spec(CONTEXT_KNOWN + CONTEXT_FREE)
DEAD_SPEC = DEAD_SPEC + _ctx_spec(CONTEXT_DEAD)

# Step 6 touch-persistence inputs: append only to team and ball-gap classifiers.
TOUCH_SPEC = [
    (("touch_signal",), lambda g: g("touch_signal")),
    (("last_touch_team",), lambda g: g("last_touch_team")),
    (("time_since_touch",), lambda g: np.minimum(g("time_since_touch"), MAX_CARRY_S) / MAX_CARRY_S),
]
if TOUCH_FEATURES:
    TEAM_SPEC = TEAM_SPEC + TOUCH_SPEC
    GAP_SPEC = GAP_SPEC + TOUCH_SPEC


# Experimental Step 3 toggle (uncomment with the helper and compute() call above):
# MICRO_KINEMATIC_SPEC = [
#     (("ball_accel",), lambda g: np.tanh(g("ctx_ball_accel_p10") / 10.0)),
#     (("ball_accel",), lambda g: np.tanh(g("ctx_ball_accel_f10") / 10.0)),
#     (("ball_heading_change",), lambda g: g("ctx_ball_heading_change_p10") / np.pi),
#     (("ball_heading_change",), lambda g: g("ctx_ball_heading_change_f10") / np.pi),
# ]
# TEAM_SPEC = TEAM_SPEC + MICRO_KINEMATIC_SPEC
# GAP_SPEC = GAP_SPEC + MICRO_KINEMATIC_SPEC


def _delta(name, sec):
    return lambda g: g("ctx_{}_f{}".format(name, sec)) - g("ctx_{}_p{}".format(name, sec))


# Change detector (phase 6): before / after contrasts of every context signal, plus the levels.
CHANGE_SPEC = [((n,), _delta(n, s)) for n in CONTEXT_KNOWN + CONTEXT_FREE for s in CONTEXT_SECONDS] + [
    (("ball_known",), lambda g: g("ball_known")),
    (("gap_pos",), lambda g: g("gap_pos")),
    (("last_margin", "next_margin"), lambda g: np.tanh(g("next_margin") / 3.0) - np.tanh(g("last_margin") / 3.0)),
    (("ctrl_run",), lambda g: g("ctrl_run")),
    (("team_margin",), lambda g: np.tanh(g("team_margin") / 3.0)),
    (("speed_med_1s",), lambda g: g("speed_med_1s")),
]


def change_design(f, rows=None, exclude=()):
    return design(CHANGE_SPEC, f, rows, exclude)


def design(spec, f, rows=None, exclude=()):
    X, sources = _cols(f, rows, spec)
    keep = [i for i, s in enumerate(sources) if not set(s) & set(exclude)]
    return X[:, keep]


def team_design(f, rows=None, exclude=()):
    return design(TEAM_SPEC, f, rows, exclude)


def gap_design(f, rows=None, exclude=()):
    return design(GAP_SPEC, f, rows, exclude)


def dead_design(f, rows=None, exclude=()):
    return design(DEAD_SPEC, f, rows, exclude)
