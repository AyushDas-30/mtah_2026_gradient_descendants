"""Command line: python -m possession <command> [options]

Phase 0 commands, in the order to run them:

  build-cache      parse the tactical ATD + Smart Tagging into cache/ and self-check the parse
  check-baseline   rebuild the four starter baselines from the cache; compare with the committed CSVs
  check-scorer     prove our metrics equal the official scorer (needs the official repo clone)
  idsse-download   fetch the IDSSE XML files from figshare
  idsse-build      parse IDSSE into clean labelled matches in cache/idsse/
  noise-profile    measure BAR-ATH tracking degradation -> cache/noise_profile.json
  simulate         degrade every IDSSE match with that profile -> cache/sim/
  evaluate         score a method on the simulated games (and, for sanity, the public window)
  phase0           run every step after idsse-download, in order, and print one verdict table
  validate FILE    check a submission CSV the way the official validator does

Every check prints PASS / FAIL / INFO lines. To report progress you only need the verdict lines --
keep the data, and anything printed from it, on your machine.
"""

import argparse
import json
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

from . import config, harness, metrics
from .io import atd, idsse, labels as label_io, tagging
from .methods import baseline as baseline_mod
from .noise import profile as noise_profile
from .noise import simulate as noise_sim
from .schema import Match, X

VERDICTS = []



# Vision ball detections as a pitch-boundary input at prediction (teammate step 5). Off: the models are
# trained on simulated games without detections, and GPU output would make the file machine-dependent.
VISION_BOUNDARY = False

def verdict(ok, what, detail=""):
    """ok: truthy -> PASS, falsy -> FAIL, None -> INFO (numpy bools included)."""
    if ok is not None:
        ok = bool(ok)
    tag = "PASS" if ok is True else ("FAIL" if ok is False else "INFO")
    VERDICTS.append((tag, what))
    print("  [{}] {}{}".format(tag, what, ("  -- " + detail) if detail else ""))
    return ok


def _cache_path(args):
    return config.cache_dir(args.cache) / (config.GAME + ".npz")


def load_bar_ath(args):
    path = _cache_path(args)
    if not path.is_file():
        sys.exit("no cache at {} -- run build-cache first".format(path))
    return Match.load(path)


# -- build-cache ---------------------------------------------------------------------------

def cmd_build_cache(args):
    t0 = time.time()
    print("parsing tactical ATD ...")
    match = atd.load_tactical(args.data)
    stats = match.meta["parse_stats"]
    print("  parsed {} lines in {:.0f}s".format(stats["lines"], time.time() - t0))
    match.save(_cache_path(args))

    verdict(stats["records"] == config.N_VIDEO_FRAMES and stats["contiguous_from_1"],
            "one record per video frame, ids 1..{}".format(config.N_VIDEO_FRAMES),
            "records {}, ids {}..{}".format(stats["records"], stats["first_id"], stats["last_id"]))
    verdict(stats["malformed"] == 0 and stats["duplicates"] == 0 and stats["out_of_range"] == 0,
            "no malformed / duplicate / out-of-range lines",
            "{} / {} / {}".format(stats["malformed"], stats["duplicates"], stats["out_of_range"]))
    verdict(match.n_slots == config.N_TRACK_SLOTS, "46 track slots", str(match.n_slots))
    verdict("barcelona" in (match.meta["home_name"] or "").lower(), "home side is Barcelona",
            match.meta["home_name"])
    verdict(None, "layout from " + match.meta["layout_source"],
            "player fields {}, ball fields {}".format(match.meta["player_fields"], match.meta["ball_fields"]))
    verdict(None if match.meta["agrees_with_player_order"] else False,
            "slot teams agree with the starter's <Player>-order assumption")
    for note in match.meta["notes"]:
        verdict(None, note)

    graded = match.graded_mask()
    present = match.ball_present()
    p_all, p_graded = present.mean(), present[graded].mean()
    verdict(abs(p_all - config.PUBLISHED["ball_present_all"]) < 0.002,
            "ball present on 56.9% of all frames", "{:.4f}".format(p_all))
    verdict(abs(p_graded - config.PUBLISHED["ball_present_graded"]) < 0.002,
            "ball present on 59.3% of graded frames", "{:.4f}".format(p_graded))
    objects = (~np.isnan(match.xy[graded, :, 0])).sum(axis=1).mean()
    verdict(abs(objects - config.PUBLISHED["mean_objects_graded"]) < 0.1,
            "about 20.5 tracked objects per graded frame", "{:.2f}".format(objects))
    verdict(None, "speed converted to m/s", "file unit: {}".format(match.meta["speed_unit_in_file"]))

    tag_path = config.tactical_paths(args.data)["smart_tagging"]
    events = tagging.load_tagging(tag_path)
    tagging.save_tagging(events, config.cache_dir(args.cache) / (config.GAME + "_tagging.json"))
    verdict(len(events) == config.PUBLISHED["n_smart_tagging"], "638 Smart Tagging instances",
            str(len(events)))
    kickoffs = [e for e in events if e["code"] == "SET PIECES"
                and e["tags"].get("Type", "").lower() == "kick off (start)"]
    contains = [any(e["start_frame"] <= f <= e["end_frame"] for e in kickoffs) for f in (11185, 113742)]
    verdict(all(contains), "tagged kickoffs contain both published kickoff frames", str(contains))


# -- check-baseline ------------------------------------------------------------------------

def cmd_check_baseline(args):
    match = load_bar_ath(args)
    repo = config.repo_dir(args.repo)
    results = repo / "evaluation" / "results"
    gt_path = repo / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
    gt = label_io.read_labels(gt_path)
    near = baseline_mod.nearest_team(match)
    for name, window in config.BASELINE_WINDOWS.items():
        ours = baseline_mod.majority(near, window)
        committed = label_io.read_labels(results / (name + ".csv"))
        agree = float((ours == committed).mean())
        verdict(agree == 1.0, "{} identical to the committed file".format(name),
                "{:.6f} of frames agree".format(agree))
        r = metrics.evaluate(gt, ours)
        exp_macro, exp_spell = config.PUBLISHED_BASELINE_PUBLIC[name]
        verdict(round(r["macro_f1"], 4) == exp_macro and round(r["spell_f1"], 4) == exp_spell,
                "{} public-window scores reproduce the README".format(name),
                "macroF1 {:.4f} spellF1 {:.4f}".format(r["macro_f1"], r["spell_f1"]))


# -- check-scorer --------------------------------------------------------------------------

def _official_metrics(repo):
    path = Path(repo) / "evaluation" / "possession_metrics.py"
    spec = importlib.util.spec_from_file_location("official_possession_metrics", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cmd_check_scorer(args):
    repo = config.repo_dir(args.repo)
    pm = _official_metrics(repo)
    gt = label_io.read_labels(repo / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv")
    rng = np.random.default_rng(0)
    cases = []
    for name in config.BASELINE_WINDOWS:
        cases.append((name, label_io.read_labels(repo / "evaluation" / "results" / (name + ".csv"))))
    graded = np.flatnonzero(gt != X)
    for i in range(args.n_random):
        pred = gt.copy()
        for _ in range(rng.integers(1, 12)):                 # random relabelled stretches
            a = int(rng.choice(graded))
            pred[a:a + int(rng.integers(1, 300))] = rng.integers(0, 4)
        cases.append(("random{}".format(i), pred))
    worst = 0.0
    for name, pred in cases:
        official = pm.evaluate(gt, pred)
        ours = metrics.evaluate(gt, pred)
        worst = max(worst, abs(official["spell_f1_mean"] - ours["spell_f1"]),
                    abs(official["macro_f1"] - ours["macro_f1"]))
    verdict(worst < 1e-12, "our metrics equal the official scorer on {} cases".format(len(cases)),
            "max abs difference {:.2e}".format(worst))


# -- IDSSE ---------------------------------------------------------------------------------

def _idsse_ids(args):
    return list(idsse.MATCHES) if args.matches == ["all"] else args.matches


def cmd_idsse_download(args):
    folder = config.idsse_dir(args.idsse, args.cache)
    for match_id in _idsse_ids(args):
        print(match_id)
        idsse.download(match_id, folder)
    verdict(True, "IDSSE files in {}".format(folder))


def cmd_idsse_build(args):
    folder = config.idsse_dir(args.idsse, args.cache)
    out = config.cache_dir(args.cache) / "idsse"
    built = 0
    for match_id in _idsse_ids(args):
        names = idsse.file_names(match_id)
        if not all((folder / n).is_file() for n in names.values()):
            verdict(None, "{} not downloaded, skipped".format(match_id))
            continue
        t0 = time.time()
        match = idsse.load_match(match_id, folder)
        match.save(out / (match_id + ".npz"))
        runs = metrics.valid_runs(match.labels)
        sp = metrics.spells(match.labels, runs)
        dead = float((match.labels == 2).sum()) / max(1, int((match.labels != X).sum()))
        verdict(len(sp) > 100, "{} parsed in {:.0f}s".format(match_id, time.time() - t0),
                "{} possessions, mean {:.1f}s, dead-ball share {:.2f}, players {}/{}, "
                "on pitch {:.1f}/{:.1f}, {} non-players dropped".format(
                    len(sp), np.mean([b - a for a, b, _ in sp]) / 25 if sp else 0, dead,
                    match.meta["n_home_persons"], match.meta["n_away_persons"],
                    *_on_pitch(match), match.meta["n_skipped_non_players"]))
        built += 1
    verdict(built > 0, "{} IDSSE matches built".format(built))


# -- noise ---------------------------------------------------------------------------------

def cmd_noise_profile(args):
    match = load_bar_ath(args)
    prof = noise_profile.measure(match)
    path = noise_profile.save(prof, config.cache_dir(args.cache) / "noise_profile.json")
    for k, v in noise_profile.summary(prof).items():
        verdict(None, "profile {}".format(k), str(v))
    verdict(path.is_file(), "noise profile written")


def cmd_state_profile(args):
    """Phase 9: measure BAR-ATH's game-state-aware ball visibility from a stage-1 submission."""
    from .noise import state as noise_state
    match = load_bar_ath(args)
    codes = label_io.read_labels(args.labels)
    prof = noise_state.measure(match, codes)
    path = noise_state.save(prof, config.cache_dir(args.cache) / "state_profile.json")
    verdict(None, "live-play ball coverage", "{:.3f}".format(prof["live_coverage"]))
    verdict(None, "stoppage ball coverage", "{:.3f}".format(prof["dead_coverage"]))
    verdict(prof["n_stoppages"] >= 30, "stoppage visibility patterns", str(prof["n_stoppages"]))
    verdict(prof["n_alive_gaps"] >= 10, "live-play gaps", str(prof["n_alive_gaps"]))
    verdict(path.is_file(), "state profile written")


def cmd_simulate(args):
    cache = config.cache_dir(args.cache)
    prof = noise_profile.load(cache / "noise_profile.json")
    state = None
    if args.world == "state":
        from .noise import state as noise_state
        sp = cache / "state_profile.json"
        if not sp.is_file():
            sys.exit("no state profile -- run state-profile first")
        state = noise_state.load(sp)
    target = noise_profile.summary(prof)
    clean = sorted((cache / "idsse").glob("*.npz"))
    if not clean:
        sys.exit("no clean IDSSE matches -- run idsse-build first")
    for path in clean:
        match = Match.load(path)
        # The simulator degrades each real player at BAR-ATH's per-player coverage, so the
        # objects-per-team it can reach depends on how many players this match really had on the
        # pitch (red cards, missing tracks). Expected = coverage x this match's on-pitch count.
        on_pitch = _on_pitch(match)
        expected = [min(0.995, max(0.3, prof["objects_per_team_mean"][t] / noise_sim.TEAM_SIZE))
                    * on_pitch[t] for t in (0, 1)]
        for seed in args.seeds:
            sim = noise_sim.simulate(match, prof, seed=seed, smooth_sigma=noise_sim.SIM_SMOOTH_SIGMA,
                                     state=state)
            sim.save(cache / "sim" / "{}_s{}.npz".format(path.stem, seed))
            if state is not None:
                from .noise import state as noise_state
                y = match.labels
                vis = ~np.isnan(sim.ball[:, 0])
                live = float(vis[y < 2].mean())
                verdict(abs(live - noise_state.ALIVE_COVERAGE) < 0.01,
                        "{} seed {} live-play ball coverage".format(path.stem, seed),
                        "{:.3f} (target {:.2f}); stoppages {:.3f}".format(
                            live, noise_state.ALIVE_COVERAGE, float(vis[y == 2].mean())))
                continue
            got = noise_profile.summary(noise_profile.measure(sim))
            ok = (abs(got["ball_present_fraction"] - target["ball_present_fraction"]) < 0.015
                  and abs(got["objects_home_mean"] - expected[0]) < 0.5
                  and abs(got["objects_away_mean"] - expected[1]) < 0.5
                  and 0.5 < got["tracklet_median_frames"] / max(1.0, target["tracklet_median_frames"]) < 2.0)
            verdict(ok, "{} seed {} matches the profile".format(path.stem, seed),
                    "ball {:.3f} vs {:.3f}, objects {:.1f}/{:.1f} vs expected {:.1f}/{:.1f} "
                    "(clean on-pitch {:.1f}/{:.1f}), tracklet median {:.0f} vs {:.0f}, hazard x{:.2f}".format(
                        got["ball_present_fraction"], target["ball_present_fraction"],
                        got["objects_home_mean"], got["objects_away_mean"], expected[0], expected[1],
                        on_pitch[0], on_pitch[1],
                        got["tracklet_median_frames"], target["tracklet_median_frames"],
                        sim.meta["hazard_scale"]))


def _on_pitch(match):
    """Mean number of tracked players per team over graded frames (clean data)."""
    graded = match.graded_mask()
    seen = ~np.isnan(match.xy[graded, :, 0])
    return [float(seen[:, match.slot_team == t].sum(axis=1).mean()) for t in (0, 1)]


# -- evaluate ------------------------------------------------------------------------------

def _method(args):
    if args.method == "baseline":
        window = int(args.window)
        return "baseline_w{}".format(window), lambda m: baseline_mod.baseline(m, window)
    raise SystemExit("unknown method {!r}".format(args.method))


def cmd_evaluate(args):
    name, method = _method(args)
    if args.on in ("sim", "both"):
        matches = harness.sim_matches(config.cache_dir(args.cache))
        if not matches:
            sys.exit("no simulated matches -- run simulate first")
        pooled, per = harness.run(method, matches, bootstrap_n=args.bootstrap)
        print("SIMULATED VALIDATION (the set to make decisions on)")
        print(harness.format_result(name, pooled, per))
        verdict(True, "{} scored on {} simulated matches".format(name, len(matches)))
    if args.on in ("public", "both"):
        match = load_bar_ath(args)
        gt = label_io.read_labels(config.repo_dir(args.repo) / "evaluation" / "gt"
                                  / "BAR-ATH_possession_gt_public.csv")
        import dataclasses
        pooled, per = harness.run(method, [dataclasses.replace(match, labels=gt)])
        print("PUBLIC 5-MINUTE WINDOW (sanity only -- never choose between versions on this)")
        print(harness.format_result(name, pooled, per))


def cmd_phase1_select(args):
    from . import phase1
    cache = config.cache_dir(args.cache)
    t0 = time.time()
    bar_path = cache / (config.GAME + ".npz")
    bar = Match.load(bar_path) if bar_path.is_file() and not args.merge else None
    if bar is None and not args.merge:
        verdict(None, "no BAR-ATH cache: shift guard skipped")
    result = phase1.select(cache, bar, folds=args.folds, merge_only=args.merge)
    del bar
    if "incomplete" in result:
        verdict(None, "folds still to run: " + ", ".join(result["incomplete"]),
                "run them (--folds ...), then merge with --merge")
        return
    path = phase1.save_json(result, cache / "phase1" / "selection.json")
    print("\nTOP CONFIGURATIONS (objective = mean pooled spellF1 over control thresholds 0/0.5/1/2 s)")
    print("  {:<34} {:>9}   {}".format("configuration", "objective", "by threshold 0 / 0.5 / 1 / 2 s"))
    shown = [r for r in result["table"] if r["key"] != "baseline_w50"][:8] + [result["baseline"]]
    for r in shown:
        print("  {:<34} {:>9.4f}   {}   spells {}/{}".format(
            r["key"], r["objective"],
            " / ".join("{:.3f}".format(v) for v in r["spell_f1_by_threshold"].values()),
            r["n_pred_t1"], r["n_gt_t1"]))
    c, b = result["chosen"], result["baseline"]
    lo, hi = result["chosen_ci_t1"]
    blo, bhi = result["baseline_ci_t1"]
    verdict(None, "chosen " + c["key"], "objective {:.6f}, 90% CI at T=1s [{:.4f}, {:.4f}]".format(
        c["objective"], lo, hi))
    verdict(None, "baseline_w50", "objective {:.4f}, 90% CI at T=1s [{:.4f}, {:.4f}]".format(
        b["objective"], blo, bhi))
    verdict(lo > bhi, "decoder beats the baseline with non-overlapping 90% intervals")
    for p in result.get("paired", []):
        verdict(None, "paired check {}: {}".format(p["family"], p["key"]),
                "better on {} of {} held-out matches, {:+.4f} pooled".format(p["wins"], p["of"], p["gain"]))
    verdict(None, "shift guard excluded: " + (", ".join(result["excluded"]) or "nothing"))
    verdict(path.is_file(), "selection written ({:.0f} min)".format((time.time() - t0) / 60))


def cmd_predict(args):
    from . import phase1
    cache = config.cache_dir(args.cache)
    sel_path = cache / "phase1" / "selection.json"
    if not sel_path.is_file():
        sys.exit("no selection -- run phase1-select first")
    selection = json.loads(sel_path.read_text(encoding="utf-8"))
    match = load_bar_ath(args)
    t0 = time.time()
    tag_path = cache / (config.GAME + "_tagging.json")
    events = tagging.read_saved_tagging(tag_path) if tag_path.is_file() else None
    boundary_xy = None
    tactical = config.tactical_paths(args.data)
    det_path = _dets_path(args)
    if VISION_BOUNDARY and tactical["homography"].is_file() and det_path.is_file():
        from .vision import homography
        z = np.load(det_path, allow_pickle=False)
        H, _, valid_H = homography.load(tactical["homography"], match.n_frames)
        boundary_xy = np.full((match.n_frames, 2), np.nan, dtype=np.float64)
        for i, frame in enumerate(z["frames"]):
            frame = int(frame)
            if not 0 <= frame < match.n_frames or not valid_H[frame]:
                continue
            candidates = z["dets"][i]
            usable = np.isfinite(candidates[:, 2])
            if not usable.any():
                continue
            candidate = candidates[np.flatnonzero(usable)[np.argmax(candidates[usable, 2])], :2]
            try:
                boundary_xy[frame] = homography.pixel_to_pitch(H[frame], candidate[None, :])[0]
            except np.linalg.LinAlgError:
                continue
    codes, info = phase1.predict(cache, match, selection, events, boundary_xy=boundary_xy)
    out = Path(args.out) / "{}.csv".format(args.team)
    label_io.write_labels(out, codes, n_expected=config.N_VIDEO_FRAMES)
    phase1.save_json(info, Path(args.out) / "{}_model.json".format(args.team))
    verdict(None, "configuration " + selection["chosen"]["key"])
    verdict(info["deterministic_in_process"], "decoding is deterministic")
    verdict((time.time() - t0) < 600, "predicted in {:.0f}s".format(time.time() - t0))
    back = label_io.read_labels(out)
    graded = np.zeros(len(back), dtype=bool)
    for _, a, b in config.GRADED_HALVES:
        graded[a:b + 1] = True
    verdict(len(back) == config.N_VIDEO_FRAMES and not np.any(back[graded] == X)
            and np.all(back[~graded] == X), "submission well-formed: 199,319 rows, H/A/D on graded, X elsewhere")
    for name, ks in info["shift_ks"]:
        excluded = name in info["excluded"]
        verdict(True if ks < phase1.KS_PASS else (None if (ks < phase1.KS_FAIL or excluded) else False),
                "feature '{}' distributed like the simulated games".format(name),
                "{} shift{}".format("small" if ks < phase1.KS_PASS else
                                    ("moderate" if ks < phase1.KS_FAIL else "LARGE"),
                                    " -- excluded from the models by the shift guard" if excluded else ""))
    cks, cname = info["context_ks_max"]
    verdict(cks < phase1.KS_PASS if cks < phase1.KS_FAIL else False,
            "all context-window features distributed like the simulated games",
            "largest KS {:.3f} ({})".format(cks, cname))
    verdict(len(info["pre_kickoff_spans"]) >= 1,
            "pre-kick-off dead ball enforced from Smart Tagging",
            "{} span(s)".format(len(info["pre_kickoff_spans"])))
    pre = [codes[a:b] for a, b in info["pre_kickoff_spans"]]
    verdict(all(np.all(p == 2) for p in pre), "every pre-kick-off frame is dead ball")
    verdict(None, "tracker speed field unit: " + info["speed_field_unit"])
    dead_share, per_min = phase1.plausibility(back, match.periods)
    verdict(phase1.DEAD_SHARE_BAND[0] <= dead_share <= phase1.DEAD_SHARE_BAND[1],
            "dead-ball share inside the external-data band")
    verdict(phase1.POSS_PER_MIN_BAND[0] <= per_min <= phase1.POSS_PER_MIN_BAND[1],
            "possessions per minute inside the external-data band")
    gt_path = config.repo_dir(args.repo) / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
    if gt_path.is_file():
        r = metrics.evaluate(label_io.read_labels(gt_path), back)
        verdict(5 <= r["n_pred_spells"] <= 60,
                "public window sanity: a plausible number of possessions (not a tuning target)")
    verdict(None, "sha256 " + phase1.sha256_of(out)[:16],
            "run predict twice: the same hash means the file is reproducible")
    verdict(out.is_file(), "written " + str(out))


def _dets_path(args):
    stem = Path(args.model).stem
    return config.cache_dir(args.cache) / "vision" / "ball_dets_{}_{}.npz".format(args.mode, stem)


def cmd_vision_detect(args):
    """Ball detection on the tactical video (needs the vision environment: torch + ultralytics)."""
    from .vision import detect
    match = load_bar_ath(args)
    frames = detect.frame_plan(match, args.mode, sample_n=args.sample_n)
    out = _dets_path(args)
    video = config.tactical_paths(args.data)["video"]
    r = detect.run(video, frames, out, model_name=args.model, conf=args.conf,
                   batch_frames=args.batch, device=args.device, half=not args.no_half,
                   expected_frames=config.N_VIDEO_FRAMES)
    verdict(r["declared"] == config.N_VIDEO_FRAMES, "video declares 199,319 frames", str(r["declared"]))
    if r["decoded_to_end"] is not None:
        verdict(r["decoded_to_end"], "decoding ended exactly at frame 199,318")
    verdict(None, "processed {} frames in {:.0f} min ({:.1f} frames/s)".format(
        r["processed"], r["seconds"] / 60, r["processed"] / max(1e-9, r["seconds"])))
    verdict(r["with_candidate"] > 0, "frames with at least one ball candidate",
            "{} of {}".format(r["with_candidate"], r["frames"]))
    verdict(Path(out).is_file(), "detections written to " + str(out))


def cmd_vision_validate(args):
    from .vision import homography, validate
    match = load_bar_ath(args)
    z = np.load(_dets_path(args), allow_pickle=False)
    H, _, valid = homography.load(config.tactical_paths(args.data)["homography"], config.N_VIDEO_FRAMES)
    verdict(bool(valid[match.graded_mask()].all()), "homography valid on every graded frame")
    rows, n_seen, n_miss = validate.report(z["dets"], z["frames"], match.ball, H, match.pitch)
    print("  checked on {} tracker-visible frames; fill rate over {} tracker-missing frames".format(n_seen, n_miss))
    print("  {:>9} {:>8} {:>10} {:>10} {:>12} {:>10}".format(
        "conf >=", "recall", "within 2m", "within 4m", "median err", "fill rate"))
    for r in rows:
        print("  {:>9.2f} {:>8.3f} {:>10.3f} {:>10.3f} {:>11.2f}m {:>10.3f}".format(
            r["threshold"], r["recall"], r["within_2m"], r["within_4m"], r["median_err_m"], r["fill_rate"]))


def cmd_validate(args):
    try:
        codes = label_io.read_labels(args.file)
    except ValueError as err:
        verdict(False, "submission format", str(err))
        return
    verdict(len(codes) == config.N_VIDEO_FRAMES, "row count {}".format(len(codes)))
    graded = np.zeros(len(codes), dtype=bool)
    for _, a, b in config.GRADED_HALVES:
        graded[a:b + 1] = True
    n_x = int((codes[graded[:len(codes)]] == X).sum()) if len(codes) == len(graded) else -1
    verdict(n_x == 0 if n_x >= 0 else False, "no X on graded frames", "{} X".format(n_x))


def cmd_phase0(args):
    steps = [("build-cache", cmd_build_cache), ("check-baseline", cmd_check_baseline),
             ("check-scorer", cmd_check_scorer), ("idsse-build", cmd_idsse_build),
             ("noise-profile", cmd_noise_profile), ("simulate", cmd_simulate),
             ("evaluate", cmd_evaluate)]
    for name, fn in steps:
        print("\n== {} ==".format(name))
        fn(args)


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", help="data bundle folder (has tactical/); env MTAH_DATA_DIR")
    common.add_argument("--repo", help="official repo clone; env MTAH_REPO_DIR")
    common.add_argument("--cache", help="cache folder; env MTAH_CACHE_DIR (default ./cache)")
    common.add_argument("--idsse", help="IDSSE XML folder; env MTAH_IDSSE_DIR")
    common.add_argument("--matches", nargs="+", default=["all"], help="IDSSE match ids or 'all'")
    common.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    common.add_argument("--method", default="baseline")
    common.add_argument("--window", default="50")
    common.add_argument("--on", choices=("sim", "public", "both"), default="both")
    common.add_argument("--bootstrap", type=int, default=1000)
    common.add_argument("--n-random", type=int, default=200)
    common.add_argument("--team", default="Gradient Descendants", help="submission file name (team name)")
    common.add_argument("--folds", nargs="+", help="phase1-select: run only these held-out matches")
    common.add_argument("--merge", action="store_true", help="phase1-select: merge saved folds only")
    common.add_argument("--mode", default="sample", choices=("sample", "missing", "all"), help="vision: frames")
    common.add_argument("--sample-n", type=int, default=3000)
    common.add_argument("--model", default="yolo11s.pt", help="vision: Ultralytics weights")
    common.add_argument("--conf", type=float, default=0.05)
    common.add_argument("--batch", type=int, default=8, help="vision: frames per GPU batch")
    common.add_argument("--device", default="0", help="vision: CUDA device or 'cpu'")
    common.add_argument("--no-half", action="store_true", help="vision: disable fp16")
    common.add_argument("--out", default="submissions", help="where predict writes the CSV")
    common.add_argument("--labels", default="cache/stage1_labels.csv",
                        help="state-profile: stage-1 submission that supplies the game state")
    common.add_argument("--world", default="state", choices=("phase6", "state"),
                        help="simulate: phase6 = state-blind dropout (phases 1-7); state = phase 9")

    parser = argparse.ArgumentParser(prog="python -m possession", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {"build-cache": cmd_build_cache, "check-baseline": cmd_check_baseline,
                "check-scorer": cmd_check_scorer, "idsse-download": cmd_idsse_download,
                "idsse-build": cmd_idsse_build, "noise-profile": cmd_noise_profile,
                "simulate": cmd_simulate, "evaluate": cmd_evaluate, "phase0": cmd_phase0,
                "state-profile": cmd_state_profile,
                "phase1-select": cmd_phase1_select, "predict": cmd_predict,
                "vision-detect": cmd_vision_detect, "vision-validate": cmd_vision_validate}
    for name in commands:
        sub.add_parser(name, parents=[common])
    val = sub.add_parser("validate", parents=[common])
    val.add_argument("file")
    commands["validate"] = cmd_validate

    args = parser.parse_args(argv)
    commands[args.command](args)
    if VERDICTS:
        fails = [w for t, w in VERDICTS if t == "FAIL"]
        print("\n{} PASS, {} FAIL, {} INFO".format(
            sum(t == "PASS" for t, _ in VERDICTS), len(fails), sum(t == "INFO" for t, _ in VERDICTS)))
        for w in fails:
            print("  FAIL: " + w)
        return 1 if fails else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
