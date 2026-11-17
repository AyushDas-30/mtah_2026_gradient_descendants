"""Restart/stop experiments. usage:
  python3 rsx.py scores <held>      fit Phase-5 emission on the other matches, save held-out scores
  python3 rsx.py edges <held>       fit restart/stop detectors on the other matches, save held-out probs
  python3 rsx.py eval <name> [...]  decode saved folds with named decoder variants, pooled objective"""
import sys, os, time, pickle, json, numpy as np, warnings
warnings.simplefilter("ignore")
P2 = os.path.expanduser("~/work/p2"); sys.path.insert(0, os.path.expanduser(os.environ.get("XPCODE", "~/work/p6")))
from pathlib import Path
from possession import phase1, features as feat, decode, relabel, metrics
from possession.gbdt import GBDT
from possession.schema import D, X
XPV = os.environ.get("XPV", "")
OUT = Path(os.path.expanduser("~/work/rs/folds" + os.environ.get("XPSET", "") + ("_" + XPV if XPV else ""))); OUT.mkdir(exist_ok=True)
def patch(v):
    if not v: return
    import numpy as _np
    if "short" in v:
        feat.CONTEXT_SECONDS = (0.5, 1, 2, 5)
        def cw(f, fps):
            out = {}; known = f["ball_known"] > 0
            for name in feat.CONTEXT_KNOWN + feat.CONTEXT_FREE + feat.CONTEXT_DEAD:
                x = _np.where(known, f[name], _np.nan) if name in feat.CONTEXT_KNOWN else f[name].astype(_np.float64)
                for sec in feat.CONTEXT_SECONDS:
                    w = int(round(sec * fps))
                    out["ctx_{}_p{}".format(name, sec)] = _np.nan_to_num(feat._window_mean(x, -w, 0))
                    out["ctx_{}_f{}".format(name, sec)] = _np.nan_to_num(feat._window_mean(x, 1, w + 1))
            return out
        feat.context_windows = cw
        nk = len(feat.CONTEXT_KNOWN + feat.CONTEXT_FREE) * 4; nd = len(feat.CONTEXT_DEAD) * 4
        feat.TEAM_SPEC = feat.TEAM_SPEC[:-nk] + feat._ctx_spec(feat.CONTEXT_KNOWN + feat.CONTEXT_FREE)
        feat.GAP_SPEC = feat.GAP_SPEC[:-nk] + feat._ctx_spec(feat.CONTEXT_KNOWN + feat.CONTEXT_FREE)
        feat.DEAD_SPEC = feat.DEAD_SPEC[:-nd] + feat._ctx_spec(feat.CONTEXT_DEAD)
    if "recv" in v:
        import recv as _recv
        _orig_r = feat.compute
        def compute3(match):
            out = _orig_r(match); out.update(_recv.features_for(match, feat, match.fps)); return out
        feat.compute = compute3
        phase1.features.compute = compute3
        feat.TEAM_SPEC = feat.TEAM_SPEC + _recv.specs()
        feat.GAP_SPEC = feat.GAP_SPEC + _recv.specs()
    if "touch" in v:
        import touch as _touch
        _orig = feat.compute
        def compute2(match):
            out = _orig(match); out.update(_touch.compute(match, feat)); return out
        feat.compute = compute2
        phase1.features.compute = compute2
        feat.TEAM_SPEC = feat.TEAM_SPEC + _touch.specs()
        feat.GAP_SPEC = feat.GAP_SPEC + _touch.specs()
patch(XPV)
CACHE = Path(os.environ.get("XPCACHE", P2 + "/cache"))
SEL = json.loads((Path(P2) / "cache" / "phase1" / "selection.json").read_text())
R = 5            # frames either side of an edge count as positive
def load():
    g = phase1._group_sims(CACHE)
    data = {k: phase1._load_featurised(v, lambda s: None) for k, v in g.items()}
    extra = os.environ.get("XPEXTRA")        # extra training-only sims (augmentation)
    if extra:
        ge = phase1._group_sims(Path(os.path.expanduser(extra)))
        for k, v in ge.items():
            for it in phase1._load_featurised(v, lambda s: None):
                it["_extra"] = True
                data.setdefault(k, []).append(it)
    return data
def edges(y, kind):
    y = np.asarray(y); a = y[:-1]; b = y[1:]
    if kind == "restart": t = np.flatnonzero((a == D) & (b < D)) + 1
    else: t = np.flatnonzero((a < D) & (b == D)) + 1
    m = np.zeros(len(y) + 1, np.int64)
    np.add.at(m, np.clip(t - R, 0, len(y)), 1); np.add.at(m, np.clip(t + R + 1, 0, len(y)), -1)
    return np.cumsum(m)[:-1] > 0
cmd = sys.argv[1] if __name__ == "__main__" else ""
if cmd in ("scores", "edges"):
    held = sys.argv[2]; t0 = time.time()
    data = load()
    train = [it for g, its in data.items() if g != held for it in its]
    if cmd == "scores":
        phase1.EDGE_MODEL = False
        model, priors = phase1.fit_models(train, SEL.get("excluded", []))
        out = [{"name": it["name"], "periods": it["periods"], "n_frames": it["n_frames"],
                "variants": it["variants"], "scores": model.scores(it["features"])} for it in data[held]
               if not it.get("_extra")]
        pickle.dump({"items": out, "priors": priors}, open(OUT / f"{held}_scores.pkl", "wb"))
    else:
        res = {}
        for kind in ("restart", "stop"):
            Xs, ys = [], []
            for it in train:
                y = it["variants"][phase1.T_TRAIN]; pos = edges(y, kind) & (y != X)
                rows = np.union1d(np.flatnonzero(y != X)[::10], np.flatnonzero(pos))
                Xs.append(feat.dead_design(it["features"], rows, SEL.get("excluded", []))); ys.append(pos[rows].astype(float))
            g = GBDT().fit(np.vstack(Xs), np.concatenate(ys))
            res[kind] = [np.round(g.predict_proba(feat.dead_design(it["features"], None, SEL.get("excluded", []))), 6) for it in data[held]]
        pickle.dump(res, open(OUT / f"{held}_edges.pkl", "wb"))
    print(cmd, held, "%.0fs" % (time.time() - t0))
