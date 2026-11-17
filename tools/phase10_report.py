"""Summary of the phase 10 variants against the Phase 9 file. Run from mtah-possession-phase9:
    python tools/phase10_report.py"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from possession import phase1  # noqa: E402
from possession.io import labels as L  # noqa: E402

base_path = Path("submissions/Gradient Descendants.csv")
base = L.read_labels(base_path)
periods = [(1, 11185, 90652), (2, 110808, 190198)]
rows = [("phase9", base_path, None)]
for d in sorted(Path("results10").glob("*")):
    f = d / "Gradient Descendants.csv"
    if f.is_file():
        rows.append((d.name, f, d / "selection.json"))
print("%-10s %-16s %-36s %8s %6s %8s %7s" % ("variant", "sha256", "chosen decoder", "sim obj", "dead", "poss/min", "agree"))
graded = base < 3
for name, f, sel in rows:
    y = L.read_labels(f)
    ds, pm = phase1.plausibility(y, periods)
    h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    key, obj = "-", float("nan")
    if sel is not None and sel.is_file():
        s = json.loads(sel.read_text())
        key, obj = s["chosen"]["key"], s["chosen"]["objective"]
    print("%-10s %-16s %-36s %8.4f %6.3f %8.2f %7.3f" % (name, h, key, obj, ds, pm, np.mean(y[graded] == base[graded])))
