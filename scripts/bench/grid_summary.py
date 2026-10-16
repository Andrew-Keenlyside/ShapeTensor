"""Compact step x texture grid: median error in two bands for the main arms. Usage: grid_summary.py TAG pid..."""
import sys
from shapetensor.validate.report import load, _get
tag, pids = sys.argv[1], sys.argv[2:]
arms = [("single", "intensity", "eigen", "ST"), ("single", "anatomix16", "tca", "a16tca"), ("single", "anatomix48", "tca", "a48tca"),
        ("single", "jet", "tca", "jet_tca"), ("hybrid", "anatomix16", "switch_cl", "sw_cl16"), ("hybrid", "anatomix48", "switch_cl", "sw_cl48"),
        ("hybrid", "anatomix48", "excise_all", "fixA"), ("hybrid", "anatomix48", "excise_cl", "fixA_cl"), ("hybrid", "anatomix48", "excise_tca_dom", "fixA_tca")]
from collections import defaultdict
import numpy as np
acc = defaultdict(list)
for kind in ("single", "hybrid"):
    for r in load(tag, kind):
        for name, sc in r["scores"].items():
            acc[(kind, r["engine"], name, r["pid"], r["rho"])].append(sc["bands"])
class _Avg(dict):
    pass
res = {}
for k, lst in acc.items():
    d = {}
    for band in lst[0]:
        d[band] = {"median": float(np.mean([_get(b, band, "median") for b in lst]))}
    res[k] = d
for band in ("0-0.1mm", "0.25-0.5mm", ">2mm"):
    print(f"\n### band {band}: median angle (deg)   columns = rho 8 | 16 | 32")
    print(f"{'pid':14s} " + " ".join(f"{a[3]:>17s}" for a in arms))
    for pid in pids:
        cells = []
        for k, e, n, lab in arms:
            v = [_get(res.get((k, e, n, pid, rho), {}), band, "median") for rho in (8.0, 16.0, 32.0)]
            cells.append(" ".join(f"{x:5.1f}" for x in v))
        print(f"{pid:14s} " + " ".join(f"{c:>17s}" for c in cells))
