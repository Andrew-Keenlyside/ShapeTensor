"""Tables from gap_probe.py's probe_results.json: per condition the targets' visibility and, per model, the single-
population axis error and miss rate, false second peaks, second/third peak detection, pair error, false peaks on
isotropic cells, unexplained order-2 energy.
    python scripts/learn/gap_probe_table.py DIR
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1]); R = json.load(open(d / "probe_results.json"))
models = sorted({k for v in R.values() if isinstance(v, dict) for k, x in v.items() if isinstance(x, dict) and "unexplained_l2" in x})
cols = [("single_deg", "1-pop °"), ("single_miss", "1-pop miss"), ("false_second", "false 2nd"), ("second_detected", "2nd det."), ("pair_deg", "pair °"),
        ("third_detected", "3rd det."), ("false_peak_iso", "false pk iso"), ("unexplained_l2", "unexpl. l2")]
f = lambda x, k: (f"{x[k]:.2f}" if isinstance(x.get(k), float) else "–")
lines = []
for grp in ("morph", "res", "cross", "artefact", "iso"):
    keys = [k for k in R if k.startswith(grp + ":")]
    if not keys:
        continue
    lines += [f"\n### {grp}\n", "| condition | n | target vis | model | " + " | ".join(c[1] for c in cols) + " | target 0/1/2/3 peaks |",
              "|" + "---|" * (len(cols) + 5)]
    for k in keys:
        v = R[k]
        for m in models:
            x = v.get(m, {}); tp = x.get("target_peaks_share", {})
            lines.append(f"| {k.split(':', 1)[1]} | {v['n']} | {v['target_orientation_visibility']:.2f} | {m.split('_')[0]} | " + " | ".join(f(x, c[0]) for c in cols)
                         + " | " + "/".join(str(tp.get(str(i), "–")) for i in range(4)) + " |")
if "limits:target_crossing_resolution" in R:
    lines += ["\n### the targets' own crossing resolution (two line populations in a cell: share with two target peaks)\n", "```", json.dumps(R["limits:target_crossing_resolution"], indent=1), "```"]
open(d / "probe_tables.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
