"""The pre-registered claims evaluated against a results tag: one row per claim and arm, with the measured value,
the threshold and PASS / FAIL, as a markdown scorecard (``SCORECARD.md``) and a JSON. The claims and thresholds
are those of docs/TCA_FIELD_REVIEW.md §7, written before the runs; changing them here is a documented act."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .report import _get, load
from .suite import RESULTS_DIR

#: (claim, kind, how to measure) -- measured on the seed mean of per-seed medians / shares
CLAIMS = [
    {"claim": "interface fixed", "pids": ["P3-40-t0.25", "P3-40-t0.5", "P3-15-t0.25"], "band": "0-0.1mm", "stat": "median",
     "kind": "single", "op": "<=", "threshold": 10.0, "unit": "deg"},
    {"claim": "nothing lost in clear tissue", "pids": ["P3-0", "P3-0-t0.5", "P3-0-t0.25", "P8", "P8-t0.5", "P8-t0.25"],
     "band": "all", "stat": "median", "kind": "single", "op": "<= ST + 1", "threshold": 1.0, "unit": "deg"},
    {"claim": "U-fibres kept", "pids": ["P2", "P2-t0.5"], "band": "0.25-0.5mm", "stat": "median", "kind": "single",
     "op": "<=", "threshold": 10.0, "unit": "deg"},
    {"claim": "vessels", "pids": ["P6", "P6-t0.5"], "band": "0-0.1mm", "stat": "median", "kind": "single", "op": "<=",
     "threshold": 15.0, "unit": "deg"},
    {"claim": "crossings resolved (both within 15 deg at >= 90% of sites, 60 deg)", "pids": ["P4-lines-c60-f1"],
     "kind": "multi", "metric": ("two", "both_within_15_share"), "op": ">=", "threshold": 0.9, "unit": "share"},
    {"claim": "crossings resolved (45 deg)", "pids": ["P4-lines-c45-f1"], "kind": "multi",
     "metric": ("two", "both_within_15_share"), "op": ">=", "threshold": 0.9, "unit": "share"},
    {"claim": "no false populations (single)", "pids": ["P4-lines-single", "P4-lines-single-t0.5"], "kind": "multi",
     "metric": ("single", "false_second_share"), "op": "<=", "threshold": 0.02, "unit": "share"},
    {"claim": "fractions (4:1)", "pids": ["P4-lines-c60-f4"], "kind": "multi", "metric": ("two", "frac_abs_err", "median"),
     "op": "<=", "threshold": 0.1, "unit": "abs"},
    {"claim": "boundary detector (PR-AUC)", "pids": ["P3-40-t0.25", "P3-40-t0.5", "P3-40"], "kind": "detector",
     "op": ">=", "threshold": 0.9, "unit": "auc"},
]


def _single_values(tag):
    """(pid, engine, readout, rho) -> {band: [per-seed medians]}"""
    acc = defaultdict(lambda: defaultdict(list))
    for kind in ("single", "hybrid"):
        for r in load(tag, kind):
            for name, sc in r["scores"].items():
                for band, s in sc["bands"].items():
                    if "median" in s:
                        acc[(r["pid"], r["engine"], name, r["rho"])][band].append(s["median"])
    return acc


def evaluate(tag: str) -> list[dict]:
    rows = []
    singles = _single_values(tag)
    multis = defaultdict(list)
    for r in load(tag, "multi"):
        for name, m in r["methods"].items():
            multis[(r["pid"], r["engine"], name, r["rho"])].append(m)
    dets = defaultdict(list)
    for r in load(tag, "single"):
        for name, d in r["detectors"].items():
            dets[(r["pid"], r["engine"], name, r["rho"])].append(_get(d, "pr_auc"))
    for c in CLAIMS:
        if c["kind"] == "single":
            arms = sorted({(e, n, rho) for (p, e, n, rho) in singles if p in c["pids"]})
            for e, n, rho in arms:
                vals = []
                for pid in c["pids"]:
                    v = singles.get((pid, e, n, rho), {}).get(c["band"], [])
                    if v:
                        vals.append(np.mean(v))
                if not vals:
                    continue
                value = float(max(vals))                                  # the worst phantom of the claim
                if c["op"] == "<= ST + 1":
                    ref = [np.mean(singles.get((pid, "intensity", "eigen", rho), {}).get(c["band"], [np.nan])) for pid in c["pids"]]
                    thr = float(np.nanmax(ref)) + c["threshold"]; ok = value <= thr
                    rows.append({**c, "arm": f"{e} {n} rho{rho:g}", "value": value, "threshold_value": thr, "pass": bool(ok)})
                else:
                    ok = value <= c["threshold"] if c["op"] == "<=" else value >= c["threshold"]
                    rows.append({**c, "arm": f"{e} {n} rho{rho:g}", "value": value, "threshold_value": c["threshold"], "pass": bool(ok)})
        elif c["kind"] == "multi":
            arms = sorted({(e, n, rho) for (p, e, n, rho) in multis if p in c["pids"]})
            for e, n, rho in arms:
                vals = []
                for pid in c["pids"]:
                    lst = multis.get((pid, e, n, rho), [])
                    if lst:
                        vals.append(np.nanmean([_get(m, *c["metric"]) for m in lst]))
                if not vals or not np.isfinite(vals).any():
                    continue
                value = float(np.nanmin(vals) if c["op"] == ">=" else np.nanmax(vals))
                ok = value >= c["threshold"] if c["op"] == ">=" else value <= c["threshold"]
                rows.append({**c, "arm": f"{e} {n} rho{rho:g}", "value": value, "threshold_value": c["threshold"], "pass": bool(ok)})
        else:
            arms = sorted({(e, n, rho) for (p, e, n, rho) in dets if p in c["pids"]})
            for e, n, rho in arms:
                vals = [np.nanmean(dets[(pid, e, n, rho)]) for pid in c["pids"] if (pid, e, n, rho) in dets]
                if not vals or not np.isfinite(vals).any():
                    continue
                value = float(np.nanmin(vals)); ok = value >= c["threshold"]
                rows.append({**c, "arm": f"{e} {n} rho{rho:g}", "value": value, "threshold_value": c["threshold"], "pass": bool(ok)})
    return rows


def write(tag: str = "main", out: str | None = None, only_passing_and_best: bool = True) -> Path:
    rows = evaluate(tag)
    md = [f"# Scorecard: tag `{tag}`\n", "Each claim is evaluated on the worst of its phantoms (seed mean of per-seed medians / shares). "
          "Arms that pass are listed in full; for failed claims the three best arms are shown.\n"]
    by = defaultdict(list)
    for r in rows:
        by[r["claim"]].append(r)
    for claim, lst in by.items():
        c = lst[0]
        md.append(f"\n## {claim}  —  threshold {c['op']} {c['threshold']} {c['unit']}, phantoms {', '.join(c['pids'])}\n")
        md.append("| arm | value | threshold | result |\n|---|---|---|---|")
        passing = [r for r in lst if r["pass"]]
        better = (lambda r: -r["value"]) if c["op"].startswith("<") else (lambda r: r["value"])
        show = passing if passing else sorted(lst, key=better, reverse=True)[:3]
        if passing and only_passing_and_best:
            show = sorted(passing, key=better, reverse=True)
        for r in show:
            md.append(f"| {r['arm']} | {r['value']:.3f} | {r['threshold_value']:.3f} | {'PASS' if r['pass'] else 'FAIL'} |")
        md.append(f"\n{len(passing)} of {len(lst)} arms pass.\n")
    path = Path(out) if out else RESULTS_DIR / tag / "SCORECARD.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md))
    (path.with_suffix(".json")).write_text(json.dumps(rows, indent=1, default=str))
    return path
