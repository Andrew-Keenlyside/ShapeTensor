"""Acceptance criteria for the crossing ladder (:mod:`~shapetensor.learn.simple_phantom`), fixed before results: a rung
passes only when every criterion holds on every held-out phantom. Measured on the crossing plane's maps (sigma 2.5
windows): the targets ``tA, tB, tI`` (A, B, background) and the model's ``mA, mB, mO, mI`` (along A, along B, other
directions, isotropic: the lobe-fit split with the density floor as background), plus the model's entities per point.

C1 level        median model / target mass of each population on its arm and in the crossing core   0.90-1.10
C2 flat         model / target at every point along each bundle's centreline (target >= half its peak)  0.85-1.15
C3 count        oriented populations per point (mass >= a third of the populations' level): background 0,
                arms 1, core 2                                                                        >= 95 % of each zone
C4 leakage      a population's model mass outside its bundle; mass in other directions               <= 5 % of the target total
C5 edges        model / target in a bundle's outer half over that in its inner half (arms)             0.90-1.10
C6 direction    angle from each population's nearest model entity axis to the true axis               median <= 10, p95 <= 20 deg
C7 background   median model / target isotropic mass in pure background and inside the bundles       0.90-1.10
"""
import json

import numpy as np
from scipy.ndimage import map_coordinates

LEVEL = (0.90, 1.10); FLAT = (0.85, 1.15); COUNT = 0.95; LEAK = 0.05; EDGE = (0.90, 1.10); DIR = (10.0, 20.0); BACK = (0.90, 1.10)
NAMES = ("C1 level", "C2 flat", "C3 count", "C4 leakage", "C5 edges", "C6 direction", "C7 background")


def _zones(ph, shape):
    """Distances from A's and B's centrelines on the plane grid (rows: local y, cols: local x) and the zones."""
    n, c, half = shape[0], ph["c"], ph["half"]
    r, q = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing="ij")
    b = ph["axes_local"]["B"].cpu().numpy()
    dA = np.abs(r); dB = np.abs(r * b[2] - q * b[1])
    inA, inB = dA <= half - 3, dB <= half - 3                                                       # interiors (past the window's blur)
    outA, outB = dA >= half + 5, dB >= half + 5                                                     # clearly outside
    return dA, dB, {"core": inA & inB, "armA": inA & outB, "armB": inB & outA, "bg": outA & outB, "inA": inA, "inB": inB, "outA": outA, "outB": outB}


def _ratio(m, t, sel):
    sel = sel & np.isfinite(t) & np.isfinite(m)
    return float(np.median(m[sel]) / max(np.median(t[sel]), 1e-9)) if sel.sum() > 5 else float("nan")


def _inside(v, lo_hi):
    return bool(np.isfinite(v) and lo_hi[0] <= v <= lo_hi[1])


def evaluate(ph, M: dict, ent: dict) -> dict:
    """Each criterion's values and pass for one phantom. ``M``: the maps (n x n, NaN outside the shown disc); ``ent``:
    per grid point (row-major) the entities' ``mass (P, C)``, ``iso (P, C)`` and world ``axis (P, C, 3)``."""
    n = M["tA"].shape[0]; c, half = ph["c"], ph["half"]
    dA, dB, Z = _zones(ph, M["tA"].shape)
    valid = np.isfinite(M["tA"])
    Z = {k: v & valid for k, v in Z.items()}
    out = {}
    # C1 level
    lv = {f"{p} {z}": _ratio(M["m" + p], M["t" + p], Z[zz]) for p in ("A", "B") for z, zz in (("arm", "arm" + p), ("core", "core"))}
    out["C1 level"] = (all(_inside(v, LEVEL) for v in lv.values()), lv)
    # C2 flat: along each centreline
    fl = {}
    for p in ("A", "B"):
        u = ph["axes_local"][p].cpu().numpy(); t = np.arange(-c + 1, c - 1)
        rows, cols = c + t * u[1], c + t * u[2]
        tv = map_coordinates(np.nan_to_num(M["t" + p], nan=-1), [rows, cols], order=1); mv = map_coordinates(np.nan_to_num(M["m" + p], nan=-1), [rows, cols], order=1)
        ok = (tv >= 0.5 * np.nanmax(M["t" + p])) & (mv >= 0)
        ra = mv[ok] / np.maximum(tv[ok], 1e-9)
        fl[p] = (float(ra.min()) if ra.size else float("nan"), float(ra.max()) if ra.size else float("nan"))
    out["C2 flat"] = (all(_inside(lo, FLAT) and _inside(hi, FLAT) for lo, hi in fl.values()), {f"{p} min/max": v for p, v in fl.items()})
    # C3 count
    lvl = min(np.median(M["tA"][Z["armA"]]) if Z["armA"].any() else 1.0, np.median(M["tB"][Z["armB"]]) if Z["armB"].any() else 1.0)
    thr = max(0.02, lvl / 3)
    cnt = ((ent["mass"] >= thr) & ~ent["iso"]).sum(1).reshape(n, n)
    cz = {z: float((cnt[Z[z]] == k).mean()) if Z[z].any() else float("nan") for z, k in (("bg", 0), ("armA", 1), ("armB", 1), ("core", 2))}
    out["C3 count"] = (all(np.isfinite(v) and v >= COUNT for v in cz.values()), cz)
    # C4 leakage
    tot = np.nansum(M["tA"]) + np.nansum(M["tB"])
    lk = {"A outside": float(np.nansum(M["mA"][Z["outA"]]) / max(np.nansum(M["tA"]), 1e-9)),
          "B outside": float(np.nansum(M["mB"][Z["outB"]]) / max(np.nansum(M["tB"]), 1e-9)), "other dirs": float(np.nansum(M["mO"]) / max(tot, 1e-9))}
    out["C4 leakage"] = (all(v <= LEAK for v in lk.values()), lk)
    # C5 edges (arms)
    ed = {}
    for p, d, arm in (("A", dA, (dA <= half) & Z["outB"]), ("B", dB, (dB <= half) & Z["outA"])):
        inner, outer = arm & (d <= half / 2), arm & (d > half / 2)
        ri = np.nansum(M["m" + p][inner]) / max(np.nansum(M["t" + p][inner]), 1e-9); ro = np.nansum(M["m" + p][outer]) / max(np.nansum(M["t" + p][outer]), 1e-9)
        ed[p] = float(ro / max(ri, 1e-9))
    out["C5 edges"] = (all(_inside(v, EDGE) for v in ed.values()), ed)
    # C6 direction: the nearest entity (mass >= thr, oriented) to each population's axis, on its arm and in the core
    dr = {}
    for p in ("A", "B"):
        a = ph["axes"][p].cpu().numpy()
        cos = np.abs(ent["axis"] @ a) * ((ent["mass"] >= thr) & ~ent["iso"])
        ang = np.degrees(np.arccos(np.clip(cos.max(1), 0, 1))).reshape(n, n)
        sel = Z["arm" + p] | Z["core"]
        dr[p] = (float(np.median(ang[sel])), float(np.percentile(ang[sel], 95))) if sel.any() else (float("nan"),) * 2
    out["C6 direction"] = (all(np.isfinite(m) and m <= DIR[0] and q <= DIR[1] for m, q in dr.values()), {f"{p} med/p95": v for p, v in dr.items()})
    # C7 background
    bk = {"background": _ratio(M["mI"], M["tI"], Z["bg"]), "inside": _ratio(M["mI"], M["tI"], Z["inA"] | Z["inB"])}
    out["C7 background"] = (all(_inside(v, BACK) for v in bk.values()), bk)
    out["PASS"] = all(v[0] for k, v in out.items())
    return out


def _fmt(v):
    if isinstance(v, tuple):
        return "/".join(f"{x:.2f}" if abs(x) < 10 else f"{x:.0f}" for x in v)
    return f"{v:.2f}" if np.isfinite(v) and abs(v) < 10 else (f"{v:.0f}" if np.isfinite(v) else "-")


def table_figure(rows: list, out_png: str, title: str):
    """One row per held-out phantom (``(label, evaluate(...))``), a column per criterion: green pass, red fail, values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = list(NAMES) + ["PASS"]
    fig, ax = plt.subplots(figsize=(3.0 * len(cols), 0.9 + 0.95 * len(rows))); ax.axis("off")
    cell = [[("PASS" if res["PASS"] else "FAIL") if k == "PASS" else "\n".join(f"{kk} {_fmt(vv)}" for kk, vv in res[k][1].items()) for k in cols]
            for _, res in rows]
    colours = [[("#b7e4c7" if (res["PASS"] if k == "PASS" else res[k][0]) else "#f4a6a6") for k in cols] for _, res in rows]
    tb = ax.table(cellText=cell, cellColours=colours, rowLabels=[r[0] for r in rows], colLabels=cols, loc="center", cellLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(7); tb.scale(1, 3.6)
    ax.set_title(title + "\nthresholds: C1 0.90-1.10 | C2 0.85-1.15 | C3 >= 95 % | C4 <= 5 % | C5 0.90-1.10 | C6 med <= 10, p95 <= 20 deg | C7 0.90-1.10", fontsize=9)
    plt.savefig(out_png, dpi=90, bbox_inches="tight"); plt.close(fig)


def save_json(rows: list, path: str):
    json.dump([{"phantom": lab, **{k: (v if k == "PASS" else {"pass": v[0], **{kk: vv for kk, vv in v[1].items()}}) for k, v in res.items()}} for lab, res in rows],
              open(path, "w"), indent=1, default=float)
