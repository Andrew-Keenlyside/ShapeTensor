"""Acceptance criteria for the crossing ladder (:mod:`~shapetensor.learn.simple_phantom`), fixed before results: a rung
passes only when every criterion holds on every held-out phantom. Measured on the crossing plane's maps (sigma 2.5
windows): the targets ``tA, tB, tI`` (A, B, background) and the model's ``mA, mB, mO, mI`` (along A, along B, other
directions, isotropic: the lobe-fit split with the density floor as background), plus the model's entities per point.
Scored only at points >= FACE_MARGIN voxels from the volume's faces -- the training queries' domain (agreed with the user
2026-10-26; the thresholds unchanged; the sigma-2.5 targets are still partly cut by the faces there, as in training).
C1 also requires each population's whole-plane total (model / target over every scored point) within LEVEL: the zones
leave a ring around each bundle unscored, where a halo or a spurious entity would otherwise go unseen (review 2026-10-26).

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
FACE_MARGIN = 2.0
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
    face = M.get("_face", np.full(M["tA"].shape, np.inf))
    valid = np.isfinite(M["tA"]) & (face >= FACE_MARGIN)
    Z = {k: v & valid for k, v in Z.items()}
    M = {k: (np.where(valid, v, np.nan) if k[0] in "tm" else v) for k, v in M.items()}
    out = {}
    # C1 level
    lv = {f"{p} {z}": _ratio(M["m" + p], M["t" + p], Z[zz]) for p in ("A", "B") for z, zz in (("arm", "arm" + p), ("core", "core"))}
    for p in ("A", "B"):
        lv[f"{p} plane"] = float(np.nansum(M["m" + p][valid]) / max(np.nansum(M["t" + p][valid]), 1e-9))
    out["C1 level"] = (all(_inside(v, LEVEL) for v in lv.values()), lv)
    # C2 flat: along each centreline
    fl = {}
    for p in ("A", "B"):
        u = ph["axes_local"][p].cpu().numpy(); t = np.arange(-c + 1, c - 1)
        rows, cols = c + t * u[1], c + t * u[2]
        tv = map_coordinates(np.nan_to_num(M["t" + p], nan=0), [rows, cols], order=1); mv = map_coordinates(np.nan_to_num(M["m" + p], nan=0), [rows, cols], order=1)
        fv = map_coordinates(np.nan_to_num(face, posinf=1e6), [rows, cols], order=1)
        okm = map_coordinates(np.isfinite(M["t" + p]).astype(float), [rows, cols], order=1) > 0.999   # every bilinear neighbour scored
        ok = (tv >= 0.5 * np.nanmax(M["t" + p])) & okm & (fv >= FACE_MARGIN)
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
    cols = [k for k in NAMES8 if k in rows[0][1]] + ["PASS"]
    fig, ax = plt.subplots(figsize=(3.0 * len(cols), 0.9 + 0.95 * len(rows))); ax.axis("off")
    cell = [[("PASS" if res["PASS"] else "FAIL") if k == "PASS" else "\n".join(f"{kk} {_fmt(vv)}" for kk, vv in res[k][1].items()) for k in cols]
            for _, res in rows]
    colours = [[("#b7e4c7" if (res["PASS"] if k == "PASS" else res[k][0]) else "#f4a6a6") for k in cols] for _, res in rows]
    tb = ax.table(cellText=cell, cellColours=colours, rowLabels=[r[0] for r in rows], colLabels=cols, loc="center", cellLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(7); tb.scale(1, 3.6)
    ax.set_title(title + "\nthresholds: C1 0.90-1.10 | C2 0.85-1.15 | C3 >= 95 % | C4 <= 5 % | C5 0.90-1.10 | C6 med <= 10, p95 <= 20 deg | C7 0.90-1.10 | C8 log-kappa error <= 0.3 | C9 >= 95 %", fontsize=9)
    plt.savefig(out_png, dpi=90, bbox_inches="tight"); plt.close(fig)


def save_json(rows: list, path: str):
    json.dump([{"phantom": lab, **{k: (v if k == "PASS" else {"pass": v[0], **{kk: vv for kk, vv in v[1].items()}}) for k, v in res.items()}} for lab, res in rows],
              open(path, "w"), indent=1, default=float)


# ---- the same criteria on generator crossings (gen_crossings.py): zones from the targets, not from a known geometry ----
# Fixed 2026-10-26 before any generator results. A population's arm: its target >= half its peak there, the other
# crossing population and every other object population absent (< 5 % of the peaks); the core: both >= half their peaks,
# no other population; background: none. C2 "flat" becomes the core-over-arm ratio of model / target (no subtraction in
# the crossing); C5 "edges" splits each arm by the distance to its population's boundary (outer vs inner half). The
# thresholds are the phantoms'.

def generic_zones(tm: dict) -> dict:
    face = tm.get("_face", np.full(tm["tA"].shape, np.inf)); valid = face >= FACE_MARGIN
    pk = {p: float(np.percentile(tm["t" + p][valid & (tm["t" + p] > 0.02)], 95)) if (valid & (tm["t" + p] > 0.02)).sum() > 10 else 1.0 for p in ("A", "B")}
    lo = max(0.02, 0.05 * min(pk.values()))
    inn = {p: tm["t" + p] >= 0.5 * pk[p] for p in ("A", "B")}; out = {p: tm["t" + p] < max(0.02, 0.05 * pk[p]) for p in ("A", "B")}
    oab = tm["tO"] < lo
    return {"armA": inn["A"] & out["B"] & oab & valid, "armB": inn["B"] & out["A"] & oab & valid, "core": inn["A"] & inn["B"] & oab & valid,
            "bg": out["A"] & out["B"] & oab & valid, "inA": inn["A"] & valid, "inB": inn["B"] & valid, "outA": out["A"] & valid, "outB": out["B"] & valid,
            "oabs": oab & valid, "valid": valid}


def evaluate_generic(tm: dict, M: dict, ent: dict) -> dict:
    """The criteria on one generator crossing: ``tm`` the slice targets (gen_crossings.slice_targets), ``M`` the model's
    maps ``mA, mB, mO, mI`` (n x n), ``ent`` per grid point the entities' ``mass``, ``iso`` and world ``axis``."""
    from scipy.ndimage import distance_transform_edt
    n = tm["tA"].shape[0]; Z = generic_zones(tm); out = {}
    T = {"A": tm["tA"], "B": tm["tB"]}; Mm = {"A": M["mA"], "B": M["mB"]}
    r = {(p, z): _ratio(Mm[p], T[p], Z[z]) for p in ("A", "B") for z in ("arm" + p, "core")}
    out["C1 level"] = (all(_inside(v, LEVEL) for v in r.values()), {f"{p} {'arm' if z != 'core' else 'core'}": v for (p, z), v in r.items()})
    fl = {p: (r[(p, "core")] / r[(p, "arm" + p)] if r[(p, "arm" + p)] > 0.05 else float("nan")) for p in ("A", "B")}   # (undefined if the arm is empty)
    out["C2 flat"] = (all(_inside(v, FLAT) for v in fl.values()), {f"{p} core/arm": v for p, v in fl.items()})
    lvl = min(np.median(T["A"][Z["armA"]]), np.median(T["B"][Z["armB"]])); thr = max(0.02, lvl / 3)
    # populations, not entities: a generator population is several resolved entities per point (fascicles, fibres)
    cnt = (M["mA"] >= thr).astype(int) + (M["mB"] >= thr) + (M["mO"] >= thr)
    cz = {z: float((cnt[Z[z]] == k).mean()) for z, k in (("bg", 0), ("armA", 1), ("armB", 1), ("core", 2))}
    out["C3 count"] = (all(v >= COUNT for v in cz.values()), cz)
    v = Z["valid"]
    lk = {"A outside": float(Mm["A"][Z["outA"]].sum() / max(T["A"][v].sum(), 1e-9)), "B outside": float(Mm["B"][Z["outB"]].sum() / max(T["B"][v].sum(), 1e-9)),
          "other dirs": float(M["mO"][Z["oabs"]].sum() / max((T["A"] + T["B"])[Z["oabs"]].sum(), 1e-9))}
    out["C4 leakage"] = (all(x <= LEAK for x in lk.values()), lk)
    ed = {}
    for p in ("A", "B"):
        d = distance_transform_edt(Z["in" + p] | ~Z["valid"]); arm = Z["arm" + p]
        if arm.sum() < 10:
            ed[p] = float("nan"); continue
        md = np.median(d[arm]); inner, outer = arm & (d > md), arm & (d <= md)
        ri = Mm[p][inner].sum() / max(T[p][inner].sum(), 1e-9); ro = Mm[p][outer].sum() / max(T[p][outer].sum(), 1e-9)
        ed[p] = float(ro / max(ri, 1e-9))
    out["C5 edges"] = (all(_inside(x, EDGE) for x in ed.values()), ed)
    dr = {}
    for p in ("A", "B"):
        u, ok = tm["axes"][p]; u = u.cpu().numpy(); ok = ok.cpu().numpy().reshape(n, n)
        cos = np.abs((ent["axis"] * u[:, None, :]).sum(-1)) * ((ent["mass"] >= thr) & ~ent["iso"])
        ang = np.degrees(np.arccos(np.clip(cos.max(1), 0, 1))).reshape(n, n)
        sel = (Z["arm" + p] | Z["core"]) & ok
        dr[p] = (float(np.median(ang[sel])), float(np.percentile(ang[sel], 95))) if sel.any() else (float("nan"),) * 2
    out["C6 direction"] = (all(np.isfinite(m) and m <= DIR[0] and q <= DIR[1] for m, q in dr.values()), {f"{p} med/p95": x for p, x in dr.items()})
    bk = {"background": _ratio(M["mI"], tm["tI"], Z["bg"] & (tm["tI"] >= 0.05)),
          "inside": _ratio(M["mI"], tm["tI"], (Z["inA"] | Z["inB"]) & Z["oabs"] & (tm["tI"] >= 0.05))}             # (where there is background)
    out["C7 background"] = (all(_inside(x, BACK) for x in bk.values()), bk)
    out["PASS"] = all(x[0] for k, x in out.items())
    return out


# ---- composable phantoms (phantom2.py): any number of populations, fibres / fans / curves / sheets / blobs ---------
# Fixed 2026-10-26 23:30 ET before results, revised 2026-10-27 after an adversarial review (a perfect model -- the targets
# fed in as the model -- failed the first version): thresholds as above, plus C8 (the population's ODF shape: the matched
# entity's concentrations against the target's, median |log((1 + k_model) / (1 + k_target))| <= ODF_TOL for kappa1 and
# kappa2, on matched points) and C9 (match rate: each population found -- a model entity of its type within 30 deg, or
# isotropic-object mass above the count threshold -- on >= COUNT of its zone's points). Model entities are matched to
# the target populations present at each point (target mass > 0.02) by type (peaked: kappa2 >= 2; girdle: kappa2 < 2 <=
# kappa1, compared by its normal; isotropic: kappa1 < ISO_K, the training's oriented threshold) and angle (<= 30 deg,
# one-to-one, nearest first); further entities of the type within 30 deg of a matched population are merged into it
# (its mass; the split rate is reported); isotropic populations share the isotropic-object mass by their target shares.
# Leakage is the model's excess over the target outside each population, over max(target total, 0.1 per scored point)
# -- the same for empty phantoms; a zone missing from the slice is n/a, not a fail.
ODF_TOL = 0.3
ISO_K = 0.5
NAMES8 = NAMES + ("C8 ODF shape", "C9 match rate")


def _types(k1, k2, present=None):
    t = np.where(k1 < ISO_K, 2, np.where(k2 < 2.0, 1, 0))                                          # 0 peaked, 1 girdle, 2 isotropic
    return t if present is None else np.where(present, t, -1)


def match_p2(tgt: dict, mdl: dict, max_deg: float = 30.0) -> dict:
    """Per grid point: each target population's matched model mass, axis angle and concentrations. ``tgt``: ``mass (P, K)``
    (populations, background excluded), ``k (P, K, 2)``, ``R (P, K, 3, 3)``; ``mdl``: entities ``mass (P, C)``, ``k (P, C, 2)``,
    ``R (P, C, 3, 3)`` and ``iso (P,)`` the isotropic-object mass."""
    P, K = tgt["mass"].shape; C = mdl["mass"].shape[1]
    pres = tgt["mass"] > 0.02
    tt = _types(tgt["k"][..., 0], tgt["k"][..., 1], pres); mt = _types(mdl["k"][..., 0], mdl["k"][..., 1], mdl["mass"] > 0)
    tax = np.where((tt == 1)[..., None], tgt["R"][..., :, 0], tgt["R"][..., :, 2])                  # girdles: the normal (a-axis)
    max_ = np.where((mt == 1)[..., None], mdl["R"][..., :, 0], mdl["R"][..., :, 2])
    cos = np.abs(np.einsum("pkd,pcd->pkc", tax, max_)); ang = np.degrees(np.arccos(np.clip(cos, 0, 1)))
    same = (tt[:, :, None] == mt[:, None, :]) & (tt[:, :, None] >= 0) & (tt[:, :, None] < 2)
    cost = np.where(same, ang, np.inf)
    out = {"m": np.zeros((P, K)), "ang": np.full((P, K), 90.0), "k": np.zeros((P, K, 2)), "matched": np.zeros((P, K), bool),
           "split": np.zeros((P, K), bool), "used": np.zeros((P, C), bool)}
    c2 = cost.copy()
    for _ in range(min(K, C)):                                                                       # greedy one-to-one, nearest first
        flat = c2.reshape(P, -1); j = flat.argmin(1); v = flat[np.arange(P), j]
        sel = v <= max_deg
        if not sel.any():
            break
        pk, pc = j // C, j % C; pi = np.nonzero(sel)[0]
        out["m"][pi, pk[pi]] = mdl["mass"][pi, pc[pi]]; out["ang"][pi, pk[pi]] = v[pi]; out["k"][pi, pk[pi]] = mdl["k"][pi, pc[pi]]
        out["matched"][pi, pk[pi]] = True; out["used"][pi, pc[pi]] = True
        c2[pi, pk[pi], :] = np.inf; c2[pi, :, pc[pi]] = np.inf
    extra = (cost <= max_deg) & out["matched"][:, :, None] & ~out["used"][:, None, :]                 # splits: merged into the matched population
    for kk in range(K):
        e = extra[:, kk] & ~out["used"]
        out["m"][:, kk] += (mdl["mass"] * e).sum(1); out["split"][:, kk] |= e.any(1); out["used"] |= e
    iso_t = tt == 2; share = np.where(iso_t, tgt["mass"], 0.0); tot = share.sum(1, keepdims=True)
    out["m"] = np.where(iso_t, mdl["iso"][:, None] * share / np.maximum(tot, 1e-9), out["m"])       # isotropic populations by their shares
    out["iso_spur"] = mdl["iso"] * (tot[:, 0] <= 0)
    out["other"] = (mdl["mass"] * (~out["used"]) * (mt >= 0) * (mt < 2)).sum(1)
    out["ttype"] = tt
    return out


def _nan_ok(vals, lo_hi):
    f = [v for v in vals if np.isfinite(v)]
    return bool(f) and all(lo_hi[0] <= v <= lo_hi[1] for v in f)


def evaluate_p2(n: int, tgt: dict, mdl: dict, face: np.ndarray, valid_disc: np.ndarray) -> dict:
    """The criteria for one composable phantom on its slice grid (``n x n`` points, row-major). ``tgt`` adds ``bg (P,)``
    and ``kinds``; ``mdl`` adds ``bg (P,)`` (the model's background mass) and optionally ``bgm`` (the head's)."""
    from scipy.ndimage import distance_transform_edt
    K = tgt["mass"].shape[1]; m = match_p2(tgt, mdl)
    valid = (face.reshape(-1) >= FACE_MARGIN) & valid_disc.reshape(-1); nv = max(int(valid.sum()), 1)
    T = tgt["mass"]; pk = np.array([np.percentile(T[valid & (T[:, k] > 0.02), k], 95) if (valid & (T[:, k] > 0.02)).sum() > 10 else 1.0 for k in range(K)])
    inn = T >= 0.5 * pk[None]; out_ = T < np.maximum(0.02, 0.05 * pk[None])
    solo = np.stack([inn[:, k] & np.all(np.delete(out_, k, 1), 1) & valid for k in range(K)], 1) if K else np.zeros((n * n, 0), bool)
    bgz = (np.all(out_, 1) & valid) if K else valid
    lvl = [np.median(T[solo[:, k], k]) for k in range(K) if solo[:, k].sum() > 5]
    thr = max(0.02, min(lvl) / 3) if lvl else 0.05
    res = {}; lv, fl, ed, dr, od, mr, lk = {}, {}, {}, {}, {}, {}, {}
    D = max(T[valid].sum(), 0.1 * nv)
    for k in range(K):
        nm = f"{k + 1}{tgt['kinds'][k][0]}"
        ov = (inn[:, k] & np.any(np.delete(inn, k, 1), 1) & valid) if K > 1 else np.zeros(n * n, bool)
        rs = _ratio(m["m"][:, k], T[:, k], solo[:, k]) if solo[:, k].sum() > 20 else float("nan")
        ro = _ratio(m["m"][:, k], T[:, k], ov) if ov.sum() > 20 else float("nan")
        lv[f"{nm} solo"] = rs; lv[f"{nm} overlap"] = ro
        sup = valid & (T[:, k] > 0.02)                                                              # where the population is present
        lv[f"{nm} plane"] = float(m["m"][sup, k].sum() / max(T[sup, k].sum(), 1e-9))               # (halos are the leakage terms)
        if np.isfinite(rs) and np.isfinite(ro):
            fl[nm] = ro / max(rs, 1e-9)
        lk[nm] = float(np.maximum(m["m"][out_[:, k] & valid, k] - T[out_[:, k] & valid, k], 0).sum() / max(max(T[valid, k].sum(), 0.1 * nv), 1e-9))
        d = distance_transform_edt((inn[:, k] | ~valid).reshape(n, n)).reshape(-1); arm = solo[:, k]
        if arm.sum() >= 10:
            md = np.median(d[arm]); a_in, a_out = arm & (d > md), arm & (d <= md)
            ed[nm] = float((m["m"][a_out, k].sum() / max(T[a_out, k].sum(), 1e-9)) / max(m["m"][a_in, k].sum() / max(T[a_in, k].sum(), 1e-9), 1e-9))
        z = (solo[:, k] | ov) & (T[:, k] > 0.02)
        if not z.any():
            continue
        oriented = np.median(m["ttype"][z, k]) < 2
        if oriented:
            hit = z & m["matched"][:, k]
            mr[nm] = float(hit.sum() / z.sum())
            if hit.sum() > 5:
                dr[nm] = (float(np.median(m["ang"][hit, k])), float(np.percentile(m["ang"][hit, k], 95)))
                e1 = np.abs(np.log1p(m["k"][hit, k, 0]) - np.log1p(tgt["k"][hit, k, 0])); e2 = np.abs(np.log1p(m["k"][hit, k, 1]) - np.log1p(tgt["k"][hit, k, 1]))
                od[nm] = (float(np.median(e1)), float(np.median(e2)))
            lk[f"{nm} split"] = float(m["split"][z, k].mean())                                      # reported, not thresholded
        else:                                                                                        # isotropic: found where the model has at
            mr[nm] = float((m["m"][z, k] >= 0.5 * T[z, k]).mean())                                   # least half its target mass
    lk["other dirs"] = float(m["other"][valid].sum() / D); lk["iso spurious"] = float(m["iso_spur"][valid].sum() / D)
    res["C1 level"] = (_nan_ok(lv.values(), LEVEL), lv) if K else (True, {"(no populations)": float("nan")})
    res["C2 flat"] = (all(_inside(v, FLAT) for v in fl.values()), fl if fl else {"(no overlap)": float("nan")})
    tc = (T >= thr).sum(1) if K else np.zeros(n * n, int)
    clean = valid & (np.all((T >= 1.5 * thr) | (T <= 0.5 * thr), 1) if K else True)
    mc = ((mdl["mass"] >= thr) & (_types(mdl["k"][..., 0], mdl["k"][..., 1]) < 2)).sum(1) + (mdl["iso"] >= thr)
    agree = float((mc[clean] == tc[clean]).mean()) if clean.sum() else float("nan")
    res["C3 count"] = (np.isfinite(agree) and agree >= COUNT, {"agree": agree, "points": float(clean.sum())})
    res["C4 leakage"] = (all(v <= LEAK for kk, v in lk.items() if not kk.endswith("split")), lk)
    res["C5 edges"] = (all(_inside(v, EDGE) for v in ed.values()), ed if ed else {"(none)": float("nan")})
    res["C6 direction"] = (all(a <= DIR[0] and b <= DIR[1] for a, b in dr.values()), {f"{k} med/p95": v for k, v in dr.items()} or {"(none)": float("nan")})
    bgr = {"background": _ratio(mdl["bg"], tgt["bg"], bgz & (tgt["bg"] >= 0.05))}
    if K:
        bgr["inside"] = _ratio(mdl["bg"], tgt["bg"], np.any(inn, 1) & valid & (tgt["bg"] >= 0.05))
        isoz = valid & np.any(inn & (m["ttype"] == 2), 1)
        if "bgm" in mdl and isoz.sum() > 20:
            bgr["head in objects"] = _ratio(mdl["bgm"], tgt["bg"], isoz & (tgt["bg"] >= 0.05))
    res["C7 background"] = (_nan_ok(bgr.values(), BACK) or not any(np.isfinite(v) for v in bgr.values()), bgr)
    res["C8 ODF shape"] = (all(a <= ODF_TOL and b <= ODF_TOL for a, b in od.values()), {f"{k} k1/k2 err": v for k, v in od.items()} or {"(none)": float("nan")})
    res["C9 match rate"] = (all(v >= COUNT for v in mr.values()), mr or {"(none)": float("nan")})
    res["PASS"] = all(v[0] for kk, v in res.items())
    return res
