"""Scores against phantom truth at the sites: angular error by distance band, boundary-parallel share, N-population
confusion and matched errors, detector precision / recall."""
from __future__ import annotations

import itertools

import numpy as np

BANDS_MM = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, np.inf))


def band_name(lo, hi):
    return f"{lo:g}-{hi:g}mm" if np.isfinite(hi) else f">{lo:g}mm"


def sample_truth(tr: dict, z: np.ndarray, x: np.ndarray) -> dict:
    """Truth arrays ``(Z, X, ...)`` at the site grid ``z (nz,)``, ``x (nx,)`` (block voxel coordinates), flattened
    to ``(nz * nx, ...)`` in C order."""
    iz = np.clip(np.floor(np.asarray(z) + 0.5).astype(int), 0, next(iter(tr.values())).shape[0] - 1)
    ix = np.clip(np.floor(np.asarray(x) + 0.5).astype(int), 0, next(iter(tr.values())).shape[1] - 1)
    IZ, IX = np.meshgrid(iz, ix, indexing="ij")
    return {k: v[IZ.ravel(), IX.ravel()] for k, v in tr.items()}


def angle_err(est: np.ndarray, tru: np.ndarray) -> np.ndarray:
    """Axial angle (deg) between unit vectors ``(N, 3)``; NaN where the truth is NaN."""
    c = np.abs((est * tru).sum(-1))
    return np.degrees(np.arccos(np.clip(c, 0, 1)))


def _summ(e: np.ndarray) -> dict:
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"n": 0}
    return {"n": int(e.size), "median": float(np.median(e)), "p90": float(np.percentile(e, 90)),
            "mean": float(e.mean()), "under10": float((e < 10).mean()), "under30": float((e < 30).mean())}


def band_summary(err: np.ndarray, dist: np.ndarray, region: np.ndarray | None = None, wm_only: bool = True) -> dict:
    """Error summaries per distance band (plus ``all`` and ``nointerface``), over WM sites (region 2) if asked."""
    ok = np.isfinite(err)
    if wm_only and region is not None:
        ok &= region == 2
    out = {"all": _summ(err[ok])}
    out["nointerface"] = _summ(err[ok & ~np.isfinite(dist)])
    for lo, hi in BANDS_MM:
        sel = ok & (dist >= lo) & (dist < hi)
        out[band_name(lo, hi)] = _summ(err[sel])
    if region is not None:
        out["grey_matter"] = _summ(err[np.isfinite(err) & (region == 1)])
    return out


def parallel_share(est: np.ndarray, bnorm: np.ndarray, dist: np.ndarray, region: np.ndarray | None = None,
                   tangent_deg: float = 30.0) -> dict:
    """Share of sites whose direction lies within ``tangent_deg`` of the interface's tangent plane
    (angle to the normal > 90 - tangent_deg), per distance band. Pass the truth as ``est`` for the expected share."""
    ang_n = angle_err(est, bnorm)
    par = ang_n > (90.0 - tangent_deg)
    ok = np.isfinite(ang_n) & np.isfinite(dist)
    if region is not None:
        ok &= region == 2
    out = {}
    for lo, hi in BANDS_MM:
        sel = ok & (dist >= lo) & (dist < hi)
        out[band_name(lo, hi)] = {"n": int(sel.sum()), "share": float(par[sel].mean()) if sel.any() else float("nan")}
    return out


def detector_pr(s: np.ndarray, dist: np.ndarray, region: np.ndarray | None, positive_mm: float = 0.5,
                thresholds: np.ndarray | None = None, s0: float | None = None) -> dict:
    """Precision / recall of ``s > t`` for "within ``positive_mm`` of an interface" over WM sites, at a sweep of
    thresholds (AUC of the PR curve by trapezoid) and at ``s0`` if given."""
    ok = np.isfinite(s) & np.isfinite(dist)
    if region is not None:
        ok &= region == 2
    s, dist = s[ok], dist[ok]
    pos = dist < positive_mm
    if thresholds is None:
        thresholds = np.quantile(s, np.linspace(0.0, 0.999, 60))
    P, R = [], []
    for t in thresholds:
        det = s > t
        tp = (det & pos).sum()
        P.append(tp / max(det.sum(), 1)); R.append(tp / max(pos.sum(), 1))
    P, R = np.array(P), np.array(R)
    o = np.argsort(R)
    auc = float(np.trapezoid(P[o], R[o])) if len(R) > 1 else float("nan")
    out = {"pr_auc": auc, "n_pos": int(pos.sum()), "n": int(ok.sum())}
    if s0 is not None:
        det = s > s0
        tp = (det & pos).sum()
        out["at_s0"] = {"s0": float(s0), "precision": float(tp / max(det.sum(), 1)), "recall": float(tp / max(pos.sum(), 1)),
                        "detected_share": float(det.mean())}
    return out


def multi_scores(fib: np.ndarray, frac: np.ndarray, cnt: np.ndarray, d1: np.ndarray, d2: np.ndarray, npop: np.ndarray,
                 frac1: np.ndarray, cross_deg: np.ndarray | None = None, region: np.ndarray | None = None) -> dict:
    """``fib (N, k, 3)``, ``frac (N, k)``, ``cnt (N,)`` against the truth. Confusion over true N in {0, 1, 2};
    for true-2 sites the two strongest estimates are matched to the truth by the assignment with the least total
    angle; ``both_within_15`` counts sites with fewer than two estimates as failures."""
    ok = np.ones(len(cnt), bool) if region is None else (region == 2)
    out = {"n": int(ok.sum())}
    conf = np.zeros((3, 4), int)
    for t in range(3):
        for e in range(4):
            conf[t, e] = int(((npop == t) & (cnt == e) & ok).sum())
    out["confusion_true_x_est"] = conf.tolist()
    tot = conf.sum(1, keepdims=True).clip(1)
    out["n_correct_share"] = {str(t): float(conf[t, min(t, 3)] / tot[t, 0]) for t in range(3) if conf[t].sum()}
    # true single population
    s1 = ok & (npop == 1)
    if s1.any():
        e = angle_err(fib[s1, 0], d1[s1])
        e[cnt[s1] == 0] = np.nan
        out["single"] = {**_summ(e), "false_second_share": float((cnt[s1] >= 2).mean()), "missed_share": float((cnt[s1] == 0).mean())}
    s0 = ok & (npop == 0)
    if s0.any():
        out["null"] = {"false_any_share": float((cnt[s0] >= 1).mean()), "n": int(s0.sum())}
    s2 = ok & (npop == 2)
    if s2.any():
        idx = np.flatnonzero(s2)
        e1 = np.full(len(idx), np.nan); e2 = np.full(len(idx), np.nan); fe = np.full(len(idx), np.nan)
        for j, i in enumerate(idx):
            if cnt[i] < 2:
                continue
            a = angle_err(fib[i, :2], np.stack([d1[i], d1[i]]))
            b = angle_err(fib[i, :2], np.stack([d2[i], d2[i]]))
            # assignment 0->d1, 1->d2 or 0->d2, 1->d1
            if a[0] + b[1] <= a[1] + b[0]:
                e1[j], e2[j] = a[0], b[1]; fe[j] = abs(frac[i, 0] - frac1[i])
            else:
                e1[j], e2[j] = b[0], a[1]; fe[j] = abs(frac[i, 1] - frac1[i])
        both = np.isfinite(e1) & (e1 < 15) & (e2 < 15)
        out["two"] = {"n": int(len(idx)), "found2_share": float((cnt[idx] >= 2).mean()),
                      "err1": _summ(e1), "err2": _summ(e2), "both_within_15_share": float(both.mean()),
                      "frac_abs_err": _summ(fe)}
        if cross_deg is not None:
            cd = cross_deg[idx]
            by = {}
            for lo, hi in ((0, 20), (20, 35), (35, 50), (50, 70), (70, 91)):
                sel = (cd >= lo) & (cd < hi)
                if sel.sum() >= 20:
                    by[f"{lo}-{hi}"] = {"n": int(sel.sum()), "found2_share": float((cnt[idx][sel] >= 2).mean()),
                                        "both_within_15_share": float(both[sel].mean()),
                                        "err_median": float(np.nanmedian(np.r_[e1[sel], e2[sel]])) if np.isfinite(e1[sel]).any() else float("nan")}
            out["two"]["by_cross_deg"] = by
    return out
