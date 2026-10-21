"""Stage 2 as fibre sets (prototypes, not in the docs): the current populations against connected sets and smooth
direction fields (``shapetensor.experimental.fibre_sets``), on the bend phantom, its straight control and the 60°
crossing at both texture settings, three seeds each.

    python scripts/prototype_fibre_sets.py [OUT_DIR]     (default /home/andrew/bridge_wmcov/shapetensor/prototypes/fibre_sets)

Everything as the cortical BRIDGE study ran it: σ 1, ρ 10, truncate 3, engine anatomix_enc2, calibrated on the
validation's twelve straight bundles of the phantom's texture setting. The tissue is the intensity mask (a global Otsu
threshold, BRIDGE's rule) on the bend and its control, and the whole block on the crossings (white matter only).
The nulls of the prototypes come from texture without fibres: the evidence's null, and the smallest population a
prototype may keep — the largest set it finds in that texture, so no set size is chosen by hand. Stages 3–4 are the
pipeline's for every method.

On the band crossings (two fascicles crossing in grey-matter-level tissue, intensity mask): per band, the share of
its activation (outside the overlap) carried by its largest population, whether that population is the same on both
sides of the overlap, whether the two bands are carried by different populations; the merged field's error outside
the overlap; in the overlap the validation's crossing scores (the first two vectors against the two directions).

Scores: on the bend, per fascicle, the share of its activation carried by its largest population, whether one
population dominates both arms, and the number of populations holding at least 10% of it; the merged field's error
on the arms and in the turn, and how far the merged field and the multi-vector field's first vector turn per 0.1 mm
there; on the straight control the same; on the crossings the validation's scores (``validate.score``).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.use_deterministic_algorithms(True, warn_only=True)               # narrows the readout fit's run-to-run variation
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

from shapetensor import st
from shapetensor.experimental import fibre_sets as FS
from shapetensor.phantoms import BEND_GRID, BEND_RADII, BEND_WIDTH, Grid, TextureParams, build, finish
from shapetensor.pipeline import apply, calibrate
from shapetensor.validate import ANGLES, SETTINGS, score

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/prototypes/fibre_sets")
if __import__("os").environ.get("QUICK") == "1":
    OUT = OUT / "quick"
elif __import__("os").environ.get("MEDIUM") == "1":
    OUT = OUT / "medium"
DEV = "cuda"
SIGMA, RHO, TRUNC = 1.0, 10.0, 3.0
import os
QUICK = os.environ.get("QUICK") == "1"                 # one seed, the bend and the 90° band crossing: a check before the full run
SEEDS = (0,) if (QUICK or os.environ.get("MEDIUM") == "1") else (0, 1, 2)   # MEDIUM: every case, one seed
STRIDE, VOX = 2, 0.02
METHODS = ("current", "connected sets", "smooth fields", "hybrid", "local", "local continuation", "windowed continuation", "windowed stage 2",
           "windowed stage 2, relative", "windowed sets", "windowed sets, relative", "windowed sets 2", "windowed sets 2, relative",
           "windowed sets 3", "windowed sets 3, relative", "windowed sets, pruned", "windowed sets, pruned, relative",
           "windowed sets, split, relative", "windowed sets, deconvolved, relative")
if os.environ.get("FOCUS") == "1":                     # only the current method and the windowed methods (the rest is measured)
    METHODS = ("current", "windowed sets, relative", "windowed sets, pruned", "windowed sets, pruned, relative")
elif os.environ.get("FOCUS") == "2":                   # the current method against the settled windowed variant only
    METHODS = ("current", "windowed sets, pruned", "windowed sets, pruned, relative")
elif os.environ.get("FOCUS") == "3":                   # + the crossing split, + the deconvolved window densities
    METHODS = ("current", "windowed sets, pruned, relative", "windowed sets, split, relative", "windowed sets, deconvolved, relative")
CASES = (("bend", "calibrated", {}), ("straight", "calibrated", {"straight": True}),
         ("crossing 90", "coherent", {"cross_deg": 90.0}), ("crossing 90", "calibrated", {"cross_deg": 90.0}),
         ("crossing 60", "coherent", {"cross_deg": 60.0}), ("crossing 60", "calibrated", {"cross_deg": 60.0}),
         ("band crossing 90", "coherent", {"cross_deg": 90.0}), ("band crossing 90", "calibrated", {"cross_deg": 90.0}),
         ("band crossing 60", "coherent", {"cross_deg": 60.0}), ("band crossing 60", "calibrated", {"cross_deg": 60.0}))
if QUICK:
    CASES = tuple(c for c in CASES if (c[0], c[1]) in (("bend", "calibrated"), ("straight", "calibrated"), ("band crossing 90", "coherent"),
                                                      ("band crossing 60", "coherent")))


def tex_for(setting):
    p = SETTINGS[setting]
    return TextureParams(dispersion_deg=p["dispersion_deg"], pitch_deg=p["pitch_deg"]), p["texture"]


def otsu(v: np.ndarray) -> float:
    h = np.bincount(v.reshape(-1), minlength=256).astype(np.float64); p = h / h.sum(); x = np.arange(256)
    w0 = np.cumsum(p); m0 = np.cumsum(p * x)
    return float(np.argmax((m0[-1] * w0 - m0) ** 2 / np.maximum(w0 * (1 - w0), 1e-12))) + 0.5


def null_volume(tex, texture, seed=7, grid=None):
    """Texture without fibres, at grey-matter level, on the validation's grid (or ``grid``)."""
    g = grid or Grid(); rng = np.random.default_rng([seed, 4242])
    t = TextureParams(**{**tex.__dict__, "amp": tex.amp * texture})
    Z, X = g.shape[0], g.shape[2]
    img = finish(g, t, np.full((Z, X), t.levels["gm"], np.float32), np.zeros((Z, X), np.float32), torch.zeros(g.shape, device=DEV), rng)
    return torch.as_tensor(img, device=DEV).float()


# ---------------------------------------------------------------------------------------------------------------
# the scores
# ---------------------------------------------------------------------------------------------------------------

def inplane_deg(d):
    return np.degrees(np.arctan2(d[0], d[2])) % 180.0


def axial_err(a, b):
    e = np.abs(a - b) % 180.0
    return np.minimum(e, 180.0 - e)


def turning(field, tr, k, grid):
    """Axial angle between a field at a voxel and 5 voxels on along the fibres (0.1 mm), over fascicle ``k``'s turn."""
    zs, ys, xs = grid.inner
    fid, s, arc = tr["fid"], tr["s"], tr["bend"]
    tan = np.stack([tr["dir1"][..., 0], tr["dir1"][..., 2]], -1)
    core = np.zeros(fid.shape, bool); core[zs, xs] = True
    zz, xx = np.nonzero(arc & (fid == k) & core)
    t = tan[zz, xx]
    z2 = np.clip(np.rint(zz + 5 * t[:, 0]).astype(int), 0, fid.shape[0] - 1); x2 = np.clip(np.rint(xx + 5 * t[:, 1]).astype(int), 0, fid.shape[1] - 1)
    ok = (fid[z2, x2] == k) & core[z2, x2]
    F = np.moveaxis(field[:, :, ys, :], 0, -1)
    a, b = F[zz[ok], :, xx[ok]], F[z2[ok], :, x2[ok]]
    c = np.abs((a * b).sum(-1)) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-9)
    return np.degrees(np.arccos(np.clip(c, 0, 1))).reshape(-1)


def score_bend(out, tr, grid):
    zs, ys, xs = grid.inner
    fid, s, arc = tr["fid"], tr["s"], tr["bend"]
    truth = (np.degrees(np.arctan2(tr["dir1"][..., 0], tr["dir1"][..., 2])) % 180.0)
    core = np.zeros(fid.shape, bool); core[zs, xs] = True
    A = out["activation"].float()
    Ay = A[:, :, ys, :].mean(2).cpu().numpy()                                               # (N, Z, X)
    vec = out["vec"].cpu().numpy(); first = out["vecs"][0].cpu().numpy()
    ang = inplane_deg(vec[:, :, ys, :])
    res = {"n_populations": int(A.shape[0]), "fascicles": {}}
    allf = (fid > 0) & core
    tot_all = Ay[:, allf].sum(1)
    res["populations carrying the fascicles"] = int((tot_all >= 0.05 * max(tot_all.sum(), 1e-12)).sum()) if len(tot_all) else 0
    for k, r0 in enumerate(BEND_RADII, 1):
        rm = r0 + BEND_WIDTH / 2
        m = (fid == k) & core
        tot = Ay[:, m].sum(1)
        share = tot / max(tot.sum(), 1e-12)
        row = {"largest_share": round(float(share.max()) if len(share) else 0.0, 3),
               "populations_over_10pct": int((share >= 0.10).sum())}
        parts = {"straight": m & ~arc} if not arc.any() else {"way in": m & ~arc & (s < 0), "turn": m & arc, "way out": m & ~arc & (s > 0)}
        for name, mm in parts.items():
            m3 = np.broadcast_to(mm[:, None, :], ang.shape)
            row[f"error {name}"] = round(float(np.median(axial_err(ang[m3], np.broadcast_to(truth[:, None, :], ang.shape)[m3]))), 1)
        if arc.any():
            dom_in = int(Ay[:, parts["way in"]].mean(1).argmax()); dom_out = int(Ay[:, parts["way out"]].mean(1).argmax())
            row["one population on both arms"] = bool(dom_in == dom_out)
            row["turning p95 merged"] = round(float(np.percentile(turning(vec, tr, k, grid), 95)), 1)
            row["turning p95 first vector"] = round(float(np.percentile(turning(first, tr, k, grid), 95)), 1)
            row["true turning"] = round(math.degrees(0.1 / rm), 1)
        res["fascicles"][f"r {rm:.2f}"] = row
    return res, Ay


def score_band(out, tr, grid):
    """Two fascicles crossing: one population per band, kept on both sides of the overlap, different for the two
    bands; the merged error outside the overlap; the first two vectors in it."""
    zs, ys, xs = grid.inner
    core = np.zeros(tr["band1"].shape, bool); core[zs, xs] = True
    b1, b2 = tr["band1"] & core, tr["band2"] & core
    ov = b1 & b2
    A = out["activation"].float()
    Ay = A[:, :, ys, :].mean(2).cpu().numpy()
    Zc, Xc = np.meshgrid(grid.coords(0), grid.coords(2), indexing="ij")
    zc, xc = grid.core_mm[0] / 2, grid.core_mm[2] / 2
    vec = out["vec"].cpu().numpy()[:, :, ys, :]
    V = out["vecs"].cpu().numpy()[:, :, :, ys, :]
    res = {"n_populations": int(A.shape[0])}
    tot_all = Ay[:, (b1 | b2)].sum(1)
    res["populations carrying the bands"] = int((tot_all >= 0.05 * max(tot_all.sum(), 1e-12)).sum()) if len(tot_all) else 0
    dom = []
    for k, (b, other) in enumerate(((b1, b2), (b2, b1)), 1):
        d = tr["dir1"][b & ~other][0]
        t = (Zc - zc) * d[0] + (Xc - xc) * d[2]                                         # position along the band
        only = b & ~other
        tot = Ay[:, only].sum(1)
        share = tot / max(tot.sum(), 1e-12)
        side = [int(Ay[:, only & (t < 0)].mean(1).argmax()) if A.shape[0] else -1, int(Ay[:, only & (t > 0)].mean(1).argmax()) if A.shape[0] else -1]
        dom.append(int(tot.argmax()) if A.shape[0] else -1)
        m3 = np.broadcast_to(only[:, None, :], vec.shape[1:])
        t3 = np.moveaxis(np.broadcast_to(tr["dir1"][:, None, :, :], vec.shape[1:] + (3,)), -1, 0)
        c = np.abs((vec * t3).sum(0))
        res[f"band {k}"] = {"largest share": round(float(share.max()) if len(share) else 0.0, 3), "same population both sides": side[0] == side[1],
                            "merged error outside the overlap": round(float(np.median(np.degrees(np.arccos(np.clip(c[m3], 0, 1))))), 1)}
    res["different populations for the two bands"] = bool(len(set(dom)) == 2)
    if V.shape[0] >= 2:
        t1 = np.moveaxis(np.broadcast_to(tr["dir1"][:, None, :, :], vec.shape[1:] + (3,)), -1, 0)
        t2 = np.moveaxis(np.broadcast_to(np.nan_to_num(tr["dir2"])[:, None, :, :], vec.shape[1:] + (3,)), -1, 0)
        ang = lambda a, b: np.degrees(np.arccos(np.clip(np.abs((a * b).sum(0)), 0, 1)))
        e11, e12, e21, e22 = ang(V[0], t1), ang(V[0], t2), ang(V[1], t1), ang(V[1], t2)
        sw = (e12 + e21) < (e11 + e22)
        ea, eb = np.where(sw, e12, e11), np.where(sw, e21, e22)
        m3 = np.broadcast_to(ov[:, None, :], ea.shape)
        res["overlap pair error"] = round(float(np.median(((ea + eb) / 2)[m3])), 1)
        res["overlap both within 15"] = round(float(((ea <= 15) & (eb <= 15))[m3].mean()), 3)
    else:
        res["overlap pair error"] = None; res["overlap both within 15"] = 0.0
    return res, Ay


def score_crossing(out, v, tr, grid, setting):
    f = {"image": v, "structure tensor": st.structure_tensor(v, sigma=SIGMA, rho=RHO, truncate=TRUNC, excise=False, order="zyx", device=DEV),
         "excised structure tensor": st.structure_tensor(v, sigma=SIGMA, rho=RHO, truncate=TRUNC, excise=True, order="zyx", device=DEV),
         "shape tensor": out["vec"], "shape tensor, multi (first)": out["vecs"][0], "shape tensor, multi": out["vecs"],
         "n_populations": int(out["activation"].shape[0])}
    r = score("crossing", tr, f, grid.inner, grid.vox)
    pair = r.get("shape tensor, multi (pair)")
    return {"n_populations": r["n_populations"],
            "merged error to nearest": r["shape tensor (nearest population)"]["median_deg"],
            "first vector error to nearest": r["shape tensor, multi (first) (nearest population)"]["median_deg"],
            "pair error": None if pair is None else pair["median_deg"], "both within 15": None if pair is None else pair["both_within_15"],
            "excised structure tensor to nearest": r["excised structure tensor (nearest population)"]["median_deg"]}


# ---------------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------------

CMAPS = ("Blues", "Oranges", "Greens", "Purples", "Reds", "Greys")
CAT = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#eda100", "#e87ba4", "#008300")


def fig_layers(rows, image, tr, grid, path, title, n_show=4):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    zs, _, xs = grid.inner
    ext = (0, grid.core_mm[2], 0, grid.core_mm[0])
    fid = tr["fid"][zs, xs] if "fid" in tr else (tr["band1"][zs, xs] * 1 + tr["band2"][zs, xs] * 2 if "band1" in tr else None)
    cols = 2 + n_show
    fig, axs = plt.subplots(len(rows), cols, figsize=(3.6 * cols, 3.4 * len(rows)), dpi=100, squeeze=False, constrained_layout=True)
    for i, (name, Ay, n) in enumerate(rows):
        ax = axs[i, 0]; ax.imshow(image, cmap="gray", origin="lower", extent=ext); ax.set_title(f"{name}: {n} population{'s' if n != 1 else ''}", fontsize=10, loc="left")
        Ac = Ay[:, zs, xs]
        top = max(float(np.percentile(Ac, 99.5)) if Ac.size else 1.0, 1e-6)
        order = np.argsort(-Ac.reshape(Ac.shape[0], -1).sum(1)) if Ac.shape[0] else []
        for j in range(n_show):
            ax = axs[i, 1 + j]
            if j < len(order):
                k = order[j]
                ax.imshow(Ac[k], cmap=CMAPS[j % len(CMAPS)], vmin=0, vmax=top, origin="lower", extent=ext)
                ax.set_title(f"population {k + 1} (largest {j + 1})", fontsize=9, color=CAT[j % len(CAT)])
            else:
                ax.axis("off")
            if fid is not None and j < len(order):
                masks = [fid == f for f in range(1, int(fid.max()) + 1)] if "band1" not in tr else [tr["band1"][zs, xs], tr["band2"][zs, xs]]
                for mk in masks:
                    ax.contour(mk, [0.5], colors="#9a9890", linewidths=0.5, extent=ext, origin="lower")
        ax = axs[i, -1]
        if Ac.shape[0]:
            rank = np.empty(Ac.shape[0], int); rank[order] = np.arange(Ac.shape[0])
            strong = np.where(Ac.max(0) > 0.5 * np.median(Ac[Ac > 0]) if (Ac > 0).any() else Ac.max(0) > 0, rank[Ac.argmax(0)] + 1, 0)
            cmap = ListedColormap([(0.12, 0.12, 0.12)] + [CAT[j % len(CAT)] for j in range(max(1, Ac.shape[0]))])
            ax.imshow(strong, cmap=cmap, vmin=-0.5, vmax=Ac.shape[0] + 0.5, origin="lower", extent=ext, interpolation="nearest")
        ax.set_title("strongest population", fontsize=9)
        for a in axs[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.savefig(path); plt.close(fig)


def fig_profiles(runs, tr, grid, path, k=2):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zs, ys, xs = grid.inner
    fid, s, arc = tr["fid"], tr["s"], tr["bend"]
    truth = np.degrees(np.arctan2(tr["dir1"][..., 0], tr["dir1"][..., 2])) % 180.0
    core = np.zeros(fid.shape, bool); core[zs, xs] = True
    m2 = (fid == k) & core
    edges = np.arange(np.nanmin(s[m2]), np.nanmax(s[m2]) + 0.05, 0.05); mid = 0.5 * (edges[1:] + edges[:-1])
    b = np.digitize(s[m2], edges) - 1
    cnt = np.bincount(b, minlength=len(mid))[: len(mid)]; ok = cnt > 0
    rm = BEND_RADII[k - 1] + BEND_WIDTH / 2

    def binned(field):
        a2 = np.radians(inplane_deg(field)) * 2
        C_ = np.bincount(b, weights=np.cos(a2).mean(1)[m2], minlength=len(mid))[: len(mid)]
        S_ = np.bincount(b, weights=np.sin(a2).mean(1)[m2], minlength=len(mid))[: len(mid)]
        return (np.degrees(np.arctan2(S_, C_) / 2) + 45.0) % 180.0 - 45.0

    tv = np.stack([np.sin(np.radians(truth)), np.zeros_like(truth), np.cos(np.radians(truth))])[:, :, None, :]
    fig, axs = plt.subplots(2, len(runs), figsize=(5.2 * len(runs), 7.6), dpi=110, sharex=True, constrained_layout=True)
    for c, (name, Ay, vec, exc) in enumerate(runs):
        ax = axs[0, c]
        tots = np.array([np.bincount(b, weights=Ay[j][m2], minlength=len(mid))[: len(mid)].sum() for j in range(Ay.shape[0])])
        for rank, j in enumerate(np.argsort(-tots)[:6]):
            act = np.bincount(b, weights=Ay[j][m2], minlength=len(mid))[: len(mid)] / np.maximum(cnt, 1)
            ax.plot(mid[ok], act[ok], color=CAT[rank % len(CAT)], lw=2, label=f"population {j + 1}")
        ax.axvspan(0, rm * math.pi / 2, color="#e8e6df", zorder=0)
        ax.set_title(f"{name}", loc="left", fontsize=11); ax.set_ylabel("activation (nulls), y and width mean")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax = axs[1, c]
        for field, colour, style, lab in ((tv, "#0b0b0b", ":", "fibres' tangent"), (exc, "#2a78d6", "-", "excised structure tensor"),
                                         (vec, "#eb6834", "-", "merged field")):
            yv = binned(field[:, :, ys, :] if field.shape[2] != 1 else field)
            ax.plot(mid[ok], yv[ok], color=colour, ls=style, lw=2, label=lab)
        ax.axvspan(0, rm * math.pi / 2, color="#e8e6df", zorder=0)
        ax.set_ylim(-45, 135); ax.set_yticks([-45, 0, 45, 90, 135])
        ax.set_xlabel("position along the fascicle (mm; grey = the turn)"); ax.set_ylabel("in-plane direction (° from x)")
        if c == 0:
            ax.legend(fontsize=8, frameon=False, loc="lower left")
        for a_ in axs[:, c]:
            for sp in ("top", "right"):
                a_.spines[sp].set_visible(False)
            a_.grid(alpha=0.25, lw=0.6)
    fig.suptitle(f"The middle fascicle (turn radius {rm:.2f} mm), seed {SEEDS[0]}: each population's activation and the merged direction",
                 x=0.01, ha="left", fontsize=12)
    fig.savefig(path); plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(*a):
        print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)

    cals, nulls = {}, {}
    for setting in ("calibrated", "coherent"):
        tex, texture = tex_for(setting); cg = Grid()
        vols = [torch.as_tensor(build("bundle", 100 + k, cg, tex, texture=texture, angle=float(a), device=DEV)[0], device=DEV).float()
                for k, a in enumerate(ANGLES)]
        torch.manual_seed(0)                                                              # the calibration's sampling draws
        cal = calibrate(vols, sigma=SIGMA, rho=RHO, truncate=TRUNC, margin=cg.margin[0], device=DEV)
        second_null = FS.peak_null(cal, vols, cg.inner) if "windowed continuation" in METHODS else 0.0   # one bundle's tallest second peak
        share_null = FS.second_share_null(cal, vols[::3], cg.inner)                       # one bundle's largest second population
        from shapetensor.phantoms import direction as _direction
        responses = globals().setdefault("RESPONSES", {})
        responses[setting] = FS.readout_response(cal, vols, [_direction(float(a)) for a in ANGLES], cg.inner)   # the readout's one-population response
        del vols
        nv = null_volume(tex, texture)
        from shapetensor.pipeline import apply as _a  # noqa: F401
        U0 = FS.lift(cal, cal.features(nv), stride=STRIDE)
        u = cal.readout.grid
        nulls[setting] = {"mass": FS.null_level(cal, nv, stride=STRIDE), "pi": FS.field_null(U0, u)}
        # the smallest population each prototype may keep: the largest set it finds where there are no fibres
        t0_ = torch.ones(U0.shape[1:], dtype=torch.bool, device=DEV)
        tn = torch.ones_like(nv, dtype=torch.bool)
        o0 = apply(cal, nv, tn, device=DEV); o0["activation"] = o0["activation"] / max(cal.tau, 1e-12)
        if "connected sets" in METHODS:
            _, r1 = FS.connected_sets(U0, u, t0_, nulls[setting]["mass"], vox_mm=VOX, stride=STRIDE, min_volume_mm3=0.0, layers=False)
            nulls[setting]["min_volume_sets"] = max([0.0] + r1["kept_volumes_mm3"][:1]) + 1e-6
        if "smooth fields" in METHODS:
            _, r2 = FS.smooth_fields(U0, u, t0_, nulls[setting]["pi"], vox_mm=VOX, stride=STRIDE, min_volume_mm3=0.0, layers=False)
            nulls[setting]["min_volume_fields"] = max([0.0] + [v_ for _, v_ in r2["sets"]]) + 1e-6
        if "hybrid" in METHODS:
            _, r3 = FS.hybrid(o0, tn, vox_mm=VOX)
            nulls[setting]["min_volume_hybrid"] = max([0.0] + r3["volumes_mm3"][:1]) + 1e-6
        if "local" in METHODS:
            _, r4 = FS.local_populations(cal, nv, tn, cal.features(nv), region=Grid().inner, vox_mm=VOX)
            nulls[setting]["min_volume_local"] = max([0.0] + r4["volumes_mm3"][:1]) + 1e-6
        # the coherence a population's tensor reaches on fibre-free texture: presence needs more
        nulls[setting]["gap"] = float(o0["gap"][:, 16:-16, 16:-16, 16:-16].reshape(-1)[::5].quantile(0.95))
        del o0
        # the share of a window a population covers on fibre-free texture: a window population must cover more
        if "local continuation" in METHODS:
            _, r5 = FS.local_continuation(cal, nv, tn, cal.features(nv), gap_null=nulls[setting]["gap"], region=Grid().inner, vox_mm=VOX,
                                          min_share=0.0, layers=False)
            nulls[setting]["window_share"] = max([0.0] + r5["shares"][:1])
        # how far a window's orientation density stands out on fibre-free texture: a window with fibres must exceed it
        Dn, _, _ = FS.window_densities(cal, cal.dictionary.prep(cal.features(nv)), Grid().inner, 32, 8)
        nulls[setting]["prominence"] = float(FS.prominence(Dn).max()); del Dn
        # the same on texture the size of the bend and band phantoms (the largest of more windows is larger)
        nb = null_volume(tex, texture, grid=BEND_GRID)
        Dn, _, _ = FS.window_densities(cal, cal.dictionary.prep(cal.features(nb)), BEND_GRID.inner, 32, 8)
        nulls[setting]["prominence_bend"] = float(FS.prominence(Dn).max()); del Dn, nb
        nulls[setting]["second_peak"] = second_null
        nulls[setting]["second_share"] = share_null
        cals[setting] = cal
        del U0, nv; torch.cuda.empty_cache()
        log(f"{setting}: calibrated (tau {cal.tau:.3f}, overlap {cal.readout_overlap:.2f}); nulls {json.dumps({k: round(v, 5) for k, v in nulls[setting].items()})}")

    results = {}
    figs = {}
    for case, setting, kw in CASES:
        label = f"{case} ({setting})"
        cal = cals[setting]; u = cal.readout.grid
        tex, texture = tex_for(setting)
        name = "bend" if case in ("bend", "straight") else ("band_crossing" if case.startswith("band") else "crossing")
        for seed in SEEDS:
            img, tr, meta = build(name, seed, tex=tex, texture=texture, device=DEV, **kw)
            grid = BEND_GRID if name in ("bend", "band_crossing") else Grid()
            v = torch.as_tensor(img, device=DEV).float()
            tissue = (v > otsu(img)) if name in ("bend", "band_crossing") else torch.ones_like(v, dtype=torch.bool)
            F = cal.features(v)
            U = FS.lift(cal, F, stride=STRIDE) if ("connected sets" in METHODS or "smooth fields" in METHODS) else None
            ts = tissue[::STRIDE, ::STRIDE, ::STRIDE]
            outs, reports = {}, {}
            o = apply(cal, v, tissue, features=F, device=DEV)
            o["activation"] = o["activation"] / max(cal.tau, 1e-12)
            pnull = nulls[setting]["prominence_bend" if grid is BEND_GRID else "prominence"]
            outs["current"] = o; reports["current"] = {"modes_deg_from_x": [round(float(a), 1) for a in inplane_deg(o["modes"].cpu().numpy().T)]}
            if "connected sets" in METHODS:
                A, rep = FS.connected_sets(U, u, ts, nulls[setting]["mass"], vox_mm=VOX, stride=STRIDE, full_shape=tuple(v.shape),
                                           min_volume_mm3=nulls[setting]["min_volume_sets"])
                outs["connected sets"] = FS.fields_from_layers(cal, v, tissue, A); reports["connected sets"] = rep
            if "smooth fields" in METHODS:
                A, rep = FS.smooth_fields(U, u, ts, nulls[setting]["pi"], vox_mm=VOX, stride=STRIDE, full_shape=tuple(v.shape),
                                          min_volume_mm3=nulls[setting]["min_volume_fields"])
                outs["smooth fields"] = FS.fields_from_layers(cal, v, tissue, A); reports["smooth fields"] = rep
            # no set-size floor for the two: on fibre-free texture the pipeline's own layers exceed its null over most of the block
            # (the null is set on the calibration bundles), so a noise-calibrated floor exceeds a fascicle's volume
            if "hybrid" in METHODS:
                A, rep = FS.hybrid(o, tissue, vox_mm=VOX, min_volume_mm3=0.0)
                outs["hybrid"] = FS.fields_from_layers(cal, v, tissue, A); reports["hybrid"] = rep
            if "local" in METHODS:
                t1 = time.time()
                A, rep = FS.local_populations(cal, v, tissue, F, region=grid.inner, vox_mm=VOX, min_volume_mm3=0.0)
                rep["seconds"] = round(time.time() - t1, 1)
                outs["local"] = FS.fields_from_layers(cal, v, tissue, A); reports["local"] = rep
            if "local continuation" in METHODS:
                t1 = time.time()
                A, rep = FS.local_continuation(cal, v, tissue, F, gap_null=nulls[setting]["gap"], region=grid.inner, vox_mm=VOX,
                                               min_share=nulls[setting]["window_share"])
                rep["seconds"] = round(time.time() - t1, 1)
                outs["local continuation"] = FS.fields_from_layers(cal, v, tissue, A); reports["local continuation"] = rep
            if "windowed continuation" in METHODS:
                t1 = time.time()
                A, rep = FS.windowed_continuation(cal, v, tissue, F, prominence_null=pnull, second_null=nulls[setting]["second_peak"],
                                                  region=grid.inner, vox_mm=VOX)
                rep["seconds"] = round(time.time() - t1, 1)
                outs["windowed continuation"] = FS.fields_from_layers(cal, v, tissue, A); reports["windowed continuation"] = rep
            A = None
            if "windowed stage 2" in METHODS or "windowed stage 2, relative" in METHODS:
                t1 = time.time()
                A, rep = FS.windowed_stage2(cal, v, tissue, F, prominence_null=pnull, share_null=nulls[setting]["second_share"],
                                            region=grid.inner, vox_mm=VOX)
                rep["seconds"] = round(time.time() - t1, 1)
                outs["windowed stage 2"] = FS.fields_from_layers(cal, v, tissue, A); reports["windowed stage 2"] = rep
                outs["windowed stage 2, relative"] = FS.fields_from_layers(cal, v, tissue, A, relative=True); reports["windowed stage 2, relative"] = rep
            if "windowed sets" in METHODS or "windowed sets, relative" in METHODS:
                t1 = time.time()
                A, rep = FS.windowed_sets(cal, v, tissue, F, prominence_null=pnull, share_null=nulls[setting]["second_share"],
                                          region=grid.inner, vox_mm=VOX)
                rep["seconds"] = round(time.time() - t1, 1)
                outs["windowed sets"] = FS.fields_from_layers(cal, v, tissue, A); reports["windowed sets"] = rep
                outs["windowed sets, relative"] = FS.fields_from_layers(cal, v, tissue, A, relative=True); reports["windowed sets, relative"] = rep
            if "windowed sets, pruned" in METHODS or "windowed sets, pruned, relative" in METHODS:
                t1 = time.time()
                A, rep = FS.windowed_sets(cal, v, tissue, F, prominence_null=pnull, share_null=nulls[setting]["second_share"],
                                          region=grid.inner, vox_mm=VOX, min_extent=32.0)
                rep["seconds"] = round(time.time() - t1, 1)
                outs["windowed sets, pruned"] = FS.fields_from_layers(cal, v, tissue, A); reports["windowed sets, pruned"] = rep
                outs["windowed sets, pruned, relative"] = FS.fields_from_layers(cal, v, tissue, A, relative=True); reports["windowed sets, pruned, relative"] = rep
            if "windowed sets, split, relative" in METHODS:
                t1 = time.time()
                A, rep = FS.windowed_sets(cal, v, tissue, F, prominence_null=pnull, share_null=nulls[setting]["second_share"],
                                          region=grid.inner, vox_mm=VOX, min_extent=32.0, split=True)
                rep["seconds"] = round(time.time() - t1, 1)
                outs["windowed sets, split, relative"] = FS.fields_from_layers(cal, v, tissue, A, relative=True); reports["windowed sets, split, relative"] = rep
            if "windowed sets, deconvolved, relative" in METHODS:
                t1 = time.time()
                A, rep = FS.windowed_sets(cal, v, tissue, F, prominence_null=pnull, share_null=nulls[setting]["second_share"],
                                          region=grid.inner, vox_mm=VOX, min_extent=32.0, split=True, response=RESPONSES[setting])
                rep["seconds"] = round(time.time() - t1, 1)
                outs["windowed sets, deconvolved, relative"] = FS.fields_from_layers(cal, v, tissue, A, relative=True); reports["windowed sets, deconvolved, relative"] = rep
            if "windowed sets 3" in METHODS or "windowed sets 3, relative" in METHODS:
                t1 = time.time()
                A, rep = FS.windowed_sets(cal, v, tissue, F, prominence_null=pnull, share_null=nulls[setting]["second_share"],
                                          region=grid.inner, vox_mm=VOX, prefer_alone=True, reach_windows=3)
                rep["seconds"] = round(time.time() - t1, 1)
                outs["windowed sets 3"] = FS.fields_from_layers(cal, v, tissue, A); reports["windowed sets 3"] = rep
                outs["windowed sets 3, relative"] = FS.fields_from_layers(cal, v, tissue, A, relative=True); reports["windowed sets 3, relative"] = rep
            if "windowed sets 2" in METHODS or "windowed sets 2, relative" in METHODS:
                t1 = time.time()
                A, rep = FS.windowed_sets2(cal, v, tissue, F, prominence_null=pnull, share_null=nulls[setting]["second_share"],
                                           region=grid.inner, vox_mm=VOX)
                rep["seconds"] = round(time.time() - t1, 1)
                outs["windowed sets 2"] = FS.fields_from_layers(cal, v, tissue, A); reports["windowed sets 2"] = rep
                outs["windowed sets 2, relative"] = FS.fields_from_layers(cal, v, tissue, A, relative=True); reports["windowed sets 2, relative"] = rep
            del F, U, A
            for m in METHODS:
                if name == "band_crossing":
                    r, Ay = score_band(outs[m], tr, grid)
                    if seed == SEEDS[0]:
                        figs.setdefault(label, {})[m] = (Ay, int(outs[m]["activation"].shape[0]), None)
                elif name == "bend":
                    r, Ay = score_bend(outs[m], tr, grid)
                    if seed == SEEDS[0]:
                        figs.setdefault(label, {})[m] = (Ay, int(outs[m]["activation"].shape[0]),
                                                          outs[m]["vec"].cpu().numpy())
                else:
                    r = score_crossing(outs[m], v, tr, grid, setting)
                    if seed == SEEDS[0]:
                        zs, ys, xs = grid.inner
                        figs.setdefault(label, {})[m] = (outs[m]["activation"][:, :, ys, :].mean(2).cpu().numpy(), int(outs[m]["activation"].shape[0]), None)
                r["report"] = reports[m]
                results.setdefault(label, {}).setdefault(m, []).append(r)
            if seed == SEEDS[0]:
                y = grid.shape[1] // 2
                figs[label]["_image"] = img[:, y, :][grid.inner[0], grid.inner[2]]
                figs[label]["_truth"] = tr
                figs[label]["_grid"] = grid
                if name == "bend":
                    figs[label]["_exc"] = st.structure_tensor(v, sigma=SIGMA, rho=RHO, truncate=TRUNC, excise=True, mask=tissue,
                                                               order="zyx", device=DEV).cpu().numpy()
            log(f"{label} seed {seed}: " + "; ".join(f"{m} N {results[label][m][-1]['n_populations']}" for m in METHODS))
            for m_ in ("local continuation", "windowed continuation", "windowed stage 2", "windowed sets", "windowed sets 2", "windowed sets, pruned",
                       "windowed sets, pruned, relative", "windowed sets, split, relative", "windowed sets, deconvolved, relative"):
                if m_ in reports:
                    log(f"   {m_}: {json.dumps(reports[m_])}")
            del outs, v, tissue; torch.cuda.empty_cache()

    # figures (seed 0)
    for label, f in figs.items():
        grid = f["_grid"]; tr = f["_truth"]
        rows = [(m, f[m][0], f[m][1]) for m in METHODS]
        tag = label.replace(" ", "_").replace("(", "").replace(")", "")
        fig_layers(rows, f["_image"], tr, grid, OUT / f"layers_{tag}.png",
                   f"{label}, seed {SEEDS[0]}: activation layers (in nulls, mean over y) and the strongest population, per method")
        if label.startswith("bend"):
            fig_profiles([(m, f[m][0], f[m][2], f["_exc"]) for m in METHODS], tr, grid, OUT / "profiles_bend.png")

    # summary
    summary = {}
    for label, per in results.items():
        summary[label] = {}
        for m, rows in per.items():
            s = {"n_populations": [r["n_populations"] for r in rows]}
            if "populations carrying the fascicles" in rows[0]:
                s["populations carrying the fascicles"] = [r["populations carrying the fascicles"] for r in rows]
            if "band 1" in rows[0]:
                s["populations carrying the bands"] = [r["populations carrying the bands"] for r in rows]
                s["bands each one population on both sides"] = f"{sum(r[b]['same population both sides'] for r in rows for b in ('band 1', 'band 2'))}/{2 * len(rows)}"
                s["bands carried by different populations"] = f"{sum(r['different populations for the two bands'] for r in rows)}/{len(rows)}"
                s["largest share per band"] = round(float(np.median([r[b]["largest share"] for r in rows for b in ("band 1", "band 2")])), 3)
                s["merged error outside the overlap"] = round(float(np.median([r[b]["merged error outside the overlap"] for r in rows for b in ("band 1", "band 2")])), 1)
                vals = [r["overlap pair error"] for r in rows if r["overlap pair error"] is not None]
                s["overlap pair error"] = round(float(np.median(vals)), 1) if vals else None
                s["overlap both within 15"] = round(float(np.median([r["overlap both within 15"] for r in rows])), 3)
            elif "fascicles" in rows[0]:
                keys = [k for k in rows[0]["fascicles"][next(iter(rows[0]["fascicles"]))] if k != "one population on both arms"]
                for k in keys:
                    vals = [fr[k] for r in rows for fr in r["fascicles"].values()]
                    s[k] = round(float(np.median(vals)), 3)
                if "one population on both arms" in rows[0]["fascicles"][next(iter(rows[0]["fascicles"]))]:
                    vals = [fr["one population on both arms"] for r in rows for fr in r["fascicles"].values()]
                    s["fascicles with one population on both arms"] = f"{sum(vals)}/{len(vals)}"
            else:
                for k in ("merged error to nearest", "first vector error to nearest", "pair error", "both within 15", "excised structure tensor to nearest"):
                    vals = [r[k] for r in rows if r[k] is not None]
                    s[k] = round(float(np.median(vals)), 3) if vals else None
            summary[label][m] = s
            log(f"{label} | {m}: {json.dumps(s)}")
    (OUT / "prototype_fibre_sets.json").write_text(json.dumps({"settings": {"sigma": SIGMA, "rho": RHO, "truncate": TRUNC, "stride": STRIDE,
                                                                            "seeds": SEEDS, "nulls": nulls},
                                                               "summary": summary, "runs": results}, indent=1, default=float) + "\n")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
