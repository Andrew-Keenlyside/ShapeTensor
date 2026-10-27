"""Paper figure: how the generator makes a training example (the training pipeline itself, :func:`scene.draw_scene`).

One scene -- two fibre bundles crossing, beside other populations, at a voxel size where some fibres are below the voxel
-- shown on the plane of its crossing: (a) its populations, (b) the true fibre direction, (c) the rendered image with two
query windows, (d) the training target at those windows (the population density along the in-plane directions, as the
loss sees it: mass x Bingham, concentration capped at the direction basis's); (e-h) the same crop through four
acquisitions (X-ray; a heavy point-spread function; MRI-like contrast; voxels twice as large).

Usage: python scripts/learn/figure_generator.py OUT_STEM --search 24     (a contact sheet of candidate scenes)
       python scripts/learn/figure_generator.py OUT_STEM --seed S        (the figure, OUT_STEM.pdf / .png)
"""
import argparse, math, sys
import numpy as np
import torch
import torch.nn.functional as Fn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import scene as SC, generator as G, bingham as BG, entity_train as ET

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--search", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--voxel", type=float, default=4.0); ap.add_argument("--half", type=int, default=48); ap.add_argument("--sigma", type=float, default=4.0)
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
CANVAS, H = 192, a.half; N = 2 * H; NZ = {1: 136, 2: 96}                                        # crops: the tilted plane fits (136 > 96 sqrt 2)
FAM = {k: 0.0 for k in SC.Scene.FAMILIES}; FAM.update({"bundle": 1.0, "tree": 0.6, "cells": 0.4})
BASE = dict(canvas=CANVAS, crops_per_scene=1, voxel_um=a.voxel, families=FAM, n_groups=(2, 3), layout_p=0.0, underlay_p=0.0, cross_p=1.0,
            p_sparse=0.0, p_cutout=0.0, p_boundary=0.0, p_warp=0.0, p_affine=0.0, p_merge=0.0, p_scramble=0.0, p_blend=0.0, p_aniso=0.0, entity_ctx=True)
BLUE, ORANGE, AQUA, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"


def scene_of(seed):
    """The cached scene of ``draw_scene``'s first call (same key) and its crossing: point, the two bundles' groups and axes."""
    sc, _, _, _ = SC.render_scene((CANVAS,) * 3, 7_000_000 + seed, dev, a.voxel, FAM, (2, 3), 0.0, "general", None, 0.0, 1.0, False, 0.0)
    gi_b = next((i for i, g in enumerate(sc.groups) if getattr(g, "cross", None) is not None), None)
    if gi_b is None:
        return sc, None
    P0, t0, d = sc.groups[gi_b].cross
    gi_a = max([i for i in range(gi_b) if sc.groups[i].name == "bundle"], default=None)
    e1 = Fn.normalize(t0, dim=0); e2 = Fn.normalize(d - (d @ e1) * e1, dim=0); P0 = P0.float()
    g = torch.arange(N, device=dev, dtype=torch.float32) - H + 0.5; U, V = torch.meshgrid(g, g, indexing="ij")
    best = None                                                                                     # the view's centre: the crossing, shifted
    for du in range(-20, 21, 4):                                                                    # (at most 20 voxels) so the plane stays
        for dv in range(-20, 21, 4):                                                                # inside both crops
            V0 = P0 + du * e1 + dv * e2; Pp = V0 + U[..., None] * e1 + V[..., None] * e2
            o1 = (V0 - NZ[1] / 2).round().clamp(0, CANVAS - NZ[1])
            bad = float((~(((Pp >= o1) & (Pp <= o1 + NZ[1] - 1)).all(-1) & ((Pp >= 0) & (Pp <= CANVAS - 1)).all(-1))).float().mean())
            key = (round(bad, 3), du * du + dv * dv)
            if best is None or key < best[0]:
                best = (key, V0)
    return sc, dict(P0=P0, V0=best[1], e1=e1, e2=e2, n=torch.linalg.cross(e1, e2), A=gi_a, B=gi_b,
                    deg=math.degrees(math.acos(min(1.0, abs(float(e1 @ Fn.normalize(d, dim=0)))))))


def draw(seed, X, zoom=1, **kw):
    ext = NZ[zoom] * zoom
    o = [int(min(max(round(float(v) - ext / 2), 0), CANVAS - ext)) for v in X.get("V0", X["P0"])]
    s, d = SC.draw_scene((NZ[zoom],) * 3, seed=seed, device=dev, zoom=zoom, crop=tuple(o), scene_seed=seed, **{**BASE, **kw})
    to_crop = lambda p: (p - torch.tensor(o, device=dev, dtype=torch.float32)) / zoom
    return s, d, to_crop(X.get("V0", X["P0"])), to_crop


def plane(vol, c, X, zoom=1, mode="bilinear"):
    """``vol (C, Z, Y, X)`` on the crossing's plane through ``c`` (crop voxels): (C, N, N), one output pixel per finest voxel;
    NaN beyond the crop."""
    g = (torch.arange(N, device=dev, dtype=torch.float32) - H + 0.5) / zoom
    U, V = torch.meshgrid(g, g, indexing="ij")
    P = c + U[..., None] * X["e1"] + V[..., None] * X["e2"]                                           # (N, N, 3) zyx
    sz = torch.tensor(vol.shape[1:], device=dev, dtype=torch.float32)
    grid = (2 * (P + 0.5) / sz - 1).flip(-1)[None, None]                                               # (1, 1, N, N, 3) xyz in [-1, 1]
    out = Fn.grid_sample(vol[None].float(), grid, mode=mode, align_corners=False)[0, :, 0]
    inside = ((P >= 0) & (P <= sz - 1)).all(-1)
    return torch.where(inside[None], out, torch.full_like(out, float("nan")))


def window(x, lo=1, hi=99):
    q = np.nanpercentile(x, [lo, hi]); return np.clip((x - q[0]) / max(q[1] - q[0], 1e-6), 0, 1)


def roles(s, d, X):
    """Each layer's colour: the crossing's two bundles blue / orange, one other population aqua, the rest grey."""
    gl = d["entity_ctx"]["group_layer"]; col = {}
    if X["A"] is not None and gl[X["A"]] >= 0:
        col[gl[X["A"]]] = BLUE
    col[gl[X["B"]]] = ORANGE
    rest = [k for k, L in enumerate(s.layers) if k not in col and L.role not in ("isotropic", "interface") and float(d["occupancy"][k].sum()) > 50]
    rest.sort(key=lambda k: -float(d["occupancy"][k].sum()))
    for j, k in enumerate(rest):
        col[k] = AQUA if j == 0 else GREY
    return col


def rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


if a.search:
    fig, axs = plt.subplots(3, (a.search + 2) // 3, figsize=(2.2 * ((a.search + 2) // 3), 6.6))
    axs = np.atleast_2d(axs)
    for k in range(a.search):
        ax = axs[k % 3, k // 3]; ax.axis("off")
        sc, X = scene_of(k)
        if X is None or X["A"] is None:
            ax.set_title(f"seed {k}: no crossing", fontsize=7); SC._scene_cache().clear(); continue
        s, d, c, _ = draw(k, X, modality="hipct", strength=0.3, acq_seed=1)
        img = plane(s.image.float()[None], c, X)[0].cpu().numpy()
        bun = [f["bundle"] for f in s.record.get("families", []) if "bundle" in f]
        tag = " ".join(("u" + (f"t{b.get('texture', 0):.1f}" if b.get("texture") else "s")) if b.get("unresolved") else "r" for b in bun)
        ax.imshow(window(img), cmap="gray"); ax.set_title(f"seed {k}: {X['deg']:.0f} deg, {tag}", fontsize=7)
        print(k, round(X["deg"]), tag, [g.name for g in sc.groups], flush=True)
        SC._scene_cache().clear(); del s, d; torch.cuda.empty_cache()                               # (one 192^3 scene at a time)
    plt.tight_layout(); plt.savefig(a.out + "_search.png", dpi=90); print("saved", a.out + "_search.png"); sys.exit()

# ---- the figure ----------------------------------------------------------------------------------------------------
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "Arial", "DejaVu Sans"], "font.size": 7,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                     "pdf.fonttype": 42, "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
sc, X = scene_of(a.seed); assert X is not None and X["A"] is not None, "no crossing of two bundles in this scene"
s, d, c, to_crop = draw(a.seed, X, modality="hipct", strength=0.3, acq_seed=1)
col = roles(s, d, X)
occ = d["occupancy"].float(); T6 = d["tensors"].float()
occ_p = np.nan_to_num(plane(occ, c, X).cpu().numpy())                                                                # (K, N, N)
over = np.ones((N, N, 3))                                                                                 # (a) populations on white
for k, h in sorted(col.items(), key=lambda kv: kv[1] == GREY, reverse=True):
    al = np.clip(occ_p[k], 0, 1)[..., None] * 0.9; over = over * (1 - al) + rgb(h) * al
line_k = [k for k in col if s.layers[k].role not in ("isotropic", "interface")]                          # (b) the fibre direction
T = torch.einsum("kzyx,kczyx->czyx", occ[line_k], T6[line_k]); Tp = torch.nan_to_num(plane(T, c, X))                      # (6, N, N) occupancy-weighted
M = torch.stack([torch.stack([Tp[0], Tp[3], Tp[4]], -1), torch.stack([Tp[3], Tp[1], Tp[5]], -1), torch.stack([Tp[4], Tp[5], Tp[2]], -1)], -2)
ev, V = torch.linalg.eigh(M + 1e-9 * torch.eye(3, device=dev)); v = V[..., -1]
dec = torch.stack([(v @ X["e1"]).abs(), (v @ X["e2"]).abs(), (v @ X["n"]).abs()], -1).cpu().numpy()
bright = np.clip(occ_p[line_k].sum(0), 0, 1)[..., None]
dec_img = dec * bright
clean = window(plane(s.image.float()[None], c, X)[0].cpu().numpy())

# (d) the training target at two query windows: on bundle A's arm and at the crossing
qx = to_crop(X["P0"]); qa = qx - 0.6 * H * X["e1"]                                                     # bundle 1's other arm, and the crossing
pts = torch.stack([qa, qx]); sig = torch.full((2,), a.sigma, device=dev)
t = TR.entity_queries(s, d, NZ[1], 0, dev, points=pts, sigma=sig)
cl = sc.groups[X["A"]].centreline.P[0]; Tc = Fn.normalize(torch.gradient(cl, dim=0)[0], dim=-1)    # bundle 1's own direction at each
q_can = torch.stack([X["P0"] - 0.6 * H * X["e1"], X["P0"]])                                           # query (canvas): angles from it
t1 = Tc[torch.cdist(q_can, cl).argmin(1)]; phi1 = torch.atan2(t1 @ X["e2"], t1 @ X["e1"])            # (bundle 1 curves: ~25 deg at 1)
th = torch.deg2rad(torch.arange(0, 180.5, 0.5, device=dev))
U = torch.cos(th[None] + phi1[:, None])[..., None] * X["e1"] + torch.sin(th[None] + phi1[:, None])[..., None] * X["e2"]   # (2, D, 3)
m = (t["e_mass"].float() * t["e_vis"].float()).clamp_min(0) * (t["e_node"] >= 0)
kap = t["e_kappa"].float().clamp(max=ET.DENS_KAPPA_MAX)
dens = np.stack([BG.mixture_pdf(U[q], t["e_R"][q].float(), kap[q], m[q]).cpu().numpy() for q in range(2)])   # (2, D)
b2_deg = (X["deg"] - math.degrees(float(phi1[1]))) % 180                                               # bundle 2 from bundle 1 at the crossing

# (e-h) acquisitions of the same crop
acq = [("X-ray phase contrast", dict(modality="hipct", strength=0.25, acq_seed=11), 1, {}),
       ("Heavy point-spread function", dict(modality="hipct", strength=1.0, acq_seed=11), 1, {"heavy_psf": 1.0}),
       ("MRI-like contrast", dict(modality="mri", strength=0.5, acq_seed=12), 1, {}),
       (f"Voxels {2 * a.voxel:g} µm", dict(modality="hipct", strength=0.5, acq_seed=11), 2, {})]
acq_img = []
for title, kw, zoom, patch in acq:
    saved = {k: G.MODALITIES["hipct"].get(k) for k in patch}
    G.MODALITIES["hipct"].update(patch)
    try:
        s2, d2, c2, _ = draw(a.seed, X, zoom=zoom, **kw)
    finally:
        for k, v in saved.items():
            (G.MODALITIES["hipct"].pop(k) if v is None else G.MODALITIES["hipct"].__setitem__(k, v))
    acq_img.append(window(plane(d2["image"].float()[None], c2, X, zoom, mode="nearest" if zoom > 1 else "bilinear")[0].cpu().numpy()))
    psf = d2["record"].get("psf"); print(title, "psf", psf, "res", d2["record"].get("resolution"), "voxel", d2["record"].get("voxel_um"), flush=True)

# ---- layout: 183 mm wide (double column) ---------------------------------------------------------------------------------
mm = 1 / 25.4
fig = plt.figure(figsize=(183 * mm, 98 * mm))
gs = fig.add_gridspec(2, 4, left=0.02, right=0.985, top=0.93, bottom=0.04, wspace=0.12, hspace=0.32)
um = a.voxel


def img_ax(r_, c_, im, title, letter, cmap=None, bar=True):
    ax = fig.add_subplot(gs[r_, c_]); ax.imshow(im, cmap=cmap, vmin=0, vmax=1, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=7, pad=3, loc="left")
    ax.text(-0.02, 1.13, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="right")
    if bar:                                                                                       # 100 um scale bar, black or white by
        L = 100 / um; y0 = N - 6; x0 = N - 6 - L                                                 # what lies under it
        under = np.nanmean(im[int(y0 - 4):int(y0 + 3), int(x0):int(x0 + L)]) if im.ndim == 2 else np.nanmean(im[int(y0 - 4):int(y0 + 3), int(x0):int(x0 + L)])
        ax.add_patch(Rectangle((x0, y0 - 1.2), L, 2.4, color="white" if (np.isfinite(under) and under < 0.45) else INK, lw=0))
    return ax


ax = img_ax(0, 0, over, "Populations", "a")
labels = [("Bundle 1", BLUE), ("Bundle 2", ORANGE)] + ([(sc.groups[[i for i, g in enumerate(sc.groups) if d["entity_ctx"]["group_layer"][i] == k][0]].name.capitalize(), AQUA)
                                                         for k in col if col[k] == AQUA][:1])
for j, (lb, h) in enumerate(labels):
    ax.add_patch(Rectangle((3, 3 + 7 * j), 4, 4, color=h, lw=0)); ax.text(9, 5.2 + 7 * j, lb, fontsize=6, va="center", color=INK)
ax = img_ax(0, 1, dec_img, "True fibre direction", "b")
ins = ax.inset_axes([0.74, 0.74, 0.24, 0.24]); ins.set_xlim(-1.1, 1.1); ins.set_ylim(-0.15, 1.1); ins.axis("off")
for k_ in range(36):                                                                              # the direction key: in-plane angle -> colour
    t0 = 5 * k_; cc = (abs(math.sin(math.radians(t0 + 2.5))), abs(math.cos(math.radians(t0 + 2.5))), 0)   # (rows: e1, red; columns: e2, green)
    ins.add_patch(Wedge((0, 0), 1.0, t0, t0 + 5, width=0.45, color=cc, lw=0))
ax = img_ax(0, 2, clean, "Rendered image", "c", cmap="gray")
for j, (q, nm) in enumerate(((qa, "1"), (qx, "2"))):                                             # the query windows (2 sigma)
    uv = [float((q - c) @ X["e1"]) + H - 0.5, float((q - c) @ X["e2"]) + H - 0.5]
    ax.add_patch(Circle((uv[1], uv[0]), 2 * a.sigma, fill=False, ec=INK, lw=0.8)); ax.text(uv[1] + 2 * a.sigma + 1.0, uv[0] - 2 * a.sigma + 1, nm, color=INK, fontsize=6.5, fontweight="bold")
ax = fig.add_subplot(gs[0, 3])
deg = np.rad2deg(th.cpu().numpy())
ax.plot(deg, dens[0] / dens.max(), color=INK2, lw=1.0, ls=(0, (3, 1.5)), label="1  one bundle")
ax.plot(deg, dens[1] / dens.max(), color=INK, lw=1.0, label="2  crossing")
for ang_, h in ((0.0, BLUE), (b2_deg, ORANGE)):
    ax.plot([ang_, ang_], [1.04, 1.12], color=h, lw=1.6, solid_capstyle="butt", clip_on=False)
ax.set_xlim(0, 180); ax.set_ylim(0, 1.12); ax.set_xticks([0, 45, 90, 135, 180]); ax.set_yticks([0, 0.5, 1])
ax.set_xlabel("Direction from bundle 1 (°)", labelpad=1.5); ax.set_ylabel("Target density (a.u.)", labelpad=1.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=6, loc="upper right", bbox_to_anchor=(1.02, 0.98), handlelength=1.8)
ax.set_title(f"Training target (window σ = {a.sigma * um:g} µm)", fontsize=7, pad=3, loc="left")
ax.text(-0.25, 1.13, "d", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="right")
for j, ((title, _, zoom, _), im) in enumerate(zip(acq, acq_img)):
    img_ax(1, j, im, title, "efgh"[j], cmap="gray")
fig.savefig(a.out + ".pdf"); fig.savefig(a.out + ".png", dpi=600)
print("saved", a.out + ".pdf", a.out + ".png", "| crossing", round(X["deg"], 1), "deg | layers", {k: v for k, v in col.items()},
      "| bundles", [f["bundle"] for f in s.record.get("families", []) if "bundle" in f])
