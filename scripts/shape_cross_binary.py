"""PROTOTYPE: a simple binary rod phantom for the two-fibre TCA test. Dark rods (radius 1.5 vox, linear partial volume at
the edge) on a flat background (180 vs 80, noise 2), 0.1 mm voxels, 128^3. Rods of one population lie on a regular 10 x
10 voxel lattice (in y and in the in-plane normal); in a crossing both populations share the y planes, so their rods
intersect (max of the two rod maps).

Rows: single populations at 90, 30 and 60 deg (to the z = const plane, in x-z), crossings 90 + 30 and 60 + 120 (neither
on a grid axis). Read-outs on the mid-y x-z slice, central 64 x 64, rho 8 vox:
  intensity ST    e3 of the image's structure tensor
  shape eigen     e3 of the 16-D anatomix shape tensor (whole block, 8 grid phases)
  TCA             CP of (space x channel x gradient axis): rank 2 -> one fibre (d1 x d2) for single populations;
                  rank 4 paired by signature + spatial-map similarity -> two fibres for crossings; and the oracle pairing

    python shape_cross_binary.py     (shapetensors env, one GPU)
"""
import json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import anatomix_shape as AS
import shape_tca as T
import shape_tensor as ST

N, VOX = 128, 0.1
C0, W = 32, 64
JY = N // 2
RHO = 8.0
BG, ROD, NOISE, R, S = 180.0, 80.0, 2.0, 1.5, 10.0
ROWS = (("single 90°", (90.0,)), ("single 30°", (30.0,)), ("single 60°", (60.0,)),
        ("crossing 90° + 30°", (90.0, 30.0)), ("crossing 60° + 120°", (60.0, 120.0)))
ROOT = Path("/home/andrew/bridge_wmcov/shape_cross_binary")


def unit(th):
    a = np.radians(th); return np.array([np.cos(a), 0.0, np.sin(a)])


def rods(th, phase=0.0):
    """Rod occupancy (0-1) of one population along unit(th): axes on a lattice of spacing S in y and in the in-plane
    normal; linear partial volume over one voxel at the radius."""
    x, y, z = np.meshgrid(*(np.arange(N, dtype=np.float32) + 0.5,) * 3, indexing="ij")
    u = unit(th); nrm = np.array([-u[2], 0.0, u[0]])
    sn = x * nrm[0] + z * nrm[2] + phase
    dn = (sn % S) - S / 2; dy = (y % S) - S / 2
    return np.clip(R + 0.5 - np.sqrt(dn ** 2 + dy ** 2), 0, 1)


def make(rng):
    vols = {}
    for name, ths in ROWS:
        occ = np.max(np.stack([rods(t, 3.3 * k) for k, t in enumerate(ths)]), 0)
        v = BG - (BG - ROD) * occ + rng.normal(0, NOISE, occ.shape)
        vols[name] = np.clip(np.rint(v), 0, 255).astype(np.float32)
    return vols


def ang(u, t):
    return np.degrees(np.arccos(np.clip(np.abs(u @ t), 0, 1)))


def pair_err(Up, tru):
    e11, e22 = ang(Up[:, 0], tru[0]), ang(Up[:, 1], tru[1]); e12, e21 = ang(Up[:, 0], tru[1]), ang(Up[:, 1], tru[0])
    swap = (e12 + e21) < (e11 + e22)
    return np.where(swap[:, None], np.stack([e21, e12], 1), np.stack([e11, e22], 1)), np.where(swap[:, None, None], Up[:, ::-1], Up)


def analyse(v, tru):
    F = AS.anatomix_clean(v, VOX); out = {}
    out["intensity ST"] = T.null_dir(T.k_slice(v[None], RHO, JY)[C0:C0 + W, C0:C0 + W].reshape(-1, 3, 3).double()).float().cpu().numpy()
    K = T.k_slice(F, RHO, JY)[C0:C0 + W, C0:C0 + W]
    Kf = K.reshape(-1, 48, 48).double(); Kf = Kf / Kf.diagonal(dim1=1, dim2=2).sum(1)[:, None, None]
    T3 = Kf.reshape(-1, 16, 3, 16, 3).diagonal(dim1=1, dim2=3).sum(-1)
    out["shape eigen"] = T.null_dir(T3).float().cpu().numpy()
    eye = torch.eye(48, device=T.DEV, dtype=Kf.dtype)
    L, info = torch.linalg.cholesky_ex(Kf + 1e-9 * eye)
    if (info > 0).any():
        L[info > 0] = torch.linalg.cholesky(Kf[info > 0] + 1e-6 * eye)
    Y = L.mT.float().reshape(-1, 48, 16, 3)
    Cg = Kf.reshape(-1, 16, 3, 16, 3).diagonal(dim1=2, dim2=4).sum(-1)
    init = (torch.linalg.eigh(Cg)[1].flip(-1).float(), torch.linalg.eigh(T3)[1].flip(-1).float())
    R_ = 2 * len(tru)
    A, B, D, err = T.cp_als(Y, R_, init, starts=4, iters=200)
    A, B, D = (x.double().cpu().numpy() for x in (A, B, D))
    if len(tru) == 1:
        c = np.cross(D[:, :, 0], D[:, :, 1]); out["TCA"] = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    else:
        na = np.linalg.norm(A, axis=1)
        sa = np.abs(np.einsum("npr,nps->nrs", A, A)) / np.maximum(na[:, :, None] * na[:, None, :], 1e-30)
        sb = np.abs(np.einsum("ncr,ncs->nrs", B, B))
        pairings = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
        U = np.zeros((len(A), 3, 2, 3)); score = np.zeros((len(A), 3)); geo = np.full((len(A), 3), np.inf)
        for j, prs in enumerate(pairings):
            for q, (r, s) in enumerate(prs):
                c = np.cross(D[:, :, r], D[:, :, s]); nc = np.linalg.norm(c, axis=1)
                U[:, j, q] = c / np.maximum(nc[:, None], 1e-12)
                score[:, j] += sa[:, r, s] + sb[:, r, s]; geo[:, j] = np.minimum(geo[:, j], nc)     # sine of the pair angle
        out["TCA"] = U[np.arange(len(A)), score.argmax(1)]
        out["TCA geometric pairing"] = U[np.arange(len(A)), geo.argmax(1)]
        tot = np.stack([pair_err(U[:, j], tru)[0].sum(1) for j in range(3)], 1)
        out["TCA oracle pairing"] = U[np.arange(len(A)), tot.argmin(1)]
    return F, out, float(np.median(err.cpu().numpy()))


def main():
    vols = make(np.random.default_rng(5)); res, keep = {}, {}
    for name, ths in ROWS:
        tru = np.stack([unit(t) for t in ths]); F, out, fit = analyse(vols[name], tru)
        r = res[name] = {"cp_rel_err": round(fit, 3)}; keep[name] = (vols[name], F, out, tru)
        for k, e in out.items():
            if e.ndim == 2:
                q = {f"to {t:g}°": round(float(np.median(ang(e, unit(t)))), 1) for t in ths}
                if len(ths) == 2:
                    b = np.stack([tru[0] + tru[1], tru[0] - tru[1]]); b /= np.linalg.norm(b, axis=1, keepdims=True)
                    q["to bisector"] = round(float(np.median(np.minimum(ang(e, b[0]), ang(e, b[1])))), 1)
            else:
                errs, _ = pair_err(e, tru)
                q = {f"fibre {ths[0]:g}°": round(float(np.median(errs[:, 0])), 1), f"fibre {ths[1]:g}°": round(float(np.median(errs[:, 1])), 1),
                     "both within 15°": round(float(np.mean((errs < 15).all(1))), 3)}
            r[k] = q; print(f"{name:20s} {k:20s} {q}", flush=True)
        print(f"{name:20s} CP rank-{2 * len(ths)} relative error {fit:.3f}", flush=True)
    (ST.OUT / "shape_cross_binary.json").write_text(json.dumps(res, indent=1))
    figure(keep, res)


def figure(keep, res):
    cols = ("image (x-z, mid y)", "shape space, RGB = PC 1-3", "intensity structure tensor", "shape tensor eigen e3",
            "TCA, pieces paired by\nsignature similarity", "TCA, pieces paired geometrically\n(most perpendicular pairs; no truth)", "TCA, oracle pairing")
    fig, axs = plt.subplots(len(keep), 7, figsize=(31, 4.7 * len(keep) + 1.2), dpi=80)
    ext = (C0 * VOX, (C0 + W) * VOX, C0 * VOX, (C0 + W) * VOX)
    xi, zi = np.meshgrid(np.arange(2, W, 5), np.arange(2, W, 5), indexing="ij"); site = (xi * W + zi).ravel()
    cx, cz = (C0 + xi.ravel() + 0.5) * VOX, (C0 + zi.ravel() + 0.5) * VOX

    def sticks(ax, e, col, L=0.42):
        inpl = np.hypot(e[:, 0], e[:, 2]); d = e[:, [0, 2]] / np.maximum(inpl[:, None], 1e-9); h = L * inpl / 2
        segs = np.stack([np.stack([cx - d[:, 0] * h, cz - d[:, 1] * h], 1), np.stack([cx + d[:, 0] * h, cz + d[:, 1] * h], 1)], 1)
        ax.add_collection(LineCollection(segs, colors="white", linewidths=3.4)); ax.add_collection(LineCollection(segs, colors=col, linewidths=1.8))

    def label(ax, q):
        ax.text(0.02, 0.02, ", ".join(f"{k} {v:.0f}" + ("%" if "within" in k else "°") if "within" not in k else f"{k} {v:.0%}" for k, v in q.items()),
                transform=ax.transAxes, fontsize=8.5, bbox=dict(boxstyle="round", fc="w", alpha=0.9, lw=0))
    for r, (name, (v, F, out, tru)) in enumerate(keep.items()):
        img = v[C0:C0 + W, JY, C0:C0 + W]
        for c in range(7):
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        axs[r, 0].imshow(img.T, cmap="gray", origin="lower", extent=ext); axs[r, 0].set_ylabel(name, fontsize=13)
        X = F[:, C0:C0 + W, JY, C0:C0 + W].reshape(16, -1); X = X - X.mean(1, keepdims=True)
        w, U = np.linalg.eigh(np.cov(X)); P = U[:, np.argsort(w)[::-1][:3]].T @ X
        lo, hi = np.percentile(P, 1, axis=1), np.percentile(P, 99, axis=1)
        axs[r, 1].imshow(np.transpose(np.clip((P - lo[:, None]) / np.maximum(hi - lo, 1e-12)[:, None], 0, 1).T.reshape(W, W, 3), (1, 0, 2)), origin="lower", extent=ext)
        for c, k in ((2, "intensity ST"), (3, "shape eigen")):
            ax = axs[r, c]; ax.imshow(img.T, cmap="gray", origin="lower", extent=ext, alpha=0.3); sticks(ax, out[k][site], "#0b0b0b"); label(ax, res[name][k])
        for c, k in ((4, "TCA"), (5, "TCA geometric pairing"), (6, "TCA oracle pairing")):
            ax = axs[r, c]; ax.imshow(img.T, cmap="gray", origin="lower", extent=ext, alpha=0.3)
            if k not in out:
                ax.text(0.5, 0.5, "single population:\nTCA rank 2 -> one fibre\n(column 5)", ha="center", va="center", transform=ax.transAxes, fontsize=11); continue
            e = out[k]
            if e.ndim == 2:
                sticks(ax, e[site], "#2a78d6")
            else:
                _, m = pair_err(e, tru); sticks(ax, m[site, 0], "#2a78d6"); sticks(ax, m[site, 1], "#eb6834")
            label(ax, res[name][k])
    for c, t in enumerate(cols):
        axs[0, c].set_title(t, fontsize=12)
    fig.suptitle("PROTOTYPE. Binary rod phantom (dark rods, radius 1.5 vox, 1 mm lattice; crossings intersect): does the 16-D shape space, "
                 f"and TCA of it, see each fibre direction? ρ {RHO:g} vox, central 6.4 x 6.4 mm.\nSticks = estimated fibre directions "
                 "(in-plane part); blue / orange = the two fibres matched to the truth. Angles to the x-y plane; medians.", fontsize=12.5,
                 x=0.01, ha="left", y=0.995)
    fig.subplots_adjust(top=0.92, wspace=0.05, hspace=0.12, left=0.05, right=0.99)
    out = ST.FIGS / "shape_cross_binary.png"; fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main()
