"""PROTOTYPE: can TCA of the anatomix shape space return N = 2 fibre orientations per window? A small crossing phantom
(16 x 9.6 x 16 mm, 0.1 mm voxels), two fibre populations in the x-z plane 60 deg apart (90 and 30 deg to the z = const
plane, as the D phantoms), D-phantom line density in total, intensity 180 darkened by the fibres, noise 8.

  A  layered, different texture    thin (blur 0.6 vox) vs thick (blur 1.4 vox) fibres in alternating 0.6 mm y-layers
  B  intermixed, different texture  thin and thick fibres at random y
  C  intermixed, same texture       both thin, at random y (the hard case)

Read-outs on the mid-y x-z slice, central 64 x 64 voxels (48 from every x and z face), rho 8 vox:
  intensity ST     e3 of the image's structure tensor (one direction)
  shape eigen      e3 of the 16-D shape tensor (one direction)
  TCA, 2 fibres    CP rank 4 of (space x channel x gradient axis) per pixel; the 4 pieces paired by similarity of
                   their channel signatures and spatial maps; each pair's fibre = d_r x d_s
  TCA, oracle      the same with the pairing that best matches the truth (upper bound: do the pieces split by
                   population at all?)
Scores: median angle of each estimate to each population (matched), and for one-direction read-outs also to the
nearer bisector of the two populations.

    python shape_cross.py        (shapetensors env, one GPU)
"""
import json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import anatomix_shape as AS
import shape_tca as T
import shape_tensor as ST

NX, NY, NZ, VOX = 160, 96, 160, 0.1
C0, W = 48, 64                                            # scored region [C0, C0 + W) in x and z
JY = NY // 2
BASE, AMP, NOISE = 180.0, 14.0, 8.0
DENS = 13.7                                               # lines per mm^2 of cross-section in total (the D phantoms)
TH = (90.0, 30.0)
LAYER = 0.6                                               # mm
ROOT = Path("/home/andrew/bridge_wmcov/shape_cross")
RHO = 8.0
TRU = np.array([[np.cos(np.radians(t)), 0.0, np.sin(np.radians(t))] for t in TH])        # (2, 3) x, y, z


def population(theta, sig, rng, layer=None):
    """Density of one population: straight lines at theta in the x-z plane, uniform across the block, each at one y
    (layer = 0 / 1: only in even / odd LAYER-thick y-layers), blurred by sig voxels, normalised to its 99th pct."""
    a = np.radians(theta); d = np.array([np.cos(a), np.sin(a)]); nrm = np.array([-d[1], d[0]])
    Lx, Lz, Ly = NX * VOX, NZ * VOX, NY * VOX
    pr = np.array([[0, 0], [Lx, 0], [0, Lz], [Lx, Lz]]) @ nrm; lo, hi = pr.min(), pr.max()
    n = int(DENS / 2 * (hi - lo) * Ly); s = np.arange(-30, 30, 0.05); cen = np.array([Lx / 2, Lz / 2])
    dens = np.zeros((NX, NY, NZ), np.float32); nl = int(Ly / LAYER)
    for _ in range(n):
        c = rng.uniform(lo, hi) * nrm + (cen @ d) * d
        p = c[None] + s[:, None] * d[None]; k = (p[:, 0] >= 0) & (p[:, 0] < Lx) & (p[:, 1] >= 0) & (p[:, 1] < Lz)
        y = rng.uniform(0, Ly) if layer is None else (2 * rng.integers(0, nl // 2) + layer) * LAYER + rng.uniform(0, LAYER)
        ix = (p[k, 0] / VOX).astype(int); iz = (p[k, 1] / VOX).astype(int)
        np.add.at(dens, (ix, np.full(ix.shape, min(int(y / VOX), NY - 1)), iz), 1.0)
    dens = ndi.gaussian_filter(dens, sig)
    return dens / np.percentile(dens[dens > 0], 99)


def make(rng):
    out = {}
    for name, sigs, layered in (("A", (0.6, 1.4), True), ("B", (0.6, 1.4), False), ("C", (0.6, 0.6), False)):
        v = np.full((NX, NY, NZ), BASE, np.float32)
        for k, (th, sg) in enumerate(zip(TH, sigs)):
            v -= AMP * np.clip(population(th, sg, rng, k if layered else None), 0, 1.5)
        v = np.clip(np.rint(v + rng.normal(0, NOISE, v.shape)), 0, 255).astype(np.uint8)
        ROOT.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(v, np.diag([VOX, VOX, VOX, 1.0])), ROOT / f"{name}.nii.gz"); out[name] = v.astype(np.float32)
        print("made", name, flush=True)
    return out


def ang(u, t):
    return np.degrees(np.arccos(np.clip(np.abs(u @ t), 0, 1)))


def analyse(v):
    """All read-outs on the scored region: {name: (N, 3) or (N, 2, 3)} plus the CP fit."""
    F = AS.anatomix_clean(v, VOX)
    out = {}
    Ki = T.k_slice(v[None], RHO, JY)[C0:C0 + W, C0:C0 + W].reshape(-1, 3, 3).double()
    out["intensity ST"] = T.null_dir(Ki)
    K = T.k_slice(F, RHO, JY)[C0:C0 + W, C0:C0 + W]
    Kf = K.reshape(-1, 48, 48).double(); Kf = Kf / Kf.diagonal(dim1=1, dim2=2).sum(1)[:, None, None]
    T3 = Kf.reshape(-1, 16, 3, 16, 3).diagonal(dim1=1, dim2=3).sum(-1); out["shape eigen"] = T.null_dir(T3)
    eye = torch.eye(48, device=T.DEV, dtype=Kf.dtype)
    L, info = torch.linalg.cholesky_ex(Kf + 1e-9 * eye)
    if (info > 0).any():
        L[info > 0] = torch.linalg.cholesky(Kf[info > 0] + 1e-6 * eye)
    Y = L.mT.float().reshape(-1, 48, 16, 3)
    Cg = Kf.reshape(-1, 16, 3, 16, 3).diagonal(dim1=2, dim2=4).sum(-1)
    A, B, D, err = T.cp_als(Y, 4, (torch.linalg.eigh(Cg)[1].flip(-1).float(), torch.linalg.eigh(T3)[1].flip(-1).float()), starts=4, iters=200)
    A, B, D = (x.double().cpu().numpy() for x in (A, B, D))
    sa = np.abs(np.einsum("npr,nps->nrs", A, A)) / np.maximum(np.linalg.norm(A, axis=1)[:, :, None] * np.linalg.norm(A, axis=1)[:, None, :], 1e-30)
    sb = np.abs(np.einsum("ncr,ncs->nrs", B, B))
    pairings = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
    U = np.zeros((len(A), 3, 2, 3)); score = np.zeros((len(A), 3))
    for j, prs in enumerate(pairings):
        for q, (r, s) in enumerate(prs):
            c = np.cross(D[:, :, r], D[:, :, s]); U[:, j, q] = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
            score[:, j] += sa[:, r, s] + sb[:, r, s]
    pick = score.argmax(1); out["TCA, 2 fibres"] = U[np.arange(len(A)), pick]
    tot = np.stack([pair_err(U[:, j])[0].sum(1) for j in range(3)], 1)
    out["TCA, 2 fibres (oracle pairing)"] = U[np.arange(len(A)), tot.argmin(1)]
    for k in ("intensity ST", "shape eigen"):
        out[k] = out[k].float().cpu().numpy()
    return F, out, float(np.median(err.cpu().numpy()))


def pair_err(Up):
    """Up (N, 2, 3) -> matched errors (N, 2) [population 1, population 2] and the matched estimates."""
    e11, e22 = ang(Up[:, 0], TRU[0]), ang(Up[:, 1], TRU[1]); e12, e21 = ang(Up[:, 0], TRU[1]), ang(Up[:, 1], TRU[0])
    swap = (e12 + e21) < (e11 + e22)
    errs = np.where(swap[:, None], np.stack([e21, e12], 1), np.stack([e11, e22], 1))
    return errs, np.where(swap[:, None, None], Up[:, ::-1], Up)


def main():
    rng = np.random.default_rng(23)
    vols = make(rng)
    bis = np.stack([TRU[0] + TRU[1], TRU[0] - TRU[1]]); bis /= np.linalg.norm(bis, axis=1, keepdims=True)
    res, keep = {}, {}
    for name, v in vols.items():
        F, out, fit = analyse(v); r = res[name] = {"cp_rank4_rel_err": round(fit, 3)}; keep[name] = (v, F, out)
        for k, e in out.items():
            if e.ndim == 2:
                a1, a2 = ang(e, TRU[0]), ang(e, TRU[1]); ab = np.minimum(ang(e, bis[0]), ang(e, bis[1]))
                r[k] = {"to_pop1_90deg": round(float(np.median(a1)), 1), "to_pop2_30deg": round(float(np.median(a2)), 1),
                        "to_bisector": round(float(np.median(ab)), 1)}
            else:
                errs, _ = pair_err(e)
                r[k] = {"pop1_90deg": round(float(np.median(errs[:, 0])), 1), "pop2_30deg": round(float(np.median(errs[:, 1])), 1),
                        "both_within_15deg": round(float(np.mean((errs < 15).all(1))), 3)}
            print(name, f"{k:32s}", r[k], flush=True)
        print(name, "CP rank-4 relative error", r["cp_rank4_rel_err"], flush=True)
    (ST.OUT / "shape_cross.json").write_text(json.dumps(res, indent=1))
    figure(keep, res)


def figure(keep, res):
    titles = {"A": "A  layered, thin vs thick", "B": "B  intermixed, thin vs thick", "C": "C  intermixed, same texture"}
    cols = ("image (x-z, mid y)", "shape space, RGB = PC 1-3", "intensity structure tensor\n(one direction)",
            "shape tensor eigen e3\n(one direction)", "TCA, 2 fibres per site\n(pieces paired by similarity)", "TCA, 2 fibres\n(oracle pairing)")
    fig, axs = plt.subplots(3, 6, figsize=(28, 14.5), dpi=80)
    ext = (C0 * VOX, (C0 + W) * VOX, C0 * VOX, (C0 + W) * VOX)
    xi, zi = np.meshgrid(np.arange(2, W, 5), np.arange(2, W, 5), indexing="ij"); site = (xi * W + zi).ravel()
    cx, cz = ((C0 + xi.ravel() + 0.5) * VOX), ((C0 + zi.ravel() + 0.5) * VOX)

    def sticks(ax, e, col, L=0.42):
        inpl = np.hypot(e[:, 0], e[:, 2]); d = e[:, [0, 2]] / np.maximum(inpl[:, None], 1e-9); h = L * inpl / 2
        segs = np.stack([np.stack([cx - d[:, 0] * h, cz - d[:, 1] * h], 1), np.stack([cx + d[:, 0] * h, cz + d[:, 1] * h], 1)], 1)
        ax.add_collection(LineCollection(segs, colors="white", linewidths=3.4)); ax.add_collection(LineCollection(segs, colors=col, linewidths=1.8))
    for r, (name, (v, F, out)) in enumerate(keep.items()):
        img = v[C0:C0 + W, JY, C0:C0 + W]
        for c in range(6):
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        axs[r, 0].imshow(img.T, cmap="gray", origin="lower", extent=ext)
        X = F[:, C0:C0 + W, JY, C0:C0 + W].reshape(16, -1); X = X - X.mean(1, keepdims=True)
        w, U = np.linalg.eigh(np.cov(X)); P = U[:, np.argsort(w)[::-1][:3]].T @ X
        lo, hi = np.percentile(P, 1, axis=1), np.percentile(P, 99, axis=1)
        axs[r, 1].imshow(np.transpose(np.clip((P - lo[:, None]) / (hi - lo)[:, None], 0, 1).T.reshape(W, W, 3), (1, 0, 2)), origin="lower", extent=ext)
        for c, k in ((2, "intensity ST"), (3, "shape eigen")):
            ax = axs[r, c]; ax.imshow(img.T, cmap="gray", origin="lower", extent=ext, alpha=0.35); sticks(ax, out[k][site], "#0b0b0b")
            q = res[name][k]; T_lab = f"to 90°: {q['to_pop1_90deg']:.0f}°, to 30°: {q['to_pop2_30deg']:.0f}°, to bisector: {q['to_bisector']:.0f}°"
            ax.text(0.02, 0.02, T_lab, transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round", fc="w", alpha=0.9, lw=0))
        for c, k in ((4, "TCA, 2 fibres"), (5, "TCA, 2 fibres (oracle pairing)")):
            ax = axs[r, c]; ax.imshow(img.T, cmap="gray", origin="lower", extent=ext, alpha=0.35)
            _, m = pair_err(out[k]); sticks(ax, m[site, 0], "#2a78d6"); sticks(ax, m[site, 1], "#eb6834")
            q = res[name][k]
            ax.text(0.02, 0.02, f"blue (90° fibres): {q['pop1_90deg']:.0f}°, orange (30°): {q['pop2_30deg']:.0f}°\n"
                    f"both within 15°: {q['both_within_15deg']:.0%} of sites", transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle="round", fc="w", alpha=0.9, lw=0))
        axs[r, 0].set_ylabel(titles[name], fontsize=13)
    for c, t in enumerate(cols):
        axs[0, c].set_title(t, fontsize=12)
    fig.suptitle("PROTOTYPE. Two crossing fibre populations (90° and 30° to the x-y plane, 60° apart) in a small phantom; can TCA of the "
                 f"16-D shape space give both directions per site? ρ {RHO:g} vox (0.8 mm), CP rank 4 (two pieces per population).\n"
                 "Sticks = estimated fibre directions (in-plane part); errors are medians over the central 6.4 x 6.4 mm. One-direction "
                 "read-outs are also scored against the bisector of the two populations.", fontsize=12.5, x=0.01, ha="left", y=0.995)
    fig.subplots_adjust(top=0.9, wspace=0.05, hspace=0.1, left=0.05, right=0.99)
    out = ST.FIGS / "shape_cross.png"; fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main()
