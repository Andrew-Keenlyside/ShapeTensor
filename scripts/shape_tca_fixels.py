"""PROTOTYPE: multi-fibre output from TCA of the 16-D shape space, with the number of orientations per site chosen by
rules borrowed from multi-fibre diffusion MRI. Per site (window rho 8 vox, central 64 x 64 of the mid-y slice):

  1. Fit CP with 2 pieces per population: ranks 2, 4, 6 (N = 1, 2, 3). Pieces are paired geometrically (the perfect
     matching whose pairs are most perpendicular: a population's gradients span the plane perpendicular to it);
     each pair's fibre = d_r x d_s.
  2. Model selection (as multi-tensor / ball-and-stick fitting): N by the Bayesian information criterion on the CP
     residual, n ln(RSS/n) + k ln n, k = 2N (P + C + I - 2) parameters, n = P C I entries.
  3. Volume-fraction floor (as bedpostx's ARD pruning): a population's fraction = its pair's share of the fitted energy;
     populations under F_MIN are dropped.
  4. Lobe merging (as fixel segmentation): populations closer than SEP deg are merged (the stronger kept).
Output: fixels = orientations + fractions per site. Phantoms from shape_cross_binary / shape_cross_texture.

    python shape_tca_fixels.py     (shapetensors env, one GPU)
"""
import itertools, json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import anatomix_shape as AS
import shape_cross_binary as B
import shape_cross_texture as CT
import shape_tca as T
import shape_tensor as ST

N, VOX, C0, W, JY, RHO = B.N, B.VOX, B.C0, B.W, B.JY, B.RHO
F_MIN, SEP, NMAX = 0.10, 20.0, 3
ROWS = (("rods, single 90°", "rods", 2.0, (90.0,)), ("rods, single 30°", "rods", 2.0, (30.0,)),
        ("rods, 90° + 30°", "rods", 2.0, (90.0, 30.0)), ("rods, 60° + 120°", "rods", 2.0, (60.0, 120.0)),
        ("rods, noise 8, 60° + 120°", "rods", 8.0, (60.0, 120.0)), ("lines, noise 8, single 30°", "lines", 8.0, (30.0,)),
        ("lines, noise 8, 60° + 120°", "lines", 8.0, (60.0, 120.0)))


def matchings(n):
    """All perfect matchings of range(n) (n even) as tuples of pairs."""
    if n == 0:
        return [()]
    out = []
    for j in range(1, n):
        rest = [k for k in range(1, n) if k != j]
        for m in matchings(len(rest)):
            out.append(((0, j),) + tuple((rest[a], rest[b]) for a, b in m))
    return out


def fixels(v):
    """Per site: list of (orientation (3,), fraction) after model selection; plus the chosen-N diagnostics."""
    F = AS.anatomix_clean(v, VOX)
    K = T.k_slice(F, RHO, JY)[C0:C0 + W, C0:C0 + W]
    Kf = K.reshape(-1, 48, 48).double(); Kf = Kf / Kf.diagonal(dim1=1, dim2=2).sum(1)[:, None, None]
    T3 = Kf.reshape(-1, 16, 3, 16, 3).diagonal(dim1=1, dim2=3).sum(-1)
    eye = torch.eye(48, device=T.DEV, dtype=Kf.dtype)
    L, info = torch.linalg.cholesky_ex(Kf + 1e-9 * eye)
    if (info > 0).any():
        L[info > 0] = torch.linalg.cholesky(Kf[info > 0] + 1e-6 * eye)
    Y = L.mT.float().reshape(-1, 48, 16, 3); P_, C_, I_ = Y.shape[1:]; n = P_ * C_ * I_
    init = (torch.linalg.eigh(Kf.reshape(-1, 16, 3, 16, 3).diagonal(dim1=2, dim2=4).sum(-1))[1].flip(-1).float(),
            torch.linalg.eigh(T3)[1].flip(-1).float())
    fits = {}
    for Np in range(1, NMAX + 1):
        A, Bm, D, err = T.cp_als(Y, 2 * Np, init, starts=4, iters=200)
        A, Bm, D, err = (x.double().cpu().numpy() for x in (A, Bm, D, err))
        G = np.einsum("npr,nps->nrs", A, A) * np.einsum("ncr,ncs->nrs", Bm, Bm) * np.einsum("nir,nis->nrs", D, D)   # block Gram
        best = None
        for m in matchings(2 * Np):
            sines = np.stack([np.linalg.norm(np.cross(D[:, :, r], D[:, :, s]), axis=1) for r, s in m], 1)
            sc = sines.min(1)
            if best is None:
                best = (sc, [m] * len(A))
            else:
                upd = sc > best[0]; best = (np.where(upd, sc, best[0]), [m if u else b for u, b in zip(upd, best[1])])
        U = np.zeros((len(A), Np, 3)); f = np.zeros((len(A), Np))
        for i, m in enumerate(best[1]):
            for q, (r, s) in enumerate(m):
                c = np.cross(D[i, :, r], D[i, :, s]); U[i, q] = c / max(np.linalg.norm(c), 1e-12)
                f[i, q] = G[i][np.ix_((r, s), (r, s))].sum()
        f = f / np.maximum(f.sum(1, keepdims=True), 1e-30)
        k = 2 * Np * (P_ + C_ + I_ - 2)
        fits[Np] = dict(U=U, f=f, bic=n * np.log(np.maximum(err, 1e-12)) + k * np.log(n), err=err)
    bic = np.stack([fits[k]["bic"] for k in range(1, NMAX + 1)], 1); nb = bic.argmin(1) + 1
    out, ndiag = [], {"N by BIC": nb}
    for i in range(len(nb)):
        Uc, fc = fits[nb[i]]["U"][i], fits[nb[i]]["f"][i]
        o = np.argsort(fc)[::-1]; keep = []
        for q in o:
            if fc[q] < F_MIN:
                continue
            if all(np.abs(Uc[q] @ Uc[j]) < np.cos(np.radians(SEP)) for j in keep):
                keep.append(q)
        fk = fc[keep] / fc[keep].sum()
        out.append([(Uc[q], float(x)) for q, x in zip(keep, fk)])
    return F, out, ndiag


def score(fx, tru):
    nf = np.array([len(x) for x in fx]); r = {"N correct": float(np.mean(nf == len(tru))), "N found": {int(k): float(np.mean(nf == k)) for k in range(1, NMAX + 1)}}
    if len(tru) == 1:
        r["error main fixel"] = float(np.median([B.ang(x[0][0], tru[0]) for x in fx]))
    else:
        e2 = np.array([min((B.ang(x[0][0], tru[0]), B.ang(x[1][0], tru[1])), (B.ang(x[0][0], tru[1]), B.ang(x[1][0], tru[0])), key=sum)
                       for x in fx if len(x) >= 2]).reshape(-1, 2)
        r["errors when 2 found"] = [float(np.median(e2[:, 0])), float(np.median(e2[:, 1]))] if len(e2) else None
        r["both within 15° (all sites)"] = float(np.sum((e2 < 15).all(1)) / len(fx)) if len(e2) else 0.0
        r["minor fraction (median, 2-fixel sites)"] = float(np.median([x[1][1] for x in fx if len(x) >= 2])) if len(e2) else None
    return r


def main():
    rng = np.random.default_rng(31); res, keep = {}, {}
    for name, kind, noise, ths in ROWS:
        v = CT.make(kind, noise, ths, rng); tru = np.stack([B.unit(t) for t in ths])
        F, fx, nd = fixels(v); res[name] = score(fx, tru); keep[name] = (v, fx, tru)
        res[name]["N by BIC alone"] = {int(k): float(np.mean(nd["N by BIC"] == k)) for k in range(1, NMAX + 1)}
        print(f"{name:28s} {res[name]}", flush=True)
    (ST.OUT / "shape_tca_fixels.json").write_text(json.dumps(res, indent=1))
    figure(keep, res)


def figure(keep, res):
    ext = (C0 * VOX, (C0 + W) * VOX, C0 * VOX, (C0 + W) * VOX)
    xi, zi = np.meshgrid(np.arange(2, W, 5), np.arange(2, W, 5), indexing="ij"); site = (xi * W + zi).ravel()
    cx, cz = (C0 + xi.ravel() + 0.5) * VOX, (C0 + zi.ravel() + 0.5) * VOX
    ncm = ListedColormap(["#ffffff", "#cde2fb", "#86b6ef", "#3987e5"]); pcol = ("#0b0b0b", "#eb6834", "#1baf7a")
    fig, axs = plt.subplots(len(keep), 2, figsize=(10.5, 4.7 * len(keep) + 1.4), dpi=80)
    for r, (name, (v, fx, tru)) in enumerate(keep.items()):
        axs[r, 0].imshow(v[C0:C0 + W, JY, C0:C0 + W].T, cmap="gray", origin="lower", extent=ext); axs[r, 0].set_ylabel(name, fontsize=12)
        ax = axs[r, 1]; ax.imshow(np.array([len(x) for x in fx]).reshape(W, W).T, cmap=ncm, vmin=-0.5, vmax=3.5, origin="lower", extent=ext, interpolation="nearest")
        for j in range(3):
            segs = []
            for s_, x_, z_ in zip(site, cx, cz):
                if len(fx[s_]) > j:
                    e, fr = fx[s_][j]; inpl = np.hypot(e[0], e[2]); d = np.array([e[0], e[2]]) / max(inpl, 1e-9); h = 0.21 * inpl * (0.4 + 0.6 * fr / fx[s_][0][1])
                    segs.append([[x_ - d[0] * h, z_ - d[1] * h], [x_ + d[0] * h, z_ + d[1] * h]])
            if segs:
                ax.add_collection(LineCollection(segs, colors="white", linewidths=3.2)); ax.add_collection(LineCollection(segs, colors=pcol[j], linewidths=1.6))
        q = res[name]
        txt = f"N = {len(tru)} at {q['N correct']:.0%} of sites; " + (f"error {q['error main fixel']:.0f}°" if "error main fixel" in q else
              (f"errors {q['errors when 2 found'][0]:.0f}° / {q['errors when 2 found'][1]:.0f}°, both < 15°: {q['both within 15° (all sites)']:.0%}"
               if q["errors when 2 found"] else "no 2-fixel sites"))
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round", fc="w", alpha=0.9, lw=0))
        for c in range(2):
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
    axs[0, 0].set_title("image (x-z, mid y)", fontsize=12); axs[0, 1].set_title("TCA fixels, N chosen per site\n(BIC + fraction floor + merge)", fontsize=12)
    fig.suptitle("PROTOTYPE. Multi-fibre output from TCA of the 16-D shape space; how many orientations per site is chosen as in\n"
                 f"multi-fibre dMRI: BIC between 1-3 populations, populations under {F_MIN:.0%} of the energy dropped (ARD-like), "
                 f"populations within {SEP:g}° merged.\nBackground = N output (light 1, mid 2, dark 3); sticks = fixels (length ~ fraction).",
                 fontsize=11.5, x=0.01, ha="left", y=0.995)
    fig.subplots_adjust(top=0.93, wspace=0.04, hspace=0.1, left=0.2, right=0.99)
    out = ST.FIGS / "shape_tca_fixels.png"; fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main()
