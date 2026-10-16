"""PROTOTYPE: finding N fibre populations per window without choosing a rank. Each population leaves a RING of gradient
directions on the sphere (everything perpendicular to its fibre). Per window: the distribution of gradient directions
(every channel, weighted by |g|^2, Gaussian window rho) in even spherical harmonics (order 8), then the Funk-Radon
transform (Funk-Hecke: order-l coefficients x 2 pi P_l(0)), which turns each ring into a peak at its fibre direction (as
q-ball imaging). N = the number of local maxima of the min-subtracted ODF above PEAK_REL x its maximum, at least SEP
apart (cap 3). Run on the image's own gradients (1 channel) and on the 16-D anatomix shape space (16 channels).

Phantoms (128^3, 0.1 mm, scored on the central 64 x 64 of the mid-y slice), from shape_cross_binary / shape_cross_texture.

    python shape_npop.py     (shapetensors env, one GPU)
"""
import json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.special import sph_harm_y

sys.path.insert(0, str(Path(__file__).parent))
import anatomix_shape as AS
import shape_cross_binary as B
import shape_cross_texture as CT
import shape_tca as T
import shape_tensor as ST
import tensor_variants as TV

N, VOX, C0, W, JY, RHO = B.N, B.VOX, B.C0, B.W, B.JY, B.RHO
LMAX, PEAK_REL, SEP, NBR = 8, 0.5, 25.0, 7.0
ROWS = (("rods, single 90°", "rods", 2.0, (90.0,)), ("rods, single 30°", "rods", 2.0, (30.0,)),
        ("rods, 90° + 30°", "rods", 2.0, (90.0, 30.0)), ("rods, 60° + 120°", "rods", 2.0, (60.0, 120.0)),
        ("rods, noise 8, 60° + 120°", "rods", 8.0, (60.0, 120.0)), ("lines, noise 8, single 30°", "lines", 8.0, (30.0,)),
        ("lines, noise 8, 60° + 120°", "lines", 8.0, (60.0, 120.0)), ("lines, noise 8, 90° + 30°", "lines", 8.0, (90.0, 30.0)))


def real_sh(theta, phi):
    """Real even spherical harmonics up to LMAX at polar theta, azimuth phi: (..., n_coef)."""
    cols = []
    for l in range(0, LMAX + 1, 2):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, abs(m), theta, phi)
            cols.append(np.sqrt(2) * (-1) ** m * Y.imag if m < 0 else (Y.real if m == 0 else np.sqrt(2) * (-1) ** m * Y.real))
    return np.stack(cols, -1).astype(np.float32)


def frt_weights():
    from scipy.special import eval_legendre
    return np.concatenate([np.full(2 * l + 1, 2 * np.pi * eval_legendre(l, 0.0)) for l in range(0, LMAX + 1, 2)]).astype(np.float32)


SPH = B.unit(0.0)  # placeholder to keep imports used


def hemisphere(n=2000):
    k = np.arange(n) + 0.5; c = 1 - k / n; r = np.sqrt(1 - c ** 2); ph = k * np.pi * (3 - np.sqrt(5))
    return np.stack([r * np.cos(ph), r * np.sin(ph), c], 1)


DIRS = hemisphere()
YD = real_sh(np.arccos(np.clip(DIRS[:, 2], -1, 1)), np.arctan2(DIRS[:, 1], DIRS[:, 0]))         # (2000, n_coef)
NB = [np.flatnonzero((np.abs(DIRS @ d) > np.cos(np.radians(NBR))) & (np.arange(len(DIRS)) != i)) for i, d in enumerate(DIRS)]


def odf_coefs(Z, rho=RHO):
    """Z (C, x, y, z) -> FRT-ODF SH coefficients (W*W, n_coef) on the scored region at mid y."""
    m = int(np.ceil(2 * rho)) + 4; r = int(TV.TR * rho + 0.5)
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / rho) ** 2); w /= w.sum()
    acc = 0
    for f in Z:
        g = np.stack(TV.grads(f[:, JY - m:JY + m + 1, :]))[:, :, m - r:m + r + 1, :]                  # (3, x, 2r+1, z)
        mag2 = (g ** 2).sum(0); u = g / np.sqrt(np.maximum(mag2, 1e-20))
        Y = real_sh(np.arccos(np.clip(u[2], -1, 1)), np.arctan2(u[1], u[0]))                         # (x, 2r+1, z, n)
        acc = acc + np.einsum("xyzn,xyz,y->nxz", Y, mag2, w.astype(np.float32))
    acc = np.stack([ndi.gaussian_filter(a, rho, mode="nearest", truncate=TV.TR) for a in acc])
    return (acc[:, C0:C0 + W, C0:C0 + W].reshape(len(acc), -1).T) * frt_weights()[None]


def peaks(coef):
    """FRT-ODF coefficients (P, n_coef) -> list of (N peak directions (n, 3)) per pixel."""
    O = coef @ YD.T; O = O - O.min(1, keepdims=True)                                                 # (P, 2000)
    out = []
    for o in O:
        mx = o.max(); cand = [i for i in np.argsort(o)[::-1][:200] if o[i] >= PEAK_REL * mx and (o[NB[i]] <= o[i]).all()]
        keep = []
        for i in cand:
            if all(np.abs(DIRS[i] @ DIRS[j]) < np.cos(np.radians(SEP)) for j in keep):
                keep.append(i)
            if len(keep) == 3:
                break
        out.append(DIRS[keep])
    return out


def score(pk, tru):
    n = np.array([len(p) for p in pk]); r = {"N correct": float(np.mean(n == len(tru))), "N found": {int(k): float(np.mean(n == k)) for k in range(4)}}
    if len(tru) == 1:
        r["error top peak"] = float(np.median([B.ang(p[0], tru[0]) for p in pk if len(p)]))
    else:
        errs = [np.sort([min(B.ang(p[0], tru[0]) + B.ang(p[1], tru[1]), B.ang(p[0], tru[1]) + B.ang(p[1], tru[0]))]) for p in pk if len(p) >= 2]
        e2 = np.array([[min((B.ang(p[0], tru[0]), B.ang(p[1], tru[1])), (B.ang(p[0], tru[1]), B.ang(p[1], tru[0])), key=sum)] for p in pk if len(p) >= 2]).reshape(-1, 2)
        r["errors when 2 found (fibre 1 / fibre 2)"] = [float(np.median(e2[:, 0])), float(np.median(e2[:, 1]))] if len(e2) else None
        r["both within 15° (all sites)"] = float(np.sum((e2 < 15).all(1)) / len(pk)) if len(e2) else 0.0
    return r


def main():
    rng = np.random.default_rng(31); res, keep = {}, {}
    for name, kind, noise, ths in ROWS:
        v = CT.make(kind, noise, ths, rng); tru = np.stack([B.unit(t) for t in ths])
        F = AS.anatomix_clean(v, VOX)
        pk = {"intensity": peaks(odf_coefs(v[None])), "shape (16-D)": peaks(odf_coefs(F))}
        res[name] = {k: score(p, tru) for k, p in pk.items()}; keep[name] = (v, pk, tru)
        for k, q in res[name].items():
            print(f"{name:28s} {k:13s} N correct {q['N correct']:.0%}  N found {q['N found']}  "
                  + (f"top-peak error {q['error top peak']:.1f}" if "error top peak" in q else
                     f"errors {q['errors when 2 found (fibre 1 / fibre 2)']}, both within 15° {q['both within 15° (all sites)']:.0%}"), flush=True)
    (ST.OUT / "shape_npop.json").write_text(json.dumps(res, indent=1))
    figure(keep, res)


def figure(keep, res):
    ext = (C0 * VOX, (C0 + W) * VOX, C0 * VOX, (C0 + W) * VOX)
    xi, zi = np.meshgrid(np.arange(2, W, 5), np.arange(2, W, 5), indexing="ij"); site = (xi * W + zi).ravel()
    cx, cz = (C0 + xi.ravel() + 0.5) * VOX, (C0 + zi.ravel() + 0.5) * VOX
    ncm = ListedColormap(["#ffffff", "#cde2fb", "#86b6ef", "#3987e5"])
    pcol = ("#0b0b0b", "#eb6834", "#1baf7a")
    fig, axs = plt.subplots(len(keep), 3, figsize=(15, 4.6 * len(keep) + 1.2), dpi=80)
    for r, (name, (v, pk, tru)) in enumerate(keep.items()):
        axs[r, 0].imshow(v[C0:C0 + W, JY, C0:C0 + W].T, cmap="gray", origin="lower", extent=ext); axs[r, 0].set_ylabel(name, fontsize=12)
        for c, k in ((1, "intensity"), (2, "shape (16-D)")):
            ax = axs[r, c]; nmap = np.array([len(p) for p in pk[k]]).reshape(W, W)
            ax.imshow(nmap.T, cmap=ncm, vmin=-0.5, vmax=3.5, origin="lower", extent=ext, interpolation="nearest")
            for j in range(3):
                segs = []
                for s_, x_, z_ in zip(site, cx, cz):
                    p = pk[k][s_]
                    if len(p) > j:
                        e = p[j]; inpl = np.hypot(e[0], e[2]); d = np.array([e[0], e[2]]) / max(inpl, 1e-9); h = 0.21 * inpl
                        segs.append([[x_ - d[0] * h, z_ - d[1] * h], [x_ + d[0] * h, z_ + d[1] * h]])
                if segs:
                    ax.add_collection(LineCollection(segs, colors="white", linewidths=3.2)); ax.add_collection(LineCollection(segs, colors=pcol[j], linewidths=1.6))
            q = res[name][k]
            txt = f"N = {len(tru)} at {q['N correct']:.0%} of sites; " + (f"error {q['error top peak']:.0f}°" if "error top peak" in q else
                  (f"errors {q['errors when 2 found (fibre 1 / fibre 2)'][0]:.0f}° / {q['errors when 2 found (fibre 1 / fibre 2)'][1]:.0f}°, both < 15°: "
                   f"{q['both within 15° (all sites)']:.0%}" if q['errors when 2 found (fibre 1 / fibre 2)'] else "no 2-peak sites"))
            ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round", fc="w", alpha=0.9, lw=0))
        for c in range(3):
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
    for c, t in enumerate(("image (x-z, mid y)", "intensity gradients:\nFRT-ODF peaks", "16-D shape-space gradients:\nFRT-ODF peaks")):
        axs[0, c].set_title(t, fontsize=12)
    fig.suptitle("PROTOTYPE. N fibre populations per site, found as peaks of the Funk-Radon transform of the gradient-direction "
                 f"distribution (ρ {RHO:g} vox, SH order {LMAX}).\nBackground = N found (white 0, light 1, mid 2, dark 3); sticks = peaks "
                 "(black 1st, orange 2nd, green 3rd).", fontsize=12, x=0.01, ha="left", y=0.995)
    fig.subplots_adjust(top=0.95, wspace=0.04, hspace=0.1, left=0.12, right=0.99)
    out = ST.FIGS / "shape_npop.png"; fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main()
