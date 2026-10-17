"""One example, start to finish: how a population's activation image is made from the shape space.

Row 1 — the image and its shape space: three encoder-2 channels (the one most tuned to each population's angle and
an untuned one) and the feature RGB.
Row 2 — the division of the shape space: the signature dictionary (channel patterns of the two populations' angles
and of the background); the orientation plane of the dictionary (the two principal directions of the angle
signatures, on which the 12 calibration angles lie on a circle: the double-angle code) with the crossing's voxels
projected into it, coloured by their true population; the per-voxel distribution's density over directions with
its modes and the truths.
Row 3 — how the activation image is created: the unmixing at one voxel (its standardised pattern = a1 S1 + a2 S2 +
a_bg S_bg + residual, as bars); the raw activation volumes of the two populations; the 3³-averaged activations
with the mask contour (the population images, and where their vectors go).

    python scripts/fig_activation_walkthrough.py [--cross 60]   -> figures/tca_field/activation_walkthrough_c<deg>.png
"""
import argparse, json, math, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fn

from shapetensor import DATA_ROOT, st
from shapetensor.features import make_engine
from shapetensor.features.cache import cached_features
from shapetensor.phantoms import make
from shapetensor.readout.layers import SignatureDictionary, activation_null, nnls_batched, unmix
from shapetensor.readout.orientation import OrientationReadoutDist, modes_from_density

FIGS = Path(__file__).resolve().parents[1] / "figures" / "tca_field"
CAT = ["#6250d6", "#e34948", "#eda100", "#2a9d8f"]
ANG = np.arange(0, 180, 15.0)
SIGMA, TR, RHO_TEACHER = 1.0, 4.0, 8.0


def build(angles, seed, tex, g):
    rng = np.random.default_rng(seed)
    sp = make.Splat(g, tex, rng, "cuda")
    for a, f in angles:
        make._add_population(sp, a, f)
    d = sp.render()
    if len(angles) > 1:
        sp1 = make.Splat(g, tex, np.random.default_rng(seed), "cuda"); make._add_population(sp1, angles[0][0], angles[0][1])
        m1 = sp1.render() > 0.25; masks = [m1, (d > 0.25) & ~m1]
    else:
        masks = [d > 0.25]
    Zm, Xm = np.meshgrid(g.coords(0), g.coords(2), indexing="ij")
    img = make._finish(g, tex, np.full(Zm.shape, 164.0, np.float32), np.full(Zm.shape, tex.amp, np.float32), d, rng)
    return torch.as_tensor(img, device="cuda").float(), (d > 0.25), masks


def unit(a):
    return torch.as_tensor(np.asarray(make._dir(a), np.float32), device="cuda")


def teacher(v):
    S = st.excised_structure_tensor_3d(v, SIGMA, RHO_TEACHER, TR)
    val, vec = st.eig_special_3d(S)
    return vec.flip(0), ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cross", type=float, default=60.0); a = ap.parse_args(); t0 = time.time()
    tex = json.load(open(f"{DATA_ROOT}/results/tex_calibrated.json"))["texture"]; tex["widths"] = [tuple(w) for w in tex["widths"]]
    tex = make.TextureParams(**{**tex, "dispersion_deg": 8.0, "pitch_deg": 4.0})
    g = make.Grid(vox=0.02, core=(96, 32, 96), margin=(32, 32, 32)); Z, Y, X = g.shape; y_mid = Y // 2
    inner = (slice(36, Z - 36), slice(12, Y - 12), slice(36, X - 36))
    eng = make_engine("anatomix_enc2", nyquist="none")
    blocks, bg = [], []
    for k, ang in enumerate(ANG):
        v, fib, _ = build([(ang, 1.0)], 100 + k, tex, g)
        F = cached_features(eng, v); d, gap = teacher(v)
        w = torch.zeros_like(gap); w[inner] = (gap[inner] > torch.quantile(gap[inner], 0.7)).float() * fib[inner].float()
        wb = torch.zeros_like(gap); wb[inner] = (gap[inner] < torch.quantile(gap[inner], 0.3)).float() * (~fib[inner]).float()
        blocks.append((F, d, w)); bg.append((F, wb))
    dic = SignatureDictionary(smooth=3).fit(blocks, background=bg)
    ro = OrientationReadoutDist(smooth=3).fit(blocks, n_sample=240_000)
    tau = activation_null(dic, blocks); del blocks, bg; torch.cuda.empty_cache()
    a1, a2 = 30.0, 30.0 + a.cross
    v, fib, masks = build([(a1, 0.5), (a2, 0.5)], 300, tex, g)
    F = cached_features(eng, v); f = dic._prep(F)
    row = (slice(36, Z - 36), y_mid, slice(36, X - 36))
    img = v[row].cpu().numpy(); tm = [m[row].cpu().numpy() for m in masks]; fibr = fib[row].cpu().numpy()
    # --- the populations present and their signatures
    P = ro.predict_dist(F)
    sl = (slice(36, Z - 36), slice(y_mid - 4, y_mid + 5), slice(36, X - 36))
    dens = (P[(slice(None),) + sl] * fib[sl].float()[None]).reshape(P.shape[0], -1).sum(1)
    modes, h, n = modes_from_density(dens, ro.grid, rel=0.3, min_sep_deg=25.0, nmax=3, mass_floor=0.1)
    t1, t2 = unit(a1), unit(a2)
    em = [[float(torch.rad2deg(torch.arccos((modes[r] * t).sum().abs().clamp(max=1)))) for t in (t1, t2)] for r in range(n)]
    order = sorted(range(n), key=lambda r: min(em[r]))[:2]
    order = sorted(order, key=lambda r: 0 if em[r][0] <= em[r][1] else 1)             # population 1 first
    S1, S2, Sb = dic.signature(modes[order[0]]), dic.signature(modes[order[1]]), dic.background
    A, R = unmix(F, dic, modes[order]); As = Fn.avg_pool3d(A[None], 3, stride=1, padding=1)[0]
    # --- the orientation plane: PCA of the angle signatures
    Dang = torch.stack([dic.signature(unit(ang)) for ang in ANG])                      # (12, C)
    mu = Dang.mean(0); U, Sv, Vt = torch.linalg.svd(Dang - mu, full_matrices=False); E = Vt[:2].T   # (C, 2)
    ang_xy = ((Dang - mu) @ E).cpu().numpy()
    Xrow = ((f[(slice(None),) + row].reshape(f.shape[0], -1).T - dic.mean) / dic.std)        # (n_vox, C) standardised
    vox_xy = ((Xrow - mu) @ E).cpu().numpy()
    lab = np.zeros(fibr.size, int); lab[tm[0].ravel()] = 1; lab[tm[1].ravel()] = 2
    both = (masks[0] & masks[1])[row].cpu().numpy().ravel(); lab[both] = 3
    s1_xy, s2_xy, sb_xy = [((S - mu) @ E).cpu().numpy() for S in (S1, S2, Sb)]
    # --- the example voxel: a population-1 voxel with a strong activation
    cand = torch.nonzero(masks[0][row] & (As[0][row] > torch.quantile(As[0][row][masks[0][row]], 0.9)))
    iz, ix = [int(c) for c in cand[len(cand) // 2]]
    x_vox = Xrow[iz * (X - 72) + ix]
    Sm = torch.stack([S1, S2, Sb], 1); coef = nnls_batched(Sm, x_vox[None])[0]
    fit = Sm @ coef; resid = x_vox - fit
    # --- channels most tuned to each population and one untuned
    chan_resp = torch.stack([dic.signature(unit(ang)) for ang in ANG]).cpu().numpy()              # (12, C)
    c1 = int(np.argmax(dic.signature(modes[order[0]]).cpu().numpy())); c2 = int(np.argmax(dic.signature(modes[order[1]]).cpu().numpy()))
    depth = (chan_resp.max(0) - chan_resp.min(0)); c0 = int(np.argmin(depth))
    rgb = f[:3][(slice(None),) + row].cpu().numpy(); lo, hi = np.percentile(rgb, 1, axis=(1, 2)), np.percentile(rgb, 99, axis=(1, 2))
    rgb = np.clip((rgb - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1).transpose(1, 2, 0)
    # ---------------------------------------------------------------- figure
    fig, axs = plt.subplots(3, 4, figsize=(24, 17), dpi=100)
    axs[0, 0].imshow(img, cmap="gray", origin="lower"); axs[0, 0].set_title(f"1a. image: {a.cross:.0f}° crossing, populations at {a1:.0f}° (1) and {a2:.0f}° (2)")
    for ax, c, name in ((axs[0, 1], c1, f"channel {c1}: the one most excited by population 1's angle"), (axs[0, 2], c2, f"channel {c2}: most excited by population 2's angle"), (axs[0, 3], c0, f"channel {c0}: an untuned channel (texture)")):
        ax.imshow(f[c][row].cpu().numpy(), cmap="magma", origin="lower"); ax.set_title("1b–d. shape space — " + name, fontsize=10)
    axs[1, 0].imshow(rgb, origin="lower", interpolation="nearest"); axs[1, 0].set_title("2a. shape space as RGB (the first three encoder-2 channels)")
    ax = axs[1, 1]
    for S, col, name in ((S1, CAT[0], "signature of population 1"), (S2, CAT[1], "signature of population 2"), (Sb, "0.4", "background signature")):
        ax.plot(S.cpu().numpy(), "o-", color=col, ms=3, lw=1, label=name)
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("encoder-2 channel"); ax.set_ylabel("standardised mean response"); ax.legend(fontsize=9); ax.set_title("2b. the dictionary: each population's channel pattern (label-free)", fontsize=10)
    ax = axs[1, 2]
    for L, col, name in ((0, "0.75", "background / no fibre"), (1, CAT[0], "population 1 voxels"), (2, CAT[1], "population 2 voxels"), (3, CAT[2], "both (intersections)")):
        m = lab == L; ax.scatter(vox_xy[m, 0], vox_xy[m, 1], s=2, color=col, alpha=0.4, label=name)
    ax.plot(np.append(ang_xy[:, 0], ang_xy[0, 0]), np.append(ang_xy[:, 1], ang_xy[0, 1]), "k-", lw=1)
    for k, ang in enumerate(ANG):
        ax.annotate(f"{ang:.0f}°", ang_xy[k], fontsize=7, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    for xy, col in ((s1_xy, CAT[0]), (s2_xy, CAT[1])):
        ax.annotate("", xy=xy, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=col, lw=2.5))
    ax.set_title("2c. the orientation plane of the shape space: the 12 calibration angles lie on a circle (double-angle code);\nthe crossing's voxels projected, coloured by truth; arrows = the two detected populations' signatures", fontsize=9)
    ax.set_xlabel("principal direction 1 of the angle signatures"); ax.set_ylabel("principal direction 2"); ax.legend(fontsize=8, markerscale=4, loc="lower left")
    ax = axs[1, 3]
    th = torch.rad2deg(torch.atan2(ro.grid[:, 0], ro.grid[:, 2])) % 180
    o = torch.argsort(th); ax.plot(th[o].cpu().numpy(), (dens / dens.max())[o].cpu().numpy(), ".", ms=3, color=CAT[0])
    for r in range(n):
        t = float(torch.rad2deg(torch.atan2(modes[r][0], modes[r][2])) % 180); ax.axvline(t, color=CAT[1] if r in order else "0.6", lw=2 if r in order else 1)
    for t in (a1, a2):
        ax.axvline(t, color="k", lw=1, ls="--")
    ax.set_xlabel("direction (deg, in the fibre plane)"); ax.set_ylabel("summed per-voxel distribution"); ax.set_title(f"2d. the clustering: modes of the summed per-voxel orientation distribution\n(red = kept populations, grey = a third candidate; dashed = truths)", fontsize=9)
    ax = axs[2, 0]
    C = len(x_vox); idx = np.arange(C)
    ax.bar(idx - 0.3, x_vox.cpu().numpy(), width=0.3, color="k", label="voxel's standardised pattern")
    ax.bar(idx, (coef[0] * S1).cpu().numpy(), width=0.3, color=CAT[0], label=f"a1 · S1 (a1 = {float(coef[0]):.2f})")
    ax.bar(idx + 0.3, (coef[1] * S2).cpu().numpy(), width=0.3, color=CAT[1], label=f"a2 · S2 (a2 = {float(coef[1]):.2f})")
    ax.plot(idx, (coef[2] * Sb).cpu().numpy(), "o", color="0.4", ms=3, label=f"a_bg · S_bg (a_bg = {float(coef[2]):.2f})")
    ax.plot(idx, resid.cpu().numpy(), "x", color=CAT[2], ms=4, label="residual")
    ax.set_xlabel("encoder-2 channel"); ax.legend(fontsize=8); ax.set_title(f"3a. the unmixing at one voxel (z {iz}, x {ix}; a population-1 voxel): pattern = a1·S1 + a2·S2 + a_bg·S_bg + residual, a ≥ 0", fontsize=9)
    for k, ax in enumerate(axs[2, 1:3]):
        ax.imshow(A[k][row].cpu().numpy(), cmap="Blues", vmin=0, vmax=float(torch.quantile(A[k][inner], 0.99)), origin="lower", interpolation="nearest")
        ax.plot(ix, iz, "o", mfc="none", mec=CAT[2], ms=10, mew=2)
        ax.set_title(f"3b/c. raw activation a{k + 1}(v): the coefficient of population {k + 1}'s signature at every voxel\n(the population's own image from the shape space)", fontsize=9)
    ax = axs[2, 3]
    ax.imshow(img, cmap="gray", origin="lower", alpha=0.45)
    for k, col in ((0, CAT[0]), (1, CAT[1])):
        m = (As[k][row] > tau).cpu().numpy()
        ax.contour(m.astype(float), levels=[0.5], colors=[col], linewidths=1.5)
    ax.set_title(f"3d. the 3³-averaged activations above the null τ = {tau:.2f}: each population's mask\n(where its vectors go; the union = any population)", fontsize=9)
    for ax in list(axs[0]) + [axs[1, 0]] + list(axs[2, 1:]):
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"How a population's activation image is made (encoder-2 shape space, {a.cross:.0f}° crossing): image → shape space → its orientation plane and the modes → per-voxel non-negative unmixing onto the modes' signatures → activation volumes → masks", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FIGS / f"activation_walkthrough_c{a.cross:.0f}.png"); plt.close(fig)
    print(f"WALKTHROUGH FIGURE DONE {time.time() - t0:.0f} s; modes {[round(min(e), 1) for e in em]} deg from truths; example voxel coefficients {coef.tolist()}", flush=True)


if __name__ == "__main__":
    main()
