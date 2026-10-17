"""Per-population layers of the shape space on the crossing phantoms: the signature dictionary (label-free), the
per-voxel unmixing onto the populations' signatures, each population's activation volume (its own image from the
shape space), the residual, and the direction field of each layer's own structure tensor against the truth.

    python scripts/fig_population_layers.py [--cross 60 90 45] [--rho 10]   -> figures/tca_field/population_layers_c<deg>.png
"""
import argparse, json, math, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shapetensor import DATA_ROOT, st
from shapetensor.features import make_engine
from shapetensor.features.cache import cached_features
from shapetensor.phantoms import make
from shapetensor.readout.layers import SignatureDictionary, layer_tensor_direction, unmix
from shapetensor.readout.orientation import OrientationReadoutDist, modes_from_density
from shapetensor.viz import sticks

FIGS = Path(__file__).resolve().parents[1] / "figures" / "tca_field"
CAT = ["#6250d6", "#e34948", "#eda100", "#2a9d8f"]
ANG = np.arange(0, 180, 15.0)
SIGMA, TR, RHO_TEACHER = 1.0, 4.0, 8.0
cmap = plt.get_cmap("Blues")


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


def angle_map(d, t):
    return torch.rad2deg(torch.arccos((d * t[:, None, None, None]).sum(0).abs().clamp(max=1)))


def teacher(v):
    S = st.excised_structure_tensor_3d(v, SIGMA, RHO_TEACHER, TR)
    val, vec = st.eig_special_3d(S)
    return vec.flip(0), ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cross", type=float, nargs="+", default=[60.0, 90.0, 45.0]); ap.add_argument("--engine", default="anatomix_enc2")
    ap.add_argument("--rho", type=float, default=10.0); ap.add_argument("--smooth", type=int, default=3)
    a = ap.parse_args(); t0 = time.time()
    tex = json.load(open(f"{DATA_ROOT}/results/tex_calibrated.json"))["texture"]; tex["widths"] = [tuple(w) for w in tex["widths"]]
    tex = make.TextureParams(**{**tex, "dispersion_deg": 8.0, "pitch_deg": 4.0})
    g = make.Grid(vox=0.02, core=(96, 32, 96), margin=(32, 32, 32)); Z, Y, X = g.shape; y_mid = Y // 2
    inner = (slice(36, Z - 36), slice(12, Y - 12), slice(36, X - 36))
    eng = make_engine(a.engine, nyquist="none")
    blocks, bg = [], []
    for k, ang in enumerate(ANG):
        v, fib, _ = build([(ang, 1.0)], 100 + k, tex, g)
        F = cached_features(eng, v); d, gap = teacher(v)
        w = torch.zeros_like(gap); w[inner] = (gap[inner] > torch.quantile(gap[inner], 0.7)).float() * fib[inner].float()
        blocks.append((F, d, w))
        wb = torch.zeros_like(gap); wb[inner] = (gap[inner] < torch.quantile(gap[inner], 0.3)).float() * (~fib[inner]).float()   # no fibre, low gap: background
        bg.append((F, wb))
    dic = SignatureDictionary(smooth=a.smooth).fit(blocks, background=bg)
    ro = OrientationReadoutDist(smooth=a.smooth).fit(blocks, n_sample=240_000)
    del blocks; torch.cuda.empty_cache()
    print(f"[{time.time() - t0:4.0f}s] dictionary and readout fitted", flush=True)
    out = {}
    for cross in a.cross:
        a1, a2 = 30.0, 30.0 + cross
        v, fib, masks = build([(a1, 0.5), (a2, 0.5)], 300, tex, g)
        F = cached_features(eng, v)
        # the populations present: the modes of the slab's summed per-voxel distribution
        P = ro.predict_dist(F)
        sl = (slice(36, Z - 36), slice(y_mid - 4, y_mid + 5), slice(36, X - 36))
        dens = (P[(slice(None),) + sl] * fib[sl].float()[None]).reshape(P.shape[0], -1).sum(1)
        modes, h, n = modes_from_density(dens, ro.grid, rel=0.3, min_sep_deg=25.0, nmax=3, mass_floor=0.1); del P
        t1, t2 = unit(a1), unit(a2)
        em = [[float(torch.rad2deg(torch.arccos((modes[r] * t).sum().abs().clamp(max=1)))) for t in (t1, t2)] for r in range(n)]
        order = sorted(range(n), key=lambda r: min(em[r]))[:2]
        # the layers: unmixing onto the two populations' signatures
        A, R = unmix(F, dic, modes[order])
        res = {"n_modes": int(n), "mode_err": [round(min(e), 1) for e in em], "residual_median": round(float(R[inner].median()), 3)}
        row = (slice(36, Z - 36), y_mid, slice(36, X - 36))
        img = v[row].cpu().numpy(); tm = [m[row].cpu().numpy() for m in masks]
        fig, axs = plt.subplots(2, 4, figsize=(22, 11), dpi=100)
        axs[0, 0].imshow(img, cmap="gray", origin="lower"); axs[0, 0].set_title(f"{cross:.0f}° crossing: image (populations at {a1:.0f}° and {a2:.0f}°)")
        axs[0, 3].imshow(R[row].cpu().numpy(), cmap=cmap, vmin=0, vmax=1, origin="lower", interpolation="nearest"); axs[0, 3].set_title(f"residual energy share (median {res['residual_median']:.2f})")
        for k, r in enumerate(order):
            tk = 0 if em[r][0] <= em[r][1] else 1; t = (t1, t2)[tk]
            act = A[k]
            d_layer, gap_layer = layer_tensor_direction(act, SIGMA, a.rho, TR)
            err = angle_map(d_layer, t)
            onpop = masks[tk]; other = masks[1 - tk]
            res[f"layer{k + 1}"] = {"truth": tk + 1, "mode_err": round(em[r][tk], 1),
                                    "activation_on_own_voxels_vs_others": [round(float(act[onpop & fib].mean()), 3), round(float(act[other & fib].mean()), 3)],
                                    "layer_dir_err_median_on_own_voxels": round(float(err[onpop & torch.zeros_like(onpop).index_fill_(0, torch.arange(36, Z - 36, device="cuda"), True)].median()), 1),
                                    "layer_dir_err_median_all_fibre_voxels": round(float(err[fib].median()), 1),
                                    "layer_dir_within_15_on_own_voxels": round(float((err[onpop & fib] <= 15).float().mean()), 3)}
            ax = axs[0, 1 + k]
            ax.imshow(act[row].cpu().numpy(), cmap=cmap, vmin=0, vmax=float(torch.quantile(act[inner], 0.99)), origin="lower", interpolation="nearest")
            ax.set_title(f"population {k + 1} activation volume (its signature's weight per voxel)\nmean on own voxels {res[f'layer{k + 1}']['activation_on_own_voxels_vs_others'][0]:.2f} vs on the other's {res[f'layer{k + 1}']['activation_on_own_voxels_vs_others'][1]:.2f}", fontsize=10)
            ax = axs[1, k]
            ax.imshow(img, cmap="gray", origin="lower", alpha=0.5)
            ax.imshow(np.ma.masked_where(~tm[tk], err[row].cpu().numpy()), cmap="Blues", vmin=0, vmax=90, origin="lower", alpha=0.8, interpolation="nearest")
            dd = d_layer[(slice(None),) + row].cpu().numpy(); nz, nx = dd.shape[1:]
            Zc, Xc = np.meshgrid(np.arange(nz) * 0.02, np.arange(nx) * 0.02, indexing="ij"); sel = np.zeros((nz, nx), bool); sel[::6, ::6] = True; sel &= tm[tk]
            sticks(ax, Zc[sel], Xc[sel], dd[:, sel].T, 6 * 0.02 * 0.9, plt.cm.Blues(np.clip(err[row].cpu().numpy()[sel] / 90, 0, 1)))
            ax.set_xlim(0, nx * 0.02); ax.set_ylim(0, nz * 0.02)
            ax.set_title(f"direction of layer {k + 1}'s own structure tensor (ρ {a.rho:g}) on population {tk + 1}'s voxels\nmedian error {res[f'layer{k + 1}']['layer_dir_err_median_on_own_voxels']:.1f}°, within 15°: {res[f'layer{k + 1}']['layer_dir_within_15_on_own_voxels']:.0%}", fontsize=10)
        # the reference: the plain structure tensor of the image on each population's voxels
        S = st.structure_tensor_3d(v, SIGMA, a.rho, TR); _, vec = st.eig_special_3d(S); d_img = vec.flip(0)
        e_img = [float(angle_map(d_img, t)[m & fib].median()) for t, m in ((t1, masks[0]), (t2, masks[1]))]
        res["image_tensor_err_on_pop_voxels"] = [round(e, 1) for e in e_img]
        ax = axs[1, 2]; ax.imshow(img, cmap="gray", origin="lower", alpha=0.5)
        for k, (m, t) in enumerate(((masks[0], t1), (masks[1], t2))):
            ax.imshow(np.ma.masked_where(~m[row].cpu().numpy(), angle_map(d_img, t)[row].cpu().numpy()), cmap="Blues", vmin=0, vmax=90, origin="lower", alpha=0.8, interpolation="nearest")
        ax.set_title(f"reference: the image's own structure tensor, error on each population's voxels: {e_img[0]:.1f}° / {e_img[1]:.1f}°", fontsize=10)
        ax = axs[1, 3]; ax.axis("off")
        ax.text(0.02, 0.95, "Validation\n" + "\n".join(f"{k}: {v}" for k, v in res.items()), va="top", fontsize=9, family="monospace", wrap=True)
        for ax in axs.ravel()[:7]:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Per-population layers of the shape space ({a.engine}): signature dictionary (label-free) → per-voxel non-negative unmixing onto the modes' signatures → one activation volume per population → that layer's own structure tensor", x=0.01, ha="left", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIGS / f"population_layers_c{cross:.0f}.png"); plt.close(fig)
        # the DEC overlay test: each layer's direction as colour, its activation as opacity, over the intensity image
        fig2, ax2 = plt.subplots(1, 4, figsize=(24, 6.3), dpi=100)
        ax2[0].imshow(img, cmap="gray", origin="lower"); ax2[0].set_title(f"{cross:.0f}° crossing: intensity")
        comb = np.zeros(img.shape + (3,)); wsum = np.zeros(img.shape)
        for k, r in enumerate(order):
            act = A[k][row].cpu().numpy(); wgt = np.clip(act / max(np.percentile(act, 99), 1e-6), 0, 1) ** 0.7
            dl, _ = layer_tensor_direction(A[k], SIGMA, a.rho, TR); dd = dl[(slice(None),) + row].cpu().numpy()
            rgbk = np.clip(np.stack([np.abs(dd[2]), np.abs(dd[1]), np.abs(dd[0])], -1), 0, 1)
            ax2[1 + k].imshow(img, cmap="gray", origin="lower"); ax2[1 + k].imshow(np.concatenate([rgbk, wgt[..., None]], -1), origin="lower", interpolation="nearest")
            ax2[1 + k].set_title(f"population {k + 1}: DEC of its layer's direction, opacity = activation", fontsize=10)
            comb += rgbk * wgt[..., None]; wsum += wgt
        ax2[3].imshow(img, cmap="gray", origin="lower"); ax2[3].imshow(np.concatenate([np.clip(comb / np.maximum(wsum, 1e-6)[..., None], 0, 1), np.clip(wsum, 0, 1)[..., None]], -1), origin="lower", interpolation="nearest")
        ax2[3].set_title("both: activation-weighted DEC, opacity = total activation", fontsize=10)
        for ax in ax2: ax.set_xticks([]); ax.set_yticks([])
        fig2.suptitle("DEC orientation per population, weighted by activation, over the image (R = |x|, G = |y|, B = |z|; fibres lie in the z–x plane: x = red, z = blue)", x=0.01, ha="left", fontsize=11)
        fig2.tight_layout(rect=[0, 0, 1, 0.93]); fig2.savefig(FIGS / f"population_layers_dec_c{cross:.0f}.png"); plt.close(fig2)
        out[f"c{cross:.0f}"] = res
        print(f"[{time.time() - t0:4.0f}s] {cross:.0f}°: {res}", flush=True)
        del F, A, R, S; torch.cuda.empty_cache()
    json.dump(out, open(f"{DATA_ROOT}/results/population_layers.json", "w"), indent=1)
    print("LAYERS DONE", flush=True)


if __name__ == "__main__":
    main()
