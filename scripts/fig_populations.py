"""Per-population images from the separated shape space (stages 2 and 3 on a crossing phantom).

The distribution readout (encoder-2 features -> a per-voxel distribution over directions, self-calibrated on the
single-fibre blocks) is applied to a crossing; the modes of the slab's summed distribution are the populations;
each voxel's membership of a population is the mass of its distribution within 20 degrees of that mode. The figure
shows, per crossing: the image, the true voxels of each population, the membership maps, the image masked by each
membership (the "population images"), with the Dice of membership > 0.5 against the truth and the mode directions'
errors; and the membership-weighted direction per population from the per-voxel distributions.

    python scripts/fig_populations.py [--cross 60 90 45] [--model dist]   -> figures/tca_field/populations_c<deg>.png
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


def angle(d, t):
    return float(torch.rad2deg(torch.arccos((d * t).sum(-1).abs().clamp(max=1))))


def teacher(v):
    S = st.excised_structure_tensor_3d(v, SIGMA, RHO_TEACHER, TR)
    val, vec = st.eig_special_3d(S)
    return vec.flip(0), ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1)


def dice(a, b):
    return float(2 * (a & b).sum() / max(1, int(a.sum() + b.sum())))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cross", type=float, nargs="+", default=[60.0, 90.0, 45.0]); ap.add_argument("--engine", default="anatomix_enc2")
    ap.add_argument("--member-deg", type=float, default=20.0)
    a = ap.parse_args(); t0 = time.time()
    tex = json.load(open(f"{DATA_ROOT}/results/tex_calibrated.json"))["texture"]; tex["widths"] = [tuple(w) for w in tex["widths"]]
    tex = make.TextureParams(**{**tex, "dispersion_deg": 8.0, "pitch_deg": 4.0})
    g = make.Grid(vox=0.02, core=(96, 32, 96), margin=(32, 32, 32)); Z, Y, X = g.shape; y_mid = Y // 2
    inner = (slice(36, Z - 36), slice(12, Y - 12), slice(36, X - 36))
    eng = make_engine(a.engine, nyquist="none")
    # the readout, self-calibrated on the single-fibre blocks (features from the cache)
    blocks = []
    for k, ang in enumerate(ANG):
        v, fib, _ = build([(ang, 1.0)], 100 + k, tex, g)
        F = cached_features(eng, v); d, gap = teacher(v)
        w = torch.zeros_like(gap); w[inner] = (gap[inner] > torch.quantile(gap[inner], 0.7)).float() * fib[inner].float()
        blocks.append((F, d, w))
    ro = OrientationReadoutDist(smooth=3).fit(blocks, n_sample=240_000); del blocks; torch.cuda.empty_cache()
    print(f"[{time.time() - t0:4.0f}s] readout fitted (target overlap {ro.fit_r2:.2f})", flush=True)
    out = {}
    for cross in a.cross:
        a1, a2 = 30.0, 30.0 + cross
        v, fib, masks = build([(a1, 0.5), (a2, 0.5)], 300, tex, g)
        F = cached_features(eng, v); P = ro.predict_dist(F)                             # (n_dirs, Z, Y, X)
        t1, t2 = unit(a1), unit(a2)
        sl = (slice(36, Z - 36), slice(y_mid - 4, y_mid + 5), slice(36, X - 36))        # a thin slab around the drawn row
        dens = (P[(slice(None),) + sl] * fib[sl].float()[None]).reshape(P.shape[0], -1).sum(1)
        modes, h, n = modes_from_density(dens, ro.grid, rel=0.3, min_sep_deg=25.0, nmax=3, mass_floor=0.1)
        # memberships: the mass of each voxel's distribution within member_deg of each mode
        c2 = (ro.grid @ modes[:max(n, 1)].T) ** 2                                            # (n_dirs, n)
        near = (c2 >= math.cos(math.radians(a.member_deg)) ** 2).float()
        M = torch.einsum("dzyx,dn->nzyx", P, near)                                           # (n, Z, Y, X) raw memberships
        Mn = M / M.sum(0, keepdim=True).clamp_min(1e-6)
        # assign the modes to the truths (nearest), report
        em = [[angle(modes[r], t) for t in (t1, t2)] for r in range(n)]
        order = sorted(range(n), key=lambda r: min(em[r]))[:2]
        res = {"n_modes": int(n), "mode_err_to_nearest_truth": [round(min(e), 1) for e in em]}
        row = (slice(None), slice(36, Z - 36), y_mid, slice(36, X - 36))
        img = v[row[1:]].cpu().numpy(); fibr = fib[row[1:]].cpu().numpy()
        tm = [m[row[1:]].cpu().numpy() for m in masks]
        fig, axs = plt.subplots(2, 4, figsize=(22, 11), dpi=100)
        axs[0, 0].imshow(img, cmap="gray", origin="lower"); axs[0, 0].set_title(f"{cross:.0f}° crossing: image (populations at {a1:.0f}° and {a2:.0f}°)")
        for k, (ax, m, col) in enumerate(zip(axs[0, 1:3], tm, CAT)):
            ax.imshow(img, cmap="gray", origin="lower", alpha=0.5); ax.imshow(np.ma.masked_where(~m, m), cmap=matplotlib.colors.ListedColormap([col]), origin="lower", alpha=0.9, interpolation="nearest")
            ax.set_title(f"truth: population {k + 1} voxels")
        axs[0, 3].axis("off")
        for k, r in enumerate(order):
            # which truth does this mode serve
            tk = 0 if em[r][0] <= em[r][1] else 1
            mem = Mn[r][row[1:]].cpu().numpy() * fibr
            dsc = dice((mem > 0.5), tm[tk])
            res[f"mode{k + 1}"] = {"truth": tk + 1, "dir_err": round(em[r][tk], 1), "dice_membership_gt_0.5_vs_truth": round(dsc, 3),
                                   "truth_voxels_with_membership_gt_0.5": round(float((mem > 0.5)[tm[tk]].mean()), 3)}
            ax = axs[1, k]
            ax.imshow(mem, cmap="viridis", vmin=0, vmax=1, origin="lower", interpolation="nearest"); ax.set_title(f"membership of population {k + 1} (mode {em[r][tk]:.1f}° from truth {tk + 1}); Dice {dsc:.2f}")
            ax = axs[1, 2 + k]
            ax.imshow(img * (0.15 + 0.85 * mem), cmap="gray", origin="lower", vmin=img.min(), vmax=img.max()); ax.set_title(f"population {k + 1} image (image × membership)")
        for ax in axs.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Per-population images from the separated shape space ({a.engine} → per-voxel orientation distribution → modes of the slab → memberships); "
                     f"{n} modes found, directions {[round(min(e), 1) for e in em]}° from the nearest truth", x=0.01, ha="left", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIGS / f"populations_c{cross:.0f}.png"); plt.close(fig)
        out[f"c{cross:.0f}"] = res
        print(f"[{time.time() - t0:4.0f}s] {cross:.0f}°: {res}", flush=True)
    json.dump(out, open(f"{DATA_ROOT}/results/populations_fig.json", "w"), indent=1)
    print("POPULATIONS DONE", flush=True)


if __name__ == "__main__":
    main()
