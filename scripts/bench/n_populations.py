"""How many populations? Phantoms with 1, 2 and 3 populations, N never specified: the modes of the per-voxel
distribution (density peaks above the single-population null, a mass floor) confirmed by the layers (a candidate
population must own a mask -- activation above the wrong-population null -- covering at least 5% of the fibre
voxels). Reports N for the slab and per window, each kept population's mode and layer-tensor direction against
the truth, the Dice of its mask against its true voxels, and exports one field (the 3-population case) as a zarr
group with the per-population masks, the union mask and the strongest-population map.

    python scripts/bench/n_populations.py     -> results/n_populations.json, figures/tca_field/n_populations.png
"""
import json, math, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from shapetensor import DATA_ROOT, st
from shapetensor.features import make_engine
from shapetensor.features.cache import cached_features
from shapetensor.phantoms import make
from shapetensor.readout.layers import (SignatureDictionary, activation_null, detect_populations, export_population_field,
                                        layer_tensor_direction)
from shapetensor.readout.orientation import OrientationReadoutDist, modes_from_density

FIGS = Path(__file__).resolve().parents[2] / "figures" / "tca_field"
CAT = ["#6250d6", "#e34948", "#eda100", "#2a9d8f"]
ANG = np.arange(0, 180, 15.0)
SIGMA, TR, RHO_TEACHER, RHO = 1.0, 4.0, 8.0, 10.0
CASES = {"1 population (30°)": [(30.0, 1.0)], "1 population (105°)": [(105.0, 1.0)],
         "2 populations 90° (30, 120)": [(30.0, 0.5), (120.0, 0.5)], "2 populations 60° (30, 90)": [(30.0, 0.5), (90.0, 0.5)],
         "3 populations (0, 60, 120)": [(0.0, 1 / 3), (60.0, 1 / 3), (120.0, 1 / 3)], "3 populations (20, 75, 130)": [(20.0, 1 / 3), (75.0, 1 / 3), (130.0, 1 / 3)]}


def build(angles, seed, tex, g):
    """The combined render and one mask per population from the same random sequence (population k's splats are
    drawn after populations < k, so rendering the first k populations with the same seed reproduces them)."""
    rng = np.random.default_rng(seed)
    sp = make.Splat(g, tex, rng, "cuda")
    for a, f in angles:
        make._add_population(sp, a, f)
    d = sp.render()
    masks, prev = [], torch.zeros_like(d, dtype=torch.bool)
    for k in range(1, len(angles) + 1):
        spk = make.Splat(g, tex, np.random.default_rng(seed), "cuda")
        for a, f in angles[:k]:
            make._add_population(spk, a, f)
        cum = spk.render() > 0.25
        masks.append(cum & ~prev); prev = prev | cum
    Zm, Xm = np.meshgrid(g.coords(0), g.coords(2), indexing="ij")
    img = make._finish(g, tex, np.full(Zm.shape, 164.0, np.float32), np.full(Zm.shape, tex.amp, np.float32), d, rng)
    return torch.as_tensor(img, device="cuda").float(), (d > 0.25), masks


def unit(a):
    return torch.as_tensor(np.asarray(make._dir(a), np.float32), device="cuda")


def ang(d, t):
    return float(torch.rad2deg(torch.arccos((d * t).sum(-1).abs().clamp(max=1))))


def teacher(v):
    S = st.excised_structure_tensor_3d(v, SIGMA, RHO_TEACHER, TR)
    val, vec = st.eig_special_3d(S)
    return vec.flip(0), ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1)


def dice(a, b):
    return float(2 * (a & b).sum() / max(1, int(a.sum() + b.sum())))


def main():
    t0 = time.time()
    tex = json.load(open(f"{DATA_ROOT}/results/tex_calibrated.json"))["texture"]; tex["widths"] = [tuple(w) for w in tex["widths"]]
    tex = make.TextureParams(**{**tex, "dispersion_deg": 8.0, "pitch_deg": 4.0})
    g = make.Grid(vox=0.02, core=(96, 32, 96), margin=(32, 32, 32)); Z, Y, X = g.shape; y_mid = Y // 2
    inner = (slice(36, Z - 36), slice(12, Y - 12), slice(36, X - 36))
    eng = make_engine("anatomix_enc2", nyquist="none")
    blocks, bg = [], []
    for k, a in enumerate(ANG):
        v, fib, _ = build([(a, 1.0)], 100 + k, tex, g)
        F = cached_features(eng, v); d, gap = teacher(v)
        w = torch.zeros_like(gap); w[inner] = (gap[inner] > torch.quantile(gap[inner], 0.7)).float() * fib[inner].float()
        wb = torch.zeros_like(gap); wb[inner] = (gap[inner] < torch.quantile(gap[inner], 0.3)).float() * (~fib[inner]).float()
        blocks.append((F, d, w)); bg.append((F, wb))
    dic = SignatureDictionary(smooth=3).fit(blocks, background=bg)
    ro = OrientationReadoutDist(smooth=3).fit(blocks, n_sample=240_000)
    tau = activation_null(dic, blocks)
    # the density null: the second mode's relative height on single-population windows (99th percentile)
    half = int(RHO * 2); zs = range(44, Z - 44, 8); xs = range(44, X - 44, 8)
    second = []
    for k, a in enumerate(ANG[::3]):
        v, fib, _ = build([(a, 1.0)], 200 + 3 * k, tex, g); P = ro.predict_dist(cached_features(eng, v))
        for z in zs:
            for x in xs:
                sl = (slice(None), slice(z - half, z + half + 1), slice(y_mid - half // 2, y_mid + half // 2 + 1), slice(x - half, x + half + 1))
                dens = (P[sl].reshape(P.shape[0], -1) * fib[sl[1:]].float().reshape(-1)[None]).sum(1)
                _, h, n = modes_from_density(dens, ro.grid, rel=0.0); second.append(float(h[1]))
    rel_thr = max(0.25, float(np.percentile(second, 99)))
    print(f"[{time.time() - t0:4.0f}s] calibrated: activation null tau {tau:.3f}, density null rel {rel_thr:.3f}", flush=True)
    del blocks, bg; torch.cuda.empty_cache()
    res = {"tau": tau, "rel_thr": rel_thr}
    fig, axs = plt.subplots(2, len(CASES), figsize=(5 * len(CASES), 10.5), dpi=100)
    for j, (name, angles) in enumerate(CASES.items()):
        v, fib, masks_t = build(angles, 300 + j, tex, g)
        F = cached_features(eng, v); P = ro.predict_dist(F)
        region = fib.clone(); region[~torch.zeros_like(fib).index_fill_(0, torch.arange(36, Z - 36, device="cuda"), True)] = False
        modes, A, R, masks, union, strongest, info = detect_populations(P, ro.grid, F, dic, tau, region=region, rel=rel_thr)
        N = int(modes.shape[0]); truths = [unit(a) for a, _ in angles]
        r = {"true_N": len(angles), "detected_N": N, "candidates": info["candidates"], "mask_shares": [round(s, 3) for s in info["mask_shares"]]}
        # per window N (the same rule on windows)
        nwin = []
        for z in zs:
            for x in xs:
                sl = (slice(None), slice(z - half, z + half + 1), slice(y_mid - half // 2, y_mid + half // 2 + 1), slice(x - half, x + half + 1))
                reg = torch.zeros_like(fib); reg[sl[1:]] = fib[sl[1:]]
                dens = (P * reg.float()[None]).reshape(P.shape[0], -1).sum(1)
                _, _, n = modes_from_density(dens, ro.grid, rel=rel_thr, mass_floor=0.1); nwin.append(n)
        r["window_N_shares"] = [round(float(np.mean(np.array(nwin) == k)), 3) for k in range(5)]
        pops = []
        dirs_all, gaps_all = [], []
        for k in range(N):
            d_layer, gap_layer = layer_tensor_direction(A[k], SIGMA, RHO, TR); dirs_all.append(d_layer); gaps_all.append(gap_layer)
            errs = [ang(modes[k], t) for t in truths]; tk = int(np.argmin(errs))
            err_map = torch.rad2deg(torch.arccos((d_layer * truths[tk][:, None, None, None]).sum(0).abs().clamp(max=1)))
            own = masks_t[tk] & region
            pops.append({"nearest_truth": tk + 1, "mode_err": round(errs[tk], 1), "layer_dir_err_on_truth_voxels": round(float(err_map[own].median()), 1),
                         "layer_dir_err_in_own_mask": round(float(err_map[masks[k] & region].median()), 1) if (masks[k] & region).any() else None,
                         "mask_dice_vs_truth": round(dice(masks[k] & region, own), 3), "mask_share_of_fibre": round(float((masks[k] & region).float().sum() / region.float().sum()), 3)})
        r["populations"] = pops; r["union_dice_vs_fibre"] = round(dice(union & region, region), 3)
        res[name] = r
        print(f"[{time.time() - t0:4.0f}s] {name}: true N {len(angles)}, detected {N} (candidates {info['candidates']}, mask shares {[round(s, 2) for s in info['mask_shares']]}); windows N shares {r['window_N_shares']}; "
              + "; ".join(f"pop{k + 1}->truth{p['nearest_truth']} mode {p['mode_err']}°, layer dir {p['layer_dir_err_on_truth_voxels']}°, Dice {p['mask_dice_vs_truth']}" for k, p in enumerate(pops)), flush=True)
        row = (slice(36, Z - 36), y_mid, slice(36, X - 36))
        ax = axs[0, j]; ax.imshow(v[row].cpu().numpy(), cmap="gray", origin="lower"); ax.set_title(f"{name}\ndetected N = {N} (true {len(angles)})", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        ax = axs[1, j]; ax.imshow(v[row].cpu().numpy(), cmap="gray", origin="lower", alpha=0.45)
        if N:
            s_row = strongest[row].cpu().numpy()
            ax.imshow(np.ma.masked_where(s_row == 0, s_row), cmap=ListedColormap(CAT[:max(N, 1)]), vmin=1, vmax=max(N, 1), origin="lower", alpha=0.85, interpolation="nearest")
        ax.set_title("strongest population per voxel (colour), union = any colour", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        if name.startswith("3 populations (0"):
            out = Path(DATA_ROOT) / "fields" / "phantom_3pop_population_field.zarr"
            export_population_field(out, modes, A, torch.stack(dirs_all), torch.stack(gaps_all), masks, union, strongest, 0.02, {"phantom": name, "tau": tau})
            r["exported"] = str(out)
        del F, P, A; torch.cuda.empty_cache()
    fig.suptitle("N populations without specifying N: density modes above the single-population null, confirmed by each candidate's layer mask (activation above the wrong-population null)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIGS / "n_populations.png"); plt.close(fig)
    json.dump(res, open(f"{DATA_ROOT}/results/n_populations.json", "w"), indent=1)
    print("N POPULATIONS DONE", flush=True)


if __name__ == "__main__":
    main()
