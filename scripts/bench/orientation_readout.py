"""Stage 2 by a self-calibrated orientation readout of the shape space (readout.orientation), on the phantoms.

Calibration (no labels): the readout is fitted on single-fibre blocks (12 angles, seed A) with the *excised
structure tensor's* direction at its confident sites as the teacher. Test: per-voxel orientation error on the
seed-B single-fibre blocks; then the 90 / 60 / 45 degree and 4:1 crossings -- the per-voxel orientation map, the
window modes (N, directions), the per-population voxel assignment, and one tensor per mode (stage 3) -- for the
encoder-2, layer-58 and jet spaces. Small blocks, GPU.

    python scripts/bench/orientation_readout.py     -> results/orientation_readout.json, figures/tca_field/orientation_readout_*.png
"""
import json, math, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shapetensor import DATA_ROOT, st
from shapetensor.features import make_engine
from shapetensor.phantoms import make
from shapetensor.readout import common, orientation
from shapetensor.readout.orientation import OrientationReadout, assign, window_modes
from shapetensor.tensor import gradients

FIGS = Path(__file__).resolve().parents[2] / "figures" / "tca_field"
CAT = ["#6250d6", "#e34948", "#eda100", "#2a9d8f"]
ANG = np.arange(0, 180, 15.0)
CROSS = {"c90": (30.0, 120.0, 0.5), "c60": (30.0, 90.0, 0.5), "c45": (30.0, 75.0, 0.5), "c60-f4": (30.0, 90.0, 0.8)}
SPACES = {"anatomix_enc2": {"nyquist": "none"}, "anatomix48": {"nyquist": "none", "n_pcs": 16}, "jet": {}}
RHO_TEACHER, RHO_WINDOW, SIGMA, TR, STRIDE = 8.0, 10.0, 1.0, 4.0, 8


def build(angles, seed, tex, g):
    rng = np.random.default_rng(seed)
    sp = make.Splat(g, tex, rng, "cuda")
    for a, f in angles:
        make._add_population(sp, a, f)
    d = sp.render()
    masks = []
    for a, f in angles:
        sp2 = make.Splat(g, tex, np.random.default_rng(seed), "cuda")
        for a2, f2 in angles:
            make._add_population(sp2, a2, f2 if a2 == a else 0.0)
        masks.append(sp2.render() > 0.25)
    Zm, Xm = np.meshgrid(g.coords(0), g.coords(2), indexing="ij")
    img = make._finish(g, tex, np.full(Zm.shape, 164.0, np.float32), np.full(Zm.shape, tex.amp, np.float32), d, rng)
    return torch.as_tensor(img, device="cuda").float(), (d > 0.25), masks


def unit(a):
    return torch.tensor([math.cos(math.radians(a)), 0.0, math.sin(math.radians(a))], device="cuda")


def angle(d, t):
    return torch.rad2deg(torch.arccos((d * t).sum(-1).abs().clamp(max=1)))


def teacher(v):
    """The excised tensor's direction (z, y, x) and gap on the block, at rho_teacher."""
    S = st.excised_structure_tensor_3d(v, SIGMA, RHO_TEACHER, TR)
    val, vec = st.eig_special_3d(S)                                                      # vec (x, y, z)
    d = vec.flip(0); gap = ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1)
    return d, gap


def main():
    t0 = time.time()
    tex = json.load(open(f"{DATA_ROOT}/results/tex_calibrated.json"))["texture"]; tex["widths"] = [tuple(w) for w in tex["widths"]]
    tex = make.TextureParams(**{**tex, "dispersion_deg": 8.0, "pitch_deg": 4.0})
    g = make.Grid(vox=0.02, core=(96, 32, 96), margin=(32, 32, 32)); Z, Y, X = g.shape; y_mid = Y // 2
    inner = (slice(36, Z - 36), slice(12, Y - 12), slice(36, X - 36))
    res = {}
    fig, axs = plt.subplots(2, 3, figsize=(18, 11), dpi=100)
    for j, (sname, ekw) in enumerate(SPACES.items()):
        eng = make_engine(sname, **ekw)
        # --- calibration blocks (seed A): features, the teacher, confident single-population voxels
        blocks = []
        for k, a in enumerate(ANG):
            v, fib, _ = build([(a, 1.0)], 100 + k, tex, g)
            F = eng(v); d, gap = teacher(v)
            w = torch.zeros_like(gap); w[inner] = (gap[inner] > torch.quantile(gap[inner], 0.7)).float() * fib[inner].float()
            blocks.append((F, d, w))
        ro = OrientationReadout(smooth=3, ridge=1.0).fit(blocks, n_sample=240_000)
        del blocks; torch.cuda.empty_cache()
        r = {"fit_r2": round(ro.fit_r2, 3)}
        # --- test on seed B single fibres: per-voxel error on fibre voxels, teacher error for reference
        errs, errs_t, confs = [], [], []
        for k, a in enumerate(ANG):
            v, fib, _ = build([(a, 1.0)], 200 + k, tex, g)
            o, conf = ro.predict(eng(v)); d, gap = teacher(v)
            m = fib & torch.zeros_like(fib).index_fill_(0, torch.arange(36, Z - 36, device="cuda"), True)
            e = angle(o.permute(1, 2, 3, 0), unit(a)); et = angle(d.permute(1, 2, 3, 0), unit(a))
            errs.append(e[m & (conf > 0)].cpu()); errs_t.append(et[m].cpu()); confs.append(conf[m].cpu())
        errs, errs_t, confs = torch.cat(errs), torch.cat(errs_t), torch.cat(confs)
        r["single_voxel_err_median"] = round(float(errs.median()), 1); r["single_voxel_within_15"] = round(float((errs <= 15).float().mean()), 3)
        r["single_teacher_err_median_rho8"] = round(float(errs_t.median()), 1); r["single_conf_median"] = round(float(confs.median()), 3)
        print(f"[{time.time() - t0:4.0f}s] {sname}: fit R2 {ro.fit_r2:.2f}; single fibres per voxel: median {errs.median():.1f} deg, within 15: {(errs <= 15).float().mean():.0%} (teacher rho 8: {errs_t.median():.1f} deg)", flush=True)
        # null for the modes: the second mode's relative height on single-fibre windows
        half = int(RHO_WINDOW * 2)
        zs = range(44, Z - 44, STRIDE); xs = range(44, X - 44, STRIDE)
        second = []
        for k, a in enumerate(ANG[::3]):
            v, fib, _ = build([(a, 1.0)], 200 + 3 * k, tex, g)
            o, conf = ro.predict(eng(v))
            for z in zs:
                for x in xs:
                    sl = (slice(None), slice(z - half, z + half + 1), slice(y_mid - half // 2, y_mid + half // 2 + 1), slice(x - half, x + half + 1))
                    oo = o[sl].reshape(3, -1).T; ww = (conf[sl[1:]] * fib[sl[1:]].float()).reshape(-1)
                    _, h, n = window_modes(oo, ww, rel=0.0)
                    second.append(float(h[1]))
        thr = float(np.percentile(second, 99)); r["mode_rel_thr_99"] = round(thr, 3)
        print(f"[{time.time() - t0:4.0f}s] {sname}: null second-mode height 99th pct {thr:.3f}", flush=True)
        # --- the crossings
        for cname, (a1, a2, f1) in CROSS.items():
            v, fib, masks = build([(a1, f1), (a2, 1 - f1)], 300, tex, g)
            F = eng(v); o, conf = ro.predict(F); G = gradients(v[None], SIGMA, TR)
            t1, t2 = unit(a1), unit(a2)
            m1, m2 = masks[0] & ~masks[1], masks[1] & ~masks[0]
            e1 = angle(o.permute(1, 2, 3, 0), t1); e2 = angle(o.permute(1, 2, 3, 0), t2)
            vox = {"within_15_of_either": round(float(((e1 <= 15) | (e2 <= 15))[fib & (conf > 0)].float().mean()), 3),
                   "pop1_voxels_to_pop1": round(float((e1 <= 15)[m1].float().mean()), 3), "pop2_voxels_to_pop2": round(float((e2 <= 15)[m2].float().mean()), 3)}
            found, both, errs_pair, nfound = 0, 0, [], []
            for z in zs:
                for x in xs:
                    sl = (slice(None), slice(z - half, z + half + 1), slice(y_mid - half // 2, y_mid + half // 2 + 1), slice(x - half, x + half + 1))
                    oo = o[sl].reshape(3, -1).T; ww = (conf[sl[1:]] * fib[sl[1:]].float()).reshape(-1)
                    modes, h, n = window_modes(oo, ww, rel=max(thr, 0.2))
                    nfound.append(n)
                    if n >= 2:
                        found += 1
                        # stage 3: a membership-weighted intensity tensor per mode
                        mem = assign(oo, modes, n)
                        gg = G[sl].reshape(3, -1)
                        dirs = []
                        for k in range(2):
                            wk = ww * mem[:, k]
                            S = (gg * wk) @ gg.T / wk.sum().clamp_min(1e-6); _, V = common.eigh(S[None]); dirs.append(V[0][:, 0])
                        d1, d2 = dirs
                        ea = max(float(angle(d1, t1)), float(angle(d2, t2))); eb = max(float(angle(d1, t2)), float(angle(d2, t1)))
                        best = min(ea, eb); errs_pair.append(best); both += int(best <= 15)
                        if cname == "c60" and z == list(zs)[len(zs) // 2] and x == list(xs)[len(xs) // 2]:
                            vox["mode_dirs_centre_err"] = [round(float(angle(modes[0], t1)), 1), round(float(angle(modes[1], t2)), 1), round(float(angle(modes[0], t2)), 1), round(float(angle(modes[1], t1)), 1)]
            nw = len(nfound)
            r[cname] = {**vox, "found2_share": round(found / nw, 3), "both_within_15_share": round(both / nw, 3), "pair_err_median": round(float(np.median(errs_pair)) if errs_pair else float("nan"), 1),
                        "n_share": [round(float(np.mean(np.array(nfound) == k)), 3) for k in range(4)]}
            print(f"[{time.time() - t0:4.0f}s] {sname} {cname}: {r[cname]}", flush=True)
            if cname == "c60":
                ax = axs[0, j]
                img = v[36:Z - 36, y_mid, 36:X - 36].cpu().numpy(); ax.imshow(img, cmap="gray", origin="lower")
                oz = o[:, 36:Z - 36, y_mid, 36:X - 36]; ang_map = torch.rad2deg(torch.atan2(oz[2], oz[0])) % 180
                msk = (fib & (conf > 0.2))[36:Z - 36, y_mid, 36:X - 36]
                ax.imshow(np.ma.masked_where(~msk.cpu().numpy(), ang_map.cpu().numpy()), cmap="hsv", vmin=0, vmax=180, origin="lower", alpha=0.75, interpolation="nearest")
                ax.set_title(f"{sname}: per-voxel orientation on the 60° crossing\nwithin 15° of either: {vox['within_15_of_either']:.0%}; pop1→30°: {vox['pop1_voxels_to_pop1']:.0%}, pop2→90°: {vox['pop2_voxels_to_pop2']:.0%}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
                ax = axs[1, j]
                ax.hist(ang_map[msk].cpu().numpy(), bins=np.arange(0, 181, 5), color=CAT[0]); ax.axvline(a1, color="k"); ax.axvline(a2, color="0.4")
                ax.set_title(f"windows: N=2 at {r[cname]['found2_share']:.0%}, both within 15°: {r[cname]['both_within_15_share']:.0%}", fontsize=10); ax.set_xlabel("per-voxel orientation (deg)"); ax.set_yticks([])
            del F, G; torch.cuda.empty_cache()
        res[sname] = r
        json.dump(res, open(f"{DATA_ROOT}/results/orientation_readout.json", "w"), indent=1)
    fig.suptitle("Stage 2 by a self-calibrated orientation readout of the shape space (teacher: the excised tensor on confident single-fibre voxels; no labels)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIGS / "orientation_readout_crossing.png"); plt.close(fig)
    print("ORIENTATION READOUT DONE", flush=True)


if __name__ == "__main__":
    main()
