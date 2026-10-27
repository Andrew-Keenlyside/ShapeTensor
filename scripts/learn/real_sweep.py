"""The readout on real HiP-CT across resolution and smoothing: the five 20 um deep-brain crops, each centred on the same
point and block-averaged by ``--pools`` (20, 40, 80, 160 um voxels: a 96^3 input spans 1.9-15 mm), read out on the middle
slice at window sigma ``--sigmas`` voxels (the default continuous readout).

Types enter as the readout's sheet weight (smooth in the ring cover): each population's line part (mass x (1 - sheet))
by its axis, its sheet part by its normal.

Figures: per specimen, a grid (rows: resolution, columns: window) of the oriented mass -- lines coloured by their mean
direction (red x, green y, blue z), sheets grey, brightness = mass; and the metrics against the window's physical size,
per resolution (means over the specimens): oriented mass, background, the sheet share of it, the share of points
holding two or more populations (mixing > 0.2), the median angle to the structure tensor (integration sigma = the
window) where the image is clearly oriented (coherence >= 0.3) -- the lines' mean direction against its line direction
where lines carry most of the mass, the sheets' mean normal against its normal where sheets do -- and the white- vs
grey-matter AUC of the oriented mass.

Usage: python scripts/learn/real_sweep.py OUT_STEM CKPT [--pools 1 2 4 8] [--sigmas 1.5 2.5 4 6]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.filters import gaussian
from shapetensor.learn import entity_infer as EI, entity_train as ET
from shapetensor.learn.model import PopulationNet
from shapetensor.learn.realdata import normalise

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("ckpt"); ap.add_argument("--name", default="overnight pilot")
ap.add_argument("--pools", type=int, nargs="+", default=[1, 2, 4, 8]); ap.add_argument("--sigmas", type=float, nargs="+", default=[1.5, 2.5, 4.0, 6.0])
ap.add_argument("--size", type=int, default=96); a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"]); sph = EI.sphere_dirs(200, dev)
CROPS = Path("/hdd/andrew/brain_atlas_20um_test_crops"); WM = (2, 41); GM = (3, 42, 10, 49, 11, 50, 12, 51)


def read_stack(folder, sl):
    import tifffile
    files = sorted(folder.glob("*.tif*"))
    return np.stack([tifffile.imread(files[z])[sl[1], sl[2]] for z in range(sl[0].start, sl[0].stop)])


def st_dir(x, pts, rho):
    xs = gaussian(x.float()[None, None], 1.0)[0, 0]; gz, gy, gx = torch.gradient(xs)
    comps = [gaussian((u * v)[None, None], rho)[0, 0] for u, v in ((gz, gz), (gy, gy), (gx, gx), (gz, gy), (gz, gx), (gy, gx))]
    idx = pts.round().long(); G = torch.stack([c[idx[:, 0], idx[:, 1], idx[:, 2]] for c in comps], -1)
    M = torch.stack([torch.stack([G[:, 0], G[:, 3], G[:, 4]], -1), torch.stack([G[:, 3], G[:, 1], G[:, 5]], -1), torch.stack([G[:, 4], G[:, 5], G[:, 2]], -1)], -2)
    ev, V = torch.linalg.eigh(M)
    return ((ev[:, 2] - ev[:, 0]) / ev.sum(-1).clamp_min(1e-9)).cpu().numpy(), V[:, :, 0].cpu().numpy(), V[:, :, 2].cpu().numpy()


def auc(p, q, n=4000):
    if len(p) < 10 or len(q) < 10:
        return float("nan")
    g = np.random.default_rng(0); p = p[g.integers(0, len(p), n)]; q = q[g.integers(0, len(q), n)]
    return float((p[:, None] > q[None]).mean() + 0.5 * (p[:, None] == q[None]).mean())


specs = [s for s in sorted(CROPS.iterdir()) if next(s.glob("*_deepbrain_L_20um_768"), None) is not None]
res = {}; imgs = {}
for spec in specs:
    img_dir = next(spec.glob("*_deepbrain_L_20um_768")); lab_dir = next(spec.glob("*_deepbrain_L_synthseg_768"))
    for pool in a.pools:
        ext = a.size * pool; o0 = (768 - ext) // 2; sl = tuple(slice(o0, o0 + ext) for _ in range(3))           # centred: the same point
        img = read_stack(img_dir, sl).astype(np.float32)
        if pool > 1:
            img = img.reshape(a.size, pool, a.size, pool, a.size, pool).mean((1, 3, 5))
        lab = read_stack(lab_dir, (slice(o0 + (a.size // 2) * pool + pool // 2, o0 + (a.size // 2) * pool + pool // 2 + 1), sl[1], sl[2]))[0][pool // 2::pool, pool // 2::pool]
        x, _ = normalise(torch.tensor(img, device=dev)); n = a.size; zc = n // 2
        gu, gv = torch.meshgrid(torch.arange(n, device=dev, dtype=torch.float32), torch.arange(n, device=dev, dtype=torch.float32), indexing="ij")
        pts = torch.stack([torch.full_like(gu, float(zc)), gu, gv], -1).view(-1, 3)
        inner = ((pts[:, 1] >= 6) & (pts[:, 1] <= n - 7) & (pts[:, 2] >= 6) & (pts[:, 2] <= n - 7)).cpu().numpy()
        wm = np.isin(lab.reshape(-1), WM) & inner; gm = np.isin(lab.reshape(-1), GM) & inner
        with torch.no_grad():
            feats = EI.features(net, x)
        for sg in a.sigmas:
            with torch.no_grad():
                ro = {k: v.float().cpu().numpy() for k, v in EI.readout(net, x, pts, torch.full((pts.shape[0],), sg, device=dev), sph, feats=feats, iso_k=ET.VOTE_KAPPA_MIN).items()}
            mass = ro["mass"]; R = ro["R"]; lm = mass * (1 - ro["sheet"]); sm = mass * ro["sheet"]      # line and sheet parts
            mean_ax = lambda m_, a_: np.linalg.eigh((m_[..., None, None] * a_[..., :, None] * a_[..., None, :]).sum(1))[1][..., -1]
            dom = mean_ax(lm, R[..., :, 2]); nrm = mean_ax(sm, R[..., :, 0])
            tot = mass.sum(1); ltot = lm.sum(1); stot = sm.sum(1); mix = 1 - (mass ** 2).sum(1) / np.maximum(tot, 1e-9) ** 2
            coh, sd, sn = st_dir(x, pts, sg)
            angle = np.degrees(np.arccos(np.clip(np.abs((dom * sd).sum(-1)), 0, 1)))
            angle_s = np.degrees(np.arccos(np.clip(np.abs((nrm * sn).sum(-1)), 0, 1)))
            clear = inner & (coh >= 0.3) & (ltot > 0.05) & (ltot >= stot); clear_s = inner & (coh >= 0.3) & (stot > 0.05) & (stot > ltot)
            key = f"{spec.name}|{pool}|{sg}"
            res[key] = {"oriented_mass": float(tot[inner].mean()), "background": float(ro["bg"][inner].mean()), "iso": float(ro["iso"][inner].mean()),
                        "sheet_share": float(stot[inner].sum() / max(tot[inner].sum(), 1e-9)),
                        "two_pop_share": float((mix[inner] > 0.2).mean()), "st_angle_median": float(np.median(angle[clear])) if clear.any() else float("nan"),
                        "st_sheet_angle_median": float(np.median(angle_s[clear_s])) if clear_s.sum() >= 20 else float("nan"),
                        "st_clear_share": float(clear.mean() / max(inner.mean(), 1e-9)), "auc_wm_gm": auc(tot[wm], tot[gm]), "wm_share": float(wm.mean() / max(inner.mean(), 1e-9))}
            rgb = np.clip(np.abs(dom[:, [2, 1, 0]]) * np.clip(ltot / 0.5, 0, 1)[:, None] + 0.6 * np.clip(stot / 0.5, 0, 1)[:, None], 0, 1).reshape(n, n, 3)
            imgs[key] = (x[zc].cpu().numpy(), rgb, np.isin(lab, WM))
            print(key, {k: round(v, 3) for k, v in res[key].items()}, flush=True)

json.dump(res, open(a.out + ".json", "w"), indent=1)
# per-specimen grids
for spec in specs:
    fig, axs = plt.subplots(len(a.pools), 1 + len(a.sigmas), figsize=(2.6 * (1 + len(a.sigmas)), 2.6 * len(a.pools)), squeeze=False, layout="constrained")
    for i, pool in enumerate(a.pools):
        for j, sg in enumerate(a.sigmas):
            im, rgb, wmm = imgs[f"{spec.name}|{pool}|{sg}"]
            if j == 0:
                axs[i, 0].imshow(im, cmap="gray"); axs[i, 0].contour(wmm.astype(float), levels=[0.5], colors="#eda100", linewidths=0.7)
                axs[i, 0].set_ylabel(f"{20 * pool} um voxels\n({a.size * 20 * pool / 1000:.1f} mm field)", fontsize=8)
            axs[i, 1 + j].imshow(rgb)
            r_ = res[f"{spec.name}|{pool}|{sg}"]
            axs[i, 1 + j].set_title(f"sigma {sg:g} vox = {sg * 20 * pool:.0f} um\nmass {r_['oriented_mass']:.2f}, sheets {r_['sheet_share'] * 100:.0f} %, 2 pops {r_['two_pop_share'] * 100:.0f} %, "
                                    f"vs ST {r_['st_angle_median']:.0f} deg", fontsize=7)
    axs[0, 0].set_title("image (yellow: SynthSeg white matter)", fontsize=8)
    for ax in axs.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{a.name} on real HiP-CT {spec.name}, centred crop: oriented mass, lines coloured by their mean direction (red x, green y, blue z), sheets grey; brightness = mass;\n"
                 f"across resolution (rows) and window (columns)", fontsize=9)
    plt.savefig(f"{a.out}_{spec.name}.png", dpi=70); plt.close(fig)
# metrics vs the window's physical size
fig, axs = plt.subplots(2, 4, figsize=(18, 8), layout="constrained"); axs = axs.ravel(); pal = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
for m_i, (met, lab_) in enumerate((("oriented_mass", "oriented mass (mean)"), ("background", "background (mean)"), ("sheet_share", "sheet share of the oriented mass"),
                                   ("two_pop_share", "share of points with 2+ populations"),
                                   ("st_angle_median", "lines' direction vs the structure tensor's\n(median deg; coherence >= 0.3, lines dominant)"),
                                   ("st_sheet_angle_median", "sheets' normal vs the structure tensor's\n(median deg; coherence >= 0.3, sheets dominant)"),
                                   ("auc_wm_gm", "AUC white vs grey matter\n(oriented mass)"))):
    ax = axs[m_i]
    for p_i, pool in enumerate(a.pools):
        xs = [sg * 20 * pool for sg in a.sigmas]
        ys = [np.nanmean([res[f"{s.name}|{pool}|{sg}"][met] for s in specs]) for sg in a.sigmas]
        ax.plot(xs, ys, "-o", color=pal[p_i % 4], lw=2, ms=4, label=f"{20 * pool} um voxels")
    ax.set_xscale("log"); ax.set_xlabel("window sigma (um)"); ax.set_title(lab_, fontsize=9); ax.grid(color="#e4e3df", lw=0.6)
    if met == "auc_wm_gm":
        ax.axhline(0.5, color="#999", ls=":", lw=0.8)
axs[0].legend(fontsize=8); axs[-1].axis("off")
fig.suptitle(f"{a.name} on real HiP-CT (5 specimens, centred crops): the readout across resolution and window size (means over the specimens)", fontsize=10)
plt.savefig(f"{a.out}_metrics.png", dpi=75); print("saved figures")
