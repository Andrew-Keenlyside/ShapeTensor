"""The population readout on real HiP-CT with independent anatomy: the 20 um deep-brain crops of five specimens with
SynthSeg labels on the same grid (``/hdd/andrew/brain_atlas_20um_test_crops/<ID>``; checks only, never training).

Per specimen, a crop holding both white matter (SynthSeg 2/41) and grey matter (cortex 3/42, thalamus 10/49, caudate
11/50, putamen 12/51), the model read out (:func:`entity_infer.readout`, the default readout) on a 3-D grid of points at
window sigma ``--sigma`` voxels. Per point: the oriented populations' total mass, their count (mass >= ``--count-min``),
the density's anisotropy (std / mean over the directions), the heaviest line population's axis. Scores per model (and
for a structure tensor as the classical baseline):

* AUC with which oriented mass / anisotropy / the structure tensor's coherence separates white from grey matter;
* the share of points with 0, 1, 2+ oriented populations in each tissue;
* the axis coherence: mean |cos| between a point's axis and its grid neighbours' ``--lag`` steps away (white vs grey).

Also a figure: per specimen, the crop's middle slice with the white-matter outline, and per model the populations as
sticks (in-plane part of each line population's axis, width = its mass, colour = its 3-D direction).

Usage: python scripts/learn/real_populations.py OUT_STEM --models NAME=CKPT[:um] ... [--size 96] [--sigma 2.5]
       (``:um`` passes the voxel size, 20 um; otherwise unknown -- the phantom-trained models never saw one)
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.filters import gaussian
from shapetensor.learn import entity_infer as EI, entity_train as ET
from shapetensor.learn.model import PopulationNet
from shapetensor.learn.realdata import normalise

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--size", type=int, default=96); ap.add_argument("--sigma", type=float, default=2.5); ap.add_argument("--step", type=int, default=3)
ap.add_argument("--pool", type=int, default=1, help="block-average the 20 um input by this factor (a wider field of view: SynthSeg's labels are 1 mm)")
ap.add_argument("--min-share", type=float, default=0.10)
ap.add_argument("--count-min", type=float, default=0.03); ap.add_argument("--lag", type=int, default=2); ap.add_argument("--per-specimen", type=int, default=1)
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
CROPS = Path("/hdd/andrew/brain_atlas_20um_test_crops"); WM = (2, 41); GM = (3, 42, 10, 49, 11, 50, 12, 51)
models = []
for spec in a.models:
    name, path = spec.split("=", 1); um = path.endswith(":um"); path = path[:-3] if um else path       # (voxel size: 20 um x pool)
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = PopulationNet.from_args(ck["args"]).to(dev).eval(); m.load_state_dict(ck["model"]); models.append((name, m, 20.0 * a.pool if um else None))
sph = EI.sphere_dirs(200, dev)


def read_stack(folder: Path, sl):
    import tifffile
    files = sorted(folder.glob("*.tif*"))
    return np.stack([tifffile.imread(files[z])[sl[1], sl[2]] for z in range(sl[0].start, sl[0].stop)])


def auc(pos, neg, n=4000):
    if pos.numel() < 10 or neg.numel() < 10:
        return float("nan")
    g = torch.Generator().manual_seed(0)
    p = pos[torch.randint(0, pos.numel(), (n,), generator=g)]; q = neg[torch.randint(0, neg.numel(), (n,), generator=g)]
    return float((p[:, None] > q[None]).float().mean() + 0.5 * (p[:, None] == q[None]).float().mean())


def st_point(x, pts):
    """Structure tensor (gradient sigma 1, window 5) at ``pts``: coherence and the least-gradient direction."""
    xs = gaussian(x.float()[None, None], 1.0)[0, 0]; gz, gy, gx = torch.gradient(xs)
    comps = [gaussian((u * v)[None, None], 5.0)[0, 0] for u, v in ((gz, gz), (gy, gy), (gx, gx), (gz, gy), (gz, gx), (gy, gx))]
    idx = pts.round().long()
    G = torch.stack([c[idx[:, 0], idx[:, 1], idx[:, 2]] for c in comps], -1)
    M = torch.stack([torch.stack([G[:, 0], G[:, 3], G[:, 4]], -1), torch.stack([G[:, 3], G[:, 1], G[:, 5]], -1), torch.stack([G[:, 4], G[:, 5], G[:, 2]], -1)], -2)
    ev, V = torch.linalg.eigh(M)
    return (ev[:, 2] - ev[:, 0]) / ev.sum(-1).clamp_min(1e-9), V[:, :, 0]


def coherence(ax, ok, shape, lag):
    """Mean |cos| between each grid point's axis and those ``lag`` steps away along the grid axes (both valid)."""
    A = ax.view(*shape, 3); V = ok.view(*shape); acc = torch.zeros(shape, device=ax.device); n = torch.zeros(shape, device=ax.device)
    for d in range(3):
        for s in (lag, -lag):
            An = torch.roll(A, s, dims=d); Vn = torch.roll(V, s, dims=d)
            valid = V & Vn
            acc += (A * An).sum(-1).abs() * valid; n += valid.float()
    return (acc / n.clamp_min(1)).view(-1), (n > 0).view(-1)


res = {}; figrows = []
for spec in sorted(CROPS.iterdir()):
    img_dir = next(spec.glob("*_deepbrain_L_20um_768"), None); lab_dir = next(spec.glob("*_deepbrain_L_synthseg_768"), None)
    if img_dir is None or lab_dir is None:
        continue
    rng = np.random.default_rng(0)
    for k in range(a.per_specimen):
        ext = a.size * a.pool; best = None
        for _ in range(60):                                                                       # both tissues, min-share each at least
            o = [int(rng.integers(0, 768 - ext + 1)) for _ in range(3)]; sl = tuple(slice(v, v + ext) for v in o)
            lab = read_stack(lab_dir, sl)[a.pool // 2::a.pool, a.pool // 2::a.pool, a.pool // 2::a.pool]
            sh = min(np.isin(lab, WM).mean(), np.isin(lab, GM).mean())
            if best is None or sh > best[0]:
                best = (sh, o, sl, lab)
            if sh >= a.min_share:
                break
        _, o, sl, lab = best
        img = read_stack(img_dir, sl).astype(np.float32)
        if a.pool > 1:
            img = img.reshape(a.size, a.pool, a.size, a.pool, a.size, a.pool).mean((1, 3, 5))
        x, tissue = normalise(torch.tensor(img, device=dev))
        n = a.size; mg = 12; g1 = torch.arange(mg, n - mg, a.step, device=dev, dtype=torch.float32); G = g1.numel()
        pts = torch.stack(torch.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3)
        labp = torch.tensor(lab.astype(np.int64), device=dev)[pts[:, 0].long(), pts[:, 1].long(), pts[:, 2].long()]
        wm = torch.isin(labp, torch.tensor(WM, device=dev)).cpu(); gm = torch.isin(labp, torch.tensor(GM, device=dev)).cpu()
        key = f"{spec.name}_{k}"; r_ = {"origin": o, "wm_share": float(wm.float().mean()), "gm_share": float(gm.float().mean()), "tissue": tissue}
        coh, sd = st_point(x, pts); coh = coh.cpu(); sd = sd.cpu()
        r_["st_auc_coh"] = auc(coh[wm], coh[gm])
        ch, okc = coherence(sd, torch.ones(pts.shape[0], dtype=torch.bool), (G, G, G), a.lag)
        r_["st_axis_coh_wm"] = float(ch[wm].mean()); r_["st_axis_coh_gm"] = float(ch[gm].mean())
        zc = int(G // 2); rowsel = torch.arange(pts.shape[0]).view(G, G, G)[zc].reshape(-1)       # the middle grid slice (figure)
        row = {"key": key, "img": x[int(g1[zc])].cpu().numpy(), "lab": lab[int(g1[zc])], "g1": g1.cpu().numpy(), "models": []}
        for name, m, um in models:
            sig = torch.full((pts.shape[0],), a.sigma, device=dev)
            with torch.no_grad():
                feats = EI.features(m, x, voxel_um=um)
                ro = EI.readout(m, x, pts, sig, sph, feats=feats, iso_k=ET.VOTE_KAPPA_MIN)
                qf = torch.cat([EI.query(m, feats, pts[j:j + 2048], sig[j:j + 2048], sph, vol_shape=x.shape)["f"].float() for j in range(0, pts.shape[0], 2048)])
            mass = ro["mass"].float().cpu(); kk = ro["k"].float().cpu(); R = ro["R"].float().cpu()
            ori = mass.sum(1); cnt = (mass >= a.count_min).sum(1); aniso = (qf.std(1) / qf.mean(1).clamp_min(1e-9)).cpu()
            top = mass.argmax(1); line = (kk[torch.arange(len(top)), top, 1] >= 2.0) & (mass.max(1).values >= a.count_min)
            axis = R[torch.arange(len(top)), top, :, 2]
            pre = name + "_"
            r_[pre + "auc_oriented_mass"] = auc(ori[wm], ori[gm]); r_[pre + "auc_anisotropy"] = auc(aniso[wm], aniso[gm])
            r_[pre + "oriented_mass_wm"] = float(ori[wm].mean()); r_[pre + "oriented_mass_gm"] = float(ori[gm].mean())
            for t, sel in (("wm", wm), ("gm", gm)):
                c_ = cnt[sel]
                r_[pre + f"pop0_{t}"] = float((c_ == 0).float().mean()); r_[pre + f"pop1_{t}"] = float((c_ == 1).float().mean()); r_[pre + f"pop2_{t}"] = float((c_ >= 2).float().mean())
            ch, okc = coherence(axis, line, (G, G, G), a.lag)
            r_[pre + "axis_coh_wm"] = float(ch[wm & line & okc].mean()) if bool((wm & line & okc).any()) else float("nan")
            r_[pre + "axis_coh_gm"] = float(ch[gm & line & okc].mean()) if bool((gm & line & okc).any()) else float("nan")
            r_[pre + "st_agree_wm"] = float((axis[wm & line] * sd[wm & line]).sum(-1).abs().mean()) if bool((wm & line).any()) else float("nan")
            row["models"].append((name, mass[rowsel].numpy(), kk[rowsel].numpy(), R[rowsel].numpy()))
        res[key] = r_; figrows.append(row)
        print(key, {k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in r_.items() if k_ != "origin"}, flush=True)

keys = sorted({k_ for r in res.values() for k_ in r if k_ not in ("origin",)})
summ = {k_: float(np.nanmean([r[k_] for r in res.values() if k_ in r])) for k_ in keys}
json.dump({"crops": res, "summary": summ, "args": vars(a)}, open(a.out + ".json", "w"), indent=1)
print(json.dumps(summ, indent=1))

nm = len(models)
fig, axs = plt.subplots(len(figrows), 2 + nm, figsize=(3.4 * (2 + nm), 3.4 * len(figrows)), squeeze=False, layout="constrained")
for i, row in enumerate(figrows):
    im = row["img"]; L = row["lab"]; g1 = row["g1"]; G = len(g1)
    wmL = np.isin(L, WM).astype(float)
    axs[i, 0].imshow(im, cmap="gray"); axs[i, 0].contour(wmL, levels=[0.5], colors="#eda100", linewidths=0.8)
    axs[i, 0].set_title(f"{row['key']}: {20 * a.pool} um HiP-CT\n(yellow: SynthSeg white matter)", fontsize=8)
    axs[i, 1].imshow(np.where(np.isin(L, WM), 2, np.where(np.isin(L, GM), 1, 0)), cmap="Greys", vmin=0, vmax=2.5)
    axs[i, 1].set_title("white (dark) / grey (light) matter", fontsize=8)
    yy, xx = np.meshgrid(g1, g1, indexing="ij"); yy = yy.reshape(-1); xx = xx.reshape(-1)
    for j, (name, mass, kk, R) in enumerate(row["models"]):
        ax = axs[i, 2 + j]; ax.imshow(im, cmap="gray", alpha=0.45); segs, cols, lws = [], [], []
        for p in range(mass.shape[0]):
            for c in range(mass.shape[1]):
                if mass[p, c] < a.count_min or kk[p, c, 1] < 2.0:
                    continue
                u = R[p, c, :, 2]; v = u[1:]; ln = 0.45 * 3 * float(np.hypot(*v))
                if ln < 0.1:
                    continue
                v = v / np.hypot(*v)
                segs.append([(xx[p] - ln * v[1], yy[p] - ln * v[0]), (xx[p] + ln * v[1], yy[p] + ln * v[0])])
                cols.append(np.abs(u[[2, 1, 0]])); lws.append(0.5 + 8 * min(float(mass[p, c]), 0.5))
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, capstyle="round"))
        ax.set_xlim(0, im.shape[1]); ax.set_ylim(im.shape[0], 0)
        ax.set_title(f"{name}: line populations\n(width = mass, colour = 3-D direction)", fontsize=8)
    for ax in axs[i]:
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Populations on real HiP-CT ({20 * a.pool} um voxels: 20 um deep brain{' block-averaged x' + str(a.pool) if a.pool > 1 else ''}; SynthSeg labels), window sigma {a.sigma} voxels ({a.sigma * 20 * a.pool:.0f} um)", fontsize=10)
plt.savefig(a.out + ".png", dpi=70); print("saved", a.out + ".png")
