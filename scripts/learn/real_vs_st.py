"""The models' predominant orientation against the structure tensor's on real HiP-CT (the 20 um deep-brain crops of
five specimens, as `real_populations.py`): at a 3-D grid of points, the angle between

* the density's peak direction (argmax over 200 directions; the model's own predominant orientation, no readout), and
* the heaviest line population's axis (the readout),

and the structure tensor's line direction (the eigenvector of its smallest eigenvalue; gradient sigma 1, integration
sigma = the model's window, ``--sigma``), in 3-D. The structure tensor's direction is only defined where the image is
anisotropic, so the agreement is reported against its coherence ((l3 - l1) / sum l). Chance (random directions): median
angle 60 deg, 13 % within 10 deg.

Usage: python scripts/learn/real_vs_st.py OUT_STEM --models NAME=CKPT[:um] ... [--pool 1 4] [--sigma 2.5]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
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
ap.add_argument("--pool", type=int, nargs="+", default=[1, 4]); ap.add_argument("--size", type=int, default=96)
ap.add_argument("--sigma", type=float, default=2.5); ap.add_argument("--step", type=int, default=3); ap.add_argument("--count-min", type=float, default=0.03)
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
CROPS = Path("/hdd/andrew/brain_atlas_20um_test_crops"); WM = (2, 41); GM = (3, 42, 10, 49, 11, 50, 12, 51)
models = []
for spec in a.models:
    name, path = spec.split("=", 1); um = path.endswith(":um"); path = path[:-3] if um else path
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = PopulationNet.from_args(ck["args"]).to(dev).eval(); m.load_state_dict(ck["model"]); models.append((name, m, um))
sph = EI.sphere_dirs(200, dev)
COH_BINS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.01]


def read_stack(folder, sl):
    import tifffile
    files = sorted(folder.glob("*.tif*"))
    return np.stack([tifffile.imread(files[z])[sl[1], sl[2]] for z in range(sl[0].start, sl[0].stop)])


def structure_tensor(x, pts, rho):
    xs = gaussian(x.float()[None, None], 1.0)[0, 0]; gz, gy, gx = torch.gradient(xs)
    comps = [gaussian((u * v)[None, None], rho)[0, 0] for u, v in ((gz, gz), (gy, gy), (gx, gx), (gz, gy), (gz, gx), (gy, gx))]
    idx = pts.round().long(); G = torch.stack([c[idx[:, 0], idx[:, 1], idx[:, 2]] for c in comps], -1)
    M = torch.stack([torch.stack([G[:, 0], G[:, 3], G[:, 4]], -1), torch.stack([G[:, 3], G[:, 1], G[:, 5]], -1), torch.stack([G[:, 4], G[:, 5], G[:, 2]], -1)], -2)
    ev, V = torch.linalg.eigh(M)
    return (ev[:, 2] - ev[:, 0]) / ev.sum(-1).clamp_min(1e-9), V[:, :, 0]


def angle(u, v):
    return torch.rad2deg(torch.acos((torch.nn.functional.normalize(u, dim=-1) * torch.nn.functional.normalize(v, dim=-1)).sum(-1).abs().clamp(max=1)))


def stats(ang):
    ang = ang[torch.isfinite(ang)]
    if ang.numel() < 10:
        return {"n": int(ang.numel())}
    return {"n": int(ang.numel()), "median": float(ang.median()), "within10": float((ang <= 10).float().mean()),
            "within20": float((ang <= 20).float().mean()), "within30": float((ang <= 30).float().mean())}


rows = {}; figdata = []
for pool in a.pool:
    for spec in sorted(CROPS.iterdir()):
        img_dir = next(spec.glob("*_deepbrain_L_20um_768"), None); lab_dir = next(spec.glob("*_deepbrain_L_synthseg_768"), None)
        if img_dir is None or lab_dir is None:
            continue
        rng = np.random.default_rng(0); ext = a.size * pool; best = None
        for _ in range(60):                                                                       # the same crops as real_populations.py
            o = [int(rng.integers(0, 768 - ext + 1)) for _ in range(3)]; sl = tuple(slice(v, v + ext) for v in o)
            lab = read_stack(lab_dir, sl)[pool // 2::pool, pool // 2::pool, pool // 2::pool]
            sh = min(np.isin(lab, WM).mean(), np.isin(lab, GM).mean())
            if best is None or sh > best[0]:
                best = (sh, o, sl, lab)
            if sh >= (0.10 if pool == 1 else 0.2):
                break
        _, o, sl, lab = best
        img = read_stack(img_dir, sl).astype(np.float32)
        if pool > 1:
            img = img.reshape(a.size, pool, a.size, pool, a.size, pool).mean((1, 3, 5))
        x, _ = normalise(torch.tensor(img, device=dev)); n = a.size; mg = 12
        g1 = torch.arange(mg, n - mg, a.step, device=dev, dtype=torch.float32); G = g1.numel()
        pts = torch.stack(torch.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3)
        labp = torch.tensor(lab.astype(np.int64), device=dev)[pts[:, 0].long(), pts[:, 1].long(), pts[:, 2].long()].cpu()
        tissue = np.where(np.isin(labp.numpy(), WM), "wm", np.where(np.isin(labp.numpy(), GM), "gm", "other"))
        coh, st_dir = structure_tensor(x, pts, a.sigma); coh = coh.cpu(); st_dir = st_dir.cpu()
        key = f"{spec.name}_x{pool}"; rec = {"coh": coh, "tissue": tissue}
        zc = G // 2; sel2 = torch.arange(pts.shape[0]).view(G, G, G)[zc].reshape(-1)
        fd = {"key": key, "img": x[int(g1[zc])].cpu().numpy(), "g1": g1.cpu().numpy(), "st": st_dir[sel2].numpy(), "coh": coh[sel2].numpy(), "models": {}}
        for name, m, um in models:
            sig = torch.full((pts.shape[0],), a.sigma, device=dev)
            with torch.no_grad():
                feats = EI.features(m, x, voxel_um=20.0 * pool if um else None)
                f = torch.cat([EI.query(m, feats, pts[j:j + 2048], sig[j:j + 2048], sph, vol_shape=x.shape)["f"].float() for j in range(0, pts.shape[0], 2048)])
                ro = EI.readout(m, x, pts, sig, sph, feats=feats, iso_k=ET.VOTE_KAPPA_MIN)
            peak = sph[f.argmax(1)].cpu(); sharp = (f.amax(1) / f.mean(1).clamp_min(1e-9)).cpu()
            mass = ro["mass"].float().cpu(); kk = ro["k"].float().cpu(); R = ro["R"].float().cpu()
            top = mass.argmax(1); ar = torch.arange(len(top))
            line = (kk[ar, top, 1] >= 2.0) & (mass[ar, top] >= a.count_min)
            axis = torch.where(line[:, None], R[ar, top, :, 2], torch.full((len(top), 3), float("nan")))
            rec[name] = {"peak": angle(peak, st_dir), "pop": angle(axis, st_dir), "sharp": sharp, "line": line}
            fd["models"][name] = (peak[sel2].numpy(), sharp[sel2].numpy(), axis[sel2].numpy())
        rows[key] = rec; figdata.append(fd)
        print(key, " | ".join(f"{nm}: peak median {float(rec[nm]['peak'].median()):.0f} deg, coh>0.3 {stats(rec[nm]['peak'][coh > 0.3]).get('median', float('nan')):.0f} deg"
                               for nm, _, _ in models), flush=True)

# summary: per pool, per model, by structure-tensor coherence and by tissue
summary = {}
for pool in a.pool:
    keys = [k for k in rows if k.endswith(f"_x{pool}")]
    coh = torch.cat([rows[k]["coh"] for k in keys]); tis = np.concatenate([rows[k]["tissue"] for k in keys])
    s_pool = {"coh_share": {f"{lo:.1f}-{hi:.1f}": float(((coh >= lo) & (coh < hi)).float().mean()) for lo, hi in zip(COH_BINS[:-1], COH_BINS[1:])}}
    for name, _, _ in models:
        for kind in ("peak", "pop"):
            ang = torch.cat([rows[k][name][kind] for k in keys])
            s_pool[f"{name}_{kind}_all"] = stats(ang)
            for lo, hi in zip(COH_BINS[:-1], COH_BINS[1:]):
                s_pool[f"{name}_{kind}_coh{lo:.1f}-{hi:.1f}"] = stats(ang[(coh >= lo) & (coh < hi)])
            for t in ("wm", "gm"):
                s_pool[f"{name}_{kind}_{t}_coh>0.2"] = stats(ang[torch.tensor(tis == t) & (coh > 0.2)])
        s_pool[f"{name}_line_share"] = float(torch.cat([rows[k][name]["line"].float() for k in keys]).mean())
    summary[f"x{pool}"] = s_pool
json.dump(summary, open(a.out + ".json", "w"), indent=1)
for pk, sp in summary.items():
    print(f"== {pk} ({20 * int(pk[1:])} um): share of points by structure-tensor coherence {sp['coh_share']}")
    for name, _, _ in models:
        for kind, lab_ in (("peak", "density peak"), ("pop", "heaviest line population")):
            line = " | ".join(f"coh {b}: {sp[f'{name}_{kind}_coh{b}'].get('median', float('nan')):.0f} deg ({sp[f'{name}_{kind}_coh{b}'].get('within20', float('nan')) * 100:.0f} % <20)"
                              for b in [f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(COH_BINS[:-1], COH_BINS[1:])])
            print(f"  {name:17s} {lab_:25s} all {sp[f'{name}_{kind}_all'].get('median', float('nan')):.0f} deg | {line}")

# figure: per pool, (a) median angle vs coherence per model; (b) a slice of each crop with the structure tensor's and the
# models' density-peak directions; (c) the angle histogram at coherence > 0.2
cols = {"overnight_pilot": "#2a78d6", "directpath_pilot": "#eb6834", "generator_model": "#1baf7a"}
for pool in a.pool:
    fds = [fd for fd in figdata if fd["key"].endswith(f"_x{pool}")]; sp = summary[f"x{pool}"]
    fig = plt.figure(figsize=(4.0 * (2 + len(models)), 3.6 * (1 + len(fds))), layout="constrained")
    gs = fig.add_gridspec(1 + len(fds), 2 + len(models))
    ax = fig.add_subplot(gs[0, 0]); mids = [(lo + min(hi, 1.0)) / 2 for lo, hi in zip(COH_BINS[:-1], COH_BINS[1:])]
    for name, _, _ in models:
        for kind, ls in (("peak", "-"), ("pop", "--")):
            ax.plot(mids, [sp[f"{name}_{kind}_coh{lo:.1f}-{hi:.1f}"].get("median", np.nan) for lo, hi in zip(COH_BINS[:-1], COH_BINS[1:])], ls,
                    color=cols.get(name, "k"), marker="o", ms=4, lw=2, label=f"{name} ({'density peak' if kind == 'peak' else 'heaviest line pop.'})")
    ax.axhline(60, color="#888", lw=0.8, ls=":"); ax.text(0.02, 61, "chance (60)", fontsize=7, color="#555")
    ax.set_xlabel("structure-tensor coherence"); ax.set_ylabel("median angle to the structure tensor (deg)"); ax.set_ylim(0, 75)
    ax.legend(fontsize=6, loc="upper right"); ax.set_title("agreement vs how oriented the image is", fontsize=9)
    ax = fig.add_subplot(gs[0, 1])
    for name, _, _ in models:
        keys = [k for k in rows if k.endswith(f"_x{pool}")]; coh = torch.cat([rows[k]["coh"] for k in keys])
        ang = torch.cat([rows[k][name]["peak"] for k in keys])[coh > 0.2]
        ax.hist(ang.numpy(), bins=np.arange(0, 91, 5), histtype="step", lw=2, color=cols.get(name, "k"), density=True, label=name)
    ax.set_xlabel("angle, density peak vs structure tensor (deg)"); ax.set_title("where coherence > 0.2", fontsize=9); ax.legend(fontsize=7)
    for i, fd in enumerate(fds):
        im = fd["img"]; g1 = fd["g1"]; yy, xx = np.meshgrid(g1, g1, indexing="ij"); yy = yy.reshape(-1); xx = xx.reshape(-1)

        def sticks(ax_, dirs, w, col_by_dir=True):
            segs, cs, lw = [], [], []
            for p in range(len(dirs)):
                u = dirs[p]
                if not np.all(np.isfinite(u)) or w[p] <= 0:
                    continue
                v = u[1:]; L = 0.45 * 3 * float(np.hypot(*v))
                if L < 0.1:
                    continue
                v = v / np.hypot(*v); segs.append([(xx[p] - L * v[1], yy[p] - L * v[0]), (xx[p] + L * v[1], yy[p] + L * v[0])])
                cs.append(np.abs(u[[2, 1, 0]])); lw.append(0.4 + 2.5 * min(w[p], 1.0))
            ax_.add_collection(LineCollection(segs, colors=cs, linewidths=lw)); ax_.set_xlim(0, im.shape[1]); ax_.set_ylim(im.shape[0], 0)

        ax = fig.add_subplot(gs[1 + i, 0]); ax.imshow(im, cmap="gray"); ax.set_title(f"{fd['key']} ({20 * pool} um)", fontsize=8)
        ax = fig.add_subplot(gs[1 + i, 1]); ax.imshow(im, cmap="gray", alpha=0.4); sticks(ax, fd["st"], np.clip(fd["coh"] / 0.4, 0, 1))
        ax.set_title("structure tensor (width = coherence)", fontsize=8)
        for j, (name, _, _) in enumerate(models):
            pk, sharp, axis = fd["models"][name]
            ax = fig.add_subplot(gs[1 + i, 2 + j]); ax.imshow(im, cmap="gray", alpha=0.4); sticks(ax, pk, np.clip((sharp - 1) / 2, 0, 1))
            ax.set_title(f"{name}: density peak (width = peak / mean)", fontsize=8)
    for ax in fig.axes[2:]:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Predominant orientation vs the structure tensor on real HiP-CT ({20 * pool} um voxels), window sigma {a.sigma} voxels; colour = 3-D direction", fontsize=10)
    plt.savefig(f"{a.out}_x{pool}.png", dpi=65); plt.close(fig); print("saved", f"{a.out}_x{pool}.png")
