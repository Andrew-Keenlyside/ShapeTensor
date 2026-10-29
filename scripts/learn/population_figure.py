"""The population figure (one fixed layout, for every comparison): generator crossing scenes, a slice through the crossing
core; per row: the image; the true populations, each its own map (the three with most occupancy in the slice); the target
mass (the entity targets on a grid, sigma 2.5); per checkpoint: its mass, and its populations -- its entities per point
(:func:`~shapetensor.learn.entity_infer.cluster`), the isotropic ones (density-moment concentration < 2: the background,
removed as the background population is meant to be) left out, the rest grouped across the slice by the votes' axes
(average linkage of the axial angle, cut at 30 deg), each group's mass a map, the three heaviest shown; the model's mass
with and without the isotropic entities. Grouping uses only the model's outputs.

Usage: python scripts/learn/population_figure.py OUT.png CKPT [CKPT ...] [--rows 5] [--seed0 1700000]
"""
import argparse, math, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
import shapetensor.learn.scene as SS
from shapetensor.learn import lobe_train as LT, losses as L, entity_infer as EI
from shapetensor.learn.model import PopulationNet

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("ckpts", nargs="+"); ap.add_argument("--rows", type=int, default=5)
ap.add_argument("--seed0", type=int, default=1_700_000); ap.add_argument("--strength", type=float, default=0.4); ap.add_argument("--level", type=float, default=0.5)
ap.add_argument("--step", type=int, default=2)
a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None
L.DESC_K = 5; L.DESC_NC = 0
gen = {**LT.difficulty_gen(a.level, {"canvas": 128, "crops_per_scene": 1, "cross_p": 1.0, "p_centre": 1.0, "deep_tree": False}), "crops_per_scene": 1}
dirs = EI.sphere_dirs(200, dev)
models = []
for c in a.ckpts:
    ck = torch.load(c, map_location=dev, weights_only=False); m = PopulationNet.from_args(ck["args"]).to(dev).eval(); m.load_state_dict(ck["model"])
    models.append((c.split("/")[-1].replace(".pt", "") if not c.endswith("model.pt") else c.split("/")[-2], m))


def model_groups(m, x, voxel_um, z, n_show=3):
    """The model's mass at the slice's grid and its populations' masses (vote-axis groups), heaviest first."""
    gz, gx = torch.meshgrid(torch.arange(2, 62, a.step), torch.arange(2, 62, a.step), indexing="ij")
    pts = torch.stack([torch.full((gz.numel(),), float(z)), gz.flatten().float(), gx.flatten().float()], 1).to(dev)
    with torch.no_grad():
        f = EI.features(m, x, voxel_um=voxel_um)
        r = EI.query(m, f, pts, torch.full((pts.shape[0],), 2.5, device=dev), dirs, vol_shape=x.shape)
        e = EI.cluster(r, dirs)
    H, W = gz.shape
    total = (e["mass"] * e["live"]).sum(1).view(H, W).cpu()
    isoe = e["kappa_odf"][..., 0].float() < 2.0                                                  # isotropic entities: the background
    objm = (e["mass"] * e["live"] * ~isoe).sum(1).view(H, W).cpu()
    live = e["live"] & (e["share"] >= 0.05) & ~isoe                                              # populations: the oriented entities
    qi, ci = torch.nonzero(live, as_tuple=True)
    if qi.numel() < 2:
        return total, objm, []
    ax = e["R_odf"][qi, ci, :, 2].float(); ms = e["mass"][qi, ci].float()
    iso = e["kappa_odf"][qi, ci, 0].float() < 2.0                                                 # isotropic entities: a group of their own
    from scipy.cluster.hierarchy import fcluster, linkage
    sub = torch.randperm(qi.numel(), generator=torch.Generator().manual_seed(0))[:2000].to(dev)
    sub = sub[~iso[sub]]
    lab = torch.full((qi.numel(),), -1, dtype=torch.long, device=dev)
    if sub.numel() >= 2:
        A = ax[sub]; D = torch.rad2deg(torch.acos((A @ A.T).abs().clamp(max=1))).cpu().numpy()
        ls = torch.as_tensor(fcluster(linkage(D[np.triu_indices(D.shape[0], 1)], "average"), 30.0, "distance") - 1, device=dev)
        K = int(ls.max()) + 1
        cent = torch.stack([torch.linalg.eigh((A[ls == k][:, :, None] * A[ls == k][:, None, :]).sum(0))[1][:, -1] for k in range(K)])
        lab = torch.where(iso, torch.full_like(lab, K), (ax @ cent.T).abs().argmax(1))
    else:
        K = 0; lab = torch.where(iso, torch.full_like(lab, 0), lab)
    groups = []
    for k in range(int(lab.max()) + 1):
        sel = lab == k
        if not bool(sel.any()):
            continue
        mp = torch.zeros(H * W, device=dev).index_add_(0, qi[sel], ms[sel]).view(H, W).cpu()
        groups.append((float(mp.sum()), mp, "isotropic" if k == K else f"axis group {k}"))
    groups.sort(key=lambda g_: -g_[0])
    return total, objm, groups[:n_show]


rows = []; seed = a.seed0
while len(rows) < a.rows and seed < a.seed0 + 400:
    SS._CACHE.clear()
    s, d = SS.draw_scene((64,) * 3, seed=seed, device=dev, strength=a.strength, entity_ctx=True, **gen); seed += 1
    keep = [k for k, L_ in enumerate(s.layers) if L_.role == "shape"]
    if len(keep) < 2:
        continue
    occ = d["occupancy"][keep].float()
    ov = (occ.unsqueeze(1) * occ.unsqueeze(0)).sum((-1, -2)) * (1 - torch.eye(len(keep), device=dev))[:, :, None]
    if float(ov.max()) < 20:
        continue
    rows.append((seed - 1, int(ov.amax((0, 1)).argmax()), s, d, keep))
nc = 1 + 3 + 1 + len(models) * 5
fig, axs = plt.subplots(len(rows), nc, figsize=(2.4 * nc, 2.6 * len(rows)), squeeze=False)
for r_, (sd, z, s, d, keep) in enumerate(rows):
    img = d["image"][z].float().cpu() / 255.0
    axs[r_, 0].imshow(img, cmap="gray"); axs[r_, 0].set_title(f"seed {sd} z={z}\n{d['record'].get('modality', '')} {d.get('voxel_um', 0):.3g} um", fontsize=6)
    occ = d["occupancy"][keep].float()[:, z].cpu(); order = occ.sum((1, 2)).argsort(descending=True)[:3]
    for j in range(3):
        if j < order.numel():
            axs[r_, 1 + j].imshow(occ[order[j]], cmap="magma", vmin=0, vmax=1); axs[r_, 1 + j].set_title(f"true population {j + 1}", fontsize=6)
    gz, gx = torch.meshgrid(torch.arange(2, 62, a.step), torch.arange(2, 62, a.step), indexing="ij")
    pts = torch.stack([torch.full((gz.numel(),), float(z)), gz.flatten().float(), gx.flatten().float()], 1).to(dev)
    TR.set_background_entity(False)                                                             # the objects' mass (background left out)
    t = TR.entity_queries(s, d, 64, sd, dev, points=pts, sigma=torch.full((pts.shape[0],), 2.5, device=dev))
    TR.set_background_entity(True)
    tm = ((t["e_mass"].float() * t["e_vis"].float()) * (t["e_node"] >= 0)).sum(1).view(gz.shape).cpu()
    axs[r_, 4].imshow(tm, cmap="viridis", vmin=0, vmax=1); axs[r_, 4].set_title("target mass (objects, no background)", fontsize=6)
    x = d["image"].float() / 255.0
    for j, (name, m) in enumerate(models):
        total, objm, groups = model_groups(m, x, d.get("voxel_um"), z)
        c0 = 5 + 5 * j
        axs[r_, c0].imshow(total, cmap="viridis", vmin=0, vmax=1); axs[r_, c0].set_title(f"{name[:20]}\nmass (all)", fontsize=6)
        axs[r_, c0 + 1].imshow(objm, cmap="viridis", vmin=0, vmax=1); axs[r_, c0 + 1].set_title(f"{name[:20]}\nmass (isotropic removed)", fontsize=6)
        for k in range(3):
            if k < len(groups):
                axs[r_, c0 + 2 + k].imshow(groups[k][1], cmap="magma", vmin=0, vmax=max(0.05, float(groups[k][1].max())))
                axs[r_, c0 + 2 + k].set_title(f"{name[:12]} pop {k + 1} ({groups[k][2]})", fontsize=6)
for ax in axs.flat:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(a.out, dpi=70); print("saved", a.out)
