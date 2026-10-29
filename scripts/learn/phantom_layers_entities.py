"""Sense check of an entity-field checkpoint: crossing phantoms with ground-truth layers, and the model's separated layers.

Phantoms from the scene generator: two bundles through one centre at a set angle (and a single bundle), resolved
fibres or sub-voxel ones, imaged simply (contrast per population, background, a 1-voxel blur, noise). The ground-truth
layers are the populations' occupancies. The model's layers: the decoder on a dense grid of the mid slice, entities per
point (:func:`~shapetensor.learn.entity_infer.cluster`), all entities of the slice grouped into populations by their
tree embeddings (average linkage, cut at one tree edge), each population's mass drawn as an image.

Usage: python scripts/learn/phantom_layers_entities.py CHECKPOINT OUT.png [--p4] [--group-vote]
(--group-vote: populations from the votes' mode axes -- average linkage of the axial angle cut at 30 deg -- instead of the
tree embedding)
"""
import json, math, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, "src")
from shapetensor.filters import gaussian
from shapetensor.learn import entity_infer as EI
from shapetensor.learn.entity_train import TREE_EDGE
from shapetensor.learn.model import PopulationNet, poincare_dist
from shapetensor.learn.scene import Scene

ck_path, out_png = sys.argv[1], sys.argv[2]; with_p4 = "--p4" in sys.argv
GROUP = "vote" if "--group-vote" in sys.argv else "tree"                          # populations by the tree embedding, or by the votes' axes
dev = "cuda"
ck = torch.load(ck_path, map_location=dev, weights_only=False)
model = PopulationNet.from_args(ck["args"]).to(dev).eval(); model.load_state_dict(ck["model"])
N = 96; VOX = 2.0


def phantom(theta, resolved, seed=3, step_bg=False):
    """Two bundles crossing at ``theta`` degrees through the volume's centre (theta None: one bundle), axes in the z-x
    plane; returns the image (0-1), the ground-truth layers (occupancy per population) and the axes."""
    sc = Scene((N,) * 3, seed, dev, VOX); c = torch.full((3,), (N - 1) / 2, device=dev)
    fib = {"rf_um": 3.0, "jitter": 0.1, "disp": 0.1} if resolved else {"rf_um": 0.6, "jitter": 0.1, "disp": 0.1}
    axes = []
    for k, t in enumerate([0.0] if theta is None else [-theta / 2, theta / 2]):
        a = torch.tensor([math.cos(math.radians(60 + t)), 0.0, math.sin(math.radians(60 + t))], device=dev)
        g = sc.bundle(None, "single", axis=a, centre=c, radius=18.0, curv=1e-4, fibre=fib)
        sc.groups.append(g); axes.append(a)
    ren = sc.render()
    occ = torch.stack([r["occupancy"].float() for r in ren["groups"]])
    tot = occ.sum(0); occ = occ / tot.clamp_min(1.0)[None]                            # the volume budget, as the generator composites
    rng = torch.Generator(device=dev).manual_seed(seed)
    con = [-0.55, -0.4][:occ.shape[0]]
    if step_bg:                                                                          # a grey / white-like step through the centre, oblique to
        co = torch.stack(torch.meshgrid(*[torch.arange(N, device=dev, dtype=torch.float32)] * 3, indexing="ij"))   # the bundle; partial volume:
        side = ((co[0] - c[0]) * 0.8 + (co[2] - c[2]) * 0.6) > 0                           # the fibre's contrast follows the local background and
        bgv = torch.where(side, torch.tensor(0.75, device=dev), torch.tensor(0.35, device=dev))   # flips sign across the step
        img = bgv * (1 - occ.sum(0)) + 0.55 * occ.sum(0)
    else:
        img = 0.75 + sum(cc * o for cc, o in zip(con, occ))
    img = gaussian(img, 1.0) + 0.03 * torch.randn(img.shape, generator=rng, device=dev)
    return img.clamp(0, 1), occ, axes


def model_layers(x, voxel_um, step=2, sigma=2.5, n_dirs=300):
    """The decoder on a dense grid of the mid z-x slice: entities per point, grouped across the slice into populations
    by their embeddings. Returns per-population mass maps, mean axes, the per-point entities and grid."""
    Z, Y, X = x.shape; y0 = Y // 2
    gz, gx = torch.meshgrid(torch.arange(4, Z - 4, step), torch.arange(4, X - 4, step), indexing="ij")
    pts = torch.stack([gz.flatten().float(), torch.full((gz.numel(),), float(y0)), gx.flatten().float()], 1).to(dev)
    dirs = EI.sphere_dirs(n_dirs, dev)
    with torch.no_grad():
        f = EI.features(model, x, voxel_um=voxel_um)
        r = EI.query(model, f, pts, torch.full((pts.shape[0],), sigma, device=dev), dirs, vol_shape=x.shape)
        e = EI.cluster(r, dirs)
    live = e["live"] & (e["share"] >= 0.1)
    if bool(live.any()):                                                                # empty background: entities of negligible mass hidden
        live &= e["mass"] >= 0.1 * torch.quantile(e["mass"][live].float(), 0.95)
    qi, ci = torch.nonzero(live, as_tuple=True)
    z = e["z"][qi, ci]; m = e["mass"][qi, ci]
    # populations: average linkage of the embeddings (a subsample), cut at one tree edge; the rest to the nearest
    from scipy.cluster.hierarchy import fcluster
    n = z.shape[0]; sub = torch.randperm(n, generator=torch.Generator().manual_seed(0))[:min(n, 2500)].to(dev)
    if GROUP == "vote":                                                                 # the votes' mode axes: average linkage of the
        from scipy.cluster.hierarchy import linkage                                     # axial angle, cut at 30 deg (the rest: nearest axis)
        a_all = e["R"][qi, ci, :, 2].float(); a = a_all[sub]
        D = torch.rad2deg(torch.acos((a @ a.T).abs().clamp(max=1))).cpu().numpy()
        iu = np.triu_indices(D.shape[0], 1)
        lab_s = torch.as_tensor(fcluster(linkage(D[iu], "average"), 30.0, "distance") - 1, device=dev)
        K = int(lab_s.max()) + 1
        cent = torch.stack([torch.linalg.eigh((a[lab_s == k][:, :, None] * a[lab_s == k][:, None, :]).sum(0))[1][:, -1] for k in range(K)])
        lab = (a_all @ cent.T).abs().argmax(1)
    else:
        lab_s = fcluster(EI.hierarchy(z[sub]), TREE_EDGE, "distance") - 1
        lab_s = torch.as_tensor(lab_s, device=dev)
        K = int(lab_s.max()) + 1
        cent = torch.stack([z[sub][lab_s == k].mean(0) for k in range(K)])
        lab = poincare_dist(z[:, None], cent[None]).argmin(1)
    msum = torch.zeros(K, device=dev).index_add_(0, lab, m); order = msum.argsort(descending=True)
    rank = torch.empty_like(order); rank[order] = torch.arange(K, device=dev); lab = rank[lab]
    H, Wd = gz.shape
    maps = torch.zeros(K, H * Wd, device=dev).index_put_((lab, qi), m, accumulate=True).view(K, H, Wd)
    ax = e["R"][qi, ci, :, 2]
    mean_ax = []
    for k in range(K):
        a = ax[lab == k]; w = m[lab == k]
        M = (w[:, None, None] * a[:, :, None] * a[:, None, :]).sum(0); mean_ax.append(torch.linalg.eigh(M)[1][:, -1])
    return maps, torch.stack(mean_ax), (pts, e, live, lab, qi, ci), (gz, gx)


def ang(a, b):
    return float(torch.rad2deg(torch.acos((a * b).sum(-1).abs().clamp(max=1))))


cases = [("single bundle, resolved", None, True, False), ("30 deg, resolved", 30, True, False), ("60 deg, resolved", 60, True, False),
         ("90 deg, resolved", 90, True, False), ("45 deg, sub-voxel fibres", 45, False, False), ("90 deg, sub-voxel fibres", 90, False, False),
         ("bundle across a grey/white-like step (contrast flips)", None, True, True)]
rows = []
for name, th, res, stp in cases:
    img, occ, axes = phantom(th, res, step_bg=stp)
    rows.append((name, img, occ, axes, VOX))
if with_p4:
    P = "/home/andrew/bridge_wmcov/shapetensor/phantoms/"
    for pid in ["P4-lines-c90-f1"]:
        z_ = np.load(P + pid + "_s0.npz")
        vol = torch.from_numpy(z_["image"][272:368, 112:208, 264:360].astype(np.float32)).to(dev)
        d1 = torch.from_numpy(z_["dir1"][320, 312]).to(dev); d2 = torch.from_numpy(z_["dir2"][320, 312]).to(dev)
        rows.append((pid + " (old texture phantom, 20 um)", EI.normalise(vol), None, [d1, d2], 20.0))
ncol = 6
fig, axs = plt.subplots(len(rows), ncol, figsize=(4 * ncol, 4 * len(rows)))
summary = {}
cols = ["tab:red", "tab:cyan", "tab:orange", "tab:green", "tab:purple"]
for i, (name, img, occ, axes, vu) in enumerate(rows):
    maps, mean_ax, (pts, e, live, lab, qi, ci), (gz, gx) = model_layers(img, vu)
    y0 = img.shape[1] // 2
    sl = img[:, y0, :].cpu().numpy()
    axs[i, 0].imshow(sl, cmap="gray", vmin=0, vmax=1); axs[i, 0].set_title(f"{name}\nimage (z-x, mid y)")
    for j in range(2):
        a_ = axs[i, 1 + j]
        if occ is not None and j < occ.shape[0]:
            a_.imshow(occ[j, :, y0, :].cpu().numpy(), cmap="magma", vmin=0, vmax=1); a_.set_title(f"ground truth layer {j + 1}")
        else:
            a_.axis("off"); a_.set_title("ground truth layer: -" if occ is None or j >= occ.shape[0] else "")
    ext = (float(gx.min()), float(gx.max()), float(gz.max()), float(gz.min()))
    K = maps.shape[0]
    for j in range(2):
        a_ = axs[i, 3 + j]
        if j < K:
            a_.imshow(maps[j].cpu().numpy(), cmap="magma", vmin=0, vmax=max(float(maps[:2].max()), 1e-3), extent=ext)
            note = ""
            if axes:
                note = " | axis vs truth " + ", ".join(f"{ang(mean_ax[j], a):.0f}" for a in axes) + " deg"
            a_.set_title(f"model layer {j + 1} (mass){note}")
        else:
            a_.axis("off"); a_.set_title(f"model layer {j + 1}: none")
    a_ = axs[i, 5]; a_.imshow(sl, cmap="gray", alpha=0.35, vmin=0, vmax=1)
    segs, cc = [], []
    for k in range(qi.numel()):
        if k % 2:
            continue
        q, c = int(qi[k]), int(ci[k]); a = e["R"][q, c, :, 2].cpu().numpy(); L = 2.2 * float(e["share"][q, c]) ** 0.5
        pz, px = float(pts[q, 0]), float(pts[q, 2]); nrm = math.hypot(a[0], a[2]) + 1e-9
        segs.append([(px - L * a[2] / nrm, pz - L * a[0] / nrm), (px + L * a[2] / nrm, pz + L * a[0] / nrm)]); cc.append(cols[min(int(lab[k]), 4)])
    a_.add_collection(LineCollection(segs, colors=cc, linewidths=0.9)); a_.set_xlim(0, img.shape[2]); a_.set_ylim(img.shape[0], 0)
    n_ent = live.sum(1).float()
    a_.set_title(f"entity axes by model layer\n{K} layers; entities/point {float(n_ent.mean()):.2f}")
    summary[name] = {"model_layers": K, "entities_per_point": round(float(n_ent.mean()), 2),
                     "layer_axes_vs_truth_deg": [[round(ang(mean_ax[j], a), 1) for a in axes] for j in range(min(K, 3))] if axes else None,
                     "layer_mass_share": [round(float(maps[j].sum() / maps.sum()), 2) for j in range(min(K, 3))]}
fig.suptitle(f"Entity model {ck_path.split('/')[-1]} (step {ck['step']}): ground-truth and separated layers", fontsize=15)
plt.tight_layout(); plt.savefig(out_png, dpi=70)
print(json.dumps({"step": ck["step"], **summary}, indent=1))
