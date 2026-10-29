"""Sense check of an entity-field checkpoint on the P4 line-crossing phantoms (20 um voxels, ground truth per (z, x))."""
import json, math, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, "src")
from shapetensor.learn import entity_infer as EI
from shapetensor.learn.model import PopulationNet

ck_path, out_png = sys.argv[1], sys.argv[2]
dev = "cuda"
ck = torch.load(ck_path, map_location=dev)
model = PopulationNet.from_args(ck["args"]).to(dev).eval(); model.load_state_dict(ck["model"])
step = ck["step"]
PH = "/home/andrew/bridge_wmcov/shapetensor/phantoms/"
pids = ["P4-lines-single", "P4-lines-c30-f1", "P4-lines-c45-f1", "P4-lines-c60-f1", "P4-lines-c90-f1"]
Z0, Z1, Y0, Y1, X0, X1 = 240, 400, 96, 224, 232, 392                      # a central crop (160 x 128 x 160)
ymid = (Y1 - Y0) // 2
dirs = EI.sphere_dirs(300, dev)
gz, gx = torch.meshgrid(torch.arange(16, 144, 4), torch.arange(16, 144, 4), indexing="ij")
pts = torch.stack([gz.flatten().float(), torch.full_like(gz.flatten().float(), ymid), gx.flatten().float()], 1).to(dev)
sig = torch.full((pts.shape[0],), 3.0, device=dev)
ang = lambda a, b: torch.rad2deg(torch.acos((a * b).sum(-1).abs().clamp(max=1)))
fig, axs = plt.subplots(len(pids), 3, figsize=(15, 5 * len(pids)))
summary = {}
t0 = time.time()
for row, pid in enumerate(pids):
    z = np.load(PH + pid + "_s0.npz")
    vol = torch.from_numpy(z["image"][Z0:Z1, Y0:Y1, X0:X1].astype(np.float32)).to(dev)
    x = EI.normalise(vol)
    with torch.no_grad():
        f = EI.features(model, x, voxel_um=20.0)
        r = EI.query(model, f, pts, sig, dirs, vol_shape=x.shape)
        e = EI.cluster(r, dirs)
    pz = pts[:, 0].long().cpu() + Z0; px = pts[:, 2].long().cpu() + X0
    d1 = torch.from_numpy(z["dir1"][pz, px]).to(dev); d2 = torch.from_numpy(z["dir2"][pz, px]).to(dev)
    npop = int(z["npop"][pz[0], px[0]])
    big = e["live"] & (e["share"] >= 0.15)
    n_ent = big.sum(1)
    ax_ = e["R"][..., :, 2]                                                  # voted mode axes (Q, C, 3)
    # errors: single -> top entity vs dir1; crossing -> best pairing of the two heaviest entities with (dir1, dir2)
    if npop == 1:
        err = ang(ax_[:, 0], d1); rec = None
        false2 = float((n_ent >= 2).float().mean())
        summary[pid] = {"entities_mean": float(n_ent.float().mean()), "false_second": false2, "axis_err_med": float(err.median())}
    else:
        e00, e11 = ang(ax_[:, 0], d1), ang(ax_[:, 1], d2); e01, e10 = ang(ax_[:, 0], d2), ang(ax_[:, 1], d1)
        pair = torch.minimum(torch.maximum(e00, e11), torch.maximum(e01, e10))
        two = n_ent >= 2
        found = two & (pair <= 15)
        top1 = torch.minimum(ang(ax_[:, 0], d1), ang(ax_[:, 0], d2))
        summary[pid] = {"entities_mean": float(n_ent.float().mean()), "two_found": float(two.float().mean()), "pair_within_15deg": float(found.float().mean()),
                        "pair_err_med": float(pair[two].median()) if bool(two.any()) else None, "top_axis_err_med": float(top1.median())}
    # panels: image slice; entity count; glyphs of the entities' axes (z-x projection), length by share
    img = x[:, ymid, :].cpu().numpy()
    axs[row, 0].imshow(img, cmap="gray"); axs[row, 0].set_title(f"{pid}: image (z-x, mid y)")
    cnt = np.full((32, 32), np.nan); cnt.flat[:] = n_ent.cpu().numpy()
    im = axs[row, 1].imshow(cnt, cmap="viridis", vmin=0, vmax=3, extent=(16, 144, 144, 16)); plt.colorbar(im, ax=axs[row, 1], fraction=0.046)
    axs[row, 1].set_title(f"entities per point (share >= 0.15), mean {summary[pid]['entities_mean']:.2f}")
    axs[row, 2].imshow(img, cmap="gray", alpha=0.35)
    segs, cols = [], []
    colors = ["tab:red", "tab:cyan", "tab:orange"]
    for q in range(pts.shape[0]):
        for c in range(min(3, e["share"].shape[1])):
            if not bool(big[q, c]):
                continue
            a = ax_[q, c].cpu().numpy(); L = 3.5 * float(e["share"][q, c]) ** 0.5
            vz, vx = a[0], a[2]; nrm = math.hypot(vz, vx) + 1e-9
            pzq, pxq = float(pts[q, 0]), float(pts[q, 2])
            segs.append([(pxq - L * vx / nrm, pzq - L * vz / nrm), (pxq + L * vx / nrm, pzq + L * vz / nrm)]); cols.append(colors[c])
    axs[row, 2].add_collection(LineCollection(segs, colors=cols, linewidths=1.2)); axs[row, 2].set_xlim(0, 160); axs[row, 2].set_ylim(160, 0)
    s_ = summary[pid]
    ttl = (f"false 2nd {s_['false_second']:.2f}, axis err {s_['axis_err_med']:.1f} deg" if npop == 1 else
           f"2 found {s_['two_found']:.2f}, both within 15 deg {s_['pair_within_15deg']:.2f}, pair err {s_['pair_err_med'] or float('nan'):.1f} deg")
    axs[row, 2].set_title(f"entity axes (red 1st, cyan 2nd, orange 3rd): {ttl}")
fig.suptitle(f"EP3 entity model, step {step}: P4 line phantoms (20 um voxels, sigma 3 voxels, 300 directions)", fontsize=14)
plt.tight_layout(); plt.savefig(out_png, dpi=80)
print(json.dumps({"step": step, "seconds": round(time.time() - t0, 1), **summary}, indent=1))
