#!/usr/bin/env python
"""Label-free real-data check of entity checkpoints (the sense-check set: checks only, never training): per crop, the image, each
checkpoint's region map and the density's peak axis on a grid (glyphs, in-plane part).

Usage: python scripts/learn/real_check_entities.py OUT.png CKPT [CKPT ...]
"""
import sys, glob, json, math, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
sys.path.insert(0, "src")
from shapetensor.learn import entity_infer as EI
from shapetensor.learn.model import PopulationNet
dev = "cuda"; out = sys.argv[1]; ckpts = sys.argv[2:]
pick = ["bigbrain_20um_0", "airyscan_hair_follicle_basement_membrane_0"]
fs = sorted(glob.glob("/home/andrew/bridge_wmcov/shapetensor/sense_check/data/*.npz"))
names = sorted({f.split("/")[-1].rsplit("_", 1)[0] for f in fs})
rng = np.random.default_rng(5); chosen = [n + "_0" for n in rng.choice(names, 10, replace=False)]
files = [f for f in fs if f.split("/")[-1][:-4] in set(pick + chosen)][:12]
models = []
for c in ckpts:
    ck = torch.load(c, map_location=dev, weights_only=False); m = PopulationNet.from_args(ck["args"]).to(dev).eval(); m.load_state_dict(ck["model"])
    models.append((c.split("/")[-1].replace(".pt", ""), m))
dirs = EI.sphere_dirs(150, dev)
fig, axs = plt.subplots(len(files), 1 + 2 * len(models), figsize=(2.6 * (1 + 2 * len(models)), 2.6 * len(files)), squeeze=False)
for i, f in enumerate(files):
    d = np.load(f, allow_pickle=True); x = torch.tensor(d["image"], device=dev).float()
    lo, hi = torch.quantile(x.flatten()[::13], torch.tensor([0.005, 0.995], device=dev)); x = ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    try:
        vu = float(json.loads(str(d["meta"])).get("voxel_um", float("nan")))
    except Exception:
        vu = float("nan")
    N = x.shape[0]; zc = N // 2
    axs[i, 0].imshow(x[zc].cpu(), cmap="gray"); axs[i, 0].set_title(f.split("/")[-1][:-4][:34] + (f"\n{vu:.3g} um" if vu == vu else ""), fontsize=6)
    g = torch.arange(6, N - 6, 4, device=dev, dtype=torch.float32)
    pts = torch.stack(torch.meshgrid(torch.tensor([float(zc)], device=dev), g, g, indexing="ij"), -1).reshape(-1, 3)
    for j, (name, m) in enumerate(models):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            P = m(x[None, None], voxel=torch.tensor([math.log2(vu) if vu == vu else float("nan")], device=dev))
            feats = EI.features(m, x, voxel_um=vu if vu == vu else None)
            q = EI.query(m, feats, pts, torch.full((pts.shape[0],), 2.0, device=dev), dirs)
        rp = torch.sigmoid(P["region_logit"].float())[0, zc].cpu()
        axs[i, 1 + 2 * j].imshow(rp, cmap="magma", vmin=0, vmax=1); axs[i, 1 + 2 * j].set_title(f"{name[:18]} region", fontsize=6)
        f_ = q["f"].float(); pk = dirs[f_.argmax(1)]; sharp = (f_.amax(1) / f_.mean(1).clamp_min(1e-9)).cpu()
        w = (q["m"].float().cpu() if "m" in q else torch.ones(pts.shape[0]))
        segs, cols = [], []
        for p_, u_, s_, w_ in zip(pts.cpu(), pk.cpu(), sharp, w):
            if w_ < 0.05 or s_ < 1.3:
                continue
            v = u_[1:]; L = 1.8 * float(min(1.0, v.norm() + 0.15))
            if float(v.norm()) > 1e-3:
                v = v / v.norm()
            segs.append([(float(p_[2] - L * v[1]), float(p_[1] - L * v[0])), (float(p_[2] + L * v[1]), float(p_[1] + L * v[0]))]); cols.append(float(min(1, w_ * 2)))
        a = axs[i, 2 + 2 * j]; a.imshow(x[zc].cpu(), cmap="gray", alpha=0.5)
        if segs:
            a.add_collection(LineCollection(segs, colors=plt.cm.autumn(np.array(cols)), linewidths=0.8))
        a.set_title(f"{name[:18]} density peak (mass>0.05)", fontsize=6)
for a in axs.flat:
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(out, dpi=60); print("saved", out)
