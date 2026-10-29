"""Held-out skill of entity-model checkpoints on the same crops: whether each knows where the material is and which way
it lies -- region correlation, mass skill over a constant, density correlation, mode-axis error -- per generator mix
(the families, or compositional populations only) and difficulty level. One row per checkpoint and setting; a figure
of a few crops (image, region target, each checkpoint's region map).

Usage: python scripts/learn/skill_eval.py OUT_STEM CKPT [CKPT ...] [--n 12] [--device cuda]
"""
import argparse, json, sys
import numpy as np
import torch
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_train as ET, lobe_train as LT, losses as L
from shapetensor.learn.model import PopulationNet
from shapetensor.learn.scene import Scene

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("ckpts", nargs="+"); ap.add_argument("--n", type=int, default=12)
ap.add_argument("--device", default="cuda"); ap.add_argument("--levels", type=float, nargs="+", default=[0.3, 0.6, 1.0])
a = ap.parse_args(); dev = a.device
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None
L.DESC_K = 5; L.DESC_NC = 0; TR.ENTITY.update({"sigma": (1.0, 6.0)}); TR.READ_KEEP = 0.25
nets = []
for c in a.ckpts:
    ck = torch.load(c, map_location=dev, weights_only=False)
    net = PopulationNet.from_args(ck["args"]).to(dev); net.load_state_dict(ck["model"]); net.eval()
    nets.append((c.split("/")[-2] if c.endswith("model.pt") else c.split("/")[-1].replace(".pt", ""), net))
mixes = {"families": {"compose": 0.0}, "compose": {k: (1.0 if k == "compose" else 0.0) for k in Scene.FAMILIES}}
rows, fig_cases = [], []
for mix, fw in mixes.items():
    for lv in a.levels:
        base = {"canvas": 128, "crops_per_scene": 1, "cross_p": 0.3, "p_centre": 0.5, "deep_tree": True, "family_w": fw}
        gen = {"scene": LT.difficulty_gen(lv, base)}
        pairs = [TR._make_pair(31_000_000 + 1000 * int(lv * 10) + i + (0 if mix == "families" else 500), 64, dev, lv, gen) for i in range(a.n)]
        acc = {name: {} for name, _ in nets}
        for i in range(0, a.n, 2):
            x, T = TR.stack(pairs[i:i + 2], aug=False)
            gv = torch.Generator(device=dev).manual_seed(123 + i); dirs = ET.sample_dirs(T, 32, g=gv)
            for name, net in nets:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                    P = net(x, voxel=T["voxel"], e_points=T["e_points"], e_sigma=T["e_sigma"], e_dirs=dirs)
                for k, v in {**ET.where_skill(P, T, dirs), **ET.axis_error_deg(P, T, dirs)}.items():
                    acc[name].setdefault(k, []).append(v)
                if i == 0 and lv == a.levels[-1]:
                    fig_cases.append((mix, name, x[0, 0, 32].float().cpu(), T["region_t"][0].float().cpu()[32],
                                      (P["region"][0].float() if "region" in P else torch.sigmoid(P["region_logit"][0].float())).cpu().reshape(64, 64, 64)[32] if "region_logit" in P else None))
        for name, _ in nets:
            r = {"ckpt": name, "mix": mix, "level": lv, **{k: round(float(np.nanmean(v)), 3) for k, v in acc[name].items()}}
            rows.append(r); print(json.dumps(r), flush=True)
json.dump(rows, open(a.out + ".json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
by = {}
for mix, name, img, rt, rp in fig_cases:
    by.setdefault(mix, {"img": img, "rt": rt, "pred": []})["pred"].append((name, rp))
fig, axs = plt.subplots(len(by), 2 + len(nets), figsize=(3.2 * (2 + len(nets)), 3.4 * len(by)), squeeze=False)
for r_, (mix, d) in enumerate(by.items()):
    axs[r_, 0].imshow(d["img"], cmap="gray"); axs[r_, 0].set_title(f"{mix}: image (level {a.levels[-1]})", fontsize=8)
    axs[r_, 1].imshow(d["rt"], cmap="magma", vmin=0, vmax=1); axs[r_, 1].set_title("region target", fontsize=8)
    for j, (name, rp) in enumerate(d["pred"]):
        if rp is not None:
            axs[r_, 2 + j].imshow(rp, cmap="magma", vmin=0, vmax=1)
        axs[r_, 2 + j].set_title(name, fontsize=8)
for ax in axs.flat:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(a.out + ".png", dpi=60); print("saved", a.out + ".png")
