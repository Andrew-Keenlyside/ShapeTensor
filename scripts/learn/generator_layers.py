"""Every layer of held-out scene-generator crops scored with the composable phantoms' criteria
(crossing_acceptance.evaluate_p2: level, flat across overlaps, count, leakage, edges, direction, background, ODF shape,
match rate): each population layer's resolved entities aggregated into one mass and one Bingham per point
(gen_crossings.population_targets), the model's entities from the shared readout (entity_infer.readout); crops whose
ground truth fails the measurement are dropped and listed. The generator's targets as the models are now trained on:
orientation visibility from the layers' oriented texture, the entity tensors' dispersion floor.

Usage: python scripts/learn/generator_layers.py OUT.png CKPT [--n 12] [--level 0.6] [--seed0 6300000] [--orient-from-contrast] [--mc-fibres]
"""
import argparse, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
import shapetensor.learn.scene as SS
import shapetensor.learn.generator as GN
from shapetensor.learn import lobe_train as LT, losses as L, entity_infer as EI, gen_crossings as GC, crossing_acceptance as CA, entity_train as ET
from shapetensor.learn.model import PopulationNet

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("ckpt"); ap.add_argument("--n", type=int, default=12)
ap.add_argument("--level", type=float, default=0.6); ap.add_argument("--seed0", type=int, default=6_300_000); ap.add_argument("--rows", type=int, default=5)
ap.add_argument("--orient-from-contrast", action="store_true"); ap.add_argument("--tag", default="")
ap.add_argument("--mc-fibres", action="store_true", help="held-out crops from the generator with Monte Carlo random fibres (learn/fibres.py)")
a = ap.parse_args(); dev = "cuda"; stem = a.out.rsplit(".", 1)[0]
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None
L.DESC_K = 5; L.DESC_NC = 0; TR.READ_KEEP = 1.0; TR.REGION_DENSITY = True; TR.set_background_entity(True)
GN.ORIENT_FROM_TEXTURE = not a.orient_from_contrast
if a.mc_fibres:
    from shapetensor.learn import fibres as FB
    FB.PACKING, FB.RELAX = "random", True
gen = {"scene": {**LT.difficulty_gen(a.level, {"canvas": 128, "crops_per_scene": 1, "cross_p": 0.6, "p_centre": 0.7, "deep_tree": True}), "crops_per_scene": 1}}
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"])
name = a.tag or (a.ckpt.split("/")[-2] if a.ckpt.endswith("model.pt") else a.ckpt.split("/")[-1].replace(".pt", ""))
sph = EI.sphere_dirs(200, dev); n = 64
gu, gv = torch.meshgrid(torch.arange(n, device=dev, dtype=torch.float32), torch.arange(n, device=dev, dtype=torch.float32), indexing="ij")
pts = torch.stack([torch.full_like(gu, n / 2), gu, gv], -1).view(-1, 3); sig = torch.full((n * n,), 2.5, device=dev)
face = torch.minimum(pts, (n - 1) - pts).amin(1).view(n, n).cpu().numpy(); disc = np.ones((n, n), bool)


def oracle(tgt):
    iso = CA._types(tgt["k"][..., 0], tgt["k"][..., 1]) == 2
    return {"mass": np.where(iso, 0.0, tgt["mass"]), "k": tgt["k"], "R": tgt["R"], "iso": (tgt["mass"] * iso).sum(1), "bg": tgt["bg"], "bgm": tgt["bg"]}


acc, dropped, shown, seed = [], [], [], a.seed0
while len(acc) + len(dropped) < a.n and seed < a.seed0 + 400:
    SS._CACHE.clear()
    s, d = SS.draw_scene((n,) * 3, seed=seed, device=dev, strength=0.5, entity_ctx=True, **gen["scene"]); seed += 1
    tgt = GC.population_targets(TR, s, d, seed - 1, pts, sig, dev)
    if tgt["mass"].shape[1] == 0:
        continue
    lab = f"seed {seed - 1} ({','.join(k[:5] for k in tgt['kinds'])})"
    orc = CA.evaluate_p2(n, tgt, oracle(tgt), face, disc)
    if not orc["PASS"]:
        dropped.append((lab, [k.split()[0] for k, v in orc.items() if k != "PASS" and not v[0]])); continue
    ro = EI.readout(net, d["image"].float() / 255.0, pts, sig, sph, voxel_um=d.get("voxel_um"), iso_k=ET.VOTE_KAPPA_MIN)
    mdl = {k: v.cpu().numpy() for k, v in ro.items()}
    res = CA.evaluate_p2(n, tgt, mdl, face, disc); acc.append((lab, res))
    if len(shown) < a.rows:
        shown.append((lab, d["image"][n // 2].float().cpu().numpy() / 255.0, tgt, mdl, CA.match_p2(tgt, mdl), res))
npass = sum(r["PASS"] for _, r in acc)
print(f"ground-truth check: {len(dropped)} of {len(acc) + len(dropped)} crops unscoreable: {dropped}", flush=True)
cols = ["image", "target pop 1", "model pop 1", "target pop 2", "model pop 2", "model: other\ndirections", "model:\nisotropic objects", "target\nbackground", "model\nbackground"]
norm = PowerNorm(0.5, vmin=0, vmax=1); cmap = matplotlib.colormaps["viridis"].copy(); cmap.set_over("red")
fig, axs = plt.subplots(max(len(shown), 1), len(cols), figsize=(2.3 * len(cols) + 1.0, 2.4 * max(len(shown), 1)), squeeze=False, layout="constrained")
for i, (lab, img, tgt, mdl, m, res) in enumerate(shown):
    K = tgt["mass"].shape[1]; sh = lambda v: v.reshape(n, n)
    axs[i, 0].imshow(img, cmap="gray"); axs[i, 0].set_ylabel(f"{lab}\n{'PASS' if res['PASS'] else 'FAIL'}", fontsize=8, color="darkgreen" if res["PASS"] else "firebrick")
    panels = []
    for k in range(2):
        panels += [(sh(tgt["mass"][:, k]), f"sum {tgt['mass'][:, k].sum():.0f}"), (sh(m["m"][:, k]), f"sum {m['m'][:, k].sum():.0f}")] if k < K else [(None, ""), (None, "")]
    panels += [(sh(m["other"]), ""), (sh(mdl["iso"]), ""), (sh(tgt["bg"]), ""), (sh(mdl["bg"]), "")]
    for j, (v, t_) in enumerate(panels):
        if v is not None:
            im = axs[i, 1 + j].imshow(v, cmap=cmap, norm=norm); axs[i, 1 + j].text(2, 61, t_, color="w", fontsize=7)
    for ax_ in axs[i]:
        ax_.set_xticks([]); ax_.set_yticks([])
for j, t_ in enumerate(cols):
    axs[0, j].set_title(t_, fontsize=9)
fig.suptitle(f"generator layers (difficulty {a.level}) -- {name}: {npass}/{len(acc)} crops pass every criterion for every layer", fontsize=10)
plt.savefig(a.out, dpi=70); print("saved", a.out)
CA.table_figure(acc, stem + "_acceptance.png", f"generator layers (difficulty {a.level}) -- {name}: {npass}/{len(acc)}")
CA.save_json(acc, stem + "_acceptance.json")
print("acceptance:", f"{npass}/{len(acc)} |", " ; ".join(f"{lab}: " + " ".join(k.split()[0] + ("+" if v[0] else "-") for k, v in r.items() if k != "PASS") for lab, r in acc))
