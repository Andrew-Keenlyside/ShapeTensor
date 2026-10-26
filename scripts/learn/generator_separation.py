"""Separation on held-out scene-generator crossings, with the crossing ladder's acceptance criteria adapted to irregular
shapes (:func:`~shapetensor.learn.crossing_acceptance.evaluate_generic`): per crossing, the slice through it (in the
plane both populations lie most in); the image with A (orange) and B (cyan) outlined; the target and the model's mass of
A and of B (the model's own entities, mass split by their lobes with the density's floor as the background, named A / B
by the nearer local axis within 30 deg); the other object populations' target and the model's mass in other
directions; the model's isotropic mass. A table of the criteria beside it (OUT_acceptance.png / .json).

Usage: python scripts/learn/generator_separation.py OUT.png CKPT [--level 0.6] [--n 8] [--rows 4] [--seed0 6100000]
"""
import argparse, math, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
import shapetensor.learn.scene as SS
from shapetensor.learn import lobe_train as LT, losses as L, entity_infer as EI, gen_crossings as GC, crossing_acceptance as CA, entity_train as ET
from shapetensor.learn.model import PopulationNet

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("ckpt"); ap.add_argument("--level", type=float, default=0.6)
ap.add_argument("--n", type=int, default=8); ap.add_argument("--rows", type=int, default=4); ap.add_argument("--seed0", type=int, default=6_100_000)
ap.add_argument("--floor", type=float, default=0.05); ap.add_argument("--tag", default="")
a = ap.parse_args(); dev = "cuda"; stem = a.out.rsplit(".", 1)[0]
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None
L.DESC_K = 5; L.DESC_NC = 0; TR.READ_KEEP = 1.0; TR.REGION_DENSITY = True; TR.set_background_entity(True)
gen = {"scene": {**LT.difficulty_gen(a.level, {"canvas": 128, "crops_per_scene": 1, "cross_p": 1.0, "p_centre": 1.0, "deep_tree": True}), "crops_per_scene": 1}}
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"])
name = a.tag or a.ckpt.split("/")[-1].replace(".pt", "") if not a.ckpt.endswith("model.pt") else a.ckpt.split("/")[-2]
cs, tried = GC.cases(TR, SS, gen, a.seed0, a.n, dev)
print(f"{len(cs)} held-out generator crossings (level {a.level}) from {tried} scenes", flush=True)
ET.VOTE_KAPPA_MIN = 0.5                                                                           # the training's oriented threshold (the readout's too)
sph = EI.sphere_dirs(200, dev); C30 = math.cos(math.radians(30))


def name_maps(tm, ms, kv, R_):
    """The model's (or the truth's) maps from entities ``ms (P, C)``, concentrations ``kv (P, C, 2)`` and frames: named A / B
    by the nearer local axis within 30 deg; isotropic below the training's oriented threshold (one threshold for both)."""
    n = tm["tA"].shape[0]
    iso = kv[..., 0] < ET.VOTE_KAPPA_MIN; ax = R_[..., :, 2]
    cA = (ax * tm["axes"]["A"][0][:, None]).sum(-1).abs() * tm["axes"]["A"][1][:, None]
    cB = (ax * tm["axes"]["B"][0][:, None]).sum(-1).abs() * tm["axes"]["B"][1][:, None]
    toA = ~iso & (cA >= C30) & (cA >= cB); toB = ~iso & (cB >= C30) & (cB > cA)
    M = {"mA": (ms * toA).sum(1), "mB": (ms * toB).sum(1), "mO": (ms * (~iso & ~toA & ~toB)).sum(1), "mI": (ms * iso).sum(1)}
    return M, iso, ax


def oracle(c):
    """The ground truth through the same naming and criteria: the targets' own entities as a perfect model's."""
    et = c["tm"]["ent_t"]; n = c["tm"]["tA"].shape[0]
    ms, kv, R_ = (torch.as_tensor(et[k], device=dev).float() for k in ("mass", "kappa", "R"))
    M, iso, ax = name_maps(c["tm"], ms, kv, R_)
    M["mI"] = M["mI"] + torch.as_tensor(et["iso_mass"], device=dev).float()
    M = {k: v.view(n, n).cpu().numpy() for k, v in M.items()}
    return CA.evaluate_generic(c["tm"], M, {"mass": ms.cpu().numpy(), "iso": iso.cpu().numpy(), "axis": ax.cpu().numpy()})


def model_maps(c):
    tm, d = c["tm"], c["d"]; n = tm["tA"].shape[0]
    with torch.no_grad():
        f = EI.features(net, d["image"].float() / 255.0, voxel_um=d.get("voxel_um"))
        r = EI.query(net, f, tm["pts"], tm["sig"], sph, vol_shape=d["image"].shape)
        e = EI.cluster(r, sph, floor=a.floor if a.floor >= 0 else None); sp = EI.split_mass(r, sph, e)
    ms = (sp["mass"] * e["live"]).float()
    M, iso, ax = name_maps(tm, ms, e["kappa"].float(), e["R"].float())
    M["mI"] = M["mI"] + sp["floor"]
    M = {k: v.view(n, n).float().cpu().numpy() for k, v in M.items()}
    return M, {"mass": ms.float().cpu().numpy(), "iso": iso.cpu().numpy(), "axis": ax.cpu().numpy()}


acc = []; shown = []; dropped = []
for c in cs:
    orc = oracle(c)
    if not orc["PASS"]:                                                                          # the truth fails the measurement: unscoreable
        dropped.append((c["seed"], [k.split()[0] for k, v in orc.items() if k != "PASS" and not v[0]])); continue
    M, ent = model_maps(c)
    acc.append((f"seed {c['seed']}, {c['lc'][4]:.0f} deg", CA.evaluate_generic(c["tm"], M, ent)))
    if len(shown) < a.rows:
        shown.append((c, M))
cols = ["image", "target A", "model A", "target B", "model B", "target: other\npopulations", "model: other\ndirections", "model: isotropic"]
norm = PowerNorm(0.5, vmin=0, vmax=1); cmap = matplotlib.colormaps["viridis"].copy(); cmap.set_over("red")
fig, axs = plt.subplots(len(shown), len(cols), figsize=(2.5 * len(cols) + 1.0, 2.6 * len(shown)), squeeze=False, layout="constrained")
for i, (c, M) in enumerate(shown):
    tm = c["tm"]; res = acc[i][1]
    axs[i, 0].imshow(tm["image"], cmap="gray")
    for j, img in enumerate((tm["tA"], M["mA"], tm["tB"], M["mB"], tm["tO"], M["mO"], M["mI"])):
        im = axs[i, 1 + j].imshow(img, cmap=cmap, norm=norm); axs[i, 1 + j].text(2, 61, f"sum {np.nansum(img):.0f}", color="w", fontsize=8)
    for j in range(len(cols)):
        for occ, col, show in ((tm["occA"], "orange", (0, 1, 2, 5, 6, 7)), (tm["occB"], "cyan", (0, 3, 4, 5, 6, 7))):
            if j in show:
                axs[i, j].contour(occ, levels=[max(0.1, 0.4 * float(occ.max()))], colors=col, linewidths=1.0 if j == 0 else 0.6)
        axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
    axs[i, 0].set_ylabel(f"seed {c['seed']}\nA x B {c['lc'][4]:.0f} deg -- {'PASS' if res['PASS'] else 'FAIL'}", fontsize=9, color="darkgreen" if res["PASS"] else "firebrick")
for j, t in enumerate(cols):
    axs[0, j].set_title(t, fontsize=10)
fig.colorbar(im, ax=axs[:, 1:].ravel().tolist(), shrink=0.5, label="mass per voxel (square-root scale)", extend="max")
npass = sum(r["PASS"] for _, r in acc)
print(f"ground-truth check: {len(dropped)} of {len(cs)} cases unscoreable (the truth fails): {dropped}", flush=True)
fig.suptitle(f"generator crossings (difficulty {a.level}) -- {name}: {npass}/{len(acc)} pass every criterion. Outlines: A orange, B cyan; "
             f"model mass split by lobe fit, density floor = background", fontsize=11)
plt.savefig(a.out, dpi=72); print("saved", a.out)
CA.table_figure(acc, stem + "_acceptance.png", f"generator crossings (difficulty {a.level}) -- {name}: {'PASS' if npass == len(acc) else 'FAIL'} ({npass}/{len(acc)}); "
                "C2 = core/arm ratio, C5 = outer/inner half by distance to the population's edge")
CA.save_json(acc, stem + "_acceptance.json")
print("acceptance:", f"{npass}/{len(acc)} pass |", " ; ".join(f"{lab}: " + " ".join(k.split()[0] + ("+" if v[0] else "-") for k, v in r.items() if k != "PASS") for lab, r in acc))
