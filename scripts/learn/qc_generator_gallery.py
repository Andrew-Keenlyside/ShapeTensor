"""QC gallery of the training data before a long run (27 Oct): random training crops drawn exactly as training draws
them (``draw_scene`` with the training's scene settings: deformation, packed tissue, crop strata, crossings), each as
three mid-slices -- the acquired image, the populations (one colour each), the true direction (rgb = |z|, |y|, |x| of
the occupancy-weighted principal direction, brightness its fill and anisotropy) -- with its record in a line: scene kind,
voxel size, deformation, stratum.

Usage: python scripts/learn/qc_generator_gallery.py OUT.png [--n 24]
"""
import argparse, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import scene as SC

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=24); ap.add_argument("--seed0", type=int, default=120_000_000)
a = ap.parse_args(); dev = "cuda"
TR.HEAD = "entities"
from shapetensor.learn import lobe_train as LT                                              # the training's own settings: its
gen = LT.difficulty_gen(1.0, dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA))   # family lists (full difficulty)
pal = plt.cm.tab20(np.arange(20))[:, :3]
cols = 8; blocks = (a.n + cols - 1) // cols
fig, axs = plt.subplots(3 * blocks, cols, figsize=(2.0 * cols, 2.15 * 3 * blocks))
for i in range(a.n):
    s, d = SC.draw_scene((64,) * 3, seed=a.seed0 + 8 * i, device=dev, strength=1.0, **gen)
    rec = d["record"]; z = 32
    occ = d["occupancy"].float(); T = d["tensors"].float()
    shp = [k for k, L in enumerate(s.layers) if L.role == "shape"]
    img = d["image"].float()[z].cpu().numpy(); q = np.percentile(img, [1, 99]); img = np.clip((img - q[0]) / max(q[1] - q[0], 1e-6), 0, 1)
    pop = np.ones((64, 64, 3))
    if shp:
        o_ = occ[shp, z].cpu().numpy()
        for j in np.argsort(-o_.reshape(len(o_), -1).sum(1)):
            al = np.clip(o_[j], 0, 1)[..., None] * 0.85; pop = pop * (1 - al) + pal[j % 20] * al
        Tw = (occ[shp][:, None] * T[shp]).sum(0)[:, z]
        M = torch.stack([torch.stack([Tw[0], Tw[3], Tw[4]], -1), torch.stack([Tw[3], Tw[1], Tw[5]], -1), torch.stack([Tw[4], Tw[5], Tw[2]], -1)], -2)
        ev, V = torch.linalg.eigh(M.double().cpu() + 1e-12 * torch.eye(3, dtype=torch.float64))
        ani = ((ev[..., 2] - ev[..., 1]) / ev.sum(-1).clamp_min(1e-12)).clamp(0, 1).numpy()
        dec = V[..., -1].abs().numpy() * np.clip(occ[shp, z].sum(0).cpu().numpy(), 0, 1)[..., None] * ani[..., None]
    else:
        dec = np.zeros((64, 64, 3))
    r0, c0 = 3 * (i // cols), i % cols
    for k, x in enumerate((img, pop, np.clip(dec * 1.4, 0, 1))):
        ax = axs[r0 + k, c0]; ax.imshow(x, cmap="gray" if k == 0 else None, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    kind = "tissue" + (" grey" if rec.get("tissue", {}).get("grey") else "") if "tissue" in rec else ("crossing" if rec.get("crossing_scene") else "scene")
    dfm = rec.get("deform") or {}
    dtag = ("aff+" if dfm.get("affine") else "") + (f"warp{dfm['warp'][0]:.0f}" if dfm.get("warp") and dfm["warp"][0] > 0 else "") if dfm else "rigid"
    st = (rec.get("crop_fill") or {}).get("stratum", "centred" if rec.get("crop_fill") is None else "")
    axs[r0, c0].set_title(f"{kind}, {rec.get('voxel_um', 0):.2g} um\n{dtag}, {st}, {rec.get('modality', '')}", fontsize=6.5)
    SC._scene_cache().clear()
for b in range(blocks):
    for k, lab in enumerate(("image", "populations", "direction")):
        axs[3 * b + k, 0].set_ylabel(lab, fontsize=7)
fig.suptitle(f"QC: {a.n} random training crops at the training's settings (mid-slices of 64^3)", fontsize=10, x=0.01, ha="left")
plt.tight_layout(); plt.savefig(a.out, dpi=90); print("saved", a.out)
