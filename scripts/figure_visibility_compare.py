"""The visibility guarantee, before and after, one scene per row (the same scene both ways; the guarantee draws from its
own stream): the clean image without it and with it, what it added (the raised populations' material, any sign), and
the foreground and faint objects in the targets (random colour per object, brightness its share of the voxel).

    PYTHONPATH=src python scripts/figure_visibility_compare.py OUT.png SEED [SEED ...]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import shapetensor.learn.train as TR
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
import shapetensor.learn.generator as G
G.ORIENT_FROM_TEXTURE = True; G.ORIENT_VIS = False; G.TARGETS_FROM_GEOMETRY = True
from shapetensor.learn import scene as SC, lobe_train as LT

out = sys.argv[1]; seeds = [int(x) for x in sys.argv[2:]]; dev = "cuda"; P_ON = SC.VIS_GUARANTEE_P
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.45)))
SC.MATRIX_P = 1.0; TR.ENTITY["u_max"] = 60_000
gen = LT.difficulty_gen(0.6, dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA))
rcol = lambda ids: torch.tensor(np.stack([np.random.default_rng(int(i) * 7919 + 13).uniform(0.2, 1.0, 3) for i in ids.tolist()]) if len(ids) else np.zeros((0, 3)),
                                device=dev, dtype=torch.float32)
norm = lambda x: np.clip((x - np.percentile(x, 1)) / max(np.percentile(x, 99) - np.percentile(x, 1), 1e-6), 0, 1)
rows = []
for seed in seeds:
    ims = {}
    for p_ in (0.0, P_ON):
        SC.VIS_GUARANTEE_P = p_
        s, d = SC.draw_scene((64,) * 3, seed=seed, device=dev, strength=0.0, entity_ctx=(p_ > 0), **gen)
        ims[p_] = ((d["image"].float() / 255).cpu(), d["record"].get("modality"), d["record"].get("vis_raised", 0))
        if p_ > 0:
            break
        SC._scene_cache().clear(); torch.cuda.empty_cache()
    ctx = d["entity_ctx"]; tree = ctx["tree"]; gl = list(ctx["group_layer"])
    # the slice holding the most foreground material
    fl = [k for k, L in enumerate(s.layers) if getattr(L, "mrole", None) in ("fore", "faint")]
    zm = sum((d["occupancy"][k].float().sum((1, 2)) for k in fl), torch.zeros(64, device=dev)); z0 = int(zm.argmax())
    fore = torch.tensor([g_ for g_ in range(len(gl)) if 0 <= gl[g_] < len(s.layers) and getattr(s.layers[gl[g_]], "mrole", "fore") in ("fore", "faint")] or [-9], device=dev)
    yy, xx = torch.meshgrid(torch.arange(64, device=dev), torch.arange(64, device=dev), indexing="ij")
    qp = torch.stack([torch.full_like(yy, z0), yy, xx], -1).reshape(-1, 3).float(); sg = torch.full((qp.shape[0],), 0.5, device=dev)
    TR.ENTITY["scales"] = 0.0
    t = TR.entity_queries(s, d, 64, 0, dev, points=qp, sigma=sg); TR.ENTITY["scales"] = None
    live = t["e_node"] >= 0; m = (t["e_mass"].float() * t["e_vis"].float()).clamp_min(0) * live
    sh = m / m.sum(-1, keepdim=True).clamp_min(1e-9); q, e = torch.nonzero(live, as_tuple=True)
    node = t["e_gnode"][q, e].long(); grp = torch.where(node < tree.n, tree.group[node.clamp(max=tree.n - 1)], torch.full_like(node, -1))
    f = torch.isin(grp, fore) & (sh[q, e] > 0.02); u, inv = torch.unique(node[f], return_inverse=True)
    tgt = torch.zeros(64 * 64, 3, device=dev).index_add_(0, q[f], sh[q, e][f][:, None] * rcol(u)[inv]).view(64, 64, 3).cpu().numpy()
    a0, a1 = ims[0.0][0][z0].numpy(), ims[P_ON][0][z0].numpy()
    tis = (d["occupancy"].float().sum(0)[z0] > 0.05).cpu().numpy()                            # one grey scale for both images, from the
    ref_ = a0[tis] if tis.sum() > 100 else a0.ravel()                                          # tissue (not a mount's bright corner)
    lo, hi = np.percentile(ref_, 1), np.percentile(ref_, 99)
    sc_ = lambda x: np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    rows.append((seed, ims[P_ON][1], ims[P_ON][2], sc_(a0), sc_(a1), np.abs(a1 - a0), tgt / max(float(tgt.max()), 1e-6), int(len(u))))
    SC._scene_cache().clear(); torch.cuda.empty_cache(); print(seed, "raised", ims[P_ON][2], flush=True)
SC.VIS_GUARANTEE_P = P_ON

fig, axs = plt.subplots(len(rows), 4, figsize=(18.4, 4.8 * len(rows))); axs = np.atleast_2d(axs)
for r_, (seed, mod, nr, a0, a1, df, tgt, nu) in enumerate(rows):
    axs[r_, 0].imshow(a0, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); axs[r_, 0].set_title(f"before ({mod})", fontsize=13)
    axs[r_, 1].imshow(a1, cmap="gray", vmin=0, vmax=1, interpolation="nearest"); axs[r_, 1].set_title(f"after ({nr} populations raised)", fontsize=13)
    axs[r_, 2].imshow(df, cmap="magma", vmin=0, vmax=max(float(df.max()), 1e-6), interpolation="nearest"); axs[r_, 2].set_title("what was added", fontsize=13)
    axs[r_, 3].imshow(np.clip(tgt, 0, 1), interpolation="nearest"); axs[r_, 3].set_title(f"foreground objects in the targets ({nu})", fontsize=13)
    for a_ in axs[r_]:
        a_.set_xticks([]); a_.set_yticks([])
fig.tight_layout(); fig.savefig(out, dpi=80); print("saved", out)
