"""Targets at several scales of collections, one scene per row: the image; the objects alone (scale 0) and collections of
objects up to 4 and 16 voxels across, each entity a random colour, brightness its share of the voxel (the trainer's own
targets at every voxel of the middle slice, no window; the isotropic ground and matrix left dark); and the ideal
direction (the main axis of each voxel's total ODF, colour = axis, brightness = line-ness). Scenes built of objects.

    PYTHONPATH=src python scripts/figure_scene_scales.py OUT.png SEED [SEED ...]   (env SCALES, default 0,4,16)
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

out = sys.argv[1]; seeds = [int(x) for x in sys.argv[2:]]; dev = "cuda"
SCALES = [float(x) for x in os.environ.get("SCALES", "0,4,16").split(",")]
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.45)))
SC.MATRIX_P = 1.0; TR.ENTITY["u_max"] = 60_000
gen = LT.difficulty_gen(0.6, dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA))
rcol = lambda ids: torch.tensor(np.stack([np.random.default_rng(int(i) * 7919 + 13).uniform(0.2, 1.0, 3) for i in ids.tolist()]) if len(ids) else np.zeros((0, 3)),
                                device=dev, dtype=torch.float32)
norm = lambda x: np.clip((x - np.percentile(x, 1)) / max(np.percentile(x, 99) - np.percentile(x, 1), 1e-6), 0, 1)
rows = []
for seed in seeds:
    try:
        s, d = SC.draw_scene((64,) * 3, seed=seed, device=dev, strength=0.0, entity_ctx=True, **gen)
    except torch.OutOfMemoryError:
        SC._scene_cache().clear(); torch.cuda.empty_cache(); print(seed, "out of memory"); continue
    ctx = d["entity_ctx"]; tree = ctx["tree"]; gl = list(ctx["group_layer"]); z0 = 32
    fore = torch.tensor([g_ for g_ in range(len(gl)) if 0 <= gl[g_] < len(s.layers) and getattr(s.layers[gl[g_]], "mrole", "fore") == "fore"] or [-9], device=dev)
    yy, xx = torch.meshgrid(torch.arange(64, device=dev), torch.arange(64, device=dev), indexing="ij")
    qp = torch.stack([torch.full_like(yy, z0), yy, xx], -1).reshape(-1, 3).float(); sg = torch.full((qp.shape[0],), 0.5, device=dev)
    panels = []
    for sc_ in SCALES:
        TR.ENTITY["scales"] = sc_
        t = TR.entity_queries(s, d, 64, 0, dev, points=qp, sigma=sg)
        live = t["e_node"] >= 0; m = (t["e_mass"].float() * t["e_vis"].float()).clamp_min(0) * live
        sh = m / m.sum(-1, keepdim=True).clamp_min(1e-9); q, e = torch.nonzero(live, as_tuple=True)
        node = t["e_gnode"][q, e].long(); grp = torch.where(node < tree.n, tree.group[node.clamp(max=tree.n - 1)], torch.full_like(node, -1))
        f = torch.isin(grp, fore) & (sh[q, e] > 0.02)
        u, inv = torch.unique(node[f], return_inverse=True)
        img = torch.zeros(64 * 64, 3, device=dev).index_add_(0, q[f], sh[q, e][f][:, None] * rcol(u)[inv]).view(64, 64, 3).cpu().numpy()
        panels.append((sc_, img / max(float(img.max()), 1e-6), int(len(u))))
        if sc_ == SCALES[0]:                                                                       # the ideal direction, from these targets
            T6 = (t["e_T6"].float() * m[..., None]).sum(1) / m.sum(-1).clamp_min(1e-9)[:, None]
            M = torch.stack([torch.stack([T6[:, 0], T6[:, 3], T6[:, 4]], -1), torch.stack([T6[:, 3], T6[:, 1], T6[:, 5]], -1), torch.stack([T6[:, 4], T6[:, 5], T6[:, 2]], -1)], -2)
            ev, E = torch.linalg.eigh(M.double()); ev = ev.float()
            cl = ((ev[:, 2] - ev[:, 1]) / ev[:, 2].clamp_min(1e-9)).clamp(0, 1)
            dec = (E[:, :, 2].float()[:, [2, 1, 0]].abs() * cl[:, None]).view(64, 64, 3).cpu().numpy()
    TR.ENTITY["scales"] = None
    rows.append((seed, d["record"].get("modality"), norm((d["image"].float() / 255)[z0].cpu().numpy()), panels, dec))
    SC._scene_cache().clear(); torch.cuda.empty_cache(); print(seed, "ok", flush=True)

nc = 2 + len(SCALES)
fig, axs = plt.subplots(len(rows), nc, figsize=(4.6 * nc, 4.8 * len(rows))); axs = np.atleast_2d(axs)
for r_, (seed, mod, img, panels, dec) in enumerate(rows):
    axs[r_, 0].imshow(img, cmap="gray", interpolation="nearest"); axs[r_, 0].set_title(f"image ({mod})", fontsize=13)
    for j, (sc_, im_, n_) in enumerate(panels):
        axs[r_, 1 + j].imshow(np.clip(im_, 0, 1), interpolation="nearest")
        axs[r_, 1 + j].set_title(f"objects alone ({n_})" if sc_ < 1 else f"collections up to {sc_:g} voxels ({n_})", fontsize=13)
    axs[r_, -1].imshow(np.clip(dec, 0, 1), interpolation="nearest"); axs[r_, -1].set_title("ideal direction", fontsize=13)
    for a_ in axs[r_]:
        a_.set_xticks([]); a_.set_yticks([])
fig.tight_layout(); fig.savefig(out, dpi=80); print("saved", out)
