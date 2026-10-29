"""Generator results, scenes built of objects (a matrix and a ground under the foreground): per scene, the clean
render, the same scene at a moderate strength, where each role's material lies (ground grey, matrix blue, foreground
orange), and the per-voxel target -- at each grid voxel the objects holding most of its material, each its own ellipse
scaled by the square root of its share. From the generator's construction records; no window. Synthetic only.

    PYTHONPATH=src python scripts/figure_generator_results.py OUT.png SEED [SEED ...]   (env STRENGTH, default 0.3; MATRIX)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
from odf_render import draw_tensors

out = sys.argv[1]; seeds = [int(x) for x in sys.argv[2:]]; dev = "cuda"; STR = float(os.environ.get("STRENGTH", 0.3))
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.45)))
SC.MATRIX_P = 1.0
if os.environ.get("MATRIX"):
    SC.MATRIX_TYPES = tuple(os.environ["MATRIX"].split(","))
gen = LT.difficulty_gen(0.6, dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA))
STEP, TOP = 4, 3
ROLE_RGB = {"ground": (0.55, 0.55, 0.55), "matrix": (0.25, 0.45, 1.0), "fore": (1.0, 0.55, 0.1)}
rows = []
for seed in seeds:
    try:
        s, d = SC.draw_scene((64,) * 3, seed=seed, device=dev, strength=0.0, entity_ctx=True, **gen)
        _, d3 = SC.draw_scene((64,) * 3, seed=seed, device=dev, strength=STR, entity_ctx=False, **gen)
    except torch.OutOfMemoryError:
        SC._scene_cache().clear(); torch.cuda.empty_cache(); print(seed, "out of memory"); continue
    ctx = d["entity_ctx"]
    if d.get("warp_grid") is not None or int(ctx["zoom"]) != 1:
        print(seed, "skipped (pull warp or zoom)"); SC._scene_cache().clear(); continue
    vox, grp, obj, w, v = ctx["entries"]; gl = list(ctx["group_layer"])
    cs = torch.as_tensor(ctx["canvas_shape"], device=dev); org = torch.as_tensor([int(o) for o in ctx["origin"]], device=dev)
    vox = vox.long(); P = torch.stack([vox // (cs[1] * cs[2]), (vox // cs[2]) % cs[1], vox % cs[2]], -1) - org
    ok = ((P >= 0) & (P < 64)).all(-1); P, grp, obj, w, v = P[ok], grp[ok].long(), obj[ok].long(), w[ok].float(), v[ok].float()
    z0 = 32; sl = P[:, 0] == z0
    roles = {}
    for g_ in torch.unique(grp).tolist():
        L = s.layers[gl[g_]] if 0 <= g_ < len(gl) and gl[g_] < len(s.layers) else None
        roles[g_] = getattr(L, "mrole", "fore") if L is not None else "fore"
    pix = P[sl, 1] * 64 + P[sl, 2]
    role_map = np.zeros((64 * 64, 3))
    for g_, rl in roles.items():
        m = grp[sl] == g_
        Wr = torch.zeros(64 * 64, device=dev).index_add_(0, pix[m], w[sl][m]).clamp(0, 1).cpu().numpy()
        role_map += Wr[:, None] * np.array(ROLE_RGB[rl])[None]
    on_grid = ((P[sl, 1] % STEP) == STEP // 2) & ((P[sl, 2] % STEP) == STEP // 2)
    key = obj[sl][on_grid] * 4096 + pix[on_grid]
    u, inv = torch.unique(key, return_inverse=True)
    Wo = torch.zeros(len(u), device=dev).index_add_(0, inv, w[sl][on_grid]); Vo = torch.zeros(len(u), 6, device=dev).index_add_(0, inv, v[sl][on_grid, :6])
    px = u % 4096; tot = torch.zeros(64 * 64, device=dev).index_add_(0, px, Wo); share = Wo / tot[px].clamp_min(1e-9)
    order = torch.argsort(px * 10 - share)                                                        # per voxel, largest share first
    px_o = px[order]; first = torch.ones_like(px_o, dtype=torch.bool); first[1:] = px_o[1:] != px_o[:-1]
    rank = torch.arange(len(px_o), device=dev) - torch.cummax(torch.where(first, torch.arange(len(px_o), device=dev), torch.zeros_like(px_o)), 0).values
    keep = order[(rank < TOP) & (share[order] > 0.05)]
    cen = torch.stack([px[keep] % 64, px[keep] // 64], -1).cpu().numpy()
    T6 = (Vo[keep] / Wo[keep][:, None]).cpu().numpy(); shk = share[keep].cpu().numpy()
    img0 = (d["image"].float() / 255)[z0].cpu().numpy(); img3 = (d3["image"].float() / 255)[z0].cpu().numpy()
    # the ideal single vector: each voxel's total ODF (every object's ellipse, weighted by its material), its main axis
    Wv = torch.zeros(64 * 64, device=dev).index_add_(0, pix, w[sl]); Vv = torch.zeros(64 * 64, 6, device=dev).index_add_(0, pix, v[sl][:, :6])
    t6 = Vv / Wv.clamp_min(1e-9)[:, None]
    M = torch.stack([torch.stack([t6[:, 0], t6[:, 3], t6[:, 4]], -1), torch.stack([t6[:, 3], t6[:, 1], t6[:, 5]], -1), torch.stack([t6[:, 4], t6[:, 5], t6[:, 2]], -1)], -2)
    ev, E = torch.linalg.eigh(M.double()); ev = ev.float(); ax_ = E[:, :, 2].float()                # (z, y, x) main axis
    cl = ((ev[:, 2] - ev[:, 1]) / ev[:, 2].clamp_min(1e-9)).clamp(0, 1) * (Wv > 0.05)
    dec = ax_[:, [2, 1, 0]].abs() * (Wv > 0.05)[:, None]                                            # R = x, G = y, B = z
    dec_raw = dec.view(64, 64, 3).cpu().numpy(); dec_w = (dec * cl[:, None]).view(64, 64, 3).cpu().numpy()
    rows.append((seed, d["record"].get("modality"), d["record"].get("matrix"), img0, img3, role_map.reshape(64, 64, 3).clip(0, 1), cen, T6, shk, dec_raw, dec_w))
    SC._scene_cache().clear(); torch.cuda.empty_cache(); print(seed, "ok", d["record"].get("modality"), d["record"].get("matrix"), flush=True)

fig, axs = plt.subplots(len(rows), 6, figsize=(25.5, 4.4 * len(rows))); axs = np.atleast_2d(axs)
for r_, (seed, mod, mat, img0, img3, rm, cen, T6, shk, dec_raw, dec_w) in enumerate(rows):
    norm = lambda x: np.clip((x - np.percentile(x, 1)) / max(np.percentile(x, 99) - np.percentile(x, 1), 1e-6), 0, 1)
    axs[r_, 0].imshow(norm(img0), cmap="gray"); axs[r_, 0].set_title(f"seed {seed}: clean render ({mod}; matrix {', '.join(mat or [])})", fontsize=9)
    axs[r_, 1].imshow(norm(img3), cmap="gray"); axs[r_, 1].set_title(f"same scene, strength {STR:g}", fontsize=9)
    axs[r_, 2].imshow(rm); axs[r_, 2].set_title("material by role: ground grey, matrix blue, foreground orange", fontsize=9)
    axs[r_, 3].imshow(norm(img0), cmap="gray", alpha=0.35)
    if len(cen):
        draw_tensors(axs[r_, 3], cen, T6, 0.62 * STEP, view="z", scale=np.sqrt(shk))
    axs[r_, 3].set_title(f"target per voxel: the {TOP} objects with most material, each its own ellipse", fontsize=9)
    axs[r_, 4].imshow(np.clip(dec_raw, 0, 1)); axs[r_, 4].set_title("ideal single vector: main axis of each voxel's total ODF\n(DEC, R x, G y, B z; unweighted)", fontsize=9)
    axs[r_, 5].imshow(np.clip(dec_w, 0, 1)); axs[r_, 5].set_title("the same, brightness = line-ness\n(isotropic background dark)", fontsize=9)
    for a_ in axs[r_]:
        a_.set_xticks([]); a_.set_yticks([]); a_.set_xlim(-0.5, 63.5); a_.set_ylim(63.5, -0.5)
fig.suptitle("Generator: scenes built of objects (foreground over a matrix over a ground), each object with its own ODF", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(out, dpi=75); print("saved", out)
