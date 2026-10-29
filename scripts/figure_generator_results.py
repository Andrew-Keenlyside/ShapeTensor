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
if os.environ.get("R_MIN"):
    TR.ENTITY["r_min"] = float(os.environ["R_MIN"])                                           # the resolution rule's floor (voxels)
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
    # per-object mass targets as training sees them: the trainer's targets at every voxel of the slice, window 0.5 (no
    # spatial averaging), after the resolution rule (objects thinner than the resolution merged into their population)
    yy_, xx_ = torch.meshgrid(torch.arange(64, device=dev), torch.arange(64, device=dev), indexing="ij")
    qp = torch.stack([torch.full_like(yy_, z0), yy_, xx_], -1).reshape(-1, 3).float()
    tq = TR.entity_queries(s, d, 64, 0, dev, points=qp, sigma=torch.full((qp.shape[0],), 0.5, device=dev))
    live = tq["e_node"] >= 0; mq = (tq["e_mass"].float() * tq["e_vis"].float()).clamp_min(0) * live
    shq = mq / mq.sum(-1, keepdim=True).clamp_min(1e-9)
    qi_, ei_ = torch.nonzero(live, as_tuple=True)
    pa, oa, sha = qi_, tq["e_gnode"][qi_, ei_].long().clamp_min(0), shq[qi_, ei_]
    T6q = tq["e_T6"].float()[qi_, ei_]
    t_iso = ((T6q[:, :3] - 1 / 3).abs().sum(-1) < 0.05)                                            # (isotropic entities, e.g. the ground)
    big = sha > 0.05; on_g = ((pa // 64) % STEP == STEP // 2) & ((pa % 64) % STEP == STEP // 2) & big
    cen = torch.stack([pa[on_g] % 64, pa[on_g] // 64], -1).cpu().numpy(); T6 = T6q[on_g].cpu().numpy(); shk = sha[on_g].cpu().numpy()
    rcol = lambda ids: torch.tensor(np.stack([np.random.default_rng(int(i) * 7919 + 13).uniform(0.15, 1.0, 3) for i in ids.tolist()]) if len(ids) else np.zeros((0, 3)),
                                    device=dev, dtype=torch.float32)                                # a random colour per object
    tree = ctx["tree"]; ent_group = lambda n_: tree.group[n_.clamp(max=tree.n - 1)] * (n_ < tree.n) + (-1) * (n_ >= tree.n)
    role_of_g = {g_: rl for g_, rl in roles.items()}
    fg_g = torch.tensor([g_ for g_, rl in role_of_g.items() if rl == "fore"] or [-99], device=dev)
    # (a) construction: every foreground object as built
    mfo = torch.isin(grp[sl], fg_g); key_c = obj[sl][mfo] * 4096 + pix[mfo]
    uc, invc = torch.unique(key_c, return_inverse=True)
    Wc = torch.zeros(len(uc), device=dev).index_add_(0, invc, w[sl][mfo]); pc, oc = uc % 4096, uc // 4096
    uo, invo = torch.unique(oc, return_inverse=True); colc = rcol(uo)[invo]
    cons = torch.zeros(64 * 64, 3, device=dev).index_add_(0, pc, Wc[:, None] * colc)
    cons = (cons / torch.zeros(64 * 64, device=dev).index_add_(0, pc, Wc).clamp_min(1.0)[:, None]).view(64, 64, 3).cpu().numpy()
    n_cons = int(len(uo))
    # (b) training targets: the separate entities at each voxel after the resolution rule, foreground only
    eg = ent_group(oa); fge = torch.isin(eg, fg_g) & (sha > 0.02)
    ue, inve = torch.unique(oa[fge], return_inverse=True); cole = rcol(ue)[inve]
    objcol = torch.zeros(64 * 64, 3, device=dev).index_add_(0, pa[fge], sha[fge][:, None] * cole).view(64, 64, 3).cpu().numpy()
    n_ent = int(len(ue))
    nobj = torch.zeros(64 * 64, device=dev).index_add_(0, pa[fge], torch.ones_like(sha[fge])).view(64, 64).cpu().numpy()
    top1 = torch.zeros(64 * 64, device=dev).scatter_reduce_(0, pa, sha, reduce="amax").view(64, 64).cpu().numpy()
    rows.append((seed, d["record"].get("modality"), d["record"].get("matrix"), img0, img3, role_map.reshape(64, 64, 3).clip(0, 1), cen, T6, shk, dec_raw, dec_w, objcol, nobj, (cons, n_cons, n_ent)))
    SC._scene_cache().clear(); torch.cuda.empty_cache(); print(seed, "ok", d["record"].get("modality"), d["record"].get("matrix"), flush=True)

MASS = bool(os.environ.get("MASS"))
fig, axs = plt.subplots(len(rows), 6, figsize=(25.5, 4.4 * len(rows))); axs = np.atleast_2d(axs)
for r_, (seed, mod, mat, img0, img3, rm, cen, T6, shk, dec_raw, dec_w, objcol, nobj, top1) in enumerate(rows):
  if MASS:                                                                                         # the mass targets
    norm = lambda x: np.clip((x - np.percentile(x, 1)) / max(np.percentile(x, 99) - np.percentile(x, 1), 1e-6), 0, 1)
    axs[r_, 0].imshow(norm(img0), cmap="gray"); axs[r_, 0].set_title(f"seed {seed}: clean render ({mod})", fontsize=9)
    cons, n_cons, n_ent = top1
    axs[r_, 1].imshow(np.clip(cons, 0, 1)); axs[r_, 1].set_title(f"construction: {n_cons} foreground objects as built,\neach a random colour, brightness = its material", fontsize=9)
    axs[r_, 2].imshow(np.clip(objcol, 0, 1)); axs[r_, 2].set_title(f"training targets: {n_ent} separate foreground entities\n(after the resolution rule), each a random colour, by share", fontsize=9)
    im_ = axs[r_, 3].imshow(nobj, cmap="viridis", vmin=0, vmax=max(3, float(nobj.max()))); plt.colorbar(im_, ax=axs[r_, 3], fraction=0.046)
    axs[r_, 3].set_title(f"foreground entities per voxel (share >= 2 %): max {nobj.max():.0f}", fontsize=9)
    axs[r_, 4].imshow(norm(img0), cmap="gray", alpha=0.35)
    if len(cen):
        draw_tensors(axs[r_, 4], cen, T6, 0.62 * STEP, view="z", scale=np.sqrt(shk))
    axs[r_, 4].set_title(f"ODF targets: the {TOP} largest objects per voxel", fontsize=9)
    axs[r_, 5].imshow(np.clip(dec_w, 0, 1)); axs[r_, 5].set_title("ideal single vector (DEC x line-ness)", fontsize=9)
    for a_ in axs[r_]:
        a_.set_xticks([]); a_.set_yticks([]); a_.set_xlim(-0.5, 63.5); a_.set_ylim(63.5, -0.5)
    continue
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
