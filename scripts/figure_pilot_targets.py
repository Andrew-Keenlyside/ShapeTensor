"""What the pilot trains on, for one scene built of objects (synthetic, construction targets, no blur): the image; one
object -- its material and its own elliptical ODF at each voxel; its population -- the population's material and its
ODF at each voxel; the populations head's targets at queries of one scale (the largest population's ODF, sized by its
share, over the background's share); and, at one marked query, every target the head is given: its four populations'
ODFs and shares, the rest ("other") and the background (the ground).

    PYTHONPATH=src python scripts/figure_pilot_targets.py OUT.png [SEED] [SCALE] [STRENGTH]
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
from shapetensor.learn import scene as SC, lobe_train as LT, entity_train as ET
from odf_render import draw_tensors

out = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 250_000_000
SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0; STRENGTH = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
dev = "cuda"; SC.MATRIX_P = 1.0; ET.POP_BG = True; TR.ENTITY["u_max"] = 60_000
gen = LT.difficulty_gen(0.6, dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA))
s, d = SC.draw_scene((64,) * 3, seed=seed, device=dev, strength=STRENGTH, entity_ctx=True, **gen)
ctx = d["entity_ctx"]; tree = ctx["tree"]; gl = list(ctx["group_layer"])
assert d.get("warp_grid") is None and int(ctx["zoom"]) == 1, "pick another seed (warped or zoomed crop)"
role = {g_: getattr(s.layers[gl[g_]], "mrole", "fore") for g_ in range(len(gl)) if 0 <= gl[g_] < len(s.layers)}
name = {g_: getattr(s.layers[gl[g_]], "name", "?") for g_ in role}

# the construction records in the crop: (voxel, population, object, volume, volume-weighted tensor)
vox, grp, obj, w, v = ctx["entries"]
cs = torch.as_tensor(ctx["canvas_shape"], device=dev); org = torch.as_tensor([int(o) for o in ctx["origin"]], device=dev)
vox = vox.long(); P = torch.stack([vox // (cs[1] * cs[2]), (vox // cs[2]) % cs[1], vox % cs[2]], -1) - org
inside = ((P >= 0) & (P < 64)).all(-1)
P, grp, obj, w, v = P[inside], grp[inside].long(), obj[inside].long(), w[inside].float(), v[inside].float()
fg = torch.tensor([g_ for g_, r_ in role.items() if r_ in ("fore", "faint")], device=dev)
isf = torch.isin(grp, fg)
zm = torch.zeros(64, device=dev).index_add_(0, P[isf, 0], w[isf]); z0 = int(zm.argmax()); sl = (P[:, 0] == z0)

def fields(mask):
    """Material (64, 64) and volume-weighted tensor (64, 64, 6) on the slice of the records in ``mask``."""
    m = mask & sl; li = P[m, 1] * 64 + P[m, 2]
    W = torch.zeros(64 * 64, device=dev).index_add_(0, li, w[m]); V = torch.zeros(64 * 64, 6, device=dev).index_add_(0, li, v[m, :6])
    return W.view(64, 64), V.view(64, 64, 6)

# one object: the foreground object with the most material in the slice
ko = isf & sl; o_ids, o_inv = torch.unique(grp[ko] * (1 << 22) + obj[ko], return_inverse=True)
o_mass = torch.zeros(len(o_ids), device=dev).index_add_(0, o_inv, w[ko]); key = int(o_ids[o_mass.argmax()])
g_obj, o_obj = key >> 22, key & ((1 << 22) - 1)
W_o, V_o = fields((grp == g_obj) & (obj == o_obj)); W_p, V_p = fields(grp == g_obj); W_all, _ = fields(torch.ones_like(grp, dtype=torch.bool))
n_obj_pop = int(torch.unique(obj[(grp == g_obj) & sl]).numel())

def glyphs(ax, W, V, step, thr, share=None):
    yy, xx = np.meshgrid(np.arange(step // 2, 64, step), np.arange(step // 2, 64, step), indexing="ij")
    Wn, Vn = W.cpu().numpy(), V.cpu().numpy(); keep = Wn[yy, xx] > thr
    if keep.any():
        T6 = Vn[yy, xx][keep] / Wn[yy, xx][keep][:, None]
        draw_tensors(ax, np.stack([xx[keep], yy[keep]], -1), T6, 0.9 * step, view="z",
                     scale=None if share is None else np.sqrt(np.clip(share[yy, xx][keep], 0, 1)))

# the populations head's targets at queries of one scale on a grid of the slice
STEP, SIG = 4, 2.0
gy, gx = torch.meshgrid(torch.arange(STEP // 2, 64, STEP, device=dev), torch.arange(STEP // 2, 64, STEP, device=dev), indexing="ij")
qp = torch.stack([torch.full_like(gy, z0), gy, gx], -1).reshape(-1, 3).float()
TR.ENTITY["scales"] = SCALE
t = TR.entity_queries(s, d, 64, 0, dev, points=qp, sigma=torch.full((qp.shape[0],), SIG, device=dev))
Tb = {k: x[None] for k, x in t.items() if torch.is_tensor(x) and x.dim() >= 1}
sh, E, valid, other, has = ET.population_targets(Tb, 4); bg = ET.population_bg(Tb)
sh, E, valid, other, bg = sh[0], E[0], valid[0], other[0], bg[0]
T6 = lambda M: torch.stack([M[..., 0, 0], M[..., 1, 1], M[..., 2, 2], M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]], -1)
# the marked query: on the object's population, the most of it there
gn = t["e_gnode"].long(); gq = torch.where((gn >= 0) & (gn < tree.n), tree.group[gn.clamp(0, tree.n - 1)], torch.full_like(gn, -1))
mq = (t["e_mass"].float() * t["e_vis"].float()) * (gq == g_obj); qm = int(mq.sum(1).argmax())

fig = plt.figure(figsize=(21, 10.6)); gs = fig.add_gridspec(2, 4, height_ratios=(1, 0.62))
ax = [fig.add_subplot(gs[0, j]) for j in range(4)]
img = (d["image"].float() / 255)[z0].cpu().numpy()
ax[0].imshow(img, cmap="gray", interpolation="nearest"); ax[0].set_title(f"image ({d['record'].get('modality')}, strength {STRENGTH:g})", fontsize=12)
ax[1].imshow(W_o.clamp(0, 1).cpu().numpy(), cmap="gray", vmin=0, vmax=1, alpha=0.6); glyphs(ax[1], W_o, V_o, 2, 0.05)
ax[1].set_title(f"one object ({name[g_obj]}): its material\nand its own ODF at each voxel", fontsize=12)
ax[2].imshow((W_p / W_all.clamp_min(1e-6)).clamp(0, 1).cpu().numpy(), cmap="magma", vmin=0, vmax=1); glyphs(ax[2], W_p, V_p, 3, 0.05)
ax[2].set_title(f"its population ({name[g_obj]}, {n_obj_pop} objects here): share of\neach voxel's material, and the population's ODF", fontsize=12)
bgm = np.full((64, 64), np.nan); qy, qx = qp[:, 1].long().cpu().numpy(), qp[:, 2].long().cpu().numpy()
for i in range(len(qy)):
    bgm[qy[i] - STEP // 2:qy[i] + STEP // 2, qx[i] - STEP // 2:qx[i] + STEP // 2] = float(bg[i])
ax[3].imshow(bgm, cmap="gray", vmin=0, vmax=1, alpha=0.55)
top = valid[:, 0]
if bool(top.any()):
    draw_tensors(ax[3], np.stack([qx, qy], -1)[top.cpu().numpy()], T6(E[:, 0])[top].cpu().numpy(), 0.95 * STEP, view="z",
                 scale=np.sqrt(sh[top, 0].cpu().numpy() / max(float(sh[:, 0].max()), 1e-6)))
ax[3].plot(qx[qm], qy[qm], "s", mfc="none", mec="cyan", ms=14, mew=2)
ax[3].set_title(f"head targets at scale {SCALE:g} (window sigma {SIG:g}): largest\npopulation's ODF (size = share), grey = background", fontsize=12)
for a_ in ax:
    a_.set_xticks([]); a_.set_yticks([]); a_.set_xlim(-0.5, 63.5); a_.set_ylim(63.5, -0.5)
# the marked query: everything the head is given there
labels = [f"population {k + 1}" for k in range(4)] + ["other", "background\n(ground)"]
vals = [float(sh[qm, k]) for k in range(4)] + [float(other[qm]), float(bg[qm])]
axg = fig.add_subplot(gs[1, :2]); axb = fig.add_subplot(gs[1, 2:])
cen = [(k * 3.0, 0.0) for k in range(4) if bool(valid[qm, k])]
if cen:
    draw_tensors(axg, np.array(cen), T6(E[qm])[valid[qm]].cpu().numpy(), 1.6, view="z")
iso = np.array([[1, 1, 1, 0, 0, 0]], dtype=float) / 3
draw_tensors(axg, np.array([[15.0, 0.0]]), iso, 1.6, view="z")
for k, lab in enumerate(["population 1", "population 2", "population 3", "population 4", "", "background (isotropic)"]):
    if lab and (k >= 4 or bool(valid[qm, k])):
        axg.text(k * 3.0, -1.55, lab, ha="center", fontsize=11)
axg.set_xlim(-1.8, 16.8); axg.set_ylim(-2.0, 1.3); axg.set_aspect("equal"); axg.axis("off")
axg.set_title(f"at the marked query (cyan square): the ODFs the head is trained to give", fontsize=12)
axb.bar(range(6), vals, color=["#4a7bd0"] * 4 + ["#999999", "#555555"])
axb.set_xticks(range(6)); axb.set_xticklabels(labels, fontsize=10); axb.set_ylim(0, 1); axb.set_ylabel("share of the window's material")
for k, v_ in enumerate(vals):
    axb.text(k, v_ + 0.02, f"{v_:.2f}", ha="center", fontsize=10)
axb.set_title("... and the shares (they sum to 1)", fontsize=12)
fig.tight_layout(); fig.savefig(out, dpi=80)
print("saved", out, "| slice", z0, "| object population", name[g_obj], "| marked query shares", [round(x, 3) for x in vals])
