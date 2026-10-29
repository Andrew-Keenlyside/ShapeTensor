"""A full synthetic scene, object by object, from the generator's own construction records (no window, no blur): for
the objects holding the most material in one slice, each object alone -- its material and, at every voxel it occupies,
its own elliptical ODF there -- then all objects together, the scene's per-voxel target (every object's ellipse at the
voxel, scaled by its share of the voxel's material) and the clean render. Synthetic only.

    PYTHONPATH=src python scripts/figure_scene_objects.py OUT.png [SEED] [N_OBJECTS]   (env LEVEL, STRENGTH)
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

out = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 210_000_000; NOBJ = int(sys.argv[3]) if len(sys.argv) > 3 else 7
dev = "cuda"; LEV = float(os.environ.get("LEVEL", 0.6))
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.3)))
gen = LT.difficulty_gen(LEV, dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA))
s, d = SC.draw_scene((64,) * 3, seed=seed, device=dev, strength=float(os.environ.get("STRENGTH", 0.0)), entity_ctx=True, **gen)
ctx = d["entity_ctx"]; zoom = int(ctx["zoom"])
assert d.get("warp_grid") is None and zoom == 1, f"pick another seed (pull warp: {d.get('warp_grid') is not None}, zoom {zoom})"
vox, grp, obj, w, v = ctx["entries"]
cs = torch.as_tensor(ctx["canvas_shape"], device=dev); org = torch.as_tensor([int(o) for o in ctx["origin"]], device=dev)
vox = vox.long(); P = torch.stack([vox // (cs[1] * cs[2]), (vox // cs[2]) % cs[1], vox % cs[2]], -1) - org
inside = ((P >= 0) & (P < 64)).all(-1)
P, grp, obj, w, v = P[inside], grp[inside].long(), obj[inside].long(), w[inside].float(), v[inside].float()
lin = (P[:, 0] * 64 + P[:, 1]) * 64 + P[:, 2]

# the slice holding the most object material
zmass = torch.zeros(64, device=dev).index_add_(0, P[:, 0], w); z0 = int(zmass.argmax())
sl = P[:, 0] == z0
o_ids, o_inv = torch.unique(obj[sl], return_inverse=True)
o_mass = torch.zeros(len(o_ids), device=dev).index_add_(0, o_inv, w[sl])
top = o_ids[o_mass.argsort(descending=True)[:NOBJ]].tolist()
layers = s.layers; gl = list(ctx["group_layer"])


def obj_fields(mask):
    """Per-voxel material and w-weighted tensor of the entries in ``mask``, on the slice z0: (64, 64), (64, 64, 6)."""
    m = mask & sl; li = (P[m, 1] * 64 + P[m, 2])
    W = torch.zeros(64 * 64, device=dev).index_add_(0, li, w[m]); V = torch.zeros(64 * 64, 6, device=dev).index_add_(0, li, v[m, :6])
    return W.view(64, 64), V.view(64, 64, 6)


def name_of(o):
    g_ = int(grp[obj == o][0]); L = layers[gl[g_]] if 0 <= g_ < len(gl) and gl[g_] < len(layers) else None
    return f"{getattr(L, 'name', None) or (L.role if L else '?')} ({L.kind if L else '?'})"


def glyphs(ax, W, V, step=2, share=None):
    yy, xx = np.meshgrid(np.arange(1, 64, step), np.arange(1, 64, step), indexing="ij")
    Wn = W.cpu().numpy(); Vn = V.cpu().numpy()
    keep = Wn[yy, xx] > 0.1
    if keep.any():
        T6 = Vn[yy, xx][keep] / Wn[yy, xx][keep][:, None]
        draw_tensors(ax, np.stack([xx[keep], yy[keep]], -1), T6, 0.62 * step, view="z", scale=None if share is None else np.sqrt(share[yy, xx][keep]))


ncol = max(NOBJ, 3) + 0
fig, axs = plt.subplots(2, max(NOBJ, 4), figsize=(4.2 * max(NOBJ, 4), 8.8)); axs = np.atleast_2d(axs)
img = (d["image"].float() / 255)[z0].cpu().numpy()
for j, o in enumerate(top):
    W, V = obj_fields(obj == o); ax = axs[0, j]
    ax.imshow(W.clamp(0, 1).cpu().numpy(), cmap="gray", vmin=0, vmax=1, alpha=0.5); glyphs(ax, W, V)
    ax.set_title(f"object {o}: {name_of(o)}\nits material and its own ellipse at each voxel", fontsize=9)
# all objects together, coloured per object
ax = axs[1, 0]; rgb = np.zeros((64, 64, 3)); cm = plt.cm.tab20
for k, o in enumerate(torch.unique(obj[sl]).tolist()):
    W, _ = obj_fields(obj == o); rgb += np.clip(W.cpu().numpy(), 0, 1)[..., None] * np.array(cm(k % 20)[:3])
ax.imshow(np.clip(rgb, 0, 1)); ax.set_title(f"all {len(torch.unique(obj[sl]))} objects in this slice (one colour each)", fontsize=9)
# the scene's per-voxel target: every object's ellipse at the voxel, scaled by its share of the voxel's material
ax = axs[1, 1]; ax.imshow(img, cmap="gray", alpha=0.4)
Wall, _ = obj_fields(torch.ones_like(obj, dtype=torch.bool))
for o in torch.unique(obj[sl]).tolist():
    W, V = obj_fields(obj == o); glyphs(ax, W, V, step=3, share=(W / Wall.clamp_min(1e-6)).clamp(0, 1).cpu().numpy())
ax.set_title("scene target, per voxel: every object's ellipse there,\nscaled by its share of the voxel's material", fontsize=9)
ax = axs[1, 2]; ax.imshow(img, cmap="gray"); ax.set_title(f"clean render (strength {os.environ.get('STRENGTH', '0')}), modality {d['record'].get('modality')}", fontsize=9)
for ax in axs.ravel():
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.5, 63.5); ax.set_ylim(63.5, -0.5)
for ax in axs[1, 3:]:
    ax.axis("off")
for ax in axs[0, len(top):]:
    ax.axis("off")
fig.suptitle(f"One synthetic scene, object by object from construction (seed {seed}, slice z = {z0}; no window: each voxel's own ODFs; "
             f"needle = line, disc = sheet; colour = direction, R x, G y, B z)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=80); print("saved", out, "| objects", [(o, name_of(o)) for o in top])
