"""How one synthetic scene is built, stage by stage, with the orientation information each stage carries.

Top row, a few single objects: each object's own volume (middle slice) and its own ODFs at a grid of points. Each ODF
comes from the object's per-voxel shape tensors (second moments of its local directions), summed in a Gaussian window,
then a moment-matched Bingham, exactly as the training targets are made.
Bottom row:
  - all objects (volume coloured by population);
  - the scene's target ODFs (the mass-weighted mixture of every non-background object, as trained);
  - the clean render;
  - the same scene after imaging corruptions;
  - the final training input after a batch augmentation route.

    PYTHONPATH=src python scripts/figure_scene_construction.py OUT.png [SEED]   (env WINDOW, LEVEL)
"""
import math
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
from shapetensor.filters import gaussian
from shapetensor.learn import scene as SC, lobe_train as LT, bingham as BG, entity_train as ET, augment as AU
from shapetensor.learn.model import fib_hemisphere, CE_GRID
from odf_render import draw, fit_sh, draw_tensors

out = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 230_000_000; dev = "cuda"
WIN = float(os.environ.get("WINDOW", 3.0)); LEV = float(os.environ.get("LEVEL", 0.6))
gen = LT.difficulty_gen(LEV, dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA))
s, d = SC.draw_scene((64,) * 3, seed=seed, device=dev, strength=0.0, entity_ctx=True, **gen)
_, d1 = SC.draw_scene((64,) * 3, seed=seed, device=dev, strength=1.0, entity_ctx=True, **gen)   # same geometry, full corruption
assert d.get("warp_grid") is None, "pick a seed without a post-render pull warp (scene deformation is fine)"
ctx = d["entity_ctx"]; cs = [int(v) for v in ctx["canvas_shape"]]; org = [int(v) for v in ctx["origin"]]
z0 = 32; grid = fib_hemisphere(CE_GRID, dev); step = 6; gs = np.arange(5, 60, step)
pts = torch.tensor([[z0, y, x] for y in gs for x in gs], dtype=torch.float32, device=dev)
c_disp = TR.ENTITY_DISPERSION


def crop(vol_canvas):                                                                           # canvas -> the output crop (no zoom here)
    return vol_canvas[..., org[0]:org[0] + 64, org[1]:org[1] + 64, org[2]:org[2] + 64]


def odfs_from_tensor(T6w, wv):
    """Per-point ODFs from a crop's w-weighted tensor sums ``T6w (6, Z, Y, X)`` and volume ``wv (Z, Y, X)``: Gaussian
    window, trace-1 tensor, the dispersion floor, moment-matched Bingham, pdf on the grid. NaN where no volume."""
    sm = lambda a: gaussian(a[None], (WIN,) * 3)[0]
    T6 = torch.stack([sm(T6w[k]) for k in range(6)]); W = sm(wv)
    zi, yi, xi = pts.long().unbind(1)
    t = T6[:, zi, yi, xi].T; m = W[zi, yi, xi]
    T = torch.stack([torch.stack([t[:, 0], t[:, 3], t[:, 4]], -1), torch.stack([t[:, 3], t[:, 1], t[:, 5]], -1),
                     torch.stack([t[:, 4], t[:, 5], t[:, 2]], -1)], 1)
    T = T / T.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1e-12)[:, None, None]
    if c_disp is not None:
        I3 = torch.eye(3, device=dev); T = c_disp * T + (1 - c_disp) * (I3 - T) / 2
    R, K = BG.fit(T.double()); p = BG.pdf(grid.view(1, -1, 3), R.float()[:, None], K.float()[:, None])
    return torch.where((m > 0.03)[:, None], p, torch.full_like(p, float("nan"))).cpu().numpy(), m.cpu().numpy(), K.cpu().numpy()


# per-object targets, exactly as training sees them: at each query the entities present (resolved at this window),
# each with its tree node, mass and Bingham ODF
tree = ctx["tree"]; gl = list(ctx["group_layer"]); layers = s.layers
dense_pts = torch.tensor([[z0, y, x] for y in range(1, 64, 2) for x in range(1, 64, 2)], dtype=torch.float32, device=dev)
win = lambda n: torch.full((n,), WIN, device=dev)
tdn = TR.entity_queries(s, d, 64, 0, dev, points=dense_pts, sigma=win(dense_pts.shape[0]))
tgl = TR.entity_queries(s, d, 64, 0, dev, points=pts, sigma=win(pts.shape[0]))


def pop_name(node):
    g_ = int(tree.group[node]) if node < tree.n else -1
    if 0 <= g_ < len(gl) and gl[g_] < len(layers):
        L = layers[gl[g_]]; return f"{getattr(L, 'name', None) or L.role} ({L.kind})"
    return "dense layer"


def node_mass(t, node):
    m = (t["e_mass"].float() * t["e_vis"].float()).clamp_min(0) * (t["e_gnode"] == node)
    return m.sum(1)


nodes = torch.unique(tdn["e_gnode"][tdn["e_gnode"] >= 0]).tolist()
tot = {n: float(node_mass(tdn, n).sum()) for n in nodes}
bgn = set(torch.unique(tdn["e_gnode"][tdn["e_bg"].bool()]).tolist())
picked, seen = [], set()
for n in sorted(tot, key=lambda n: -tot[n]):                                                   # the largest entity of each population
    pn = pop_name(n)
    if n in bgn or pn in seen:
        continue
    picked.append(n); seen.add(pn)
    if len(picked) == 5:
        break

fig = plt.figure(figsize=(26, 12)); gsp = fig.add_gridspec(2, 5)
U_img = (d["image"].float() / 255).cpu().numpy()
for j, n in enumerate(picked):
    mm = node_mass(tdn, n).view(32, 32).cpu().numpy()
    ax = fig.add_subplot(gsp[0, j]); ax.imshow(mm, cmap="magma", vmin=0, vmax=max(float(mm.max()), 1e-6), extent=(0, 64, 64, 0), interpolation="nearest")
    sel = (tgl["e_gnode"] == n) & ((tgl["e_mass"].float() * tgl["e_vis"].float()) > 0.03)
    q_i, e_i = torch.nonzero(sel, as_tuple=True)
    T6 = tgl["e_T6"].float()[q_i, e_i].cpu().numpy()
    if len(q_i):
        draw_tensors(ax, pts.cpu().numpy()[q_i.cpu().numpy()][:, [2, 1]], T6, 0.46 * step, view="z")
        ev = np.linalg.eigvalsh(np.stack([[[t[2], t[5], t[4]], [t[5], t[1], t[3]], [t[4], t[3], t[0]]] for t in T6]))
        ev = ev / ev.sum(1, keepdims=True); lin = float(np.median(ev[:, 2] - ev[:, 1])); iso = float(np.median(3 * ev[:, 0]))
    else:
        lin = iso = float("nan")
    rad = float(tree.radius[n]) if n < tree.n else float("nan")
    lvl = "object" if n < tree.n and int(tree.kind[n]) == 0 else "population-level entity"
    ax.set_title(f"one {lvl}: {pop_name(n)}, radius {rad:.1f} vox\nits mass + its own elliptical ODFs (median line-ness {lin:.2f}, isotropy {iso:.2f})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 64); ax.set_ylim(64, 0)
# all objects and the scene's target
ax = fig.add_subplot(gsp[1, 0]); occ = d["occupancy"].float()[:, z0].cpu().numpy()
cmap = plt.cm.tab10; rgb = np.zeros((64, 64, 3))
for li in range(occ.shape[0]):
    rgb += np.clip(occ[li], 0, 1)[..., None] * np.array(cmap(li % 10)[:3])
ax.imshow(np.clip(rgb, 0, 1)); ax.set_title(f"all populations ({occ.shape[0]} layers: {', '.join((getattr(L, 'name', None) or L.role) for L in layers)})", fontsize=9)
ax.set_xticks([]); ax.set_yticks([])
t = tgl
mm = (t["e_mass"].float() * t["e_vis"].float()).clamp_min(0) * (t["e_node"] >= 0); mo = mm * (~t["e_bg"].bool()).float()
pt = BG.pdf(grid.view(1, -1, 1, 3), t["e_R"].float()[:, None], t["e_kappa"].float()[:, None].clamp(max=ET.DENS_KAPPA_MAX)); pt = (mo[:, None] * pt).sum(-1)
ax = fig.add_subplot(gsp[1, 1]); ax.imshow(U_img[z0], cmap="gray", alpha=0.35)
mt = mo.sum(1).clamp_min(1e-9); q_i, e_i = torch.nonzero(mo > 0.02, as_tuple=True)
if len(q_i):
    draw_tensors(ax, pts.cpu().numpy()[q_i.cpu().numpy()][:, [2, 1]], t["e_T6"].float()[q_i, e_i].cpu().numpy(), 0.46 * step, view="z",
                 scale=torch.sqrt(mo[q_i, e_i] / mt[q_i]).cpu().numpy())
ax.set_title(f"scene target: every object's elliptical ODF, stacked\n(scaled by its share of the mass; window {WIN:g} voxels)", fontsize=9)
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.5, 63.5); ax.set_ylim(63.5, -0.5)
rec = d.get("record", {}); rec1 = d1.get("record", {})
ax = fig.add_subplot(gsp[1, 2]); ax.imshow(U_img[z0], cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
dk = rec.get("deform", rec.get("warp_kinds", ""))
ax.set_title(f"clean render (strength 0), modality {rec.get('modality', '?')}\nscene deformation before rendering: {str(dk)[:60]}", fontsize=9)
x1 = d1["image"].float() / 255
ax = fig.add_subplot(gsp[1, 3]); ax.imshow(x1[z0].cpu().numpy(), cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"same geometry, full corruption (strength 1)\n{', '.join(k for k in rec1 if k not in ('modality', 'window'))[:70]}", fontsize=9)
rng = np.random.default_rng(seed); r_ = {}
try:
    xa = AU.contrast_route(x1, rng, strength=1.0, rec=r_)
except Exception as e:                                                                          # the route's signature differs by version
    xa = x1; r_ = {"route": f"not applied ({type(e).__name__})"}
ax = fig.add_subplot(gsp[1, 4]); ax.imshow(xa[z0].float().cpu().numpy(), cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"training input after a batch augmentation route\n{', '.join(map(str, r_.keys()))[:70]}", fontsize=9)
fig.suptitle(f"One synthetic scene, built object by object (seed {seed}; elliptical ODFs: needle = line, disc = sheet, sphere = isotropic; DEC R = x, G = y, B = z)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=95); print("saved", out)
print("picked:", [(n, pop_name(n), round(tot[n], 1)) for n in picked])
