"""Does the generator give every object an elliptical ODF, and do artefacts leave the ground truth alone?

Part A (one family per row, clean images): the middle slice and, at a fine and a coarse window, every object's
elliptical ODF at a grid of points: the trace-1 second moment of its material directions within the window, drawn as an
ellipsoid (a line is a needle, a sheet a disc, isotropic a sphere; DEC colour). Objects that share a point are drawn
stacked, each scaled by the square root of its share of the mass.
Part B (one scene): the same geometry rendered clean and with displacement artefacts, CT streaks and oriented speckle,
each overlaid with the same targets, and the largest target difference against the clean render.

    PYTHONPATH=src python scripts/figure_generator_check.py OUT_PREFIX
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

out = sys.argv[1]; dev = "cuda"; z0 = 32; step = 8
gs = np.arange(4, 64, step); pts = torch.tensor([[z0, y, x] for y in gs for x in gs], dtype=torch.float32, device=dev)
CLEAN = dict(canvas=128, crops_per_scene=8, p_centre=1.0, deep_tree=True, n_groups=(1, 1), underlay_p=0.0, cross_p=0.0,
             p_warp=0.0, p_affine=0.0, p_boundary=0.0, p_grain=0.0, p_merge=0.0, p_blend=0.0, p_scramble=0.0, fill_quota=SC.FILL_QUOTA)


def targets(s, d, sigma):
    t = TR.entity_queries(s, d, 64, 0, dev, points=pts, sigma=torch.full((pts.shape[0],), sigma, device=dev))
    m = (t["e_mass"].float() * t["e_vis"].float()).clamp_min(0) * (t["e_node"] >= 0) * (~t["e_bg"].bool()).float()
    return t, m


def overlay(ax, t, m, title):
    mt = m.sum(1).clamp_min(1e-9)
    q_i, e_i = torch.nonzero(m > 0.02, as_tuple=True)
    if len(q_i):
        sc = torch.sqrt(m[q_i, e_i] / mt[q_i]).cpu().numpy()
        cen = pts[q_i].cpu().numpy()[:, [2, 1]]
        draw_tensors(ax, cen, t["e_T6"].float()[q_i, e_i].cpu().numpy(), 0.46 * step, view="z", scale=sc)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.5, 63.5); ax.set_ylim(63.5, -0.5)


def img(ax, d, title=""):
    x = (d["image"].float() / 255)[z0].cpu().numpy(); lo, hi = np.percentile(x, [1, 99])
    ax.imshow(np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray", alpha=1.0 if title else 0.45, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])


# ---- Part A: one family at a time
FAM = [("bundle", "bundle of fibres"), ("fascicles", "fascicles"), ("tubules", "hollow tubules / vessels"), ("sheet", "sheets"),
       ("cells", "cells (solids)"), ("texture", "oriented texture field")]
fig, axs = plt.subplots(len(FAM), 3, figsize=(15, 5 * len(FAM)))
for r, (fam, label) in enumerate(FAM):
    d = None
    for k in range(6):                                                                         # a crop that contains the family
        try:
            s, d = SC.draw_scene((64,) * 3, seed=240_000_000 + 1000 * r + 8 * k, device=dev, strength=0.0, entity_ctx=True,
                                 families={fam: 1.0}, **CLEAN)
        except Exception as e:
            print(fam, "draw failed:", type(e).__name__, str(e)[:120]); d = None; continue
        t, m = targets(s, d, 1.5)
        if float((m.sum(1) > 0.05).float().mean()) > 0.2:
            break
    if d is None:
        continue
    img(axs[r, 0], d, f"{label}: clean image (middle slice)")
    for c, sg in ((1, 1.5), (2, 6.0)):
        t, m = targets(s, d, sg); img(axs[r, c], d)
        n_obj = int(torch.unique(t["e_gnode"][(m > 0.02)]).numel())
        overlay(axs[r, c], t, m, f"per-object elliptical ODFs, window {sg:g} voxels ({n_obj} entities)")
    SC._scene_cache().clear(); torch.cuda.empty_cache()
fig.suptitle("Every object carries an elliptical ODF (needle = line, disc = sheet, sphere = isotropic; DEC R = x, G = y, B = z)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.985)); fig.savefig(out + "_families.png", dpi=80); print("saved", out + "_families.png")

# ---- Part B: artefacts change the image, never the targets
SEED = 250_000_000
base = LT.difficulty_gen(0.7, dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA))
s0, d0 = SC.draw_scene((64,) * 3, seed=SEED, device=dev, strength=0.0, entity_ctx=True, **base)
t0, m0 = targets(s0, d0, 3.0)
WANT = [("displacement (drift / misalignment / slice motion / ghost / fold)", ("drift", "misalign", "slice", "ghost", "fold", "held", "section"), None),
        ("CT streaks / beam hardening / metal", ("bh_streak", "metal", "streak", "scatter"), "ct"),
        ("oriented speckle / oriented noise", ("speckle", "oriented_noise"), None)]
found = [("clean render", d0, [])]
for label, keys, mod in WANT:
    for a in range(250):                                                                       # (the acquisition seed also steers
        kw = dict(base); kw.update({"modality": mod} if mod else {})                           # some scene-composition draws: keep
        s1, d1 = SC.draw_scene((64,) * 3, seed=SEED, acq_seed=7000 + 131 * a, device=dev, strength=1.0, entity_ctx=True, **kw)   # only
        hits = [k for k in d1.get("record", {}) if any(w in k for w in keys)]                  # draws with this exact geometry)
        if hits:
            t1, m1 = targets(s1, d1, 3.0)
            if float((t1["e_T6"].float() - t0["e_T6"].float()).abs().max() + (m1 - m0).abs().max()) < 1e-3:
                found.append((label, d1, hits)); break
    else:
        print("no draw with", label)
fig, axs = plt.subplots(1, len(found), figsize=(5.2 * len(found), 5.6)); axs = np.atleast_1d(axs)
for ax, (label, d1, hits) in zip(axs, found):
    t1, m1 = targets(s0, d1, 3.0)
    diff = float((t1["e_T6"].float() - t0["e_T6"].float()).abs().max() + (m1 - m0).abs().max())
    img(ax, d1); overlay(ax, t1, m1, f"{label}\n{', '.join(hits)[:60]}\nmax target change vs clean: {diff:.1e}")
fig.suptitle("One scene: the artefacts change the image, the ground truth (the same targets, overlaid) does not", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out + "_artefacts.png", dpi=85); print("saved", out + "_artefacts.png")
