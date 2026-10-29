"""Generator check, in-plane: for clean scenes of one family each (strength 0, no warp), the slice in which the scene's
dominant fibres lie in the plane (the axis least along them, at the slice holding most line material), shown three
ways: the image, the true fibre centrelines within 3 voxels of it (construction), and the targets (the populations'
elliptical ODFs, window 3, scaled by the square root of their share). Synthetic only.

    PYTHONPATH=src python scripts/figure_generator_inplane.py OUT.png   (env FAMS, DEV)
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
from shapetensor.learn import scene as SC, entity_train as ET
from odf_render import draw_tensors

out = sys.argv[1]; dev = os.environ.get("DEV", "cuda" if torch.cuda.is_available() else "cpu")
if dev == "cuda":
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.2)))
CLEAN = dict(canvas=128, crops_per_scene=8, p_centre=1.0, deep_tree=True, n_groups=(1, 1), underlay_p=0.0, cross_p=0.0, p_warp=0.0,
             p_affine=0.0, p_boundary=0.0, p_cutout=0.0, p_sparse=0.0, p_grain=0.0, p_merge=0.0, p_blend=0.0, p_scramble=0.0,
             p_texture=0.0, p_aniso=0.0, fill_quota=SC.FILL_QUOTA)
FAM = [("bundle", None), ("bundle", 20.0), ("fascicles", 5.0), ("network", None), ("texture", None), ("tissue", 20.0)]
if os.environ.get("FAMS"):
    FAM = [f for f in FAM if f[0] in os.environ["FAMS"].split(",")]
m6 = lambda M: torch.stack([M[..., 0, 0], M[..., 1, 1], M[..., 2, 2], M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]], -1)
VIEW = {0: "z", 1: "y", 2: "x"}; PLANE = {0: (2, 1), 1: (2, 0), 2: (1, 0)}                          # slice axis -> (column, row) axes
rows = []
for r_, (fam, vum) in enumerate(FAM):
    for k in range(12):
        kw = dict(CLEAN); kw["families"] = {**{k_: 0.0 for k_ in SC.Scene.FAMILIES}, fam: 1.0}
        if fam == "tissue":
            kw["families"] = {**{k_: 0.0 for k_ in SC.Scene.FAMILIES}, "tissue": 1.0, "bundle": 1e-6}; kw["n_groups"] = (0, 0)
        if vum:
            kw["voxel_um"] = vum
        try:
            s, d = SC.draw_scene((64,) * 3, seed=320_000_000 + 1000 * r_ + 8 * k, device=dev, strength=0.0, entity_ctx=True, **kw)
        except Exception as e:
            print(fam, "draw failed", type(e).__name__); SC._scene_cache().clear(); continue
        fams = [list(f.keys())[0] for f in d["record"].get("families", [])] + (["tissue"] if "tissue" in d["record"] else [])
        if fam not in fams:
            SC._scene_cache().clear(); continue
        g3 = np.arange(4, 64, 8)                                                                    # a coarse 3D grid: where the lines are
        pts = torch.tensor([[z, y, x] for z in g3 for y in g3 for x in g3], dtype=torch.float32, device=dev)
        t = TR.entity_queries(s, d, 64, 0, dev, points=pts, sigma=torch.full((pts.shape[0],), 3.0, device=dev))
        tt = {k_: (v[None] if torch.is_tensor(v) and k_.startswith("e_") and k_ not in ("e_n", "e_anc") else v) for k_, v in t.items()}
        sh, E, valid, other, has = ET.population_targets(tt, 4)
        ev, V = torch.linalg.eigh(E[0].double()); lin = (ev[..., 2] - ev[..., 1]).float()
        wl = (sh[0] * valid[0] * (lin > 0.6)).float()
        if float(wl.sum()) < 1.0:
            SC._scene_cache().clear(); continue
        ax = V[..., 2].float(); M = torch.einsum("qk,qki,qkj->ij", wl, ax, ax)
        dom = torch.linalg.eigh(M)[1][:, 2]; a = int(dom.abs().argmin())                              # slice across the axis least along them
        prof = torch.zeros(64, device=dev); prof.index_add_(0, pts[:, a].long(), wl.sum(-1))
        c = int(prof.argmax())
        cA, rA = PLANE[a]; gs = np.arange(2, 64, 4)
        P = np.zeros((len(gs) ** 2, 3), dtype=np.float32); P[:, a] = c
        P[:, cA] = np.repeat(gs, len(gs)); P[:, rA] = np.tile(gs, len(gs))
        pts2 = torch.tensor(P, device=dev)
        t2 = TR.entity_queries(s, d, 64, 0, dev, points=pts2, sigma=torch.full((pts2.shape[0],), 3.0, device=dev))
        tt2 = {k_: (v[None] if torch.is_tensor(v) and k_.startswith("e_") and k_ not in ("e_n", "e_anc") else v) for k_, v in t2.items()}
        sh2, E2, valid2, _, _ = ET.population_targets(tt2, 4)
        img = (d["image"].float() / 255).cpu().numpy(); sl = np.take(img, c, axis=a)
        if a in (1, 2):
            pass                                                                                    # take keeps (z, x) / (z, y): rows z
        sc = list(SC._scene_cache().values())[-1][0]; org = np.array([int(v) for v in d["entity_ctx"]["origin"]], dtype=np.float32)
        lines = []
        for g in sc.groups:
            for cv in g.curves:
                for p_ in (cv.P.float().cpu().numpy() - org):
                    near = np.abs(p_[:, a] - c) < 3
                    seg = np.split(np.arange(len(p_)), np.where(np.diff(near.astype(int)) != 0)[0] + 1)
                    lines += [p_[idx][:, [cA, rA]] for idx in seg if len(idx) > 1 and near[idx[0]]]
        q, kk = torch.nonzero(valid2[0] & (sh2[0] > 0.02), as_tuple=True)
        rows.append((f"{fam}{' @ ' + str(int(vum)) + ' um' if vum else ''}: slice {VIEW[a]} = {c}", sl, lines,
                     P[q.cpu().numpy()][:, [cA, rA]], m6(E2[0][q, kk]).cpu().numpy(), sh2[0][q, kk].cpu().numpy(), VIEW[a]))
        SC._scene_cache().clear()
        print(fam, "slice", VIEW[a], c, flush=True)
        break
    if dev == "cuda":
        torch.cuda.empty_cache()
fig, axs = plt.subplots(len(rows), 3, figsize=(15, 5 * len(rows))); axs = np.atleast_2d(axs)
for r_, (title, sl, lines, cen, T6, shr, view) in enumerate(rows):
    lo, hi = np.percentile(sl, [1, 99]); im = np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1)
    for j in range(3):
        axs[r_, j].imshow(im, cmap="gray", alpha=1.0 if j == 0 else 0.5); axs[r_, j].set_xticks([]); axs[r_, j].set_yticks([])
        axs[r_, j].set_xlim(-0.5, 63.5); axs[r_, j].set_ylim(63.5, -0.5)
    for L in lines:
        axs[r_, 1].plot(L[:, 0], L[:, 1], color="yellow", lw=0.6, alpha=0.8)
    if len(cen):
        draw_tensors(axs[r_, 2], cen, T6, 1.9, view=view, scale=np.sqrt(np.clip(shr, 0, 1)))
    axs[r_, 0].set_title(title + " (clean image)", fontsize=9); axs[r_, 1].set_title("true fibre centrelines within 3 voxels", fontsize=9)
    axs[r_, 2].set_title("targets: elliptical ODFs (window 3), scaled by sqrt share", fontsize=9)
fig.suptitle("Generator: image, construction, targets in the plane of the fibres (strength 0)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.985)); fig.savefig(out, dpi=80); print("saved", out)
