"""Generator phantoms: do the targets follow the objects? (synthetic only)

For simple, clean scenes of one family (no warp, no corruption), an independent ground truth is computed straight from
the construction: at each query point, the Gaussian-weighted (window sigma) second moment of the true fibre tangents
of every curve sample nearby (each sample weighted by its cross-section area and segment length). It is compared with
the targets the trainer uses (the populations' elliptical ODFs), and with the clean image's structure tensor.
Reported: the angle between the true axis and the target's (and the image's) at single-line queries, and a figure:
the image with the true centrelines near the slice, and the target ellipses.

    PYTHONPATH=src python scripts/check_generator_orientation.py OUT_PREFIX
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
from shapetensor.learn.model import st_products, st_matrix, window_mean
from odf_render import draw_tensors

out = sys.argv[1]; dev = os.environ.get("DEV", "cuda" if torch.cuda.is_available() else "cpu"); WIN = 3.0
if dev == "cuda":
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.45)))
CLEAN = dict(canvas=128, crops_per_scene=8, p_centre=1.0, deep_tree=True, n_groups=(1, 1), underlay_p=0.0, cross_p=0.0, p_warp=0.0,
             p_affine=0.0, p_boundary=0.0, p_cutout=0.0, p_sparse=0.0, p_grain=0.0, p_merge=0.0, p_blend=0.0, p_scramble=0.0,
             p_texture=0.0, p_aniso=0.0, fill_quota=SC.FILL_QUOTA)
ang = lambda a, b: torch.rad2deg(torch.acos((a * b).sum(-1).abs().clamp(max=1)))


def latest_scene():
    c = SC._scene_cache(); return list(c.values())[-1][0]


def true_tensors(sc, pts_canvas, sigma):
    """(Q, 3, 3): the window-weighted second moment of every curve's tangents near each point (canvas voxels)."""
    T = torch.zeros(pts_canvas.shape[0], 3, 3, device=dev); W = torch.zeros(pts_canvas.shape[0], device=dev)
    for g in sc.groups:
        for cv in g.curves:
            P = cv.P.float().to(dev)                                                              # (N, S, 3)
            if P.shape[1] < 2:
                continue
            seg = P[:, 1:] - P[:, :-1]; L = seg.norm(dim=-1).clamp_min(1e-6); tg = seg / L[..., None]
            mid = 0.5 * (P[:, 1:] + P[:, :-1])
            r = cv.rad.float().to(dev).view(-1, 1).expand_as(L) if cv.rad.dim() <= 1 else cv.rad.float().to(dev)[..., :L.shape[1]]
            wv = (L * r ** 2).reshape(-1); mid = mid.reshape(-1, 3); tg = tg.reshape(-1, 3)
            for a in range(0, pts_canvas.shape[0], 256):
                d2 = ((pts_canvas[a:a + 256, None, :] - mid[None]) ** 2).sum(-1)
                k = torch.exp(-d2 / (2 * sigma ** 2)) * wv[None]
                T[a:a + 256] += torch.einsum("qn,ni,nj->qij", k, tg, tg); W[a:a + 256] += k.sum(1)
    return T / W.clamp_min(1e-9)[:, None, None], W


FAM = [("bundle", None), ("bundle", 20.0), ("tubules", None), ("tree", None), ("fascicles", 5.0), ("helix", None), ("network", None),
       ("texture", None), ("tissue", 20.0)]                                                         # (bundle at 20 um: unresolved)
DENSE = ("texture", "tissue")                                                                     # strands drawn as fields: no curves to read the
                                                                                                  # truth from -- the targets (verified against
                                                                                                  # the construction) are the reference
if os.environ.get("FAMS"):                                                                          # (env FAMS: a subset, e.g. "tree,network")
    FAM = [f for f in FAM if f[0] in os.environ["FAMS"].split(",")]
rows, report = [], []
NS = int(os.environ.get("NSCENES", 3))                                                              # scenes pooled per family
for r_, (fam, vum) in enumerate(FAM):
    e_ts, e_is, first, used = [], [], None, 0
    for k in range(4 * NS):
        if used >= NS:
            break
        kw = dict(CLEAN); kw["families"] = {**{k_: 0.0 for k_ in SC.Scene.FAMILIES}, fam: 1.0}
        if fam == "tissue":
            kw["families"] = {**{k_: 0.0 for k_ in SC.Scene.FAMILIES}, "tissue": 1.0, "bundle": 1e-6}; kw["n_groups"] = (0, 0)
        if vum:
            kw["voxel_um"] = vum
        try:
            s, d = SC.draw_scene((64,) * 3, seed=310_000_000 + 1000 * r_ + 8 * k, device=dev, strength=0.0, entity_ctx=True, **kw)
        except Exception as e:
            print(fam, "draw failed", type(e).__name__, str(e)[:100]); continue
        fams = [list(f.keys())[0] for f in d["record"].get("families", [])] + (["tissue"] if "tissue" in d["record"] else [])
        if not (fam in fams and (fam != "bundle" or not vum or any(f.get("bundle", {}).get("unresolved") for f in d["record"]["families"]))):
            SC._scene_cache().clear(); continue
        sc = latest_scene(); used += 1
        org = torch.tensor([int(v) for v in d["entity_ctx"]["origin"]], device=dev, dtype=torch.float32)
        step = int(os.environ.get("STEP", 4)); gs = np.arange(4, 64, step); gz = (12, 22, 32, 42, 52)   # five slices (more single-line queries)
        pts = torch.tensor([[z, y, x] for z in gz for y in gs for x in gs], dtype=torch.float32, device=dev); sig = torch.full((pts.shape[0],), WIN, device=dev)
        t = TR.entity_queries(s, d, 64, 0, dev, points=pts, sigma=sig)
        st_, E, valid, other, has = ET.population_targets({k_: (v[None] if torch.is_tensor(v) and k_.startswith("e_") and k_ not in ("e_n", "e_anc") else v) for k_, v in t.items()}, 4)
        E0, s0 = E[0], st_[0]
        Tt, Wt = true_tensors(sc, pts + org, WIN)
        ev, V = torch.linalg.eigh(Tt.double()); lin_true = ((ev[:, 2] - ev[:, 1]) / ev[:, 2].clamp_min(1e-12)).float(); ax_true = V[:, :, 2].float()
        evt, Vt = torch.linalg.eigh(E0.double()); lin_t = ((evt[..., 2] - evt[..., 1]) / evt[..., 2].clamp_min(1e-12)).float()
        cand = (lin_t > 0.6) & valid[0]; sc_ = torch.where(cand, s0, torch.zeros_like(s0)); kbest = sc_.argmax(-1)
        q = torch.arange(pts.shape[0], device=dev)
        ax_t = Vt[q, kbest, :, 2].float()
        if fam in DENSE:                                                                             # the target (= construction) as the reference
            ax_true = ax_t; lin_true = lin_t[q, kbest]; Wt = s0[q, kbest]
            ok = cand.any(-1) & (lin_true > 0.6) & (Wt > 0.05) & (((cand & (s0 > 0.02)).sum(-1)) == 1)
        else:
            ok = cand.any(-1) & (lin_true > 0.6) & (Wt > 1e-3)
        j6 = window_mean(st_products((d["image"].float() / 255)[None, None])[0], pts[None].expand(6, -1, -1).contiguous(), sig[None].expand(6, -1).contiguous()).T
        ax_img = torch.linalg.eigh(st_matrix(j6).double())[1][..., :, 0].float()
        e_ts.append(ang(ax_t, ax_true)[ok].cpu()); e_is.append(ang(ax_img, ax_true)[ok].cpu())
        if first is None:
            sl = pts[:, 0] == 32
            first = (fam, d, sc, org, pts[sl], t, E0[sl], s0[sl], valid[0][sl], step)
        else:
            SC._scene_cache().clear()
        if dev == "cuda":
            torch.cuda.empty_cache()
    if first is None:
        print(fam, ": family not drawn", flush=True); continue
    e_t = torch.cat(e_ts); e_i = torch.cat(e_is)
    line = (f"{fam}{'@' + str(vum) + 'um' if vum else ''}: scenes {used}, n {len(e_t)}, target vs true axis median {float(e_t.median()) if len(e_t) else float('nan'):.1f} deg "
            f"(<10 deg {float((e_t < 10).float().mean()) * 100 if len(e_t) else float('nan'):.0f} %), image structure tensor vs true {float(e_i.median()) if len(e_i) else float('nan'):.1f} deg"
            f" (per scene: {', '.join(f'{float(x.median()):.0f}' if len(x) else '-' for x in e_is)})")
    print(line, flush=True); report.append(line)
    rows.append(first)
    SC._scene_cache().clear()

fig, axs = plt.subplots(len(rows), 3, figsize=(15, 5 * len(rows))); axs = np.atleast_2d(axs)
for r_, (fam, d, sc, org, pts, t, E0, s0, val, step) in enumerate(rows):
    x = (d["image"].float() / 255)[32].cpu().numpy(); lo, hi = np.percentile(x, [1, 99]); im = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    for c in range(3):
        axs[r_, c].imshow(im, cmap="gray", alpha=1.0 if c == 0 else 0.5); axs[r_, c].set_xticks([]); axs[r_, c].set_yticks([])
        axs[r_, c].set_xlim(-0.5, 63.5); axs[r_, c].set_ylim(63.5, -0.5)
    axs[r_, 0].set_title(f"{fam}: clean image (middle slice)", fontsize=9)
    for g in sc.groups:                                                                             # true centrelines within 3 voxels of the slice
        for cv in g.curves:
            P = (cv.P.float().to(dev) - org).cpu().numpy()
            for p in P:
                near = np.abs(p[:, 0] - 32) < 3
                seg = np.split(np.arange(len(p)), np.where(np.diff(near.astype(int)) != 0)[0] + 1)
                for idx in seg:
                    if len(idx) > 1 and near[idx[0]]:
                        axs[r_, 1].plot(p[idx, 2], p[idx, 1], color="yellow", lw=0.6, alpha=0.8)
    axs[r_, 1].set_title("true fibre centrelines near the slice (construction)", fontsize=9)
    q, k = torch.nonzero(val & (s0 > 0.02), as_tuple=True)
    m6 = lambda M: torch.stack([M[..., 0, 0], M[..., 1, 1], M[..., 2, 2], M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]], -1)
    if len(q):
        draw_tensors(axs[r_, 2], pts[q].cpu().numpy()[:, [2, 1]], m6(E0[q, k]).cpu().numpy(), 0.46 * step, view="z", scale=np.sqrt(s0[q, k].cpu().numpy()))
    axs[r_, 2].set_title("targets: populations' elliptical ODFs (window 3)", fontsize=9)
fig.suptitle("Generator phantoms: image, true centrelines, targets", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.985)); fig.savefig(out + ".png", dpi=80); print("saved", out + ".png")
open(out + ".txt", "w").write("\n".join(report) + "\n")
