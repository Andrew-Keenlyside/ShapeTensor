"""Populations, truth against model: at a grid of points on one slice, every population as its elliptical ODF (needle =
line, disc = sheet, sphere = isotropic; DEC), stacked, each scaled by the square root of its share. Truth: the
construction's populations (synthetic) or the phantom's axes (two needles at a crossing, a sphere in fibre-free
texture). Model: the slots with existence > 0.5.

    PYTHONPATH=src python scripts/figure_populations.py CKPT OUT.png   (env WINDOW, STRENGTH)
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
from shapetensor import phantoms as PH
from shapetensor.learn import scene as SC, entity_infer as EI, lobe_train as LT, entity_train as ET
from shapetensor.learn.model import PopulationNet
from odf_render import draw_tensors

ckp, out = sys.argv[1], sys.argv[2]; dev = "cuda"; WIN = float(os.environ.get("WINDOW", 3.0)); STR = float(os.environ.get("STRENGTH", 0.3))
ck = torch.load(ckp, map_location=dev, weights_only=False); net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"])
ET.configure_from_args(ck["args"])                                                                 # (its targets: background apart, matching)
SC.MATRIX_P = float(os.environ.get("MATRIX", 0))                                                   # 1: scenes built of objects
dirs8 = EI.sphere_dirs(8, dev)


def m6(M):                                                                                          # (..., 3, 3) -> (..., 6) zz yy xx zy zx yx
    return torch.stack([M[..., 0, 0], M[..., 1, 1], M[..., 2, 2], M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]], -1)


def stack(ax, img, cen, T6, share, step, title, view):
    ax.imshow(img, cmap="gray", alpha=0.5, interpolation="nearest")
    if len(cen):
        draw_tensors(ax, cen, T6, 0.46 * step, view=view, scale=np.sqrt(np.clip(share, 0, 1)))
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, img.shape[1] - 0.5); ax.set_ylim(img.shape[0] - 0.5, -0.5)


def model_pops(x, pts, sig):
    with torch.no_grad():
        f = EI.features(net, x)
        r = EI.query(net, f, pts, sig, dirs8, vol_shape=x.shape,
                     scale=torch.full_like(sig, float(os.environ.get("EVAL_SCALE", 4))) if ck["args"].get("scales") else None)
    on = r["pop_exist"] > 0.5; q, k = torch.nonzero(on, as_tuple=True)
    return q, m6(r["pop_T"][q, k]), r["pop_share"][q, k]


rows = []
base = dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA)
for seed in (290_000_000, 290_000_016):
    s, d = SC.draw_scene((64,) * 3, seed=seed, device=dev, strength=STR, entity_ctx=True, **LT.difficulty_gen(0.4 + 0.6 * STR, base))
    step = 8; gs = np.arange(4, 64, step); pts = torch.tensor([[32, y, x] for y in gs for x in gs], dtype=torch.float32, device=dev)
    sig = torch.full((pts.shape[0],), WIN, device=dev)
    t = TR.entity_queries(s, d, 64, 0, dev, points=pts, sigma=sig)
    s_, E, valid, other, has = ET.population_targets({k: (v[None] if torch.is_tensor(v) and k.startswith("e_") and k not in ("e_n", "e_anc") else v) for k, v in t.items()}, 4)
    qt, kt = torch.nonzero(valid[0], as_tuple=True)
    x = d["image"].float() / 255; qm, Tm, sm = model_pops(x, pts, sig)
    P = pts.cpu().numpy()[:, [2, 1]]
    rows.append((f"synthetic, strength {STR}, {d['record'].get('modality')}", x[32].cpu().numpy(),
                 (P[qt.cpu().numpy()], m6(E[0][qt, kt]).cpu().numpy(), s_[0][qt, kt].cpu().numpy()),
                 (P[qm.cpu().numpy()], Tm.cpu().numpy(), sm.cpu().numpy()), step, "z"))
    SC._scene_cache().clear()
for name, kw in (("crossing", {"cross_deg": 90.0}), ("crossing", {"cross_deg": 45.0}), ("bundle", {"angle": 60.0}), ("texture", {})):
    img, tr, meta = PH.build(name, seed=0, texture=0.5, device=dev, **kw)
    img = torch.as_tensor(img, device=dev).float() / 255; Z, Y, X = img.shape; c0 = [(n - 96) // 2 for n in (Z, Y, X)]
    x = img[c0[0]:c0[0] + 96, c0[1]:c0[1] + 96, c0[2]:c0[2] + 96].contiguous()
    step = 10; gs = np.arange(8, 92, step); pts = torch.tensor([[z, 48, x_] for z in gs for x_ in gs], dtype=torch.float32, device=dev)
    sig = torch.full((pts.shape[0],), WIN + 1.0, device=dev)
    zi = (pts[:, 0] + c0[0]).long().cpu().numpy(); xi = (pts[:, 2] + c0[2]).long().cpu().numpy()
    d1, d2, npop = np.asarray(tr["dir1"])[zi, xi], np.asarray(tr["dir2"])[zi, xi], np.asarray(tr["npop"])[zi, xi]
    cen, T6, sh = [], [], []
    P = pts.cpu().numpy()[:, [2, 0]]
    for i, n in enumerate(npop):
        dl = [d1[i]] + ([d2[i]] if n == 2 else []) if n > 0 else []
        if name == "texture":
            cen.append(P[i]); T6.append([1 / 3, 1 / 3, 1 / 3, 0, 0, 0]); sh.append(1.0); continue
        for dv in dl:
            a = np.nan_to_num(np.asarray(dv, dtype=np.float64)); M = 0.94 * np.outer(a, a) + 0.02 * np.eye(3)
            cen.append(P[i]); T6.append([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]]); sh.append(1.0 / len(dl))
    qm, Tm, sm = model_pops(x, pts, sig)
    label = f"phantom: {name} {' '.join(f'{k} {v:g}' for k, v in kw.items())} (texture 0.5)"
    rows.append((label, x[:, 48, :].cpu().numpy(), (np.array(cen), np.array(T6), np.array(sh)), (P[qm.cpu().numpy()], Tm.cpu().numpy(), sm.cpu().numpy()), step, "y"))

fig, axs = plt.subplots(len(rows), 3, figsize=(15, 5 * len(rows))); axs = np.atleast_2d(axs)
for r, (label, img, tru, mod, step, view) in enumerate(rows):
    lo, hi = np.percentile(img, [1, 99]); im = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    axs[r, 0].imshow(im, cmap="gray"); axs[r, 0].set_title(label, fontsize=9); axs[r, 0].set_xticks([]); axs[r, 0].set_yticks([])
    stack(axs[r, 1], im, *tru, step, "truth: populations (elliptical ODFs, scaled by share)", view)
    stack(axs[r, 2], im, *mod, step, f"model (step {ck.get('step')}): slots with existence > 0.5", view)
fig.suptitle(f"Populations, truth against model (window {WIN:g} voxels; needle = line, disc = sheet, sphere = isotropic; DEC)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.985)); fig.savefig(out, dpi=80); print("saved", out)
