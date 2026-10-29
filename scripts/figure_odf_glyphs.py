"""ODF glyphs, side by side: the image, the truth, the model's density, and the structure tensor's axis, at the same
query points (window sigma) on one slice. Glyphs are DEC-coloured (R = x, G = y, B = z; each surface point coloured by
its own direction), radius = q / max q, both hemispheres; grey = near flat (peak/mean < 1.3).

  synthetic crops (truth: the non-background material's ODF from construction; middle z slice):
    PYTHONPATH=src python scripts/figure_odf_glyphs.py synth OUT.png ckpt [N] [profile]   (env STRENGTH, LEVEL, SEED0)
  phantoms (truth: Watson lobes on the phantom's true axes; a y slice, the fibres' x-z plane):
    PYTHONPATH=src python scripts/figure_odf_glyphs.py phantom OUT.png ckpt                 (env TEXTURE)
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import shapetensor.learn.train as TR
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
TR.ENTITY["p_cross"] = 0.2
import shapetensor.learn.generator as G
G.ORIENT_FROM_TEXTURE = True; G.ORIENT_VIS = False
from shapetensor.learn import scene as SC, entity_infer as EI, lobe_train as LT, bingham as BG, entity_train as ET
from shapetensor.learn.model import PopulationNet, fib_hemisphere, st_products, st_matrix, window_mean, CE_GRID

mode, out, ckp = sys.argv[1], sys.argv[2], sys.argv[3]; dev = "cuda"
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.5)))
ck = torch.load(ckp, map_location=dev, weights_only=False); net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"])
mname = f"{os.path.basename(os.path.dirname(ckp))} @ {ck.get('step')}"
grid = fib_hemisphere(CE_GRID, dev); U = torch.cat([grid, -grid]).cpu().numpy(); DEC = np.clip(np.abs(U[:, ::-1]), 0, 1)   # (x, y, z) -> RGB
SIG = 3.0


def glyphs(ax, img, P, Q, rax, cax, step, title):
    """``P (n, 3)`` centres (z, y, x), ``Q (n, G)`` densities on the grid; rows along axis ``rax``, columns ``cax``."""
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, alpha=0.55, interpolation="nearest")
    for p, q in zip(P, Q):
        if q is None or not np.isfinite(q).all():
            continue
        q2 = np.concatenate([q, q]); r = 0.48 * step * q2 / max(q2.max(), 1e-12)
        flat = q.max() / max(q.mean(), 1e-12) < 1.3
        o = np.argsort(U[:, 3 - rax - cax])                                                       # back to front along the viewing axis
        ax.scatter(p[cax] + r[o] * U[o, cax], p[rax] + r[o] * U[o, rax], s=0.6, c=("0.6" if flat else DEC[o]), linewidths=0)
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])


def sticks(ax, img, P, D, L, rax, cax, step, title):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, alpha=0.55, interpolation="nearest")
    for p, d, l in zip(P, D, L):
        if not np.isfinite(d).all() or l <= 0.02:
            continue
        h = 0.48 * step * l
        ax.plot([p[cax] - h * d[cax], p[cax] + h * d[cax]], [p[rax] - h * d[rax], p[rax] + h * d[rax]], color=np.clip(np.abs(d[::-1]), 0, 1), lw=1.6)
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])


def model_q(x, pts, sig):
    with torch.no_grad():
        f = EI.features(net, x); r = EI.query(net, f, pts, sig, grid, vol_shape=x.shape)
    return torch.exp(r["logq"].float())


def st_axis(x, pts, sig):
    j6 = window_mean(st_products(x[None, None])[0], pts[None].expand(6, -1, -1).contiguous(), sig[None].expand(6, -1).contiguous()).T
    ev, V = torch.linalg.eigh(st_matrix(j6).double()); ev = ev.float()
    return V[..., :, 0].float(), ((ev[:, 1] - ev[:, 0]) / ev[:, 2].clamp_min(1e-12)).clamp(0, 1)


rows = []
if mode == "synth":
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 4; prof = sys.argv[5] if len(sys.argv) > 5 else None
    STR = float(os.environ.get("STRENGTH", 0.5)); LEV = float(os.environ.get("LEVEL", 0.7)); seed0 = int(os.environ.get("SEED0", 190_000_000))
    base = dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA)
    if prof:
        base["profile"] = prof
    gen = LT.difficulty_gen(LEV, base); step = 6; gs = np.arange(5, 60, step)
    pts = torch.tensor([[32, y, x] for y in gs for x in gs], dtype=torch.float32, device=dev); sig = torch.full((pts.shape[0],), SIG, device=dev)
    cand = []
    for i in range(3 * N):                                                                        # the crops with the most line material
        try:
            s, d = SC.draw_scene((64,) * 3, seed=seed0 + 8 * i, device=dev, strength=STR, entity_ctx=True, **gen)
        except torch.OutOfMemoryError:
            SC._scene_cache().clear(); torch.cuda.empty_cache(); continue
        t = TR.entity_queries(s, d, 64, 0, dev, points=pts, sigma=sig)
        m = (t["e_mass"].float() * t["e_vis"].float()).clamp_min(0) * (t["e_node"] >= 0); mo = m * (~t["e_bg"].bool()).float()
        line = (t["e_kappa"][..., 1] >= 2) & (mo > 0); nl = int(((mo * line).sum(1) > 0.05).sum())
        rho = BG.pdf(grid.view(1, -1, 1, 3), t["e_R"].float()[:, None], t["e_kappa"].float()[:, None].clamp(max=ET.DENS_KAPPA_MAX)); rho = (mo[:, None] * rho).sum(-1)
        rho = torch.where((mo.sum(1) > ET.MASS_EPS)[:, None], rho, torch.full_like(rho, float("nan")))
        x = d["image"].float() / 255
        cand.append((nl, d.get("record", {}).get("modality", "?"), x, rho.cpu().numpy()))
        SC._scene_cache().clear()
    cand.sort(key=lambda c: -c[0])
    for nl, mod, x, rho in cand[:N]:
        qm = model_q(x, pts, sig).cpu().numpy(); ax_, lin = st_axis(x, pts, sig)
        rows.append((f"{mod}, strength {STR}", x[32].cpu().numpy(), pts.cpu().numpy(), rho, qm, ax_.cpu().numpy(), lin.cpu().numpy(), 1, 2, step))
else:
    from shapetensor import phantoms as PH
    tx = float(os.environ.get("TEXTURE", 0.5)); step = 8
    for name, kw in [("bundle", {"angle": 60.0}), ("crossing", {"cross_deg": 90.0}), ("crossing", {"cross_deg": 45.0}), ("texture", {})]:
        img, tr, meta = PH.build(name, seed=0, texture=tx, device=dev, **kw)
        img = torch.as_tensor(img, device=dev).float() / 255; Z, Y, X = img.shape; c0 = [(n - 96) // 2 for n in (Z, Y, X)]
        x = img[c0[0]:c0[0] + 96, c0[1]:c0[1] + 96, c0[2]:c0[2] + 96].contiguous(); y0 = 48
        gs = np.arange(6, 92, step); pts = torch.tensor([[z, y0, x_] for z in gs for x_ in gs], dtype=torch.float32, device=dev)
        sig = torch.full((pts.shape[0],), SIG, device=dev)
        zi = (pts[:, 0] + c0[0]).long().cpu().numpy(); xi = (pts[:, 2] + c0[2]).long().cpu().numpy()
        d1 = np.asarray(tr["dir1"])[zi, xi]; d2 = np.asarray(tr["dir2"])[zi, xi]; npop = np.asarray(tr["npop"])[zi, xi]
        g_ = grid.cpu().numpy(); w = lambda d: np.exp(20 * ((g_ @ np.nan_to_num(d)) ** 2 - 1))
        truth = [None if n == 0 else (w(a) + (w(b) if n == 2 else 0)) for a, b, n in zip(d1, d2, npop)]
        if name == "texture":
            truth = [np.ones(g_.shape[0]) for _ in npop]                                         # nothing to orient: flat
        qm = model_q(x, pts, sig).cpu().numpy(); ax_, lin = st_axis(x, pts, sig)
        label = f"{name} {', '.join(f'{k} {v:g}' for k, v in kw.items())} (texture {tx})"
        rows.append((label, x[:, y0, :].cpu().numpy(), pts.cpu().numpy(), truth, qm, ax_.cpu().numpy(), lin.cpu().numpy(), 0, 2, step))

fig, axs = plt.subplots(len(rows), 4, figsize=(17, 4.3 * len(rows))); axs = np.atleast_2d(axs)
for k, (label, img, P, truth, qm, sta, lin, rax, cax, step) in enumerate(rows):
    lo, hi = np.percentile(img, [1, 99]); im = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    axs[k, 0].imshow(im, cmap="gray", interpolation="nearest"); axs[k, 0].set_title(label, fontsize=8); axs[k, 0].set_xticks([]); axs[k, 0].set_yticks([])
    glyphs(axs[k, 1], im, P, [None if (t is None or not np.isfinite(np.asarray(t)).all()) else np.asarray(t) for t in truth], rax, cax, step, "truth (from construction)")
    glyphs(axs[k, 2], im, P, list(qm), rax, cax, step, f"model {mname}")
    sticks(axs[k, 3], im, P, sta, lin, rax, cax, step, "structure tensor axis (length = line-ness)")
fig.suptitle(f"ODFs at window sigma {SIG}: DEC glyphs (R = x, G = y, B = z), grey = near flat", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(out, dpi=100); print("saved", out)
