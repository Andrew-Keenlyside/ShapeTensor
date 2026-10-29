"""Orientation glyphs on held-out Monte Carlo crossings: the image on the plane of both bundles, and over it, at points
every 3 voxels, the strongest one or two directions of the density (lines along the direction projected into the plane,
length by how far the peak stands above the point's median density; a direction leaving the plane draws shorter) --
for the target and for each model given.

Usage: python scripts/learn/crossing_glyphs.py OUT.png CKPT [CKPT ...] [--names a b] [--cases 3]
"""
import argparse, math, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_infer as EI, phantom2 as P2, bingham as BG, inferability as IF
from shapetensor.learn.model import PopulationNet

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("ckpts", nargs="+"); ap.add_argument("--names", nargs="+")
ap.add_argument("--cases", type=int, default=3); ap.add_argument("--seed0", type=int, default=5_000); ap.add_argument("--sigma", type=float, default=2.5)
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
SPH = EI.sphere_dirs(1000, dev); N = 64; C0 = N / 2.0


def crossing(seed):
    """The fit screen's held-out crossings (``crossing_fit_screen.crossing``, same seeds)."""
    g = np.random.default_rng(seed)
    A = g.normal(size=3); A /= np.linalg.norm(A); q = np.cross(A, g.normal(size=3)); q /= np.linalg.norm(q)
    ang = math.radians(g.uniform(45, 90)); B = math.cos(ang) * A + math.sin(ang) * q; nrm = np.cross(A, B); nrm /= np.linalg.norm(nrm)
    mc = lambda ax: {"kind": "fibres", "axis": ax.tolist(), "normal": nrm.tolist(), "half": float(g.uniform(9, 13)), "radius": float(g.uniform(0.7, 1.2)),
                     "fvf": float(g.uniform(0.2, 0.35)), "wander": float(g.uniform(0.2, 0.8)), "conc": 0.94}
    ph = P2.make([mc(A), mc(B)], seed=seed, look={"bg": 0.1, "contrast": [0.8, 0.8], "noise": 0.01}, device=dev)
    return ph, A, B, math.degrees(ang)


def load(path):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"]); return net


def glyphs(f, pts2, e1, e2, rel=1.5, second=0.5):
    """Line segments for the density ``f (Q, S)`` at plane points ``pts2 (Q, 2)``: up to two peaks per point, each kept if it
    stands ``rel`` x above the point's median and (the second) at least ``second`` of the first's height above it."""
    segs, hts = [], []
    for i in range(f.shape[0]):
        fi = f[i]; med = float(fi.median())
        ax, v = IF.peak_axes(fi, SPH, k=2, min_sep=25.0)
        top = None
        for u, h in zip(ax, v.tolist()):
            if h < rel * med:
                continue
            if top is None:
                top = h - med
            elif h - med < second * top:
                continue
            segs.append((pts2[i], np.array([float(u @ e1), float(u @ e2)]))); hts.append(h - med)
    return segs, hts


nets = [load(c) for c in a.ckpts]; names = a.names or [c.rsplit("/", 1)[-1] for c in a.ckpts]
cols = ["image", "target"] + names
fig, axs = plt.subplots(a.cases, len(cols), figsize=(3.6 * len(cols), 3.6 * a.cases + 0.4), squeeze=False, layout="constrained")
for r in range(a.cases):
    ph, A, B, ang = crossing(a.seed0 + r)
    e1 = torch.tensor(A, dtype=torch.float32, device=dev); Bp = B - (B @ A) * A; e2 = torch.tensor(Bp / np.linalg.norm(Bp), dtype=torch.float32, device=dev)
    c = torch.full((3,), C0, device=dev)
    t = torch.arange(-24.0, 24.01, 3.0, device=dev); U, V = torch.meshgrid(t, t, indexing="ij")
    pts = c + U.reshape(-1, 1) * e1 + V.reshape(-1, 1) * e2; pts2 = torch.stack([U.reshape(-1), V.reshape(-1)], 1).cpu().numpy()
    sig = torch.full((pts.shape[0],), a.sigma, device=dev)
    tf = torch.arange(-27.0, 27.01, 0.5, device=dev); Uf, Vf = torch.meshgrid(tf, tf, indexing="ij")
    P = c + Uf[..., None] * e1 + Vf[..., None] * e2
    grid = (2 * P / (N - 1) - 1).flip(-1)[None, None]
    sl = torch.nn.functional.grid_sample(ph["image"][None, None].float(), grid, align_corners=True)[0, 0, 0].cpu().numpy()
    tg = P2.targets(ph, pts, sig)
    dens = {"target": BG.mixture_pdf(SPH, tg["e_R"], tg["e_kappa"], tg["e_mass"], tg["e_node"] >= 0)}
    for nm, net in zip(names, nets):
        with torch.no_grad():
            f = EI.features(net, ph["image"]); dens[nm] = EI.query(net, f, pts, sig, SPH, vol_shape=ph["image"].shape)["f"].float()
    ext = [-27, 27, -27, 27]
    for j, col in enumerate(cols):
        ax_ = axs[r, j]; ax_.imshow(sl.T, cmap="gray", origin="lower", extent=ext, alpha=1.0 if col == "image" else 0.45)
        if col != "image":
            segs, hts = glyphs(dens[col], pts2, e1, e2)
            if segs:
                hm = max(hts); lines = []
                for (p0, d), h in zip(segs, hts):
                    L = 1.4 * (0.35 + 0.65 * h / hm)
                    lines.append([p0 - L * d, p0 + L * d])
                cc = [plt.cm.hsv((math.atan2(d[1], d[0]) % math.pi) / math.pi) for (_, d) in segs]
                ax_.add_collection(LineCollection(lines, colors=cc, linewidths=1.6))
        ax_.set_xlim(-27, 27); ax_.set_ylim(-27, 27); ax_.set_xticks([]); ax_.set_yticks([])
        ax_.set_title(col if r else f"{col}", fontsize=10)
    axs[r, 0].set_ylabel(f"held-out crossing {r + 1}: {ang:.0f} deg\n(bundle A along x)", fontsize=9)
fig.suptitle(f"Held-out Monte Carlo crossings on the plane of both bundles: the one or two strongest density directions every 3 voxels "
             f"(sigma {a.sigma}; colour = in-plane angle; shorter = leaves the plane)", fontsize=10)
plt.savefig(a.out, dpi=80); print("saved", a.out)
