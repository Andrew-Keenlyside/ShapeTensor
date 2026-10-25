"""The crossing figure (one fixed layout, for every comparison): generator crossing scenes, a slice through the crossing
core; per row: the image; the true populations (occupancy, coloured); the target density's peaks under each visibility
setting; each checkpoint's density peaks. Peaks at a grid of points (sigma 2.5): the strongest red, a second cyan where
it reaches 0.3 of the first and lies > 20 deg from it; glyph length by the peak's height (the point's max = full), the
in-plane part of the direction (faint where the direction leaves the slice).

Model glyphs only where the model's own mass output reaches ``--mass-min`` (never masked by the truth).

Usage: python scripts/learn/crossing_figure.py OUT.png [CKPT ...] [--rows 6] [--seed0 1700000] [--bundles]
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
import shapetensor.learn.generator as GN
import shapetensor.learn.scene as SS
from shapetensor.learn import entity_train as ET, lobe_train as LT, losses as L, bingham as BG, entity_infer as EI
from shapetensor.learn.model import PopulationNet

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("ckpts", nargs="*"); ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--seed0", type=int, default=1_700_000); ap.add_argument("--strength", type=float, default=0.4); ap.add_argument("--level", type=float, default=0.5)
ap.add_argument("--bundles", action="store_true", help="crossing bundles only (no other families)")
ap.add_argument("--mass-min", type=float, default=0.05, help="model glyphs where the model's own mass output reaches this")
a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None
L.DESC_K = 5; L.DESC_NC = 0; TR.set_background_entity(False)                                  # the oriented populations' targets
gen = {**LT.difficulty_gen(a.level, {"canvas": 128, "crops_per_scene": 1, "cross_p": 1.0, "p_centre": 1.0, "deep_tree": False}), "crops_per_scene": 1}
if a.bundles:
    gen["families"] = {k: (1.0 if k == "bundle" else 0.0) for k in SS.Scene.FAMILIES}; gen["underlay_p"] = 0.0
dirs = EI.sphere_dirs(200, dev); nb = EI.dir_neighbours(dirs, 8)
SETTINGS = [("target: old visibility (window 5)", (1,), lambda fw: 1.0),
            ("target: windows 5-10-20", (1, 2, 4), lambda fw: 1.0),
            ("target: windows 5-10-20, penalised", (1, 2, 4), lambda fw: math.sqrt(fw))]
models = []
for c in a.ckpts:
    ck = torch.load(c, map_location=dev, weights_only=False); m = PopulationNet.from_args(ck["args"]).to(dev).eval(); m.load_state_dict(ck["model"])
    models.append((c.split("/")[-1].replace(".pt", ""), m))

def peaks(f):
    """Up to two peaks per point: (index, height) pairs, the second >= 0.3 of the first and > 20 deg away."""
    ismax = (f[:, :, None] >= f[:, nb]).all(-1); val = torch.where(ismax, f, torch.full_like(f, -1e9)); top = val.topk(3, dim=1)
    out = []
    for q in range(f.shape[0]):
        i0, v0 = int(top.indices[q, 0]), float(top.values[q, 0]); ks = [(i0, v0)]
        for j in (1, 2):
            ij, vj = int(top.indices[q, j]), float(top.values[q, j])
            if vj >= 0.3 * v0 and float(torch.rad2deg(torch.acos((dirs[ij] @ dirs[i0]).abs().clamp(max=1)))) > 20:
                ks.append((ij, vj)); break
        out.append(ks)
    return out

def glyphs(ax, img, pts, pk, scale=1.7):
    ax.imshow(img, cmap="gray", alpha=0.45)
    for rank, col in ((0, "red"), (1, "cyan")):
        segs, al = [], []
        for p_, ks in zip(pts, pk):
            if len(ks) <= rank:
                continue
            i, v = ks[rank]; u = dirs[i].cpu(); vmax = ks[0][1]
            if vmax <= 0:
                continue
            Lh = scale * min(1.0, v / vmax); inpl = float(torch.sqrt(u[1] ** 2 + u[2] ** 2))
            y, x = float(p_[1]), float(p_[2])
            segs.append([(x - Lh * float(u[2]), y - Lh * float(u[1])), (x + Lh * float(u[2]), y + Lh * float(u[1]))]); al.append(0.25 + 0.75 * inpl)
        if segs:
            ax.add_collection(LineCollection(segs, colors=[(1, 0, 0, a_) if col == "red" else (0, 0.8, 1, a_) for a_ in al], linewidths=1.0))
    ax.set_xlim(-0.5, img.shape[1] - 0.5); ax.set_ylim(img.shape[0] - 0.5, -0.5)

rows = []
seed = a.seed0
while len(rows) < a.rows and seed < a.seed0 + 400:
    SS._CACHE.clear(); SS.VIS_WINDOWS = (1,)
    s, d = SS.draw_scene((64,) * 3, seed=seed, device=dev, strength=a.strength, entity_ctx=True, **gen)
    keep = [k for k, L_ in enumerate(s.layers) if L_.role == "shape"]
    seed += 1
    if len(keep) < 2:
        continue
    occ = d["occupancy"][keep].float()
    ov = (occ.unsqueeze(1) * occ.unsqueeze(0)).sum((-1, -2))                                     # (K, K, Z): overlap per slice
    ov = ov * (1 - torch.eye(len(keep), device=dev))[:, :, None]
    if float(ov.max()) < 20:                                                                     # no real crossing in this crop
        continue
    z = int(ov.amax((0, 1)).argmax())
    rows.append((seed - 1, z))
nc = 2 + len(SETTINGS) + len(models)
fig, axs = plt.subplots(len(rows), nc, figsize=(3.1 * nc, 3.2 * len(rows)), squeeze=False)
cmap = plt.get_cmap("tab10")
for r_, (sd, z) in enumerate(rows):
    g = torch.stack(torch.meshgrid(torch.tensor([float(z)], device=dev), torch.arange(4, 60, 4.0, device=dev), torch.arange(4, 60, 4.0, device=dev), indexing="ij"), -1).reshape(-1, 3)
    for c_, (name, wl, pen) in enumerate(SETTINGS):
        SS._CACHE.clear(); SS.VIS_WINDOWS = wl; GN.WINDOW_PENALTY = pen
        s, d = SS.draw_scene((64,) * 3, seed=sd, device=dev, strength=a.strength, entity_ctx=True, **gen)
        keep = [k for k, L_ in enumerate(s.layers) if L_.role == "shape"]
        img = d["image"][z].float().cpu() / 255.0
        if c_ == 0:
            axs[r_, 0].imshow(img, cmap="gray"); axs[r_, 0].set_title(f"seed {sd}, slice z={z}\n{d['record'].get('modality', '')} {d.get('voxel_um', 0):.3g} um", fontsize=7)
            occ = d["occupancy"][keep].float()[:, z].cpu(); rgb = np.zeros(occ.shape[1:] + (3,))
            for j in range(occ.shape[0]):
                rgb += occ[j].numpy()[..., None] * np.array(cmap(j)[:3])[None, None]
            axs[r_, 1].imshow(np.clip(rgb, 0, 1)); axs[r_, 1].set_title(f"true populations ({len(keep)})", fontsize=7)
        t = TR.entity_queries(s, d, 64, sd, dev, points=g, sigma=torch.full((g.shape[0],), 2.5, device=dev))
        mass = (t["e_mass"].float() * t["e_vis"].float()) * (t["e_node"] >= 0)
        rho = (mass[:, None] * BG.pdf(dirs[None, :, None, :], t["e_R"][:, None], t["e_kappa"][:, None].clamp(max=ET.DENS_KAPPA_MAX))).sum(-1)
        glyphs(axs[r_, 2 + c_], img, g.cpu(), peaks(rho)); axs[r_, 2 + c_].set_title(name, fontsize=7)
    SS.VIS_WINDOWS = (1, 2, 4); GN.WINDOW_PENALTY = lambda fw: 1.0
    x = d["image"].float()[None, None] / 255.0
    for j, (name, m) in enumerate(models):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            P = m(x, voxel=torch.tensor([math.log2(d["voxel_um"])], device=dev), e_points=g[None], e_sigma=torch.full((1, g.shape[0]), 2.5, device=dev),
                  e_dirs=dirs[None, None].expand(1, g.shape[0], -1, -1))
        pk = peaks(P["ent"]["f"][0].float())
        if "m" in P["ent"]:                                                                      # only where the model itself puts material
            mm = P["ent"]["m"][0].float().cpu(); pk = [k_ if float(mm[q]) >= a.mass_min else [] for q, k_ in enumerate(pk)]
        glyphs(axs[r_, 2 + len(SETTINGS) + j], img, g.cpu(), pk); axs[r_, 2 + len(SETTINGS) + j].set_title(f"model {name} (its mass >= {a.mass_min})", fontsize=7)
for ax in axs.flat:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(a.out, dpi=70); print("saved", a.out, "rows", [r for r in rows])
