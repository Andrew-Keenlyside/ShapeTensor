"""Per-voxel orientations on held-out phantoms: on the plane of the phantom's bundles, at every voxel (every 2 voxels in
the full view, every voxel in the zoom on the core), the true orientation of each population present (from the labels:
each population's own tensor where it holds at least 20 % of the voxel -- two lines where two populations share it)
next to the model read out per voxel (its density at sigma 1: up to two lobes per voxel, the second at least 25 deg
from the first and at least half its height above the mean), the model's lobes coloured by the entity they are grouped
into (in-plane continuity: neighbouring lobes agreeing within 20 deg; pieces whose axes agree within 15 deg merged).

Usage: python scripts/learn/voxel_glyphs.py OUT.png CKPT [--final] [--rows cross:0 cross:1 fine:0 fan:0]
"""
import argparse, math, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_infer as EI, phantom2 as P2
from shapetensor.learn.model import PopulationNet

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("ckpt"); ap.add_argument("--final", action="store_true")
ap.add_argument("--rows", nargs="+", default=["cross:0", "cross:1", "fine:0", "fan:0"]); ap.add_argument("--name", default="direct-path pilot, read out per voxel")
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
FAMS = ("cross", "fan", "curve", "sheet", "blob", "mix", "empty", "fine")
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"])
SPH = EI.sphere_dirs(300, dev)
POP_COL = ["#e41a1c", "#4daf4a", "#377eb8", "#984ea3"]
ENT_COL = ["#ff7f00", "#1f78b4", "#33a02c", "#e31a1c", "#6a3d9a", "#b15928", "#a6cee3", "#fb9a99"]


def phantom(fam, i):
    S0, fi = (1_900_000 if a.final else 900_000), FAMS.index(fam)
    spec, look = P2.random_spec(fam, torch.Generator().manual_seed(S0 + 100 * fi + i))
    return P2.make(spec, seed=S0 + 50_000 + 100 * fi + i, look=look, device=dev, ss=2 if fam == "fine" else 1)


def plane(ph):
    spec = ph["spec"]; nrm = torch.nn.functional.normalize(torch.tensor(spec[0]["normal"], dtype=torch.float32), dim=0)
    e1, e2 = P2._frame(nrm)
    if spec[0]["kind"] != "sheet" and "axis" in spec[0]:                                          # x along the first bundle
        e1 = torch.nn.functional.normalize(torch.tensor(spec[0]["axis"], dtype=torch.float32), dim=0); e2 = torch.linalg.cross(nrm, e1)
    return e1.to(dev), e2.to(dev)


def sample(vol, P):
    n = vol.shape[-1]; g = (P / (n - 1) * 2 - 1).flip(-1).view(1, -1, 1, 1, 3)
    return torch.nn.functional.grid_sample(vol[None].float() if vol.dim() == 4 else vol[None, None].float(), g, align_corners=True)[0, :, :, 0, 0]


def t33(t6):
    T = torch.zeros(t6.shape[1], 3, 3, device=dev)
    for (i, j), c in zip(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)), range(6)):
        T[:, i, j] = t6[c]; T[:, j, i] = t6[c]
    return T


def truth(ph, P):
    """Per point, per population with >= 20 % of the voxel: (point index, population, direction)."""
    occ = ph["occ_f"] if "occ_f" in ph else ph["occ"]; tens = ph["tens_f"] if "tens_f" in ph else ph["tens"]
    ss = int(ph.get("ss", 1)); Pq = (P + 0.5) * ss - 0.5
    out = []
    for k in range(occ.shape[0]):
        o = sample(occ[k], Pq)[0]; T = sample(tens[k], Pq)
        u = torch.linalg.eigh(t33(T))[1][..., -1]
        for i in torch.nonzero(o >= 0.2)[:, 0].tolist():
            out.append((i, k, u[i]))
    return out


def model_lobes(ph, P):
    with torch.no_grad():
        f = EI.features(net, ph["image"]); r = EI.query(net, f, P, torch.ones(P.shape[0], device=dev), SPH, vol_shape=ph["image"].shape, chunk=2048)
    F = r["f"].float(); mu = F.mean(1); i1 = F.argmax(1); h1 = F.gather(1, i1[:, None])[:, 0]
    far = (SPH @ SPH[i1].T).abs().T < math.cos(math.radians(25.0))
    F2 = torch.where(far, F, torch.zeros_like(F)); i2 = F2.argmax(1); h2 = F2.gather(1, i2[:, None])[:, 0]
    mass = r["m"].float() if "m" in r else mu
    ori = (h1 >= 1.5 * mu) & (mass >= 0.1)
    two = ori & (h2 - mu >= 0.5 * (h1 - mu)) & (h2 >= 1.5 * mu)
    lobes = [(i, SPH[i1[i]], 1.0) for i in torch.nonzero(ori)[:, 0].tolist()]
    sh = ((h2 - mu) / ((h1 - mu) + (h2 - mu)).clamp_min(1e-9))
    lobes = [(i, d, (1 - float(sh[i])) if bool(two[i]) else 1.0) for i, d, _ in lobes] + [(i, SPH[i2[i]], float(sh[i])) for i in torch.nonzero(two)[:, 0].tolist()]
    return lobes


def group(lobes, m, join=20.0, merge=15.0):
    """Entity labels for the lobes on an ``m x m`` plane grid (8-neighbour continuity, then axis merging)."""
    idx = {}
    for li, (i, d, s) in enumerate(lobes):
        idx.setdefault(i, []).append(li)
    rows, cols = [], []; cj = math.cos(math.radians(join))
    for li, (i, d, s) in enumerate(lobes):
        y, x = divmod(i, m)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy, dx) == (0, 0) or not (0 <= y + dy < m and 0 <= x + dx < m):
                    continue
                for lj in idx.get((y + dy) * m + x + dx, []):
                    if abs(float(d @ lobes[lj][1])) >= cj:
                        rows.append(li); cols.append(lj)
    nc, lab = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(lobes), len(lobes))), directed=False)
    D = np.stack([d.cpu().numpy() for _, d, _ in lobes]); W = np.array([s for _, _, s in lobes])
    mass = np.bincount(lab, W, nc); ax = []
    for c in range(nc):
        T = (W[lab == c, None, None] * D[lab == c, :, None] * D[lab == c, None, :]).sum(0); ax.append(np.linalg.eigh(T)[1][:, -1])
    heads, remap = [], {}
    for c in np.argsort(-mass):
        if mass[c] < 0.03 * mass.sum():
            remap[c] = -1; continue
        for h in heads:
            if abs(ax[c] @ ax[h]) >= math.cos(math.radians(merge)):
                remap[c] = h; break
        else:
            heads.append(c); remap[c] = c
    order = {h: k for k, h in enumerate(heads)}
    return [order.get(remap[l], -1) if remap[l] >= 0 else -1 for l in lab]


fig, axs = plt.subplots(len(a.rows), 5, figsize=(20, 4.1 * len(a.rows)), squeeze=False, layout="constrained")
for r_, spec in enumerate(a.rows):
    fam, i = spec.split(":"); ph = phantom(fam, int(i)); n = ph["n"]; c = n / 2.0; e1, e2 = plane(ph)
    t = torch.arange(-27.0, 27.01, 1.0, device=dev); m = t.numel(); U, V = torch.meshgrid(t, t, indexing="ij")
    P = c + U.reshape(-1, 1) * e1 + V.reshape(-1, 1) * e2
    img = sample(ph["image"], P)[0].view(m, m).cpu().numpy()
    tru = truth(ph, P); lob = model_lobes(ph, P); lab = group(lob, m)
    uv = np.stack([U.reshape(-1).cpu().numpy(), V.reshape(-1).cpu().numpy()], 1)
    proj = lambda d: np.array([float(d @ e1), float(d @ e2)])
    n_shared_t = sum(1 for k, v in __import__("collections").Counter(i_ for i_, _, _ in tru).items() if v >= 2)
    n_two_m = sum(1 for k, v in __import__("collections").Counter(i_ for i_, _, _ in lob).items() if v >= 2)

    def draw(ax_, items, cols, step, box, L=0.45):
        segs, cc = [], []
        for (i_, d, s), col in zip(items, cols):
            x, y = uv[i_]
            if not (box[0] <= x <= box[1] and box[0] <= y <= box[1]) or (x % step or y % step):
                continue
            p = proj(d); ln = L * step * (0.4 + 0.6 * s)
            segs.append([(x - ln * p[0], y - ln * p[1]), (x + ln * p[0], y + ln * p[1])]); cc.append(col)
        ax_.add_collection(LineCollection(segs, colors=cc, linewidths=1.3 if step > 1 else 1.8))

    full, zoom = (-27, 27), (-8, 8)
    for j, (title, box, step) in enumerate((("truth (per population)", full, 2), ("model per voxel (per entity)", full, 2),
                                             ("truth, core zoom (every voxel)", zoom, 1), ("model, core zoom (every voxel)", zoom, 1))):
        ax_ = axs[r_, 1 + j]; ax_.imshow(img.T, cmap="gray", origin="lower", extent=[-27.5, 27.5, -27.5, 27.5], alpha=0.35)
        if "truth" in title:
            draw(ax_, [(i_, d, 1.0) for i_, k, d in tru], [POP_COL[k % 4] for _, k, _ in tru], step, box)
        else:
            draw(ax_, lob, [ENT_COL[l % 8] if l >= 0 else "#bbbbbb" for l in lab], step, box)
        ax_.set_xlim(box[0] - 0.5, box[1] + 0.5); ax_.set_ylim(box[0] - 0.5, box[1] + 0.5); ax_.set_xticks([]); ax_.set_yticks([])
        if j == 1:
            title += f"\n{max(lab) + 1 if lab else 0} entities; {n_two_m} voxels with two vectors"
        if j == 0:
            title += f"\n{len(set(k for _, k, _ in tru))} populations; {n_shared_t} voxels shared"
        ax_.set_title(title, fontsize=9)
        if j in (0, 1):
            ax_.add_patch(plt.Rectangle((zoom[0] - 0.5, zoom[0] - 0.5), zoom[1] - zoom[0] + 1, zoom[1] - zoom[0] + 1, fill=False, ec="k", lw=0.8, ls="--"))
    axs[r_, 0].imshow(img.T, cmap="gray", origin="lower", extent=[-27.5, 27.5, -27.5, 27.5]); axs[r_, 0].set_xticks([]); axs[r_, 0].set_yticks([])
    axs[r_, 0].set_title(f"{fam} phantom {i} ({', '.join(ph['kinds'])}): image", fontsize=9)
fig.suptitle(f"Per-voxel orientations on held-out phantoms ({'final' if a.final else 'selection'} set), on the plane of the bundles. "
             f"Truth: each population's direction where it holds >= 20 % of the voxel (colour = population). Model: {a.name} (colour = entity)", fontsize=10)
plt.savefig(a.out, dpi=75); print("saved", a.out)
