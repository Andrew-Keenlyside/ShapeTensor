"""Populations as the readout gives them, against the truth, on held-out phantoms: on the plane of the phantom's first
population, at points every 3 voxels, each population's direction (projected into the plane; a direction leaving the
plane draws shorter) with its mass as the line's width -- the target's populations (colour: which population) and the
model's entities at the same window (colour: the target population each is matched to by evaluate_p2's rule, grey:
unmatched), the isotropic objects' mass as grey disc size. One row per phantom; the models side by side.

Usage: python scripts/learn/entity_glyphs.py OUT.png CKPT [CKPT ...] [--names ...] [--rows cross:0 fine:0 mix:1 fan:0 curve:0 sheet:0]
       [--final] [--sigma 2.5] [--mass maps]
"""
import argparse, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_infer as EI, entity_train as ET, phantom2 as P2, crossing_acceptance as CA
from shapetensor.learn.model import PopulationNet

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("ckpts", nargs="+"); ap.add_argument("--names", nargs="+")
ap.add_argument("--rows", nargs="+", default=["cross:0", "cross:1", "fine:0", "mix:1", "fan:0", "curve:0", "sheet:0", "blob:0"])
ap.add_argument("--final", action="store_true"); ap.add_argument("--sigma", type=float, default=2.5); ap.add_argument("--mass", default="maps")
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
FAMS = ("cross", "fan", "curve", "sheet", "blob", "mix", "empty", "fine")
POP = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]; GREY = "#9a9994"
nets = []
for c in a.ckpts:
    ck = torch.load(c, map_location=dev, weights_only=False)
    n_ = PopulationNet.from_args(ck["args"]).to(dev).eval(); n_.load_state_dict(ck["model"]); nets.append(n_)
names = a.names or [c.rsplit("/", 1)[-1].rsplit(".", 1)[0] for c in a.ckpts]
sph = EI.sphere_dirs(200, dev)


def phantom(fam, i):
    S0, fi = (1_900_000 if a.final else 900_000), FAMS.index(fam)
    spec, look = P2.random_spec(fam, torch.Generator().manual_seed(S0 + 100 * fi + i))
    return P2.make(spec, seed=S0 + 50_000 + 100 * fi + i, look=look, device=dev, ss=2 if fam == "fine" else 1)


def frame(ph):
    spec = ph["spec"]
    if spec and "normal" in spec[0]:
        nrm = torch.nn.functional.normalize(torch.tensor(spec[0]["normal"], dtype=torch.float32), dim=0); e1, e2 = P2._frame(nrm)
        if spec[0]["kind"] != "sheet" and "axis" in spec[0]:
            e1 = torch.nn.functional.normalize(torch.tensor(spec[0]["axis"], dtype=torch.float32), dim=0); e2 = torch.linalg.cross(nrm, e1)
        if spec[0]["kind"] == "sheet":                                                             # sheets: the plane across the stack
            e2 = nrm
        return e1.to(dev), e2.to(dev)
    return torch.tensor([0.0, 0.0, 1.0], device=dev), torch.tensor([0.0, 1.0, 0.0], device=dev)


def glyphs(ax, P2d, axes, mass, col, step=3.0, scale=1.0):
    segs, cols, lws = [], [], []
    for (x, y), d, m, c in zip(P2d, axes, mass, col):
        L = 0.45 * step * float(np.hypot(*d))
        if L < 0.05 or m < 0.01:
            continue
        u = d / max(np.hypot(*d), 1e-9)
        segs.append([(x - L * u[0], y - L * u[1]), (x + L * u[0], y + L * u[1])]); cols.append(c); lws.append(0.6 + 6.0 * min(m * scale, 1.0))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, capstyle="round"))


ncol = 2 + len(nets)
fig, axs = plt.subplots(len(a.rows), ncol, figsize=(3.3 * ncol, 3.3 * len(a.rows)), squeeze=False, layout="constrained")
for r_, spec_ in enumerate(a.rows):
    fam, i = spec_.split(":"); ph = phantom(fam, int(i)); n = ph["n"]; c = n / 2.0; e1, e2 = frame(ph)
    t = torch.arange(-24.0, 24.01, 3.0, device=dev); U, V = torch.meshgrid(t, t, indexing="ij")
    P = c + U.reshape(-1, 1) * e1 + V.reshape(-1, 1) * e2; uv = np.stack([U.reshape(-1).cpu().numpy(), V.reshape(-1).cpu().numpy()], 1)
    tf = torch.linspace(-27, 27, 109, device=dev); UF, VF = torch.meshgrid(tf, tf, indexing="ij"); PF = c + UF.reshape(-1, 1) * e1 + VF.reshape(-1, 1) * e2
    img = torch.nn.functional.grid_sample(ph["image"][None, None], (PF / (n - 1) * 2 - 1).flip(-1).view(1, -1, 1, 1, 3), align_corners=True).view(109, 109).cpu().numpy()
    sig = torch.full((P.shape[0],), a.sigma, device=dev); tg = P2.targets(ph, P, sig); K = ph["occ"].shape[0]
    tgt = {"mass": tg["e_mass"][:, :K].cpu().numpy(), "k": tg["e_kappa"][:, :K].cpu().numpy(), "R": tg["e_R"][:, :K].cpu().numpy(), "bg": tg["e_mass"][:, K].cpu().numpy(), "kinds": ph["kinds"]}
    proj = lambda v: np.stack([v @ e1.cpu().numpy(), v @ e2.cpu().numpy()], -1)
    tt = CA._types(tgt["k"][..., 0], tgt["k"][..., 1])
    for col_i in range(ncol):
        ax = axs[r_, col_i]; ax.imshow(img.T, cmap="gray", origin="lower", extent=[-27.25, 27.25, -27.25, 27.25], alpha=1.0 if col_i == 0 else 0.3)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-26, 26); ax.set_ylim(-26, 26)
    axs[r_, 0].set_title(f"{fam} {i}: {', '.join(ph['kinds']) or 'empty'}", fontsize=9)
    ax = axs[r_, 1]; axs[r_, 1].set_title("truth (width: mass)", fontsize=9)
    for k in range(K):
        if tt[:, k].min() == 2 and tt[:, k].max() == 2:                                           # isotropic: discs
            ax.scatter(uv[:, 0], uv[:, 1], s=60 * tgt["mass"][:, k], color=GREY, alpha=0.7, lw=0); continue
        ax_ = np.where((tt[:, k] == 1)[:, None], tgt["R"][:, k, :, 1], tgt["R"][:, k, :, 2])         # girdles: an in-plane line (b-axis)
        glyphs(ax, uv, proj(ax_), tgt["mass"][:, k], [POP[k % 4]] * len(uv))
    for j, net in enumerate(nets):
        ro = EI.readout(net, ph["image"], P, sig, sph, iso_k=ET.VOTE_KAPPA_MIN, mass=a.mass if getattr(net, "splith", None) is not None or a.mass != "maps" else "density")
        mdl = {k: v.cpu().numpy() for k, v in ro.items()}; m = CA.match_p2(tgt, mdl)
        mt = CA._types(mdl["k"][..., 0], mdl["k"][..., 1]); ax = axs[r_, 2 + j]
        ax.set_title(names[j], fontsize=9)
        own = np.full(mdl["mass"].shape, -1)                                                       # which target population each entity matched
        for k in range(K):
            ang_t = np.where((tt[:, k] == 1)[:, None], tgt["R"][:, k, :, 0], tgt["R"][:, k, :, 2])
            ang_m = np.where((mt == 1)[..., None], mdl["R"][..., :, 0], mdl["R"][..., :, 2])
            cos = np.abs(np.einsum("pd,pcd->pc", ang_t, ang_m)); ok = (cos >= np.cos(np.radians(30))) & (mt == tt[:, k:k + 1]) & (tgt["mass"][:, k:k + 1] > 0.02) & (own < 0)
            own = np.where(ok, k, own)
        for cc in range(mdl["mass"].shape[1]):
            ax_ = np.where((mt[:, cc] == 1)[:, None], mdl["R"][:, cc, :, 1], mdl["R"][:, cc, :, 2])
            glyphs(ax, uv, proj(ax_), mdl["mass"][:, cc], [POP[o % 4] if o >= 0 else GREY for o in own[:, cc]])
        ax.scatter(uv[:, 0], uv[:, 1], s=60 * mdl["iso"], color=GREY, alpha=0.7, lw=0)
fig.suptitle(f"Populations on held-out phantoms ({'final' if a.final else 'selection'} set), sigma {a.sigma}\n"
             f"each population's direction in the plane, line width = its mass in the window\n"
             f"model entities coloured by the true population they match (grey: none); grey discs: isotropic objects' mass", fontsize=9)
plt.savefig(a.out, dpi=70); print("saved", a.out)
