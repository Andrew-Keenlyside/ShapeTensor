"""The README example: one synthetic crop in three panels -- the image (a slice in the plane of its main fibres), the
populations from construction, and the model's populations -- each population an elliptical ODF (needle = line, disc =
sheet, small sphere = isotropic; coloured by direction), scaled by the square root of its share. Synthetic only.

    PYTHONPATH=src python scripts/figure_example.py CKPT OUT.png [SEED] [STRENGTH]   (env WINDOW, default 3; GLYPH glyph size; NORM=1: every population the same size)
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
from shapetensor.learn import scene as SC, entity_infer as EI, lobe_train as LT, entity_train as ET
from shapetensor.learn.model import PopulationNet
from odf_render import draw_tensors

ckp, out = sys.argv[1], sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
S = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2; WIN = float(os.environ.get("WINDOW", 3.0)); dev = "cuda"
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.3)))
ck = torch.load(ckp, map_location=dev, weights_only=False); net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"])
ET.configure_from_args(ck["args"])
m6 = lambda M: torch.stack([M[..., 0, 0], M[..., 1, 1], M[..., 2, 2], M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]], -1)
VIEW = {0: "z", 1: "y", 2: "x"}; PLANE = {0: (2, 1), 1: (2, 0), 2: (1, 0)}                          # slice axis -> (column, row) axes

base = dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=1.0, deep_tree=True, fill_quota=SC.FILL_QUOTA)
s, d = SC.draw_scene((64,) * 3, seed=seed, device=dev, strength=S, entity_ctx=True, **LT.difficulty_gen(0.4 + 0.6 * S, base))
g3 = np.arange(4, 64, 8)                                                                            # where the lines are, and which way
P3 = torch.tensor([[z, y, x] for z in g3 for y in g3 for x in g3], dtype=torch.float32, device=dev)
t = TR.entity_queries(s, d, 64, 0, dev, points=P3, sigma=torch.full((P3.shape[0],), WIN, device=dev))
tt = {k: (v[None] if torch.is_tensor(v) and k.startswith("e_") and k not in ("e_n", "e_anc") else v) for k, v in t.items()}
sh, E, valid, _, _ = ET.population_targets(tt, 4)
ev, V = torch.linalg.eigh(E[0].double()); lin = (ev[..., 2] - ev[..., 1]).float()
wl = (sh[0] * valid[0] * (lin > 0.5)).float(); ax = V[..., 2].float()
a = int(torch.linalg.eigh(torch.einsum("qk,qki,qkj->ij", wl, ax, ax))[1][:, 2].abs().argmin()) if float(wl.sum()) > 0 else 0
prof = torch.zeros(64, device=dev); prof.index_add_(0, P3[:, a].long(), wl.sum(-1) + 1e-3); c = int(prof.argmax())

cA, rA = PLANE[a]; gs = np.arange(2, 64, 4)
P = np.zeros((len(gs) ** 2, 3), dtype=np.float32); P[:, a] = c; P[:, cA] = np.repeat(gs, len(gs)); P[:, rA] = np.tile(gs, len(gs))
pts = torch.tensor(P, device=dev); sig = torch.full((pts.shape[0],), WIN, device=dev)
t2 = TR.entity_queries(s, d, 64, 0, dev, points=pts, sigma=sig)
tt2 = {k: (v[None] if torch.is_tensor(v) and k.startswith("e_") and k not in ("e_n", "e_anc") else v) for k, v in t2.items()}
sh2, E2, valid2, _, _ = ET.population_targets(tt2, 4)
x = d["image"].float() / 255
with torch.no_grad():
    f = EI.features(net, x); r = EI.query(net, f, pts, sig, EI.sphere_dirs(8, dev), vol_shape=x.shape)
on = r["pop_exist"] > 0.5
img = np.take(x.cpu().numpy(), c, axis=a); lo, hi = np.percentile(img, [1, 99]); im = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
C2 = P[:, [cA, rA]]
qt, kt = torch.nonzero(valid2[0] & (sh2[0] > 0.02), as_tuple=True); qm, km = torch.nonzero(on, as_tuple=True)
fig, axs = plt.subplots(1, 3, figsize=(15, 5.3))
axs[0].imshow(im, cmap="gray"); axs[0].set_title("synthetic crop", fontsize=12)
for j, (title, q, k, T, w) in enumerate((("ground truth (construction)", qt, kt, E2[0], sh2[0]), ("model", qm, km, r["pop_T"], r["pop_share"][:, :-1]))):
    ax_ = axs[j + 1]; ax_.imshow(im, cmap="gray", alpha=0.45)
    if len(q):
        draw_tensors(ax_, C2[q.cpu().numpy()], m6(T[q, k]).cpu().numpy(), float(os.environ.get("GLYPH", 2.6)), view=VIEW[a], scale=None if os.environ.get("NORM") else np.sqrt(w[q, k].clamp(0, 1).cpu().numpy()))
    ax_.set_title(title, fontsize=12)
for ax_ in axs:
    ax_.set_xticks([]); ax_.set_yticks([]); ax_.set_xlim(-0.5, 63.5); ax_.set_ylim(63.5, -0.5)
fig.tight_layout(); fig.savefig(out, dpi=90, bbox_inches="tight"); print("saved", out, "| modality", d["record"].get("modality"), "| slice", VIEW[a], c)
