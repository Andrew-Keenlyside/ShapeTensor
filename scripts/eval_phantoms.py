"""Known-truth phantoms (shapetensor.phantoms): the model's orientation density against the structure tensor.

At query points inside the central crop of each phantom, with window sigma, compare:
  * single-population regions (npop = 1): the angle between the model's density peak and the true axis, and the
    structure tensor's line axis (its weakest eigenvector; gradients at 1, window sigma) and the true axis;
  * crossings (npop = 2): the share of queries where the model's two strongest peaks (at least 25 deg apart, the second
    at least a quarter of the first above the density's minimum) both lie within 15 deg of the two true axes -- the
    structure tensor reports one direction, so it cannot (its error to the nearer axis is given instead);
  * fibre-free texture: the model's peak-to-mean ratio and floor, and the tensor's line-ness (how much each claims).
The phantoms' texture parameters are calibrated to a HiP-CT crop; every phantom is run at several texture levels so
that no result rests on that one setting.

    PYTHONPATH=src python scripts/eval_phantoms.py ckpt [ckpt ...]   (env SIGMA, TEXTURES, SEEDS, MEMFRAC)
"""
import math
import os
import sys

import numpy as np
import torch

from shapetensor import phantoms as PH
from shapetensor.learn import entity_infer as EI
from shapetensor.learn.model import PopulationNet, fib_hemisphere, st_products, st_matrix, window_mean, CE_GRID

dev = "cuda"; torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.5)))
SIG = float(os.environ.get("SIGMA", 4.0)); TEX = [float(t) for t in os.environ.get("TEXTURES", "0.5,1.0").split(",")]
SEEDS = int(os.environ.get("SEEDS", 1))
CASES = [("bundle", {"angle": 60.0}), ("bundle", {"angle": 15.0}), ("crossing", {"cross_deg": 90.0}), ("crossing", {"cross_deg": 60.0}),
         ("crossing", {"cross_deg": 45.0}), ("crossing", {"cross_deg": 30.0}), ("texture", {})]
grid = fib_hemisphere(CE_GRID, dev)
nb = ((grid @ grid.T).abs() > math.cos(math.radians(18))) & ~torch.eye(grid.shape[0], dtype=torch.bool, device=dev)


def ang(a, b):
    return torch.rad2deg(torch.acos((a * b).sum(-1).abs().clamp(max=1)))


def two_peaks(lq):
    """``(Q, 2, 3)`` the two strongest local maxima of each density (NaN where there is no second)."""
    q = torch.exp(lq); qn = q[:, None, :].expand(-1, q.shape[1], -1)
    nbv = torch.where(nb[None], qn, torch.full_like(qn, -1.0)).amax(-1)
    peak = q >= nbv
    val = torch.where(peak, q, torch.full_like(q, -1.0)); o = val.argsort(1, descending=True)
    p1 = grid[o[:, 0]]; out2 = torch.full_like(p1, float("nan")); qmin = q.amin(1)
    for j in range(1, 6):                                                                         # the next peak far enough from the first
        cand = grid[o[:, j]]; vj = val.gather(1, o[:, j:j + 1])[:, 0]
        ok = (ang(cand, p1) >= 25) & ((vj - qmin) >= 0.25 * (q.amax(1) - qmin)) & torch.isnan(out2[:, 0])
        out2 = torch.where(ok[:, None], cand, out2)
    return torch.stack([p1, out2], 1)


nets = {}
for c in sys.argv[1:]:
    ck = torch.load(c, map_location=dev, weights_only=False); net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"])
    nets[os.path.basename(os.path.dirname(c))] = net
print(f"phantoms at window sigma {SIG}, textures {TEX}, {SEEDS} seed(s); crop 96^3 at the centre; queries every 6 voxels")
hdr = f"{'case':22s} {'tex':>4s} {'method':28s} {'single: median axis err':>24s} {'crossing: both within 15':>25s} {'cross: nearer-axis err':>23s} {'texture: peak/mean':>19s}"
print(hdr)
for name, kw in CASES:
    for tx in TEX:
        res = {}
        for sd in range(SEEDS):
            img, tr, meta = PH.build(name, seed=sd, texture=tx, device=dev, **kw)
            img = torch.as_tensor(img, device=dev).float() / 255
            Z, Y, X = img.shape; c0 = [max(0, (n - 96) // 2) for n in (Z, Y, X)]
            x = img[c0[0]:c0[0] + 96, c0[1]:c0[1] + 96, c0[2]:c0[2] + 96].contiguous()
            gs = torch.arange(12, 85, 6, device=dev).float()
            pts = torch.stack(torch.meshgrid(gs, gs, gs, indexing="ij"), -1).view(-1, 3); sig = torch.full((pts.shape[0],), SIG, device=dev)
            zi = (pts[:, 0] + c0[0]).long(); xi = (pts[:, 2] + c0[2]).long()
            d1 = torch.as_tensor(np.asarray(tr["dir1"]), device=dev).float()[zi, xi]; d2 = torch.as_tensor(np.asarray(tr["dir2"]), device=dev).float()[zi, xi] if tr.get("dir2") is not None else torch.full_like(d1, float("nan"))
            npop = torch.as_tensor(np.asarray(tr["npop"]), device=dev)[zi, xi]
            j6 = window_mean(st_products(x[None, None])[0], pts[None].expand(6, -1, -1).contiguous(), sig[None].expand(6, -1).contiguous()).T
            ev, V = torch.linalg.eigh(st_matrix(j6).double()); st_ax = V[..., :, 0].float(); ev = ev.float()
            lin = ((ev[:, 1] - ev[:, 0]) / ev[:, 2].clamp_min(1e-12))
            meth = {"structure tensor": None, **nets}
            for mn, net in meth.items():
                r_ = res.setdefault(mn, {"single": [], "both": [], "near": [], "pm": []})
                if net is None:
                    pk1 = st_ax; pk = torch.stack([st_ax, torch.full_like(st_ax, float("nan"))], 1); pm = 1 + lin
                else:
                    with torch.no_grad():
                        f = EI.features(net, x); r = EI.query(net, f, pts, sig, grid, vol_shape=x.shape)
                    lq = r["logq"].float(); pk = two_peaks(lq); pk1 = pk[:, 0]; pm = torch.exp(lq).amax(1) / torch.exp(lq).mean(1)
                s1 = npop == 1; s2 = npop == 2
                if bool(s1.any()):
                    r_["single"].append(ang(pk1[s1], d1[s1]).cpu())
                if bool(s2.any()):
                    a11, a22 = ang(pk[s2, 0], d1[s2]), ang(pk[s2, 1], d2[s2]); a12, a21 = ang(pk[s2, 0], d2[s2]), ang(pk[s2, 1], d1[s2])
                    both = ((a11 < 15) & (a22 < 15)) | ((a12 < 15) & (a21 < 15))
                    r_["both"].append(both.float().cpu()); r_["near"].append(torch.minimum(a11, a12).cpu())
                if name == "texture":
                    r_["pm"].append(pm.cpu())
        for mn, r_ in res.items():
            cat = lambda k: torch.cat(r_[k]) if r_[k] else torch.zeros(0)
            s1, b, nr, pm = cat("single"), cat("both"), cat("near"), cat("pm")
            f = lambda t, fn: f"{fn(t):.1f}" if len(t) else "-"
            print(f"{name + ' ' + ','.join(f'{k}={v:g}' for k, v in kw.items()):22s} {tx:4.2f} {mn:28s} {f(s1, lambda t: float(t.median())):>24s} "
                  f"{(f'{float(b.mean()) * 100:.0f} %' if len(b) else '-'):>25s} {f(nr, lambda t: float(t.median())):>23s} {f(pm, lambda t: float(t.median())):>19s}")
