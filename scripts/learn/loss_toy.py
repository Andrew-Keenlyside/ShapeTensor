"""Fast checks of the density loss's framing (notes/framing_review_2026-10-27.md, finding 1), in seconds to minutes,
without images or the trunk:

A. **Optimum under positional uncertainty** (no network): a free density on the sphere fitted to crossing targets whose two
   axes are jittered by an angular noise (the same "input" meaning several worlds). Does the loss's optimum keep lobes at
   the true axes, or go flat -- for the current loss (L1 in log space per direction) and with the target smoothed on the
   sphere?
B. **Learning with perfect information** (a decoder-sized MLP): given the two true axes (as their outer products, in random
   order), predict the density's harmonics (order 8, the model's direct path), trained on random crossings in random 3-D
   orientations under each loss (and with the axes given 10 deg noisy, mild ambiguity). Does it learn two lobes at any
   orientation, and how fast?

Scored by the share of the density within 12 deg of each true axis (target ~0.12-0.13 at kappa 17; uniform 0.022).

Usage: python scripts/learn/loss_toy.py OUT_STEM
"""
import argparse, json, math, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
from shapetensor.learn import bingham as BG, entity_infer as EI, entity_train as ET
from shapetensor.learn.sh import real_sh, n_coef

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--steps", type=int, default=1500)
a = ap.parse_args(); dev = "cuda"; torch.manual_seed(0)
SPH = EI.sphere_dirs(600, dev); CAP = math.cos(math.radians(12.0)); KAPPA = 17.0; BGM = 0.25; FLOOR = ET.DENS_FLOOR


def rand_unit(n, g=None):
    return torch.nn.functional.normalize(torch.randn(n, 3, device=dev, generator=g), dim=-1)


def crossings(n, lo=30.0, hi=90.0, g=None):
    """``n`` random crossings: axes ``A, B (n, 3)`` at lo-hi deg."""
    A = rand_unit(n, g); q = torch.nn.functional.normalize(torch.linalg.cross(A, rand_unit(n, g)), dim=-1)
    ang = torch.deg2rad(lo + (hi - lo) * torch.rand(n, 1, device=dev, generator=g))
    return A, torch.cos(ang) * A + torch.sin(ang) * q


def jitter(v, deg, g=None):
    """Each axis turned by an angle ~ |N(0, deg)| about a random perpendicular."""
    if deg <= 0:
        return v
    p = torch.nn.functional.normalize(torch.linalg.cross(v, rand_unit(v.shape[0], g)), dim=-1)
    t = torch.deg2rad(deg * torch.randn(v.shape[0], 1, device=dev, generator=g))
    return torch.cos(t) * v + torch.sin(t) * p


def frame(axis):
    """A Bingham frame whose mode (third column) is ``axis (n, 3)``."""
    e1 = torch.nn.functional.normalize(torch.linalg.cross(axis, rand_unit(axis.shape[0])), dim=-1); e2 = torch.linalg.cross(axis, e1)
    return torch.stack([e1, e2, axis], -1)


def target(A, B, ks=None):
    """The target density over SPH ``(n, D)``: two lobes (mass 0.375 each, kappa 17, smoothed by a kernel ``ks``) on a
    uniform background (mass 0.25)."""
    k = KAPPA if ks is None else KAPPA * ks / (KAPPA + ks)
    kap = torch.tensor([k, k], device=dev)
    pa = BG.pdf(SPH[None], frame(A)[:, None], kap); pb = BG.pdf(SPH[None], frame(B)[:, None], kap)
    return 0.375 * (pa + pb) + BGM


def loss(f, rho):
    """The model's density shape loss: L1 of log(f / mean f + floor) against log(rho / mean rho + floor), every direction."""
    fp = torch.log(f / f.mean(-1, keepdim=True).clamp_min(1e-9) + FLOOR); ft = torch.log(rho / rho.mean(-1, keepdim=True) + FLOOR)
    return (fp - ft).abs().mean()


def share(f, A, B):
    """Mean share of the density within 12 deg of A and of B."""
    f = f.clamp_min(0); tot = f.sum(-1).clamp_min(1e-12)
    return float(np.mean([float(((f * ((SPH @ v.T).abs().T >= CAP)).sum(-1) / tot).mean()) for v in (A, B)]))


out = {"uniform": 1 - CAP}
t0 = time.time()
# ---- A. the loss's optimum under positional uncertainty
A0, B0 = crossings(1, 60, 60)
truth = share(target(A0, B0), A0, B0)
resA = []
for deg in (0, 5, 10, 15, 20, 30):
    row = {"noise_deg": deg}
    for name, ks in (("current", None), ("smoothed ks=3", 3.0), ("smoothed ks=8", 8.0)):
        g = torch.Generator(device=dev).manual_seed(1)
        AA = jitter(A0.expand(64, 3), deg, g); BB = jitter(B0.expand(64, 3), deg, g)
        rho = target(AA, BB, ks)                                                       # 64 worlds consistent with one "input"
        z = torch.zeros(1, SPH.shape[0], device=dev, requires_grad=True); opt = torch.optim.Adam([z], lr=0.05)
        for _ in range(1500):
            f = torch.exp(z).expand(64, -1); l = loss(f, rho); opt.zero_grad(); l.backward(); opt.step()
        row[name] = round(share(torch.exp(z).detach(), A0, B0), 4)
    resA.append(row); print("A", json.dumps(row), flush=True)
out["A"] = {"truth_share": truth, "rows": resA}

# ---- B. learning with perfect information
LM = 8; NC = n_coef(LM); Y = real_sh(SPH, LM)
def feats(A, B):
    """The two axes as outer products (6 + 6), in random order (no 'first' bundle)."""
    o = lambda v: torch.stack([v[:, 0] ** 2, v[:, 1] ** 2, v[:, 2] ** 2, v[:, 0] * v[:, 1], v[:, 0] * v[:, 2], v[:, 1] * v[:, 2]], -1)
    sw = torch.rand(A.shape[0], 1, device=dev) < 0.5
    return torch.cat([torch.where(sw, o(B), o(A)), torch.where(sw, o(A), o(B))], -1)

gh = torch.Generator(device=dev).manual_seed(123); Ah, Bh = crossings(512, 30, 90, gh)
resB = {}
for name, anneal, noise in (("current", None, 0.0), ("annealed 2->200", 2.0, 0.0), ("current, axes 10 deg noisy", None, 10.0), ("annealed, axes 10 deg noisy", 2.0, 10.0)):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(12, 256), torch.nn.SiLU(), torch.nn.Linear(256, 256), torch.nn.SiLU(), torch.nn.Linear(256, NC)).to(dev)
    with torch.no_grad():
        net[-1].weight.zero_(); net[-1].bias.zero_(); net[-1].bias[0] = -2.0 / float(Y[:, 0].mean())   # little density at the start, flat
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3); g = torch.Generator(device=dev).manual_seed(7); curve = []
    for step in range(a.steps + 1):
        if step % 100 == 0:
            with torch.no_grad():
                curve.append((step, share(torch.nn.functional.softplus(net(feats(Ah, Bh)) @ Y.T), Ah, Bh)))
        if step == a.steps:
            break
        A, B = crossings(128, 30, 90, g)
        ks = None if anneal is None else anneal * (200.0 / anneal) ** (step / a.steps)
        x = feats(jitter(A, noise, g), jitter(B, noise, g))
        f = torch.nn.functional.softplus(net(x) @ Y.T)
        l = loss(f, target(A, B, ks)); opt.zero_grad(); l.backward(); opt.step()
    resB[name] = curve; print("B", name, [(s, round(v, 3)) for s, v in curve[::3]], flush=True)
with torch.no_grad():
    tB = share(target(Ah, Bh), Ah, Bh)
out["B"] = {"target_share": tB, "curves": resB}
print(f"done in {time.time() - t0:.0f}s; A truth share {truth:.3f}, B target share {tB:.3f}, uniform {1 - CAP:.3f}", flush=True)
json.dump(out, open(a.out + ".json", "w"), indent=1)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
for name in ("current", "smoothed ks=3", "smoothed ks=8"):
    axs[0].plot([r["noise_deg"] for r in resA], [r[name] for r in resA], "-o", label=name)
axs[0].axhline(truth, color="k", ls="--", label="truth (no noise)"); axs[0].axhline(1 - CAP, color="grey", ls=":", label="uniform")
axs[0].set_xlabel("uncertainty of each lobe's position (deg)"); axs[0].set_ylabel("optimum's share within 12 deg of A, B"); axs[0].legend(fontsize=8)
axs[0].set_title("A. the loss's optimum under positional uncertainty", fontsize=10)
for name, c in resB.items():
    axs[1].plot([s for s, _ in c], [v for _, v in c], label=name)
axs[1].axhline(tB, color="k", ls="--", label="target"); axs[1].axhline(1 - CAP, color="grey", ls=":")
axs[1].set_xlabel("step"); axs[1].set_ylabel("held-out share within 12 deg"); axs[1].legend(fontsize=8)
axs[1].set_title("B. MLP given the true axes: learning two lobes at any orientation", fontsize=10)
plt.savefig(a.out + ".png", dpi=85); print("saved", a.out + ".png")
