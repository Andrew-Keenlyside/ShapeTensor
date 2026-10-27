"""How much crossing information the decoder's direct oriented path receives from the image, per filter-bank setting,
without training: at the cores of Monte Carlo crossings (random 3-D orientations, 45-90 deg), the bank's line energy
interpolated to each direction (as :meth:`EntityDecoder.od_feats` does) -- its value along the true axes A and B against
its median over all directions ("contrast", x), and whether the two strongest directions are A and B within 15 deg.
The same for the true line energies (the oracle that the decoder learned from, 27 Oct).

Usage: python scripts/learn/bank_check.py [--n 8]
"""
import argparse, math, sys
import numpy as np
import torch
sys.path.insert(0, "src")
from shapetensor.filters import gaussian
from shapetensor.learn import phantom2 as P2, entity_infer as EI, inferability as IF

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8); ap.add_argument("--raw", action="store_true", help="pool the energies before normalising")
ap.add_argument("--configs", default="48,3,0.7;96,4,0.6;192,5,0.6;256,6,0.6;256,8,0.6")
a = ap.parse_args(); dev = "cuda"; RAW = a.raw; SPH = EI.sphere_dirs(1000, dev); N = 64


def fib_dirs(n):
    i = torch.arange(n, dtype=torch.float64) + 0.5; zc = 1 - i / n; rc = torch.sqrt(1 - zc ** 2); ph = math.pi * (3 - math.sqrt(5)) * i
    return torch.stack([zc, rc * torch.sin(ph), rc * torch.cos(ph)], 1).float().to(dev)


def energies(img, dirs, sp, se, hp=3.0):
    x = img.float() - gaussian(img.float()[None, None], hp)[0, 0]
    F = torch.fft.fftn(x)
    k = torch.stack(torch.meshgrid(*[torch.fft.fftfreq(n, device=dev) * 2 * math.pi for n in x.shape], indexing="ij"), -1)
    kp2 = (k @ dirs.T).movedim(-1, 0) ** 2; k2 = (k ** 2).sum(-1)
    H = torch.exp(-0.5 * (sp ** 2 * kp2 + se ** 2 * (k2[None] - kp2)))
    E = torch.fft.ifftn(F[None] * H, dim=(-3, -2, -1)).real ** 2
    return E if RAW else E / (E.mean(0, keepdim=True) + 1e-8)


def crossing(seed):
    g = np.random.default_rng(seed)
    A = g.normal(size=3); A /= np.linalg.norm(A); q = np.cross(A, g.normal(size=3)); q /= np.linalg.norm(q)
    ang = math.radians(g.uniform(45, 90)); B = math.cos(ang) * A + math.sin(ang) * q; nrm = np.cross(A, B)
    mc = lambda ax: {"kind": "fibres", "axis": ax.tolist(), "normal": nrm.tolist(), "half": float(g.uniform(9, 13)), "radius": float(g.uniform(0.7, 1.2)),
                     "fvf": float(g.uniform(0.2, 0.35)), "wander": float(g.uniform(0.2, 0.8)), "conc": 0.94}
    return P2.make([mc(A), mc(B)], seed=seed, look={"bg": 0.1, "contrast": [0.8, 0.8], "noise": 0.01}, device=dev), torch.tensor(np.stack([A, B]), dtype=torch.float32, device=dev)


def score(E, dirs, axes, core, level):
    """E (D, n, n, n) relative energies -> at the core voxels, after pooling at ``level``, interpolated to SPH."""
    Ep = gaussian(E, level) if level > 0 else E
    if RAW:                                                                              # pooled first, then relative to the mean direction
        Ep = Ep / (Ep.mean(0, keepdim=True) + 1e-12)
    v = Ep[:, core].mean(1)                                                              # the core's mean relative energy per bank direction
    kk = 1.0 / (1 - math.cos(math.sqrt(2 * math.pi / dirs.shape[0])))
    w = torch.softmax(kk * (SPH @ dirs.T).abs(), -1)                                    # (S, D)
    e = w @ v                                                                            # (S,)
    at = lambda u: float(e[(SPH @ u).abs().argmax()])
    med = float(e.median())
    est, _ = IF.peak_axes(e, SPH, k=2)
    return min(at(axes[0]), at(axes[1])) / med, max(IF.axis_errors(est, axes)) <= 15


cases = [crossing(5_000 + i) for i in range(a.n)]
zz = torch.stack(torch.meshgrid(*[torch.arange(N, device=dev, dtype=torch.float32)] * 3, indexing="ij"), -1); core = (zz - N / 2).norm(dim=-1) <= 4
configs = [tuple(float(v) if i else int(v) for i, v in enumerate(c.split(','))) for c in a.configs.split(';')]
print("setting (bank dirs, sigma along, sigma across) | level -> weaker-axis contrast (median over cases) | both axes found", flush=True)
for D, sp, se in configs:
    dirs = fib_dirs(D)
    Es = [energies(ph["image"], dirs, sp, se) for ph, _ in cases]
    for lv in (1.0, 2.0, 4.0):
        r = [score(E, dirs, ax, core, lv) for E, (ph, ax) in zip(Es, cases)]
        print(f"bank {D:3d}, {sp}, {se} | level {lv}: contrast {np.median([c for c, _ in r]):.2f}  found {sum(f for _, f in r)}/{len(r)}", flush=True)
# the oracle (the true energies, as the screen's doracle)
for D in (48,):
    dirs = fib_dirs(D)
    for lv in (1.0, 2.0, 4.0):
        r = []
        for ph, ax in cases:
            T = ph["tens"]; occ = ph["occ"]; raw = 0
            for k in range(occ.shape[0]):
                t = T[k]; d0, d1, d2 = (dirs[:, i, None, None, None] for i in range(3))
                q = d0 ** 2 * t[0] + d1 ** 2 * t[1] + d2 ** 2 * t[2] + 2 * (d0 * d1 * t[3] + d0 * d2 * t[4] + d1 * d2 * t[5])
                raw = raw + occ[k][None] * q.clamp(0, 1) ** 8
            R = (raw + 0.02) / (raw.mean(0, keepdim=True) + 0.02)
            r.append(score(R, dirs, ax, core, lv))
        print(f"oracle {D} | level {lv}: contrast {np.median([c for c, _ in r]):.2f}  found {sum(f for _, f in r)}/{len(r)}", flush=True)
