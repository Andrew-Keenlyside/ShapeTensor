"""The domain gap between real crops and the generator's, and the generator settings closest to a real dataset.

    python scripts/learn/domain_gap.py --real PATH[@LEVEL] [--n 48] [--size 64] [--out FILE.json]

Per crop (robustly normalised to 0-1) a vector of image statistics that do not depend on labels: intensity quantiles;
the radial power spectrum in 8 octave-ish bands (shares of the total); gradient-magnitude quantiles; structure-tensor
coherence quantiles at two windows; the Laplacian's kurtosis. The gap between two sets is the Frechet distance
between Gaussians fitted to their vectors (standardised by the real set's spread), and per statistic the
standardised mean difference (what differs). Settings scanned: each imaging physics x acquisition strength, at the
scene generator's defaults otherwise -- a fit of the generator's discrete and scalar settings to the real data, and
the measure to track as the generator changes.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as Fn

from shapetensor.filters import gaussian

DEV = "cuda"
NAMES = ([f"q{q}" for q in (5, 25, 50, 75, 95)] + [f"psd{b}" for b in range(8)] + [f"grad{q}" for q in (25, 50, 75, 95)]
         + [f"coh3_{q}" for q in (25, 50, 75)] + [f"coh8_{q}" for q in (25, 50, 75)] + ["lap_kurt"])


def stats(x: torch.Tensor) -> np.ndarray:
    """The statistics vector of a crop ``x (Z, Y, X)`` in 0-1."""
    x = x.float(); q = lambda v, qs: [float(torch.quantile(v.flatten()[:: max(1, v.numel() // 100000)], qq / 100)) for qq in qs]
    out = q(x, (5, 25, 50, 75, 95))
    F = torch.fft.fftn(x - x.mean()); P = (F.real ** 2 + F.imag ** 2)
    n = x.shape[0]; k = torch.fft.fftfreq(n, device=x.device)
    K = torch.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)
    edges = torch.tensor([0, 0.02, 0.04, 0.07, 0.1, 0.15, 0.22, 0.32, 0.9], device=x.device)
    tot = P.sum().clamp_min(1e-12)
    out += [float(P[(K > edges[i]) & (K <= edges[i + 1])].sum() / tot) for i in range(8)]
    xs = gaussian(x, 1.0); g = torch.stack(torch.gradient(xs)); gm = g.norm(dim=0)
    out += q(gm, (25, 50, 75, 95))
    for w in (3.0, 8.0):
        G = [gaussian(g[i] * g[j], w) for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))]
        M = torch.stack([torch.stack([G[0], G[3], G[4]]), torch.stack([G[3], G[1], G[5]]), torch.stack([G[4], G[5], G[2]])]).permute(2, 3, 4, 0, 1)[::2, ::2, ::2]
        ev = torch.linalg.eigvalsh(M); coh = (ev[..., 2] - ev[..., 0]) / ev.sum(-1).clamp_min(1e-9)
        out += q(coh, (25, 50, 75))
    lap = sum(torch.gradient(torch.gradient(xs, dim=a)[0], dim=a)[0] for a in range(3))
    l = lap.flatten(); l = (l - l.mean()) / l.std().clamp_min(1e-9); out.append(float((l ** 4).mean()))
    return np.array(out)


def frechet(A: np.ndarray, B: np.ndarray, ref: np.ndarray) -> float:
    """The Frechet distance between Gaussians fitted to ``A`` and ``B`` (rows: crops), both standardised by ``ref``."""
    from scipy.linalg import sqrtm
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    a, b = (A - mu) / sd, (B - mu) / sd
    ca, cb = np.cov(a, rowvar=False) + 1e-6 * np.eye(a.shape[1]), np.cov(b, rowvar=False) + 1e-6 * np.eye(a.shape[1])
    cs = sqrtm(ca @ cb).real
    return float(((a.mean(0) - b.mean(0)) ** 2).sum() + np.trace(ca + cb - 2 * cs))


def coverage(R: np.ndarray, S: np.ndarray, ref: np.ndarray, k: int = 3) -> float:
    """The share of real crops with a synthetic crop inside their k-nearest-real-neighbour ball (Naeem et al. 2020):
    whether the real data lie inside the training distribution, which a wider (augmented) distribution can improve
    even when its Frechet distance grows."""
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    r, s_ = (R - mu) / sd, (S - mu) / sd
    Drr = np.linalg.norm(r[:, None] - r[None], axis=-1); np.fill_diagonal(Drr, np.inf)
    rad = np.sort(Drr, 1)[:, k - 1]
    Drs = np.linalg.norm(r[:, None] - s_[None], axis=-1)
    return float((Drs.min(1) <= rad).mean())


def full_mix(n: int, size: int, seed0: int, routes: bool):
    """Crops from the training mixture at full difficulty (every family, physics and draw-time route), with or without
    the batch-time contrast routes."""
    from shapetensor.learn.augment import contrast_route
    from shapetensor.learn.lobe_train import difficulty_gen
    from shapetensor.learn.scene import draw_scene
    gen = difficulty_gen(1.0, {"canvas": 2 * size, "crops_per_scene": 4}); out = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        _, d = draw_scene((size,) * 3, seed=seed0 + i, device=DEV, strength=float(rng.uniform(0.3, 1.0)), **gen)
        x = d["image"].float() / 255
        out.append(contrast_route(x, rng, 1.0) if routes else x)
    return out


def real_crops(src: str, n: int, size: int, seed: int = 0):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from shapetensor.learn.lobe_train import RealBuffer
    b = RealBuffer([src], size, DEV, n=n, seed=seed)
    return [it.float() / 255 for it in b.items]


def synth_crops(n: int, size: int, physics: str, strength: float, seed0: int):
    from shapetensor.learn.scene import draw_scene
    out = []
    for i in range(n):
        _, d = draw_scene((size,) * 3, seed=seed0 + i, device=DEV, strength=strength, physics=physics, canvas=2 * size, crops_per_scene=4, modality="hipct" if physics in ("xray", "darkfield", "additive") else None)
        out.append(d["image"].float() / 255)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--real", required=True); ap.add_argument("--n", type=int, default=48); ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--out"); ap.add_argument("--mix-only", action="store_true", help="only the training mixture, with and without routes")
    a = ap.parse_args()
    R = np.stack([stats(x) for x in real_crops(a.real, a.n, a.size)])
    R2 = np.stack([stats(x) for x in real_crops(a.real, a.n, a.size, seed=1)])
    res = {"real_vs_real": frechet(R, R2, R), "settings": {}}
    print("real vs real (the floor):", round(res["real_vs_real"], 2), flush=True)
    res["real_vs_real_coverage"] = coverage(R, R2, R)
    print("real vs real coverage (the ceiling):", round(res["real_vs_real_coverage"], 2), flush=True)
    settings = [(f"{ph}_s{st}", lambda ph=ph, st=st: synth_crops(a.n, a.size, ph, st, 60_000_000 + 1000 * int(10 * st)))
                for ph in ("additive", "xray", "darkfield", "histology", "pli", "mri") for st in (0.3, 0.6, 1.0)] if not a.mix_only else []
    settings += [("mix", lambda: full_mix(3 * a.n, a.size, 61_000_000, False)), ("mix+routes", lambda: full_mix(3 * a.n, a.size, 61_000_000, True))]
    for name, fn in settings:
        S = np.stack([stats(x) for x in fn()])
        fd = frechet(R, S, R); cv = coverage(R, S, R); smd = (S.mean(0) - R.mean(0)) / (R.std(0) + 1e-6)
        worst = [NAMES[i] for i in np.argsort(-np.abs(smd))[:4]]
        res["settings"][name] = {"frechet": fd, "coverage": cv, "worst": worst, "smd": dict(zip(NAMES, smd.round(2).tolist()))}
        print(f"{name:16s}: frechet {fd:8.2f}  coverage {cv:.2f}  most different: {worst}", flush=True)
    best = sorted(res["settings"].items(), key=lambda kv: kv[1]["frechet"])[:3]
    res["best_coverage"] = max(res["settings"].items(), key=lambda kv: kv[1]["coverage"])[0]
    res["closest"] = [k for k, _ in best]; print("closest settings:", res["closest"])
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
