"""Is the orientation in the image? A classical, model-free check of what the targets ask for.

:func:`orientation_scores`: for each direction u of a hemisphere grid, the image (high-passed) is smoothed by a Gaussian
elongated along u (``sig_par`` along, ``sig_perp`` across) and its energy in a region measured -- a line structure along
u survives the smoothing, structures along other directions are averaged out. Peaks of the score over the sphere are the
line directions present (:func:`peak_axes`): two populations crossing give two peaks. This is the line-orientation score
of orientation-score methods (an oriented filter bank), done in the Fourier domain. Where it recovers both crossing axes
the information is local and a network should be able to use it; where it does not, the targets may be asking for more
than the image holds (the 27 Oct finding on generator arms).

:func:`girdle_axes` (k-planes on gradient directions) is kept for isolated fibres; packed crossings defeat it (their
gradients concentrate in the crossing plane and the fit collapses onto its normal, 26 Oct).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..filters import gaussian


def gradients(img: torch.Tensor, sigma: float = 0.8) -> torch.Tensor:
    """Gaussian-derivative gradients ``(3, Z, Y, X)`` of ``img (Z, Y, X)`` (z, y, x order)."""
    sm = gaussian(img.float()[None, None], sigma, truncate=3.0)[0, 0]
    return torch.stack(torch.gradient(sm), 0)


def girdle_axes(g: torch.Tensor, w: torch.Tensor, k: int = 2, starts: int = 24, iters: int = 25, seed: int = 0):
    """``k`` fibre axes ``(k, 3)`` from gradients ``g (n, 3)`` with weights ``w (n,)``: k-planes clustering of the gradient
    directions (each axis the normal of its girdle). Returns the axes, each girdle's weight share and the residual (the
    weighted mean squared cosine between a gradient and its girdle's axis: 0 for perfect girdles, 1/3 for isotropic)."""
    dev = g.device
    u = g / g.norm(dim=1, keepdim=True).clamp_min(1e-9); w = w / w.sum().clamp_min(1e-12)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best = None
    for s_ in range(starts):
        N = torch.nn.functional.normalize(torch.randn(k, 3, generator=gen), dim=1).to(dev)
        for _ in range(iters):
            c = (u @ N.T) ** 2                                                    # (n, k) squared cosine to each axis
            a = c.argmin(1)
            N_new = []
            for j in range(k):
                m = a == j
                if int(m.sum()) < 3:
                    N_new.append(N[j]); continue
                M = (w[m, None, None] * u[m, :, None] * u[m, None, :]).sum(0)
                N_new.append(torch.linalg.eigh(M)[1][:, 0])
            N_new = torch.stack(N_new)
            if torch.allclose(N_new.abs(), N.abs(), atol=1e-6):
                N = N_new; break
            N = N_new
        c = (u @ N.T) ** 2; a = c.argmin(1)
        res = float((w * c.min(1).values).sum())
        if best is None or res < best[2]:
            share = torch.stack([w[a == j].sum() for j in range(k)])
            best = (N, share, res)
    return best


def axis_errors(est: torch.Tensor, true: torch.Tensor) -> list:
    """Angles (degrees) between true axes ``(m, 3)`` and their best-matched estimates ``(k, 3)`` (axial; each estimate
    used once when k >= m)."""
    c = (torch.nn.functional.normalize(true, dim=1) @ torch.nn.functional.normalize(est, dim=1).T).abs().clamp(max=1.0)
    ang = torch.rad2deg(torch.acos(c)).cpu().numpy()
    import itertools
    m, k = ang.shape
    if k >= m:
        best = min(itertools.permutations(range(k), m), key=lambda p: sum(ang[i, p[i]] for i in range(m)))
        return [float(ang[i, best[i]]) for i in range(m)]
    return [float(ang[i].min()) for i in range(m)]


def axes_in_region(img: torch.Tensor, mask: torch.Tensor, k: int = 2, sigma: float = 0.8, max_n: int = 20000, seed: int = 0):
    """:func:`girdle_axes` on the gradients of ``img`` inside ``mask`` (weights: squared gradient magnitude; at most
    ``max_n`` voxels, the strongest)."""
    G = gradients(img, sigma)
    g = G[:, mask].T; w = (g ** 2).sum(1)
    if g.shape[0] > max_n:
        keep = w.topk(max_n).indices; g, w = g[keep], w[keep]
    if g.shape[0] < 10:
        return None
    return girdle_axes(g, w, k=k, seed=seed)


def orientation_scores(img: torch.Tensor, dirs: torch.Tensor, mask: torch.Tensor, sig_par: float = 3.0, sig_perp: float = 0.6,
                       hp: float = 3.0, chunk: int = 32) -> torch.Tensor:
    """``(D,)`` line-orientation scores of ``img (Z, Y, X)`` over ``dirs (D, 3)`` (z, y, x) in ``mask``: the mean square of
    the high-passed image (less its blur at ``hp``) after a Gaussian elongated along each direction, normalised by the
    high-passed image's mean square there (1: no direction preferred beyond what any line filter keeps)."""
    x = img.float(); x = x - gaussian(x[None, None], hp, truncate=3.0)[0, 0]
    F = torch.fft.fftn(x)
    k = torch.stack(torch.meshgrid(*[torch.fft.fftfreq(n, device=x.device) * 2 * math.pi for n in x.shape], indexing="ij"), -1)   # (Z, Y, X, 3)
    k2 = (k ** 2).sum(-1); base = float((x[mask] ** 2).mean().clamp_min(1e-12))
    out = []
    for a in range(0, dirs.shape[0], chunk):
        u = dirs[a:a + chunk].float()
        kp2 = (k @ u.T).movedim(-1, 0) ** 2                                      # (c, Z, Y, X)
        H = torch.exp(-0.5 * (sig_par ** 2 * kp2 + sig_perp ** 2 * (k2[None] - kp2)))
        y = torch.fft.ifftn(F[None] * H, dim=(-3, -2, -1)).real
        out.append((y[:, mask] ** 2).mean(1) / base)
    return torch.cat(out)


def peak_axes(score: torch.Tensor, dirs: torch.Tensor, k: int = 2, min_sep: float = 20.0):
    """The ``k`` highest local maxima of ``score`` over the axial directions ``dirs`` at least ``min_sep`` degrees apart:
    ``(axes (k', 3), scores (k',))``."""
    order = torch.argsort(score, descending=True); picked = []
    cs = math.cos(math.radians(min_sep))
    for i in order.tolist():
        if all(abs(float(dirs[i] @ dirs[j])) < cs for j in picked):
            picked.append(i)
        if len(picked) == k:
            break
    idx = torch.tensor(picked, device=dirs.device)
    return dirs[idx], score[idx]


def hessian_line_dirs(img: torch.Tensor, mask: torch.Tensor, dirs: torch.Tensor, sigma: float = 1.0, bandwidth: float = 8.0):
    """Per-voxel line directions from the Hessian at the fibre scale ``sigma``, as a density over ``dirs (D, 3)``: in
    ``mask``, each voxel whose two largest curvatures share a sign (a line, bright or dark) votes for its least-curved
    eigenvector with weight ``min(|l2|, |l3|) - |l1|`` (lineness: a wall or a blob votes little); the votes are spread
    over ``dirs`` by an axial kernel of ``bandwidth`` degrees. Returns ``(D,)`` (unit sum)."""
    sm = gaussian(img.float()[None, None], sigma, truncate=3.0)[0, 0]
    g = torch.gradient(sm)
    H = torch.stack([torch.stack(torch.gradient(g[i])[:], 0) for i in range(3)], 0)       # (3, 3, Z, Y, X)
    Hm = H[:, :, mask].permute(2, 0, 1); Hm = 0.5 * (Hm + Hm.transpose(1, 2))
    lam, V = torch.linalg.eigh(Hm)                                                      # ascending by value
    o = lam.abs().argsort(1); lam = torch.gather(lam, 1, o); V = torch.gather(V, 2, o[:, None, :].expand(-1, 3, -1))   # by |l|
    line = (torch.sign(lam[:, 1]) == torch.sign(lam[:, 2]))
    w = ((torch.minimum(lam[:, 1].abs(), lam[:, 2].abs()) - lam[:, 0].abs()).clamp_min(0) * line).float()
    t = V[:, :, 0]                                                                       # least-curved: along the line
    k = 1.0 / (1 - math.cos(math.radians(bandwidth)))
    dens = (w[:, None] * torch.exp(k * ((t @ dirs.T).abs() - 1))).sum(0)
    return dens / dens.sum().clamp_min(1e-12)
