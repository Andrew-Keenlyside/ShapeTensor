"""X-ray CT reconstruction artefacts from the content: a slice-by-slice parallel-beam scan of the image and its filtered
back-projection, so streaks, dark bands and halos come from wherever the dense material is.

The image (a grey-level volume, before the acquisition) is read as an attenuation map -- its range stretched so dense
material (bone, calcification, a stained inclusion) stands out, optionally with a few very dense beads (metal,
calcifications: image-only, no target). Each slice across a random axis is projected over ``n`` angles (a Radon
transform by rotation and summation, batched on the device), the projections go through what a real scan does to them,
and the slice is reconstructed by filtered back-projection:

* **beam hardening** -- a polychromatic beam hardens along a path, so the measured line integral falls short of the
  true one, more so the denser the path: ``p - k p^2``. Between two dense objects this gives the dark band, around one
  the cupping and the dark halo;
* **photon starvation** -- counts ``I0 exp(-p)`` drawn from a Poisson law: through dense paths almost nothing arrives and
  the log of a few photons is noise -- the bright and dark streaks along those rays;
* **scatter** -- a constant fraction of the open-beam signal added to every detector reading: dense paths read too
  bright, so dark streaks between dense objects and an overall cupping;
* **few views** -- 40-360 angles over 180 degrees: angular undersampling streaks far from the centre;
* **reconstruction** -- the ramp filter with a Hann or Shepp-Logan window (sharp or smooth kernel).

The reconstruction is mapped back to the input's grey levels by a least-squares line (the artefacts stay as the
residual). Every magnitude scales with ``strength`` (0: the scan is ideal).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn


def _rot_grids(angles: torch.Tensor, D: int, inverse: bool = False) -> torch.Tensor:
    """Sampling grids rotating a ``D x D`` image by each angle (or back): ``(n, D, D, 2)``."""
    c, s = torch.cos(angles), torch.sin(angles)
    if inverse:
        s = -s
    th = torch.stack([torch.stack([c, -s, torch.zeros_like(c)], -1), torch.stack([s, c, torch.zeros_like(c)], -1)], 1)   # (n, 2, 3)
    return Fn.affine_grid(th, (angles.shape[0], 1, D, D), align_corners=False)


def radon(x: torch.Tensor, angles: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    """Parallel-beam projections of slices ``x (A, D, D)`` at ``angles``: ``(A, n, D)`` (line integrals in voxels)."""
    A, D, _ = x.shape; out = []
    for i in range(0, angles.shape[0], chunk):
        g = _rot_grids(angles[i:i + chunk], D); k = g.shape[0]
        r = Fn.grid_sample(x[:, None].repeat(1, k, 1, 1).reshape(A * k, 1, D, D), g.repeat(A, 1, 1, 1), align_corners=False)
        out.append(r.reshape(A, k, D, D).sum(2))                                                   # sum along the rotated rows
    return torch.cat(out, 1)


def fbp(p: torch.Tensor, angles: torch.Tensor, window: str = "hann", chunk: int = 8) -> torch.Tensor:
    """Filtered back-projection of ``p (A, n, D)`` taken at ``angles``: slices ``(A, D, D)``."""
    A, n, D = p.shape; m = 2 * D
    f = torch.fft.rfftfreq(m, device=p.device); ramp = 2 * f.abs()
    if window == "hann":
        ramp = ramp * (0.5 + 0.5 * torch.cos(math.pi * f / f.max()))
    elif window == "shepp":
        ramp = ramp * torch.sinc(f / f.max() / 2)
    q = torch.fft.irfft(torch.fft.rfft(p, n=m) * ramp, n=m)[..., :D]                             # zero-padded: no wrap-around
    out = torch.zeros(A, D, D, device=p.device)
    for i in range(0, n, chunk):
        g = _rot_grids(angles[i:i + chunk], D, inverse=True); k = g.shape[0]
        sm = q[:, i:i + k, None, :].expand(A, k, D, D).reshape(A * k, 1, D, D)                    # each projection smeared along its rays
        out += Fn.grid_sample(sm, g.repeat(A, 1, 1, 1), align_corners=False).reshape(A, k, D, D).sum(1)
    return out * (math.pi / (2 * n))


def scan(img: torch.Tensor, rng, rec: dict, strength: float = 1.0) -> torch.Tensor:
    """``img (Z, Y, X)`` scanned and reconstructed slice by slice across a random axis with beam hardening, photon
    starvation, scatter and few views (see the module docstring); returns the image in its own grey range. ``rng``:
    the scene's :class:`~shapetensor.learn.scene.Rng` (``u``, ``lu``, ``i``, ``p``, ``rand``, ``g``)."""
    sg = float(min(max(strength, 0.0), 1.0)); dev = img.device
    ax = rng.i(0, 2); x = img.movedim(ax, 0).float(); A, B, C = x.shape
    flat = x.flatten()[:: max(1, x.numel() // 100000)]
    lo, hi = torch.quantile(flat, torch.tensor([0.001, 0.999], device=dev))
    mu = ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1.2) ** rng.u(1.0, 2.0)                  # dense material stands out
    beads = 0
    if rng.p(0.25 * sg):                                                                        # metal / calcifications: very dense, small
        beads = rng.i(1, 4); D_ = torch.zeros_like(mu)
        idx = (rng.rand(beads, 3) * torch.tensor([A, B, C], device=dev, dtype=torch.float32)).long()
        D_[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
        from ..filters import gaussian
        D_ = gaussian(D_, rng.u(0.8, 2.0)); mu = mu + rng.u(3, 10) * D_ / D_.max().clamp_min(1e-6)
    D = int(math.ceil(math.sqrt(2) * max(B, C))) + 2; pb, pc = (D - B) // 2, (D - C) // 2
    m = Fn.pad(mu[:, None], (pc, D - C - pc, pb, D - B - pb), mode="replicate")[:, 0]             # the specimen continues past the slice
    n = rng.i(40, 360) if rng.p(0.5) else 180
    ang = torch.arange(n, device=dev, dtype=torch.float32) * (math.pi / n) + rng.u(0, math.pi / n)
    p = radon(m, ang)
    pmax = rng.lu(1.5, 7.0) * sg + 0.3                                                          # the densest path's attenuation (e-folds)
    p = p * (pmax / p.amax().clamp_min(1e-6))
    k = rng.u(0.0, 0.12) * sg; p_bh = p - k * p * p                                            # beam hardening
    I0 = rng.lu(3e2, 3e5) / max(sg, 0.05)
    s = rng.u(0.0, 0.08) * sg                                                                   # scatter: a share of the open beam
    I = I0 * torch.exp(-p_bh) + s * I0 * float(torch.exp(-p_bh).mean())
    I = torch.poisson(I, generator=rng.g) if sg > 0 else I                                      # photon starvation through dense paths
    pm = -torch.log(I.clamp_min(0.5) / (I0 * (1 + s * float(torch.exp(-p_bh).mean()))))
    win = ["hann", "shepp", "ramlak"][rng.i(0, 2)]
    y = fbp(pm, ang, win)[:, pb:pb + B, pc:pc + C]
    Xm = torch.stack([y.flatten(), torch.ones_like(y.flatten())], 1)                            # back to the input's grey levels
    sol = torch.linalg.lstsq(Xm[::7], x.flatten()[::7, None]).solution
    out = (y * sol[0, 0] + sol[1, 0])
    rec["ct_scan"] = {"axis": ax, "views": n, "pmax": round(pmax, 2), "hardening": round(k, 3), "photons": round(I0), "scatter": round(s, 3),
                      "window": win, "beads": beads}
    return out.movedim(0, ax).contiguous()
