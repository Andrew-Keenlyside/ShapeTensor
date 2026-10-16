"""The no-network control engine: a multiscale Gaussian derivative jet, locally contrast-normalised.

Channels: at each scale ``sigma`` in ``scales``, the three first derivatives (order 1) and the six second
derivatives (order 2) of the image -- 9 per scale, 27 for three scales. Each channel is divided by its own local RMS
over a Gaussian window ``norm_rho`` (plus a floor at ``eps`` x its global RMS), so a step edge, which would otherwise
dominate every window it touches, has the same local energy as the texture channels around it; then, optionally, the
channel vector is L2-normalised per voxel, as the anatomix engines' outputs are.

The hypothesis this engine tests (TCA_FIELD_REVIEW.md, arm C3): what makes the 48-D anatomix layer robust at an
interface is that the step becomes "one small change among many texture channels". If a linear jet with local
normalisation gives TCA the same robustness, the network is not needed.
"""
from __future__ import annotations

import itertools

import torch

from ..filters import gaussian


class JetEngine:
    name = "jet"
    margin = 0

    def __init__(self, scales=(1.0, 2.0, 4.0), orders=(1, 2), norm_rho: float = 8.0, l2: bool = True,
                 eps: float = 0.05, truncate: float = 3.0, include_smooth: bool = False):
        self.scales = tuple(float(s) for s in scales)
        self.orders = tuple(int(o) for o in orders)
        self.norm_rho = float(norm_rho)
        self.l2 = bool(l2)
        self.eps = float(eps)
        self.truncate = float(truncate)
        self.include_smooth = bool(include_smooth)
        per_scale = (1 if include_smooth else 0) + (3 if 1 in self.orders else 0) + (6 if 2 in self.orders else 0)
        self.n_channels = per_scale * len(self.scales)
        self.margin = int(self.truncate * (max(self.scales) + self.norm_rho) + 1)

    def _derivative_orders(self):
        if self.include_smooth:
            yield (0, 0, 0)
        if 1 in self.orders:
            for ax in range(3):
                o = [0, 0, 0]
                o[ax] = 1
                yield tuple(o)
        if 2 in self.orders:
            for a, b in itertools.combinations_with_replacement(range(3), 2):
                o = [0, 0, 0]
                o[a] += 1
                o[b] += 1
                yield tuple(o)

    def __call__(self, vol: torch.Tensor, out_dtype=torch.float16) -> torch.Tensor:
        """Channels are computed one at a time in float32 and stored in ``out_dtype``; the per-voxel L2 norm is
        accumulated in float32, so the peak is one field in ``out_dtype`` plus a few float32 maps."""
        v = vol.float()
        F = torch.empty((self.n_channels,) + tuple(v.shape), dtype=out_dtype, device=v.device)
        k = 0
        for s in self.scales:
            for o in self._derivative_orders():
                c = gaussian(v, s, order=o, truncate=self.truncate)
                if self.norm_rho > 0:
                    glob = c.pow(2).mean().sqrt()
                    loc = gaussian(c * c, self.norm_rho, truncate=self.truncate).clamp_min(0).sqrt_()
                    c = c / (loc + self.eps * glob)
                    del loc
                F[k] = c.to(out_dtype)
                k += 1
                del c
        if self.l2:
            nrm = torch.zeros(tuple(v.shape), dtype=torch.float32, device=v.device)
            for k in range(self.n_channels):
                nrm += F[k].float().pow(2)
            nrm = nrm.sqrt_().clamp_min_(1e-6)
            for k in range(self.n_channels):
                F[k] = (F[k].float() / nrm).to(out_dtype)
            del nrm
        return F

    def describe(self) -> dict:
        return {"engine": self.name, "scales": self.scales, "orders": self.orders, "norm_rho": self.norm_rho,
                "l2": self.l2, "eps": self.eps, "include_smooth": self.include_smooth, "n_channels": self.n_channels}
