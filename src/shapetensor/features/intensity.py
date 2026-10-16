"""The one-channel engine: the image itself. With it, TCA on ``K`` is the intensity structure tensor exactly
(``K`` is 3 x 3 and the only identifiable read-out is its eigendecomposition), so this engine is the control every
other engine is compared against, and the source of the hybrid's clear-tissue direction."""
from __future__ import annotations

import torch


class IntensityEngine:
    name = "intensity"
    n_channels = 1
    margin = 0

    def __init__(self, log: bool = False):
        self.log = bool(log)

    def __call__(self, vol: torch.Tensor) -> torch.Tensor:
        v = vol.float()
        if self.log:
            v = torch.log1p(v.clamp_min(0))
        return v[None]

    def describe(self) -> dict:
        return {"engine": self.name, "log": self.log}
