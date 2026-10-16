"""The feature-engine contract.

An engine maps an image block ``(Z, Y, X)`` float32 (grey levels, 0-255 for uint8 sources) to a feature field
``(C, Z, Y, X)`` on the SAME grid. The caller gives the block real context (margins) and crops after the tensor pass;
``margin`` says how many voxels from a block face the engine's own output is unreliable (its receptive field),
so the caller can refuse blocks with too little context. Engines are deterministic given their arguments.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class FeatureEngine(Protocol):
    name: str
    n_channels: int
    margin: int

    def __call__(self, vol: torch.Tensor) -> torch.Tensor: ...

    def describe(self) -> dict: ...
