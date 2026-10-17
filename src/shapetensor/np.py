"""``from shapetensor import np as st``: the one-call API returning numpy arrays (computed on the GPU, copied back
once). ``image`` may be numpy, cupy or torch."""
from __future__ import annotations

from . import shapetensor as _shapetensor, structure_tensor as _structure_tensor
from .st import _to_torch


def shapetensor(image, **kwargs):
    return _shapetensor(_to_torch(image, kwargs.get("device", "cuda")), **kwargs).cpu().numpy()


def structure_tensor(image, **kwargs):
    return _structure_tensor(_to_torch(image, kwargs.get("device", "cuda")), **kwargs).cpu().numpy()
