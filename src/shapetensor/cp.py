"""``from shapetensor import cp as st``: the one-call API returning cupy arrays (computed on the GPU in torch, handed
over through DLPack without a copy). ``image`` may be numpy, cupy or torch."""
from __future__ import annotations

from . import shapetensor as _shapetensor, structure_tensor as _structure_tensor
from .st import _to_torch


def _cupy(t):
    import cupy
    return cupy.from_dlpack(t.contiguous())


def shapetensor(image, **kwargs):
    return _cupy(_shapetensor(_to_torch(image, kwargs.get("device", "cuda")), **kwargs))


def structure_tensor(image, **kwargs):
    return _cupy(_structure_tensor(_to_torch(image, kwargs.get("device", "cuda")), **kwargs))
