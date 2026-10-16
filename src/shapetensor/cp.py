"""``from shapetensor import cp as st; vec = st.shapetensor(image, ...)``: the one-call API returning cupy arrays
(computed on the GPU in torch, handed over through DLPack without a copy). ``image`` may be numpy, cupy or torch."""
from __future__ import annotations

from . import st as _st


def shapetensor(image, **kwargs):
    import cupy
    out = _st.shapetensor(_st._to_torch(image, kwargs.get("device", "cuda")), **kwargs)
    return cupy.from_dlpack(out.contiguous())


shapetensor.__doc__ = (_st.shapetensor.__doc__ or "") + "\n    Returned as a cupy array."
__all__ = ["shapetensor"]
