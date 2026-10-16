"""``from shapetensor import np as st; vec = st.shapetensor(image, ...)``: the one-call API returning numpy arrays
(computed on the GPU in torch, copied back once). ``image`` may be numpy, cupy or torch."""
from __future__ import annotations

from . import st as _st


def shapetensor(image, **kwargs):
    out = _st.shapetensor(_st._to_torch(image, kwargs.get("device", "cuda")), **kwargs)
    return out.cpu().numpy()


shapetensor.__doc__ = (_st.shapetensor.__doc__ or "") + "\n    Returned as a numpy array."
__all__ = ["shapetensor"]
