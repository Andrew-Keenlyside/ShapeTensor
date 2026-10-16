"""Feature engines, by name. ``make_engine("anatomix48", n_pcs=16)`` etc.

| name | channels | what |
|---|---|---|
| ``intensity`` | 1 | the image: TCA degenerates to the structure tensor (the control) |
| ``jet`` | 27 (3 scales x 9) | multiscale Gaussian derivatives, locally contrast-normalised (no network) |
| ``anatomix16`` | 16 | the anatomix output layer, run cleanly |
| ``anatomix48`` | 48 -> ``n_pcs`` | anatomix layer 58 (texture + context), run cleanly, PCA-compressed |
"""
from __future__ import annotations

from .base import FeatureEngine
from .intensity import IntensityEngine
from .jet import JetEngine

ENGINES = {
    "intensity": IntensityEngine,
    "jet": JetEngine,
}


def _anatomix16(**kw):
    from .anatomix import AnatomixEngine
    kw.setdefault("n_pcs", None)
    return AnatomixEngine(layer="out16", **kw)


def _anatomix48(**kw):
    from .anatomix import AnatomixEngine
    kw.setdefault("n_pcs", 16)
    return AnatomixEngine(layer="l58", **kw)


ENGINES["anatomix16"] = _anatomix16
ENGINES["anatomix48"] = _anatomix48


def make_engine(name: str, **kwargs) -> FeatureEngine:
    try:
        return ENGINES[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown engine {name!r}; known: {sorted(ENGINES)}") from None


__all__ = ["FeatureEngine", "IntensityEngine", "JetEngine", "ENGINES", "make_engine"]
