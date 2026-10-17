"""Feature engines, by name: ``make_engine("anatomix_enc2")``.

| name | channels | what |
|---|---|---|
| ``anatomix_enc2`` | 32 | anatomix encoder block 2 — the orientation-selective layer; the pipeline's default |
| ``anatomix48`` | 48 → ``n_pcs`` | anatomix layer 58 (the decoder's full-resolution texture + context layer) |
| ``anatomix16`` | 16 | the anatomix output layer |
| ``jet`` | 27 | a multiscale Gaussian derivative jet, locally contrast-normalised: the no-network control |
| ``intensity`` | 1 | the image itself |

An engine maps a block ``(Z, Y, X)`` to features ``(C, Z, Y, X)`` on the same grid; ``engine.margin`` is how far from
a block face its output is unreliable.
"""
from __future__ import annotations

from .base import FeatureEngine
from .intensity import IntensityEngine
from .jet import JetEngine


def _anatomix(layer: str, n_pcs):
    def make(**kw):
        from .anatomix import AnatomixEngine
        kw.setdefault("n_pcs", n_pcs)
        return AnatomixEngine(layer=layer, **kw)
    return make


ENGINES = {
    "anatomix_enc2": _anatomix("enc2", None),
    "anatomix48": _anatomix("l58", 16),
    "anatomix16": _anatomix("out16", None),
    "jet": JetEngine,
    "intensity": IntensityEngine,
}


def make_engine(name: str, **kwargs) -> FeatureEngine:
    try:
        return ENGINES[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown engine {name!r}; known: {sorted(ENGINES)}") from None


__all__ = ["FeatureEngine", "IntensityEngine", "JetEngine", "ENGINES", "make_engine"]
