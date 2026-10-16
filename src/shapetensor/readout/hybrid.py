"""The hybrid single-direction field: the intensity structure tensor where no interface is detected, the shape-space
TCA direction where one is (H1, a Gram blend), or the intensity tensor with gradients excised only near detected
interfaces (H2, in :mod:`shapetensor.tensor` via ``excision_weights(gate=...)``).

The switch is a boundary-strength ``s`` per site (any detector of :func:`shapetensor.readout.single.detectors`) and
a null-calibrated threshold ``s0``: the ``q``-quantile of ``s`` on phantoms with no interface, so the false-switch
rate in clear tissue is ``1 - q`` by construction.
"""
from __future__ import annotations

import torch

from . import common


def calibrate_s0(s_null: torch.Tensor, q: float = 0.99) -> float:
    """``s0``: the ``q``-quantile of the detector on interface-free sites."""
    return float(torch.quantile(s_null.float().flatten(), q))


def blend(T_int: torch.Tensor, G_shape: torch.Tensor, s: torch.Tensor, s0: float, ds: float | None = None) -> dict:
    """H1: ``w = sigmoid((s - s0) / ds)``; direction = null vector of the trace-normalised blend
    ``(1 - w) T_int + w G_shape``. ``ds`` defaults to ``s0 / 4``. Returns ``dir``, ``w``."""
    ds = s0 / 4.0 if ds is None else ds
    w = torch.sigmoid((s - s0) / max(ds, 1e-6))
    Ti = common.normalise_trace(T_int.double())
    Gs = common.normalise_trace(G_shape.double())
    M = (1 - w)[:, None, None].double() * Ti + w[:, None, None].double() * Gs
    d = common.canonical_sign(common.null_dir(M).float())
    return {"dir": d, "w": w}


def switch(dir_int: torch.Tensor, dir_shape: torch.Tensor, s: torch.Tensor, s0: float) -> dict:
    """The hard version of H1: the shape direction where ``s > s0``, else the intensity direction."""
    w = (s > s0)
    return {"dir": torch.where(w[:, None], dir_shape, dir_int), "w": w.float()}
