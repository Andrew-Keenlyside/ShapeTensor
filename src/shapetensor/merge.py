"""Stage 4 — merging the population fields. No masks: every output covers the whole volume and carries the
activation as its weight, so a downstream consumer weights or thresholds as it sees fit and the field has no cuts.

* :func:`single_field` — one vector per voxel: the direction of the merged population tensors ``Σ_r a_r S_r / tr S_r``
  (each population's tensor weighted by its activation), its gap, and the total activation as weight. An incoherent
  layer has a near-isotropic tensor and adds no direction, so it is discounted by itself. This is the shape tensor's
  single field, the like-for-like alternative to the structure tensor's eigenvector.
* :func:`multi_field` — several vectors per voxel: every population's direction, ordered by activation, with the
  activation as weight. Where one population is present the others' weights are at the noise level; the null
  ``tau`` (exported with the field) is the activation below which a population is absent.

:func:`export` writes a population field to a zarr group.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .tensors import direction_and_gap


def merged_tensor(S: torch.Tensor, A: torch.Tensor, floor: float = 0.0) -> torch.Tensor:
    """``T (6, Z, Y, X) = Σ_r (a_r + floor) S_r / tr S_r``: the population tensors, each normalised to unit trace, weighted
    by their activations. Linear in the activations, so tensors of overlapping blocks can be blended before the
    eigensolution (the whole-volume driver does)."""
    Aw = A.clamp_min(0)
    return (S * ((Aw + floor) / S[:, :3].sum(1).clamp_min(1e-12))[:, None]).sum(0)


def single_field(S: torch.Tensor, A: torch.Tensor, floor: float = 0.0, return_tensor: bool = False):
    """``S (N, 6, Z, Y, X)`` the population tensors, ``A (N, Z, Y, X)`` the activations -> the direction ``(3, Z, Y, X)``
    in (z, y, x), its gap and its weight (the total activation), and with ``return_tensor`` the merged tensor too.
    ``floor`` is added to every activation in the merge (a tenth of the null in the pipeline), so that where every
    activation is zero the field is the mean of the population tensors rather than undefined."""
    T = merged_tensor(S, A, floor)
    d, gap = direction_and_gap(T)
    return (d, gap, A.clamp_min(0).sum(0), T) if return_tensor else (d, gap, A.clamp_min(0).sum(0))


def multi_field(dirs: torch.Tensor, A: torch.Tensor):
    """``dirs (N, 3, Z, Y, X)`` the populations' directions, ``A (N, Z, Y, X)`` their activations -> ``vecs (N, 3, Z, Y, X)``
    ordered by activation at every voxel and ``weights (N, Z, Y, X)`` (the activations, same order)."""
    Aw = A.clamp_min(0)
    order = Aw.argsort(0, descending=True)
    return dirs.gather(0, order[:, None].expand(-1, 3, -1, -1, -1)), Aw.gather(0, order)


def export(path, out: dict, voxel_mm: float, attrs: dict | None = None) -> Path:
    """Write the result of :func:`shapetensor.pipeline.apply` as a zarr group: ``modes (N, 3)``, ``activation``,
    ``dirs``, ``gap``, ``masks``, ``tissue_mask``, ``strongest``, ``vec``, ``vec_gap``, ``weight``, ``vecs``, ``weights``;
    vectors in (z, y, x); ``tau`` in the attributes."""
    import numpy as np
    import zarr
    g = zarr.open_group(str(path), mode="w")
    half = ("activation", "dirs", "gap", "vec", "vec_gap", "weight", "vecs", "weights")
    for k in ("modes", "activation", "dirs", "gap", "masks", "tissue_mask", "strongest", "vec", "vec_gap", "weight", "vecs", "weights"):
        v = out[k].cpu().numpy()
        g.create_array(k, data=v.astype(np.float16) if k in half else v)
    g.attrs.update({"voxel_mm": voxel_mm, "n_populations": int(out["modes"].shape[0]), "component_order": "zyx", "tau": out["tau"], **(attrs or {})})
    return Path(path)
