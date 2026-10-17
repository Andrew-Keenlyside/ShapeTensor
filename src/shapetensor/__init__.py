"""shapetensor: fibre orientation fields of high-resolution white-matter volumes from a pretrained shape-feature
space, on the GPU — shape features → fibre populations separated in the shape space → one tensor per population →
the merged field (one vector per voxel, or several where populations cross) — with the classic and boundary-robust
structure tensor as the baseline.

    import shapetensor as st
    vec  = st.shapetensor(image, rho=10.0)                 # (3, Z, Y, X): the shape tensor's single field, (x, y, z)
    vecs = st.shapetensor(image, rho=10.0, multi=True)     # (N, 3, Z, Y, X): several vectors where populations cross
    vec  = st.structure_tensor(image, rho=10.0)            # the excised structure tensor's direction, the baseline
    out  = st.population_field(image, rho=10.0)            # everything: modes, activation layers, masks, both fields
"""
from __future__ import annotations

import os as _os

__version__ = "0.2.0"

#: phantoms, caches and results go here (``SHAPETENSOR_DATA``)
DATA_ROOT = _os.environ.get("SHAPETENSOR_DATA", _os.path.join(_os.path.expanduser("~"), "shapetensor_data"))

from .st import eig_special_3d, excised_structure_tensor_3d, structure_tensor, structure_tensor_3d  # noqa: E402


def shapetensor(image, *, rho: float = 10.0, sigma: float = 1.0, multi: bool = False, weights: bool = False, tissue=None, nmax: int = 3,
                order: str = "xyz", engine: str = "anatomix_enc2", device: str = "cuda", **calibration):
    """The fibre direction field of ``image (Z, Y, X)`` from its shape features, in one call, as the caller's array
    kind (numpy, cupy or torch). Returns ``vec (3, Z, Y, X)``, one unit axial direction per voxel (the merged
    population tensors), or with ``multi=True`` ``vecs (N, 3, Z, Y, X)``: every population's direction at every
    voxel, ordered by activation. With ``weights=True`` also the activation weights (``(Z, Y, X)`` the total, or
    ``(N, Z, Y, X)``). No mask is applied: the weights say where a population is present. Components in ``order``
    (``"xyz"`` as the ``structure-tensor`` package, or ``"zyx"``). ``sigma`` / ``rho``: the derivative and integration
    scales in voxels, used alike by the teacher, every population's tensor and the baseline; ``tissue``: an optional
    bool mask of the tissue to read; ``engine``: the feature engine. The block is calibrated on itself (see
    :func:`shapetensor.pipeline.calibrate`); give it context on every face."""
    from .pipeline import population_field
    from .st import _like, _to_torch
    out = population_field(image, tissue, nmax=nmax, device=device, sigma=sigma, rho=rho, engine=engine, **calibration)
    v = out["vecs"] if multi else out["vec"]
    if order == "xyz":
        v = v.flip(1) if multi else v.flip(0)
    if weights:
        return _like(v, image), _like(out["weights"] if multi else out["weight"], image)
    return _like(v, image)


def population_field(image, tissue=None, **kwargs) -> dict:
    """The whole pipeline on one block: see :func:`shapetensor.pipeline.population_field`."""
    from .pipeline import population_field as _pf
    return _pf(image, tissue, **kwargs)


__all__ = ["shapetensor", "structure_tensor", "population_field", "structure_tensor_3d", "excised_structure_tensor_3d", "eig_special_3d",
           "DATA_ROOT", "__version__"]
