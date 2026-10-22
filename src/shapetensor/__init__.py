"""shapetensor: orientation fields of any array, one or several directions per voxel, on the GPU. The array is lifted
to positions × directions by its local lagged autocorrelation, the populations present are separated there (N from
the data), each is carried forward along its own direction, and the field is read at every voxel: a weighted single
field everywhere and co-located vectors with identities where populations cross. The classic and boundary-robust
structure tensor are the baseline.

    import shapetensor as st
    vec        = st.shapetensor(image, scale=10.0)                            # (3, Z, Y, X): the strongest direction, (x, y, z)
    vecs, w, l = st.shapetensor(image, scale=10.0, multi=True, weights=True, labels=True)   # co-located vectors, weights, identities
    vec        = st.structure_tensor(image, rho=10.0)                         # the excised structure tensor's direction
    out        = st.orientations(image, scale=10.0)                           # everything, including the lattice stages
"""
from __future__ import annotations

import os as _os

__version__ = "0.3.0"

#: phantoms, caches and results go here (``SHAPETENSOR_DATA``)
DATA_ROOT = _os.environ.get("SHAPETENSOR_DATA", _os.path.join(_os.path.expanduser("~"), "shapetensor_data"))

from .st import eig_special_3d, excised_structure_tensor_3d, structure_tensor, structure_tensor_3d  # noqa: E402


def shapetensor(image, *, scale: float = 10.0, multi: bool = False, weights: bool = False, labels: bool = False, order: str = "xyz",
                spacing=None, device: str = "cuda", **kwargs):
    """The orientation field of ``image`` -- ``(Z, Y, X)``, or ``(K, Z, Y, X)`` channels that add -- in one call, as the
    caller's array kind (numpy, cupy or torch). Returns ``vec (3, Z, Y, X)``: one unit axial direction per voxel (the
    strongest population everywhere, never cut), or with ``multi=True`` ``vecs (nmax, 3, Z, Y, X)``: the co-located
    vectors where populations cross, largest first, zero where absent. With ``weights=True`` also the weights
    (``(Z, Y, X)`` the presence weight, or ``(nmax, Z, Y, X)`` per vector, 0 = absent); with ``labels=True`` (multi) also
    the population identities ``(nmax, Z, Y, X)``. Components in ``order`` (``"xyz"`` as the ``structure-tensor``
    package, or ``"zyx"``). ``scale``: the window (voxels) every other length is tied to; ``spacing (dz, dy, dx)`` for
    anisotropic voxels; other keywords are :class:`shapetensor.field.Orientations` parameters. Give the block context
    on every face (the field is unreliable within about ``3 * scale`` voxels of a face)."""
    from .field import orientations
    from .st import _like
    out = orientations(image, spacing=spacing, device=device, scale=scale, **kwargs)
    v = out["vecs"] if multi else out["vec"]
    if order == "xyz":
        v = v.flip(1) if multi else v.flip(0)
    res = [_like(v, image)]
    if weights:
        res.append(_like(out["weights"] if multi else out["weight"], image))
    if labels and multi:
        res.append(_like(out["labels"], image))
    return res[0] if len(res) == 1 else tuple(res)


def orientations(image, **kwargs) -> dict:
    """Everything the chain produces on one block (vectors in (z, y, x)): see :func:`shapetensor.field.orientations`."""
    from .field import orientations as _o
    return _o(image, **kwargs)


def population_field(image, tissue=None, **kwargs) -> dict:
    """The whole pipeline on one block: see :func:`shapetensor.pipeline.population_field`."""
    from .pipeline import population_field as _pf
    return _pf(image, tissue, **kwargs)


__all__ = ["shapetensor", "orientations", "structure_tensor", "population_field", "structure_tensor_3d", "excised_structure_tensor_3d",
           "eig_special_3d", "DATA_ROOT", "__version__"]
