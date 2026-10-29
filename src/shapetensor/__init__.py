"""shapetensor: a general model for extracting object orientations from intensity volumes.

A 3D network maps any intensity array to a shape latent. Light query heads read, at any point and window, the
orientation distribution functions (ODFs) of the objects present there, at a chosen grouping level, as normalised
densities over directions. Trained only on synthetic volumes whose orientations are known by construction.
"""
from __future__ import annotations

import os as _os

__version__ = "0.1.0"

DATA_ROOT = _os.environ.get("SHAPETENSOR_DATA", _os.path.join(_os.path.expanduser("~"), "shapetensor_data"))
