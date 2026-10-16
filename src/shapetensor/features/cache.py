"""An on-disk cache of engine outputs, so an experiment that revisits a block pays for the network once.

The key is the engine's configuration plus a hash of the volume (shape, dtype and a strided sample of its bytes),
so the same block through the same engine hits the cache and anything else misses. Files are float16 ``.npy``
under ``$SHAPETENSOR_DATA/cache/features``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .. import DATA_ROOT

CACHE_DIR = Path(DATA_ROOT) / "cache" / "features"


def volume_key(vol) -> str:
    a = vol.detach().cpu().numpy() if isinstance(vol, torch.Tensor) else np.asarray(vol)
    h = hashlib.blake2b(digest_size=12)
    h.update(str(a.shape).encode()); h.update(str(a.dtype).encode())
    h.update(np.ascontiguousarray(a.reshape(-1)[::max(1, a.size // 200_000)]).tobytes())
    return h.hexdigest()


def engine_key(engine) -> str:
    cfg = engine.config() if hasattr(engine, "config") else {"name": getattr(engine, "name", type(engine).__name__)}
    return hashlib.blake2b(json.dumps(cfg, sort_keys=True, default=str).encode(), digest_size=8).hexdigest()


def cached_features(engine, vol: torch.Tensor, tag: str = "", device=None) -> torch.Tensor:
    """``engine(vol)`` from the cache when present, else computed and stored. ``tag`` is a readable prefix."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{tag + '_' if tag else ''}{getattr(engine, 'name', 'engine')}_{engine_key(engine)}_{volume_key(vol)}.npy"
    path = CACHE_DIR / name
    dev = device or (vol.device if isinstance(vol, torch.Tensor) else "cuda")
    if path.exists():
        return torch.as_tensor(np.load(path, mmap_mode="r"), device=dev)
    F = engine(vol)
    np.save(path, F.detach().cpu().numpy().astype(np.float16))
    return F
