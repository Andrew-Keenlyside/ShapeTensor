"""The anatomix-dev-vit engine: the contrast-agnostic anatomix family's vision transformer (PrimusV2, 26.2 M parameters),
read at its 32-channel output at full resolution.

The network cuts a fixed 128³ input into 8×8×8-voxel patches (4,096 tokens), runs 12 transformer layers over them
(width 396, 6 heads, 8 register tokens) and decodes every token back to its patch with three stride-2 transposed
convolutions (396 -> 216 -> 120 -> 32 channels). Two consequences shape how it is run:

* **the input is exactly 128³** (its positional embeddings are sized for it): a block is covered by tiles of 128³ whose
  central ``core`` voxels are kept, with ``(128 - core) / 2`` voxels of real context on every face, the block itself
  reflect-padded by that context (the network's own padding would put a step at the faces);
* **the decoder never mixes patches**: every output voxel is a fixed, position-specific function of one token, so the
  features carry the 8-voxel patch grid. ``shifts`` runs the block at several grid offsets and averages them (by
  default the offset (0, 0, 0) and the half-patch (4, 4, 4)); the crop moves back by the offset, so each pass is
  aligned on the content.

Inputs are grey levels 0-255, scaled to [0, 1] as the other anatomix engines; bf16 autocast on the GPU; features are
L2-normalised per voxel after the average. Several tiles go through the network at once (``batch``).
"""
from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

TILE = 128                 # the network's fixed input
PATCH = 8                  # its patch size

_NET = {}


def load_devvit(device: str = "cuda"):
    """The pretrained network on ``device`` (cached), frozen. Needs ``dynamic_network_architectures`` and ``timm``."""
    if device not in _NET:
        from anatomix.model.load_from_hf import load_from_hf
        net = load_from_hf("anatomix-dev-vit").to(device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        _NET[device] = net
    return _NET[device]


def _pad(x: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Reflect-pad ``x (Z, Y, X)`` by ``lo`` before and ``hi`` after every axis; replicate where the block is too
    small to reflect that far."""
    pads = (lo, hi, lo, hi, lo, hi)
    mode = "reflect" if all(max(lo, hi) < n for n in x.shape) else "replicate"
    return F.pad(x[None, None], pads, mode=mode)[0, 0]


class DevVitEngine:
    def __init__(self, core: int = 64, shifts=((0, 0, 0), (4, 4, 4)), l2: bool = True, device: str = "cuda", amp: bool = True,
                 batch: int = 4, out_dtype=torch.float16, **_ignored):
        if core <= 0 or core > TILE or (TILE - core) % 2:
            raise ValueError(f"core must be in (0, {TILE}] with (128 - core) even, got {core}")
        self.core = int(core); self.ctx = (TILE - self.core) // 2
        self.shifts = [tuple(int(s) for s in sh) for sh in shifts]
        if any(s < 0 or s > self.ctx for sh in self.shifts for s in sh):
            raise ValueError(f"shifts must lie in [0, {self.ctx}] (the tile's context)")
        self.l2, self.device, self.amp, self.batch, self.out_dtype = bool(l2), device, bool(amp), int(batch), out_dtype
        self.name, self.n_channels, self.margin = "anatomix_devvit", 32, self.ctx

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x (B, 128, 128, 128)`` in [0, 1] -> ``(B, 32, 128, 128, 128)`` float32."""
        net = load_devvit(self.device)
        cuda = str(self.device).startswith("cuda")
        with torch.no_grad(), torch.autocast(device_type="cuda" if cuda else "cpu", dtype=torch.bfloat16, enabled=self.amp and cuda):
            y = net(x[:, None])
        return y.float()

    def _tiles(self, shape):
        axes = [[(lo, min(lo + self.core, n)) for lo in range(0, n, self.core)] for n in shape]
        return list(itertools.product(*axes))

    def __call__(self, vol: torch.Tensor) -> torch.Tensor:
        v = (vol.to(self.device).float() / 255.0).clamp_(0, 1)
        shape = tuple(v.shape)
        smax = max(max(sh) for sh in self.shifts)
        vp = _pad(v, self.ctx, TILE - self.ctx + smax); del v
        acc = torch.zeros((self.n_channels,) + shape, dtype=torch.float32, device=self.device)
        tiles = self._tiles(shape)
        for sh in self.shifts:
            for b0 in range(0, len(tiles), self.batch):
                group = tiles[b0:b0 + self.batch]
                x = torch.stack([vp[z0 + sh[0]:z0 + sh[0] + TILE, y0 + sh[1]:y0 + sh[1] + TILE, x0 + sh[2]:x0 + sh[2] + TILE]
                                 for (z0, _), (y0, _), (x0, _) in group])
                f = self._forward(x)
                c = [self.ctx - s for s in sh]                                                     # content voxel z0 sits at window offset ctx - shift
                for k, ((z0, z1), (y0, y1), (x0, x1)) in enumerate(group):
                    acc[:, z0:z1, y0:y1, x0:x1] += f[k, :, c[0]:c[0] + (z1 - z0), c[1]:c[1] + (y1 - y0), c[2]:c[2] + (x1 - x0)]
                del f, x
        del vp
        acc /= len(self.shifts)
        if self.l2:
            acc /= acc.norm(dim=0, keepdim=True).clamp_min(1e-6)
        return acc.to(self.out_dtype)

    def describe(self) -> dict:
        return {"engine": self.name, "tile": TILE, "core": self.core, "context": self.ctx, "shifts": self.shifts, "l2": self.l2,
                "n_channels": self.n_channels, "amp": self.amp, "batch": self.batch}
