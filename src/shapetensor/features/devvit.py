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

A third consequence: **every output voxel depends on its whole tile** (the transformer attends over all 4,096 tokens),
so neighbouring tiles disagree where they meet. With ``blend="core"`` (the first version) each tile keeps its central
``core``³ and the features step at every core face (64-voxel blocks, see the results log). The other blends keep each
tile's whole output weighted by a window that fades to its faces and normalise by the summed weights, tiles every
``stride`` voxels (default 64, the same tiles the core blend runs, so no extra network passes):

* ``"hann"`` -- separable cos² per axis; at a stride of half the tile the windows sum to a constant (a seamless linear
  cross-fade between neighbours);
* ``"sphere"`` -- radial cos² out to the inscribed sphere: isotropic, no cubic faces or corners;
* ``"gauss"`` -- separable Gaussian, sigma = tile / 8 (as nnU-Net's sliding window).

What blending leaves is the patch grid (periods 8, 4 and 2 along each axis; a half-patch shift only cancels the odd
harmonics). ``grid_filter=True`` removes it after the average: a separable 9-tap filter, an 8-voxel box convolved with
a 2-voxel one (centred), which has exact zeros at every harmonic of the 8-voxel period -- about the token's own scale.

Defaults (``scripts/diag_tiling.py`` on the HiP-CT VOI, results log): ``blend="sphere"``, ``stride=48``,
``grid_filter=True`` -- no step at the tile faces (step at the core faces / median step 0.99, against 2.48 for the core
blend), no patch grid (1.02 against 1.39), features of a shifted volume half as far from the shifted features (0.49
against 1.11), 1.5x the time. ``blend="core"`` with ``grid_filter=False`` is the first version.

Inputs are grey levels 0-255, scaled to [0, 1] as the other anatomix engines; bf16 autocast on the GPU; features are
L2-normalised per voxel after the average. Several tiles go through the network at once (``batch``).
"""
from __future__ import annotations

import itertools
import math

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


def tile_window(kind: str, n: int = TILE, device="cpu") -> torch.Tensor:
    """The blending weight over one tile ``(n, n, n)``: ``"hann"``, ``"sphere"`` or ``"gauss"`` (see the module notes)."""
    i = torch.arange(n, device=device, dtype=torch.float32) + 0.5
    if kind == "hann":
        h = torch.sin(math.pi * i / n) ** 2
        return h[:, None, None] * h[None, :, None] * h[None, None, :]
    if kind == "gauss":
        g = torch.exp(-0.5 * ((i - n / 2) / (n / 8)) ** 2)
        return g[:, None, None] * g[None, :, None] * g[None, None, :]
    if kind == "sphere":
        c = i - n / 2
        r = torch.sqrt(c[:, None, None] ** 2 + c[None, :, None] ** 2 + c[None, None, :] ** 2)
        return torch.where(r < n / 2, torch.cos(math.pi * r / n) ** 2, torch.zeros_like(r))
    raise ValueError(f"blend must be 'core', 'hann', 'sphere' or 'gauss', got {kind!r}")


class DevVitEngine:
    def __init__(self, core: int = 64, shifts=((0, 0, 0), (4, 4, 4)), l2: bool = True, device: str = "cuda", amp: bool = True,
                 batch: int = 4, out_dtype=torch.float16, blend: str = "sphere", stride: int | None = 48, grid_filter: bool = True,
                 **_ignored):
        if core <= 0 or core > TILE or (TILE - core) % 2:
            raise ValueError(f"core must be in (0, {TILE}] with (128 - core) even, got {core}")
        self.core = int(core); self.ctx = (TILE - self.core) // 2
        self.shifts = [tuple(int(s) for s in sh) for sh in shifts]
        if any(s < 0 or s > self.ctx for sh in self.shifts for s in sh):
            raise ValueError(f"shifts must lie in [0, {self.ctx}] (the tile's context)")
        self.blend = str(blend)
        if self.blend != "core":
            tile_window(self.blend, 4)                                                             # validates the name
        self.stride = self.core if self.blend == "core" else (int(stride) if stride is not None else TILE // 2)
        self.l2, self.device, self.amp, self.batch, self.out_dtype = bool(l2), device, bool(amp), int(batch), out_dtype
        self.grid_filter = bool(grid_filter)
        self.name, self.n_channels, self.margin = "anatomix_devvit", 32, self.ctx + (4 if self.grid_filter else 0)
        self._win = None

    def _grid_null(self, acc: torch.Tensor) -> torch.Tensor:
        """``acc (C, Z, Y, X)`` filtered along each axis with ``box8 * box2`` (9 taps, zeros at every k / 8), replicate-padded."""
        k = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], dtype=acc.dtype, device=acc.device) / 16.0
        x = acc[:, None]                                                                                # (C, 1, Z, Y, X)
        for ax in range(3):
            shape = [1, 1, 1, 1, 1]; shape[2 + ax] = 9
            pad = [0, 0, 0, 0, 0, 0]; pad[2 * (2 - ax)] = 4; pad[2 * (2 - ax) + 1] = 4
            x = F.conv3d(F.pad(x, pad, mode="replicate"), k.view(shape))
        return x[:, 0]

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
        if self.blend != "core":
            return self._blended(vol)
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
        if self.grid_filter:
            acc = self._grid_null(acc)
        if self.l2:
            acc /= acc.norm(dim=0, keepdim=True).clamp_min(1e-6)
        return acc.to(self.out_dtype)

    def _blended(self, vol: torch.Tensor) -> torch.Tensor:
        """Every tile's whole output, weighted by the blend window, summed and normalised; tiles every ``stride`` voxels
        starting ``ctx`` voxels before the block (the block reflect-padded as for the core blend)."""
        v = (vol.to(self.device).float() / 255.0).clamp_(0, 1)
        shape = tuple(v.shape); S, P = self.stride, self.ctx
        starts = []
        for n in shape:
            k = max(1, math.ceil((n + 2 * P - TILE) / S) + 1)
            starts.append([-P + i * S for i in range(k)])
        smax = max(max(sh) for sh in self.shifts)
        pad_hi = max(st[-1] + TILE - n for st, n in zip(starts, shape)) + smax
        vp = _pad(v, P, pad_hi); del v                                                             # content voxel 0 at padded index P
        if self._win is None or self._win.device != torch.device(self.device):
            self._win = tile_window(self.blend, TILE, self.device)
        W = self._win
        acc = torch.zeros((self.n_channels,) + shape, dtype=torch.float32, device=self.device)
        wsum = torch.zeros(shape, dtype=torch.float32, device=self.device)
        tiles = list(itertools.product(*starts))
        for sh in self.shifts:
            for b0 in range(0, len(tiles), self.batch):
                group = tiles[b0:b0 + self.batch]
                x = torch.stack([vp[z + P + sh[0]:z + P + sh[0] + TILE, y + P + sh[1]:y + P + sh[1] + TILE, xx + P + sh[2]:xx + P + sh[2] + TILE]
                                 for z, y, xx in group])
                f = self._forward(x)
                for k, (z, y, xx) in enumerate(group):
                    o = (z + sh[0], y + sh[1], xx + sh[2])                                          # the window's content origin
                    lo = [max(0, a) for a in o]; hi = [min(n, a + TILE) for a, n in zip(o, shape)]
                    if any(h <= l for l, h in zip(lo, hi)):
                        continue
                    ws = tuple(slice(l - a, h - a) for l, h, a in zip(lo, hi, o)); cs = tuple(slice(l, h) for l, h in zip(lo, hi))
                    w = W[ws]
                    acc[(slice(None),) + cs] += f[(k, slice(None)) + ws] * w[None]
                    wsum[cs] += w
                del f, x
        del vp
        acc /= wsum.clamp_min(1e-6)[None]
        if self.grid_filter:
            acc = self._grid_null(acc)
        if self.l2:
            acc /= acc.norm(dim=0, keepdim=True).clamp_min(1e-6)
        return acc.to(self.out_dtype)

    def decoder1(self, vol: torch.Tensor) -> torch.Tensor:
        """The first decoder stage (216 channels, a quarter of the resolution: one cell per 4³ voxels), blended across
        tiles as :meth:`_blended` blends the output (the blend window at the cells' resolution), the patch grid (period 2
        cells) removed by ``[1, 2, 1] / 4`` per axis: ``(216, ceil(Z/4), ceil(Y/4), ceil(X/4))`` float32, cell ``i``
        covering voxels ``4i .. 4i + 3``. Tiles start every ``stride`` voxels (a multiple of 4 is required)."""
        if self.stride % 4 or any(sh % 4 for s_ in self.shifts for sh in s_) or self.ctx % 4:
            raise ValueError("decoder1 needs stride, shifts and context in multiples of 4")
        net = load_devvit(self.device)
        mod = dict(net.named_modules())["up_projection.decode.0"]
        store = {}
        h = mod.register_forward_hook(lambda m, i, o: store.__setitem__("f", o.detach().float()))
        try:
            v = (vol.to(self.device).float() / 255.0).clamp_(0, 1)
            shape = tuple(v.shape); S, P = self.stride, self.ctx
            cshape = tuple(-(-n // 4) for n in shape)
            starts = []
            for n in shape:
                k = max(1, math.ceil((n + 2 * P - TILE) / S) + 1)
                starts.append([-P + i * S for i in range(k)])
            smax = max(max(sh) for sh in self.shifts)
            pad_hi = max(st[-1] + TILE - n for st, n in zip(starts, shape)) + smax + 4
            vp = _pad(v, P, pad_hi); del v
            W = tile_window(self.blend if self.blend != "core" else "hann", TILE // 4, self.device)
            acc = torch.zeros((216,) + cshape, dtype=torch.float32, device=self.device)
            wsum = torch.zeros(cshape, dtype=torch.float32, device=self.device)
            tiles = list(itertools.product(*starts))
            for sh in self.shifts:
                for b0 in range(0, len(tiles), self.batch):
                    group = tiles[b0:b0 + self.batch]
                    x = torch.stack([vp[z + P + sh[0]:z + P + sh[0] + TILE, y + P + sh[1]:y + P + sh[1] + TILE, xx + P + sh[2]:xx + P + sh[2] + TILE]
                                     for z, y, xx in group])
                    self._forward(x); f = store["f"]                                                  # (B, 216, 32, 32, 32)
                    for k, (z, y, xx) in enumerate(group):
                        o = ((z + sh[0]) // 4, (y + sh[1]) // 4, (xx + sh[2]) // 4)                    # the window's origin in cells
                        lo = [max(0, a) for a in o]; hi = [min(n, a + TILE // 4) for a, n in zip(o, cshape)]
                        if any(hh <= ll for ll, hh in zip(lo, hi)):
                            continue
                        ws = tuple(slice(ll - a, hh - a) for ll, hh, a in zip(lo, hi, o)); cs = tuple(slice(ll, hh) for ll, hh in zip(lo, hi))
                        acc[(slice(None),) + cs] += f[(k, slice(None)) + ws] * W[ws][None]
                        wsum[cs] += W[ws]
                    del f, x
            acc /= wsum.clamp_min(1e-6)[None]
        finally:
            h.remove()
        k3 = torch.tensor([1.0, 2.0, 1.0], device=acc.device) / 4.0
        x = acc[:, None]
        for ax in range(3):
            shp = [1, 1, 1, 1, 1]; shp[2 + ax] = 3
            pad = [0, 0, 0, 0, 0, 0]; pad[2 * (2 - ax)] = 1; pad[2 * (2 - ax) + 1] = 1
            x = F.conv3d(F.pad(x, pad, mode="replicate"), k3.view(shp))
        return x[:, 0]

    def describe(self) -> dict:
        return {"engine": self.name, "tile": TILE, "core": self.core, "context": self.ctx, "shifts": self.shifts, "l2": self.l2,
                "n_channels": self.n_channels, "amp": self.amp, "batch": self.batch, "blend": self.blend, "stride": self.stride, "grid_filter": self.grid_filter}
