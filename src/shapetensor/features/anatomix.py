"""The anatomix engines: the pretrained contrast-agnostic 3-D UNet, read at its 16-channel output (``out16``) or at
the 48-channel full-resolution layer 58 before the output head (``l58``), run cleanly.

Clean means (docs/SHAPE_SPACE_HANDOFF.md, "Running anatomix cleanly"):

* **tiles with discarded margins, never sliding windows**: each tile of ``tile`` core voxels is run with ``margin``
  voxels of real context on every side (the padded tile is at most 384^3, the one-pass ceiling: 448^3 overflows
  INT_MAX in the upsampling), and only its core is kept. Measured 2026-10-16 (fp32, TF32 off): a single-voxel
  perturbation changes layer 58 by < 0.1% of its peak beyond 27 voxels, but a tile EDGE is a half-space
  perturbation and its effect decays slowly: relative to the peak, the share of values off by > 1e-2 is 4.3% at a
  32-voxel margin, 1.6% at 48 and 0.7% at 64 (tile 128), with the maximum ~3e-2 at any margin -- the same order as
  the bf16-vs-fp32 difference (rms 2.4e-3, max 3.5e-2). The default is therefore the largest affordable margin
  (tile 256 + 64 = 384) and the read-out-level effect (angles) is what the tests check, not bit equality;
* **reflect padding** of the block (the network's own zero padding puts a step at the faces) and of each tile up to a
  multiple of 32 (four max-pools);
* **the period-2 checkerboard** removed either by averaging the ``phases``^3 one-voxel grid phases
  (``nyquist="phase"``, 8 passes for ``phases=2``) or by a [1 2 1]/4 filter along each axis after a single pass
  (``nyquist="filter"``, 8x cheaper; equivalence to be shown on the phantoms), or left (``"none"``);
* **L2 normalisation per voxel** after the average, as chunkreg runs it.

An optional per-volume PCA (``n_pcs``) is fitted on a single-phase pass over a sample of tiles and applied tile by
tile, so a 48-channel field never has to be held whole; the projection is linear, so it commutes with the phase
average and the later derivative filters (but not with the L2 normalisation, which is then done in PC space).
Inputs are grey levels 0-255 (uint8 images); they are scaled to [0, 1] as chunkreg does.
"""
from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F

from ..filters import nyquist121
from ..pca import PCA

NET_MULTIPLE = 32          # four max-pools: tile dims must be multiples of 16; 32 is what chunkreg pads to
#: one forward pass's ceiling per tap: the 48-channel full-resolution tensors must stay under 2^31 elements
#: (448^3 fails in upsample_nearest3d at layer 58; the output head's padding fails at 384^3: 48 x 384^3 > 2^31)
MAX_PASS = {"l58": 384, "out16": 352}
LAYER48 = 58               # decoder output at full resolution after the first skip concat: 32 + 16 channels

_NET = {}


def load_net(device="cuda"):
    if device not in _NET:
        from anatomix.model.load_from_hf import load_from_hf
        net = load_from_hf("anatomix").to(device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        _NET[device] = net
    return _NET[device]


class AnatomixEngine:
    def __init__(self, layer: str = "l58", phases: int = 2, nyquist: str = "phase", tile: int = 256,
                 margin: int = 64, n_pcs: int | None = 16, l2: bool = True, device: str = "cuda",
                 pca_tiles: int = 6, pca_sample: int = 300_000, seed: int = 0, out_dtype=torch.float16,
                 amp: bool = True):
        if layer not in ("l58", "out16"):
            raise ValueError("layer must be 'l58' or 'out16'")
        if nyquist not in ("phase", "filter", "none"):
            raise ValueError("nyquist must be 'phase', 'filter' or 'none'")
        max_pass = MAX_PASS[layer]
        if tile + 2 * margin > max_pass:
            tile = (max_pass - 2 * margin) // NET_MULTIPLE * NET_MULTIPLE      # shrink the tile, keep the margin
            if tile <= 0:
                raise ValueError(f"margin {margin} leaves no tile under the {max_pass} one-pass ceiling")
        self.layer, self.phases, self.nyquist = layer, int(phases), nyquist
        self.tile, self.margin, self.l2 = int(tile), int(margin), bool(l2)
        self.device, self.pca_tiles, self.pca_sample, self.seed = device, int(pca_tiles), int(pca_sample), int(seed)
        self.out_dtype = out_dtype
        self.amp = bool(amp)                       # bf16 autocast (as chunkreg runs it); False = float32
        raw = 48 if layer == "l58" else 16
        self.n_pcs = None if (n_pcs is None or n_pcs >= raw) else int(n_pcs)
        self.n_channels = self.n_pcs or raw
        self.raw_channels = raw
        self.name = f"anatomix{raw}"
        self.pca: PCA | None = None

    # -- the network on one padded tile ----------------------------------------------------------------------
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x (z, y, x)`` in [0, 1] -> ``(C_raw, z, y, x)`` float32. Reflect-pads up to multiples of 32 and crops."""
        net = load_net(self.device)
        shape = x.shape
        pad = [(-n) % NET_MULTIPLE for n in shape]
        xp = x[None, None]
        if any(pad):
            xp = F.pad(xp, (0, pad[2], 0, pad[1], 0, pad[0]), mode="reflect")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
            if self.layer == "l58":
                f = net(xp, layers=[LAYER48], encode_only=True)[0][0]
            else:
                f = net(xp)[0]
        f = f.float()[:, :shape[0], :shape[1], :shape[2]]
        return f

    def _tiles(self, shape):
        """Core intervals per axis: [lo, hi) covering the block in steps of ``tile``."""
        axes = []
        for n in shape:
            edges = list(range(0, n, self.tile)) + [n]
            axes.append([(edges[i], edges[i + 1]) for i in range(len(edges) - 1)])
        return list(itertools.product(*axes))

    def _run_block(self, vp: torch.Tensor, shape, shift, acc: torch.Tensor, project) -> None:
        """One pass over the block at grid phase ``shift``: ``vp`` is the block reflect-padded by ``margin``; each
        tile's input starts ``shift`` voxels later, so the network's internal grid sits ``shift`` voxels differently
        on the content, and the crop is moved back by ``shift`` so ``acc[:, z, y, x]`` always receives the response
        at content voxel (z, y, x). Accumulated in place (no second field-sized buffer, no roll)."""
        m = self.margin
        for (z0, z1), (y0, y1), (x0, x1) in self._tiles(shape):
            lo = (z0 + shift[0], y0 + shift[1], x0 + shift[2])
            hi = (min(z1 + shift[0] + 2 * m, vp.shape[0]), min(y1 + shift[1] + 2 * m, vp.shape[1]), min(x1 + shift[2] + 2 * m, vp.shape[2]))
            t = vp[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            f = self._forward(t)
            c = [m - s for s in shift]
            f = f[:, c[0]:c[0] + (z1 - z0), c[1]:c[1] + (y1 - y0), c[2]:c[2] + (x1 - x0)]
            acc[:, z0:z1, y0:y1, x0:x1] += project(f)
            del f

    def _fit_pca(self, vp: torch.Tensor, shape):
        """A single-phase pass over up to ``pca_tiles`` tiles (spread over the block), sampled, fitted."""
        tiles = self._tiles(shape)
        g = torch.Generator().manual_seed(self.seed)
        pick = [tiles[i] for i in torch.randperm(len(tiles), generator=g)[:self.pca_tiles].tolist()]
        m = self.margin
        samples = []
        per = max(1, self.pca_sample // len(pick))
        for (z0, z1), (y0, y1), (x0, x1) in pick:
            f = self._forward(vp[z0:z1 + 2 * m, y0:y1 + 2 * m, x0:x1 + 2 * m])[:, m:m + (z1 - z0), m:m + (y1 - y0), m:m + (x1 - x0)]
            X = f.reshape(f.shape[0], -1)
            idx = torch.randperm(X.shape[1], generator=g)[:per].to(X.device)
            samples.append(X[:, idx])
            del f
        S = torch.cat(samples, 1)
        self.pca = PCA.fit(S.reshape(S.shape[0], -1, 1), self.n_pcs, n_sample=S.shape[1], seed=self.seed)

    def __call__(self, vol: torch.Tensor) -> torch.Tensor:
        v = (vol.to(self.device).float() / 255.0).clamp_(0, 1)
        shape = tuple(v.shape)
        m = self.margin
        vp = F.pad(v[None, None], (m, m, m, m, m, m), mode="reflect")[0, 0]
        del v
        if self.n_pcs is not None and self.pca is None:
            self._fit_pca(vp, shape)
        project = (lambda f: self.pca(f)) if self.n_pcs is not None else (lambda f: f)
        if self.nyquist == "phase":
            shifts = list(itertools.product(range(self.phases), repeat=3))
        else:
            shifts = [(0, 0, 0)]
        acc = torch.zeros((self.n_channels,) + shape, dtype=torch.float32, device=self.device)
        for s in shifts:
            self._run_block(vp, shape, s, acc, project)
        del vp
        acc /= len(shifts)
        if self.nyquist == "filter":
            acc = nyquist121(acc)
        if self.l2:
            acc /= acc.norm(dim=0, keepdim=True).clamp_min(1e-6)
        return acc.to(self.out_dtype)

    def describe(self) -> dict:
        d = {"engine": self.name, "layer": self.layer, "phases": self.phases, "nyquist": self.nyquist, "tile": self.tile,
             "margin": self.margin, "n_pcs": self.n_pcs, "l2": self.l2, "n_channels": self.n_channels, "amp": self.amp}
        if self.pca is not None:
            d["pca_explained"] = [round(float(x), 4) for x in self.pca.explained.tolist()]
        return d
