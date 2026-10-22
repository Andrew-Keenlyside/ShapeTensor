"""The population model: image -> populations per lattice cell.

A small, strictly local, fully convolutional network whose output *is* the stride-4 lattice:

* three levels (16 / 32 / 64 channels at strides 1 / 2 / 4), residual blocks of two 3³ convolutions (one at stride 1,
  two at stride 2, one at stride 4) with a learned zero-initialised gain (no normalisation layer: nothing depends on the chunk's statistics, so a volume run in
  chunks with a halo of the receptive field equals one pass);
* anti-aliased downsampling (an even binomial kernel at stride 2, cells centred where the target's are), no
  transposed convolution anywhere: no checkerboard or lattice-phase artefacts; the stride-2 level is blurred and subsampled once more and concatenated at the
  lattice, so the heads see both scales;
* heads at the lattice (1³ convolutions): ``K`` population slots -- a shape tensor (``B Bᵀ / tr``,
  symmetric positive semi-definite with trace 1, from 6 numbers), a mass (``softplus``, read as √mass), kind logits
  (line / sheet / blob / interface) and an identity embedding (unit length) -- and the cell's scalars: the shape
  fraction (logit), the class shares (logits over line / sheet / blob / interface / isotropic) and the log size.

The receptive field is :func:`receptive_field` voxels (45 for the default; a chunked run needs a halo of half of it),
enough for the target's window (``sigma_w`` = 5: 5 sigma = 25 voxels).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

TRIL = ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2))


class Block(nn.Module):
    """``x + g * conv(act(conv(act(x))))``, ``g`` initialised at zero (ReZero): trains stably without normalisation."""

    def __init__(self, c: int):
        super().__init__()
        self.a = nn.Conv3d(c, c, 3, padding=1); self.b = nn.Conv3d(c, c, 3, padding=1)
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.g * self.b(Fn.silu(self.a(Fn.silu(x))))


class Down(nn.Module):
    """Anti-aliased downsampling by 2: the even binomial [1 3 3 1]/8 at stride 2 (replicate edges), then a 1³
    convolution to the next width. Output j is centred at input 2j + 1/2, so two of them centre cell i at voxel
    4i + 3/2 -- the centre of the 4 voxels the target averages, and a point the cube group maps to another cell centre
    (flips stay exact)."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        k = torch.tensor([1.0, 3.0, 3.0, 1.0]) / 8
        self.register_buffer("k", (k[:, None, None] * k[None, :, None] * k[None, None, :])[None, None])
        self.proj = nn.Conv3d(c_in, c_out, 1)

    def forward(self, x):
        c = x.shape[1]
        x = Fn.conv3d(Fn.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate"), self.k.expand(c, 1, 4, 4, 4), stride=2, groups=c)
        return self.proj(x)


class PopulationNet(nn.Module):
    """Image ``(B, 1, Z, Y, X)`` (grey levels scaled to 0-1) -> a dict of lattice outputs at stride 4 (``Z/4``...)."""

    def __init__(self, K: int = 4, widths=(16, 32, 64), embed: int = 8, n_kinds: int = 4, n_classes: int = 5):
        super().__init__()
        c0, c1, c2 = widths
        self.K, self.embed, self.n_kinds = K, embed, n_kinds
        self.stem = nn.Conv3d(1, c0, 3, padding=1)
        self.l0 = Block(c0)
        self.d1 = Down(c0, c1); self.l1 = nn.Sequential(Block(c1), Block(c1))
        self.d2 = Down(c1, c2); self.l2 = Block(c2)
        self.skip = Down(c1, c2)                                                                 # the stride-2 level, brought to the lattice
        self.mix = nn.Conv3d(2 * c2, c2, 1)
        self.per_slot = 6 + 1 + n_kinds + embed
        self.slots = nn.Conv3d(c2, K * self.per_slot, 1)
        self.scalars = nn.Conv3d(c2, 1 + n_classes + 1, 1)

    def forward(self, x: torch.Tensor) -> dict:
        h0 = self.l0(self.stem(x))
        h1 = self.l1(self.d1(h0))
        h2 = self.l2(self.d2(h1))
        h = Fn.silu(self.mix(torch.cat([h2, self.skip(h1)], 1)))
        B = x.shape[0]; cs = h.shape[2:]
        s = self.slots(h).view(B, self.K, self.per_slot, *cs)
        L = torch.zeros(B, self.K, 3, 3, *cs, device=x.device, dtype=s.dtype)
        for i, (r, c) in enumerate(TRIL):
            L[:, :, r, c] = s[:, :, i]
        M = torch.einsum("bkij...,bklj...->bkil...", L, L)                                       # B Bᵀ: symmetric PSD
        tr = (M[:, :, 0, 0] + M[:, :, 1, 1] + M[:, :, 2, 2]).clamp_min(1e-6)
        T = torch.stack([M[:, :, 0, 0], M[:, :, 1, 1], M[:, :, 2, 2], M[:, :, 0, 1], M[:, :, 0, 2], M[:, :, 1, 2]], 2) / tr[:, :, None]
        root_mass = Fn.softplus(s[:, :, 6])
        kind = s[:, :, 7:7 + self.n_kinds]
        e = Fn.normalize(s[:, :, 7 + self.n_kinds:], dim=2)
        sc = self.scalars(h)
        return {"slots": T, "mass": root_mass ** 2, "root_mass": root_mass, "kind": kind, "embed": e,
                "fraction_logit": sc[:, 0], "class_logits": sc[:, 1:-1], "log_size": sc[:, -1]}


def receptive_field(model: PopulationNet, size: int = 96) -> int:
    """The receptive field (voxels, along one axis) of a lattice output, measured: the extent of the input that changes
    the centre cell's output."""
    import copy
    m = copy.deepcopy(model).eval(); dev = next(m.parameters()).device
    for p in m.modules():
        if isinstance(p, Block):
            p.g.data.fill_(1.0)                                                                  # measure with the residual branches open
    x = torch.randn(1, 1, size, size, size, device=dev, requires_grad=True)
    out = m(x); c = size // 8
    (out["slots"][0, :, :, c, c, c].sum() + out["fraction_logit"][0, c, c, c]).backward()
    gnz = (x.grad[0, 0].abs() > 0).nonzero()
    return int(gnz[:, 0].max() - gnz[:, 0].min() + 1)
