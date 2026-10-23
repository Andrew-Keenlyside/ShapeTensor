"""The population model: image -> populations per lattice cell.

A small, strictly local, fully convolutional network whose output *is* the stride-4 lattice:

* three levels (16 / 32 / 64 channels at strides 1 / 2 / 4), residual blocks of two 3³ convolutions (one at stride 1,
  two at stride 2, one at stride 4) with a learned gain (0.25 at the start), a nonlinearity after every projection (no normalisation layer: nothing depends on the chunk's statistics, so a volume run in
  chunks with a halo of the receptive field equals one pass);
* max-blur downsampling (a centred 3³ max, then an even binomial kernel at stride 2: the nonlinearity orientation
  needs, anti-aliased, cells centred where the target's are), no transposed convolution anywhere: no checkerboard or
  lattice-phase artefacts; the stride-2 level is blurred and subsampled once more and concatenated at the
  lattice, so the heads see both scales;
* heads at the lattice (1³ convolutions): ``K`` (6) population slots -- a shape tensor (``B Bᵀ / tr``,
  symmetric positive semi-definite with trace 1, from 6 numbers), a mass (``softplus``, read as √mass), a presence
  logit (is the slot a population at all: a soft mass never reaches zero, so emptiness is decided here), kind logits
  (line / sheet / blob / interface) and an identity embedding (unit length) -- and the cell's scalars: the shape
  fraction (logit), the class shares (logits over line / sheet / blob / interface / isotropic), the log size and the
  mixture tensor (the populations' mass-weighted mean tensor: a permutation-free orientation signal for training).

The receptive field is :func:`receptive_field` voxels (54 for the default; a chunked run needs a halo of half of it);
the target's support is ±16.5 voxels (σ_w = 5 truncated at 3σ, pooled by 4).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

TRIL = ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2))


class Block(nn.Module):
    """``x + g * conv(act(conv(act(x))))`` with a learned gain ``g`` (initialised at ``g0`` = 0.25): stable without
    normalisation, and open from the start (at ``g`` = 0 the branches stayed shut through a whole smoke run)."""

    def __init__(self, c: int, g0: float = 0.25):
        super().__init__()
        self.a = nn.Conv3d(c, c, 3, padding=1); self.b = nn.Conv3d(c, c, 3, padding=1)
        self.g = nn.Parameter(torch.full((1,), float(g0)))

    def forward(self, x):
        return x + self.g * self.b(Fn.silu(self.a(Fn.silu(x))))


class Down(nn.Module):
    """Anti-aliased downsampling by 2: the even binomial [1 3 3 1]/8 at stride 2 (replicate edges), then a 1³
    convolution to the next width. Output j is centred at input 2j + 1/2, so two of them centre cell i at voxel
    4i + 3/2 -- the centre of the 4 voxels the target averages, and a point the cube group maps to another cell centre
    (flips stay exact). With ``max_first`` (the default), a centred 3³ max (stride 1) comes before the blur: max-blur
    pooling. The max is the nonlinearity orientation needs -- with a linear (blur-only) downsampling the network did
    not learn orientation at all (held-out error ~60°, chance), with it 5° on both seeds tried -- and the blur keeps
    the anti-aliasing and the exact lattice."""

    def __init__(self, c_in: int, c_out: int, max_first: bool = True):
        super().__init__()
        k = torch.tensor([1.0, 3.0, 3.0, 1.0]) / 8
        self.register_buffer("k", (k[:, None, None] * k[None, :, None] * k[None, None, :])[None, None])
        self.proj = nn.Conv3d(c_in, c_out, 1)
        self.max_first = max_first

    def forward(self, x):
        c = x.shape[1]
        if self.max_first:
            x = Fn.max_pool3d(Fn.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate"), 3, stride=1)
        x = Fn.conv3d(Fn.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate"), self.k.expand(c, 1, 4, 4, 4), stride=2, groups=c)
        return self.proj(x)


class PopulationNet(nn.Module):
    """Image ``(B, 1, Z, Y, X)`` (grey levels scaled to 0-1) -> a dict of lattice outputs at stride 4 (``Z/4``...)."""

    def __init__(self, K: int = 6, widths=(16, 32, 64), embed: int = 8, n_kinds: int = 4, n_classes: int = 5, sh_lmax: int = 0,
                 lcn: float = 0.0, query: bool = False, desc_lmax: int = 0, axis_head: bool = False, head: str = "slots", J: int = 4,
                 context_level: bool = False, voxel_cond: bool = False, identity_dim: int = 0, points: bool = False, nd: int = 5,
                 free_sh: bool = False):
        super().__init__()
        self.sh_lmax, self.lcn, self.has_query, self.desc_lmax, self.axis_head = sh_lmax, lcn, bool(query), desc_lmax, bool(axis_head)
        self.head, self.J, self.nd, self.free_sh = head, J, (nd if head == "lobes" else 0), bool(free_sh)
        self.context_level, self.voxel_cond, self.identity_dim, self.has_points = bool(context_level), bool(voxel_cond), int(identity_dim), bool(points)
        c0, c1, c2 = widths
        self.K, self.embed, self.n_kinds = K, embed, n_kinds
        self.stem = nn.Conv3d(2 if lcn else 1, c0, 3, padding=1)                                  # with ``lcn``: the image and its local contrast
        self.l0 = Block(c0)
        self.d1 = Down(c0, c1); self.l1 = nn.Sequential(Block(c1), Block(c1))
        self.d2 = Down(c1, c2); self.l2 = Block(c2)
        self.skip = Down(c1, c2)                                                                 # the stride-2 level, brought to the lattice
        if self.context_level:                                                                   # a stride-8 level: twice the context, upsampled back
            self.d3 = Down(c2, c2); self.l3 = nn.Sequential(Block(c2), Block(c2))
        self.mix = nn.Conv3d((3 if self.context_level else 2) * c2, c2, 1)
        if self.voxel_cond:                                                                      # the voxel size (log2 um, and whether known): FiLM
            self.film = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 2 * c2))
            nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
        if head == "lobes":                                                                      # J lobes per cell (axis, mass, kappa, spare, descriptors)
            from .sh import n_coef as _nc
            # the ablation: the same trunk and targets with free harmonics (density to order sh_lmax, descriptors to desc_lmax)
            self.lobes = nn.Conv3d(c2, (_nc(sh_lmax or 8) + self.nd * _nc(desc_lmax or 4)) if self.free_sh else J * (6 + self.nd), 1)
            self.frac = nn.Conv3d(c2, 1, 1)
            # an auxiliary linear density head (free harmonics): a convex, permutation-free orientation signal for the
            # trunk from the first step (the mixture head's role in the slot model), the gate's signal, and a free-SH
            # readout of the same trunk to compare with the lobes
            self.aux = nn.Conv3d(c2, _nc(sh_lmax or 8), 1)
            if self.has_points:
                self.points = nn.Conv3d(c2, 2, 1); nn.init.constant_(self.points.bias, -4.0)     # near zero at the start: points are rare
            if self.identity_dim:                                                                # an embedding at stride 2 from both levels
                self.id_head = nn.Sequential(nn.Conv3d(c1 + c2, c1, 1), nn.SiLU(), nn.Conv3d(c1, self.identity_dim, 1))
            if self.has_query:
                from .query import QueryDecoder
                self.query_dec = QueryDecoder(c0, c1, c2, J * (6 + self.nd))
            return
        self.per_slot = 6 + 1 + 1 + n_kinds + embed                                             # tensor, root mass, presence logit, kind logits, embedding
        self.slots = nn.Conv3d(c2, K * self.per_slot, 1)
        self.scalars = nn.Conv3d(c2, 1 + n_classes + 1 + 6, 1)                                     # fraction, classes, log size, the mixture tensor
        if self.axis_head:                                                                         # a committed axis per slot and for the mixture
            self.axes = nn.Conv3d(c2, 3 * (K + 1), 1)
        if sh_lmax:                                                                                # the populations' density in even SH (permutation-free)
            from .sh import n_coef
            self.sh = nn.Conv3d(c2, n_coef(sh_lmax) + (3 * n_coef(desc_lmax) if desc_lmax else 0), 1)   # + descriptor densities
        if self.has_query:                                                                         # the density at any point and description scale
            from .query import QueryDecoder
            from .sh import n_coef
            assert sh_lmax, "the query decoder outputs SH: set sh_lmax"
            self.query_dec = QueryDecoder(c0, c1, c2, n_coef(sh_lmax) + (3 * n_coef(desc_lmax) if desc_lmax else 0))

    @classmethod
    def from_args(cls, args: dict | None) -> "PopulationNet":
        """The network a checkpoint's ``args`` describe (widths, SH order)."""
        args = args or {}
        return cls(widths=tuple(args.get("widths", (16, 32, 64))), sh_lmax=int(args.get("sh_lmax", 0) or 0), lcn=float(args.get("lcn", 0) or 0),
                   query=bool(args.get("query", False)), desc_lmax=int(args.get("desc_lmax", 0) or 0),
                   axis_head=bool(args.get("axis_head", False)), head=args.get("head", "slots") or "slots", J=int(args.get("lobes_J", 4) or 4),
                   context_level=bool(args.get("context_level", False)), voxel_cond=bool(args.get("voxel_cond", False)),
                   identity_dim=int(args.get("identity_dim", 0) or 0), points=bool(args.get("points", False)), free_sh=bool(args.get("free_sh", False)))

    def local_contrast(self, x: torch.Tensor) -> torch.Tensor:
        """``(x - mean) / sd`` in a Gaussian window of sigma ``lcn`` voxels (sd floored at 0.01 of the 0-1 range), /3:
        the same structure at any texture amplitude, background level or modality's contrast scale."""
        from ..filters import gaussian
        xf = x.float(); mu = gaussian(xf, self.lcn); var = gaussian((xf - mu) ** 2, self.lcn)
        return ((xf - mu) / (var + 1e-4).sqrt() / 3).to(x.dtype)

    def features(self, x: torch.Tensor):
        """The three feature levels: stride 1, stride 2 and the stride-4 lattice."""
        if self.lcn:
            x = torch.cat([x, self.local_contrast(x)], 1)
        h0 = self.l0(Fn.silu(self.stem(x)))                                                      # nonlinear from the first layer: local energies
        h1 = self.l1(Fn.silu(self.d1(h0)))                                                       # (squared-gradient-like features) need no open branch
        h2 = self.l2(Fn.silu(self.d2(h1)))
        cat = [h2, Fn.silu(self.skip(h1))]
        if self.context_level:                                                                   # stride 8 and back (trilinear: no checkerboard)
            h3 = self.l3(Fn.silu(self.d3(h2)))
            cat.append(Fn.interpolate(h3, size=h2.shape[2:], mode="trilinear", align_corners=False))
        h = Fn.silu(self.mix(torch.cat(cat, 1)))
        return h0, h1, h

    def condition(self, h: torch.Tensor, voxel: torch.Tensor | None) -> torch.Tensor:
        """FiLM of the lattice features by the voxel size: ``voxel (B,)`` log2 µm, NaN where unknown."""
        if not self.voxel_cond:
            return h
        B = h.shape[0]
        v = torch.full((B,), float("nan"), device=h.device) if voxel is None else voxel.float().to(h.device)
        known = torch.isfinite(v)
        z = torch.stack([torch.where(known, (v - 3.0) / 3.0, torch.zeros_like(v)), known.float()], 1)
        gb = self.film(z).to(h.dtype); c = h.shape[1]
        return h * (1 + gb[:, :c].view(B, c, 1, 1, 1)) + gb[:, c:].view(B, c, 1, 1, 1)

    def forward(self, x: torch.Tensor, points: torch.Tensor | None = None, sigma: torch.Tensor | None = None,
                context: torch.Tensor | None = None, voxel: torch.Tensor | None = None) -> dict:
        """The lattice outputs; with ``points (B, Q, 3)`` (input voxel coordinates) and ``sigma (B, Q)`` (voxels) and a
        query decoder, also ``q_sh (B, Q, n_coef)``: the populations' density at those points and scales. ``voxel``
        (B,): log2 of the voxel size in µm (NaN: unknown) for a voxel-conditioned model."""
        h0, h1, h = self.features(x)
        h = self.condition(h, voxel)
        B = x.shape[0]; cs = h.shape[2:]
        if self.head == "lobes":
            out = {"lobes_raw": self.lobes(h), "fraction_logit": self.frac(h)[:, 0], "aux_sh": self.aux(h)}
            if self.has_points:
                out["points"] = Fn.softplus(self.points(h))
            if self.identity_dim:
                hu = Fn.interpolate(h, size=h1.shape[2:], mode="trilinear", align_corners=False)
                out["identity"] = self.id_head(torch.cat([h1, hu], 1))
            if points is not None and self.has_query:
                out["q_lobes_raw"] = self.query_dec((h0, h1, h), points.float(), sigma.float(), None if context is None else context.float())
            return out
        s = self.slots(h).view(B, self.K, self.per_slot, *cs)
        L = torch.zeros(B, self.K, 3, 3, *cs, device=x.device, dtype=s.dtype)
        for i, (r, c) in enumerate(TRIL):
            L[:, :, r, c] = s[:, :, i]
        M = torch.einsum("bkij...,bklj...->bkil...", L, L)                                       # B Bᵀ: symmetric PSD
        tr = (M[:, :, 0, 0] + M[:, :, 1, 1] + M[:, :, 2, 2]).clamp_min(1e-6)
        T = torch.stack([M[:, :, 0, 0], M[:, :, 1, 1], M[:, :, 2, 2], M[:, :, 0, 1], M[:, :, 0, 2], M[:, :, 1, 2]], 2) / tr[:, :, None]
        root_mass = Fn.softplus(s[:, :, 6])
        presence = s[:, :, 7]
        kind = s[:, :, 8:8 + self.n_kinds]
        e = Fn.normalize(s[:, :, 8 + self.n_kinds:], dim=2)
        sc = self.scalars(h)
        n_cls = sc.shape[1] - 8
        out = {"slots": T, "mass": root_mass ** 2, "root_mass": root_mass, "presence_logit": presence, "kind": kind, "embed": e,
               "fraction_logit": sc[:, 0], "class_logits": sc[:, 1:1 + n_cls], "log_size": sc[:, 1 + n_cls], "mixture": sc[:, 2 + n_cls:]}
        if self.axis_head:
            a = self.axes(h).view(B, self.K + 1, 3, *cs)
            out["axis"], out["mix_axis"] = a[:, :self.K], a[:, self.K]
        if self.sh_lmax:
            out["sh"] = self.sh(h)
        if points is not None and self.has_query:
            out["q_sh"] = self.query_dec((h0, h1, h), points.float(), sigma.float(), None if context is None else context.float())
        return out


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
    head = out["slots"][0, :, :, c, c, c].sum() if "slots" in out else out["lobes_raw"][0, :, c, c, c].sum()
    (head + out["fraction_logit"][0, c, c, c]).backward()
    gnz = (x.grad[0, 0].abs() > 0).nonzero()
    return int(gnz[:, 0].max() - gnz[:, 0].min() + 1)
