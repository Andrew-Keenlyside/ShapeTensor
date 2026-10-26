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

import math
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
        if self.max_first:
            x = Fn.max_pool3d(Fn.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate"), 3, stride=1)
        # the binomial [1 3 3 1] / 8 at stride 2 as three box-2 averages (two at stride 1, the last at stride 2): the same
        # numbers as the depthwise 4^3 convolution (k, kept for old checkpoints), without cuDNN's slow depthwise 3-D path
        x = Fn.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate")
        x = Fn.avg_pool3d(Fn.avg_pool3d(x, 2, stride=1), 2, stride=1)
        x = Fn.avg_pool3d(x, 2, stride=2)
        return self.proj(x)


def poincare(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """A vector into the open Poincaré ball: ``tanh(|x|) x / |x|`` (radius below 1 - eps)."""
    n = x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return x / n * torch.tanh(n) * (1 - eps)


def poincare_dist(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Hyperbolic distance between points of the Poincaré ball (last dim the coordinates)."""
    du = (u - v).pow(2).sum(-1); a = (1 - u.pow(2).sum(-1)).clamp_min(1e-6); b = (1 - v.pow(2).sum(-1)).clamp_min(1e-6)
    return torch.acosh((1 + 2 * du / (a * b)).clamp_min(1 + 1e-7))


def window_mean(r: torch.Tensor, points: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """The Gaussian-window mean of a map ``r (B, Z, Y, X)`` at ``points (B, Q, 3)`` (voxels) and per-query ``sigma (B, Q)``
    (voxels): zero beyond the map (as the entity targets count nothing beyond the crop), blurred at sigma steps of sqrt 2
    over the batch's range, sampled trilinearly and interpolated in log sigma. Differentiable: ``(B, Q)``."""
    B, Z, Y, X = r.shape
    s_lo = float(sigma.min().clamp_min(0.5)); s_hi = float(sigma.max().clamp_min(s_lo * 1.0001))
    n_lv = max(2, int(math.ceil(math.log(s_hi / s_lo) / math.log(math.sqrt(2)))) + 1)
    lv = torch.exp(torch.linspace(math.log(s_lo), math.log(s_hi), n_lv, device=r.device))
    size = torch.tensor([Z, Y, X], device=r.device, dtype=torch.float32)
    g = (points / (size - 1) * 2 - 1).flip(-1).view(B, -1, 1, 1, 3)                                # grid_sample's (x, y, z), align_corners
    S = []
    for s_ in lv.tolist():
        rad = int(math.ceil(3 * s_)); t = torch.arange(-rad, rad + 1.0, device=r.device); k = torch.exp(-0.5 * (t / s_) ** 2); k = k / k.sum()
        v = r[:, None]
        for ax in range(3):
            sh = [1, 1, 1, 1, 1]; sh[2 + ax] = -1
            v = Fn.conv3d(v, k.view(*sh).to(v.dtype), padding=tuple(rad if a == ax else 0 for a in range(3)))
        S.append(Fn.grid_sample(v, g, mode="bilinear", align_corners=True).view(B, -1))
    S = torch.stack(S)                                                                              # (n_lv, B, Q)
    t = (torch.log(sigma.clamp_min(s_lo)) - math.log(s_lo)) / max(math.log(s_hi / s_lo), 1e-6) * (n_lv - 1)
    i0 = t.floor().long().clamp(0, n_lv - 2); a = (t - i0.float()).clamp(0, 1)
    return (1 - a) * S.gather(0, i0[None])[0] + a * S.gather(0, (i0 + 1)[None])[0]


class Feats(tuple):
    """The trunk's levels ``(h0, h1, h)`` and, for the oriented probe, ``scal``: the image's rotation-invariant scalar
    fields (:meth:`PopulationNet.scalar_fields`) -- a tuple, so ``h0, h1, h = feats`` keeps working."""
    scal = None


def with_scal(feats, scal) -> Feats:
    f = Feats(tuple(feats)); f.scal = scal; return f


class EntityDecoder(nn.Module):
    """The entity field at any point x, scale sigma and direction u: from the trunk's three levels at x (trilinear over
    the lattice corners, as :class:`~shapetensor.learn.query.QueryDecoder`), log sigma and the visible context, a point
    code; with the even SH basis of u (order ``lmax``), per direction: density ``f >= 0``, a Poincaré embedding ``z``
    (the owning entity's place in the tree), its Bingham frame ``R`` (6-D rotation) and concentrations ``kappa``
    (kappa1 >= kappa2 >= 0), and its 5 descriptors (log size, relative wall, lumen, flattening, angularity)."""

    def __init__(self, c0: int, c1: int, c2: int, emb: int = 16, hidden: int = 256, lmax: int = 8, unc: bool = False, mass_head: bool = False,
                 dens_sh: bool = False, dens_mlp: bool = False, z_vote: bool = False, dir_sh: int = 0, ray: bool = False,
                 kappa_log: bool = False, oprobe: bool = False, dens_log: bool = False, oprobe_only: bool = False):
        super().__init__()
        from .sh import n_coef
        self.emb, self.lmax, self.unc = emb, lmax, bool(unc)
        # the vote's concentrations in log space (5 tanh(y / 5): 0.007-150): ``softplus(y) * 10`` collapses -- most
        # owned directions are the isotropic background's (kappa 0), the softplus saturates there and the lobes' own
        # directions cannot pull it back; with kappa ~1 votes 90 deg apart are the same vote to the readout
        self.kappa_log = bool(kappa_log)
        # the density as exp of the output (its log), not its softplus: softplus is linear above ~2, so a sharp lobe on a
        # floor had to be drawn in the output itself (angular detail the decoder under-draws: tips came out blunted);
        # in log space a lobe on a floor is a smooth bump
        self.dens_log = bool(dens_log)
        # the ray probe: per (point, direction), the features along a line through the point in that direction -- short
        # (full resolution, +-K0 voxels) and long (the lattice, +-K2 x 4 voxels) -- folded axially (u, -u: sum and |diff|):
        # where two packed bundles cross, the core is featureless but a ray along either bundle runs into it outside
        self.ray = bool(ray); self.K0, self.K2 = 3, 5
        if self.ray:
            self.ray0 = nn.Linear(c0 * self.K0 * 2, 64); self.ray2 = nn.Linear(c2 * self.K2 * 2, 64)
        rin = 128 if self.ray else 0
        # the oriented probe: per (point, direction), the image's rotation-invariant scalar fields (local contrast at three
        # scales) sampled on a cylinder along the direction -- axial positions, rings around the axis reduced to their
        # angular-frequency magnitudes (the same whatever the roll about u), folded axially: what the material looks like
        # *along u*, the same input at every orientation (the trunk's features are not rotation-aware: each orientation
        # was learned apart, a crossing fitted in ~100 steps at one orientation took thousands over all of them)
        # ``oprobe_only``: the density from the probe alone (rotation-equivariant by construction; the other outputs keep
        # the point code)
        self.oprobe = bool(oprobe) or bool(oprobe_only); self.oprobe_only = bool(oprobe_only)
        self.o_t = (2.0, 4.0, 6.0); self.o_r = (1.5, 3.0); self.o_m = 8; self.o_c = 3
        if self.oprobe:
            nf = self.o_c * (1 + 3 * len(self.o_r)) * (1 + 2 * len(self.o_t))
            self.o_net = nn.Sequential(nn.Linear(nf, 128), nn.SiLU(), nn.Linear(128, 64))
            self.odens = (nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1)) if self.oprobe_only else nn.Linear(64, 1))   # a direct
            last = self.odens[-1] if self.oprobe_only else self.odens                                                                # path to the density
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        rin += 64 if self.oprobe else 0
        self.mass = nn.Linear(hidden, 1) if mass_head else None                              # where: the material's mass at (x, sigma)
        # the density's shape as harmonics of the point code: log-density c(p) . Y(u), a direct path from where to which
        # way (with only the concatenated MLP, a flat start is a saddle: the shape loss's gradient on the point code
        # cancels over the sphere until the MLP has found p x u interactions on its own)
        if dens_sh and dens_mlp:                                                                  # (the harmonics a nonlinear function of the point
            self.dens = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_coef(lmax)))   # code: a lobe's
        else:                                                                                     # coefficients are polynomials of its axis)
            self.dens = nn.Linear(hidden, n_coef(lmax)) if dens_sh else None
        if self.dens is not None:
            last = self.dens[-1] if isinstance(self.dens, nn.Sequential) else self.dens
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        # the embedding sees the vote's axes (axial tensors of the mode and the normal, the vote detached): populations
        # that lie different ways separate in the tree from the start of what the vote knows (zero init)
        self.zv = nn.Linear(12, emb) if z_vote else None
        if self.zv is not None:
            nn.init.zeros_(self.zv.weight); nn.init.zeros_(self.zv.bias)
        # the other per-direction outputs (embedding, vote frame, concentrations, descriptors) get the same direct path as
        # the density: harmonics of the point code up to order ``dir_sh``, added to the MLP's output -- an owner that
        # switches between a crossing's lobes is a function of p x u the concatenated MLP finds slowly (the density's saddle)
        self.n_out = 1 + emb + 6 + 2 + 5 + int(self.unc); self.dir_sh = int(dir_sh)
        self.dsh = nn.Linear(hidden, (self.n_out - 1) * n_coef(self.dir_sh)) if self.dir_sh > 0 else None
        if self.dsh is not None:
            nn.init.zeros_(self.dsh.weight); nn.init.zeros_(self.dsh.bias)
        self.point = nn.Sequential(nn.Linear(c2 + c1 + c0 + 3 + 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.dir = nn.Sequential(nn.Linear(hidden + n_coef(lmax) + rin, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1 + emb + 6 + 2 + 5 + int(self.unc)))
        nn.init.constant_(self.dir[-1].bias[0], -2.0)                                             # little density at the start

    def encode(self, feats, points: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        from .query import _sample
        h0, h1, h = feats
        B, Q = points.shape[:2]
        f0 = torch.stack([_sample(h0[b], points[b]) for b in range(B)])
        f1 = torch.stack([_sample(h1[b], (points[b] - 0.5) / 2) for b in range(B)])
        u = (points - 1.5) / 4; size = torch.tensor(h.shape[-3:], device=points.device); base = u.floor(); hf = h.flatten(2)
        ls = torch.log(sigma.clamp_min(1e-3) / 3.0)[..., None]
        if context is None:
            size_v = torch.tensor(h0.shape[-3:], device=points.device, dtype=points.dtype)
            context = torch.minimum(points, size_v - 1 - points).amin(-1)
        lc = torch.log(context.clamp(1.0, 64.0) / 32.0)[..., None]
        out = 0; wsum = 0
        for o in range(8):
            off = torch.tensor([(o >> 2) & 1, (o >> 1) & 1, o & 1], device=points.device, dtype=points.dtype)
            idx = (base + off).clamp(min=torch.zeros(3, device=points.device), max=(size - 1).to(points.dtype))
            rel = u - idx; wgt = (1 - rel.abs()).clamp_min(0).prod(-1)
            lin = ((idx[..., 0] * size[1] + idx[..., 1]) * size[2] + idx[..., 2]).long()
            fc = torch.gather(hf, 2, lin[:, None].expand(-1, hf.shape[1], -1))
            z = torch.cat([fc, f1, f0], 1).transpose(1, 2)
            out = out + wgt[..., None] * self.point(torch.cat([z.to(rel.dtype), rel, ls, lc], -1)); wsum = wsum + wgt
        return out / wsum.clamp_min(1e-6)[..., None]                                                # corners clamped at the faces: weights renormalised

    def ray_pairs(self, feats, pts: torch.Tensor, u: torch.Tensor, b: int = 0, chunk: int = 16384) -> torch.Tensor:
        """The ray features ``(N, 128)`` of volume ``b`` of ``feats`` at pairs of points ``pts (N, 3)`` (input voxels) and unit
        directions ``u (N, 3)``."""
        h0, _, h = feats
        out = []
        for a in range(0, pts.shape[0], chunk):
            p_, u_ = pts[a:a + chunk].float(), u[a:a + chunk].float(); n = p_.shape[0]
            parts = []
            for F, K, step, lat, lin in ((h0[b:b + 1], self.K0, 1.0, False, self.ray0), (h[b:b + 1], self.K2, 4.0, True, self.ray2)):
                t = torch.arange(1, K + 1, device=p_.device, dtype=p_.dtype) * step                    # (K,)
                pos = torch.cat([p_[:, None] + t[None, :, None] * u_[:, None], p_[:, None] - t[None, :, None] * u_[:, None]], 1)   # (n, 2K, 3)
                if lat:
                    pos = (pos - 1.5) / 4.0                                                          # lattice coordinates (encode's)
                size = torch.tensor(F.shape[-3:], device=p_.device, dtype=p_.dtype)
                g = (pos / (size - 1).clamp_min(1) * 2 - 1).flip(-1).reshape(1, -1, 1, 1, 3)
                smp = Fn.grid_sample(F.float(), g, mode="bilinear", padding_mode="border", align_corners=True)   # (1, C, n 2K, 1, 1)
                smp = smp[0, :, :, 0, 0].T.reshape(n, 2, K, -1)                                      # (n, +/-, K, C)
                fold = torch.cat([smp[:, 0] + smp[:, 1], (smp[:, 0] - smp[:, 1]).abs()], 1)       # (n, 2K, C): axial
                parts.append(Fn.silu(lin(fold.reshape(n, -1).to(lin.weight.dtype))))
            out.append(torch.cat(parts, -1))
        return torch.cat(out)

    def oprobe_feats(self, feats, pts: torch.Tensor, u: torch.Tensor, b: int = 0, chunk: int = 16384) -> torch.Tensor:
        """The oriented probe's code ``(N, 64)`` for item ``b`` at points ``pts (N, 3)`` and directions ``u (N, 3)``: the
        scalar fields at the axis positions 0, +-t (``o_t``) and on rings of ``o_m`` points at radii ``o_r`` around the
        axis; per ring its mean and the magnitudes of its first two angular frequencies; +-t folded (sum, |difference|)."""
        S = feats.scal[b:b + 1].float(); C = S.shape[1]
        size = torch.tensor(S.shape[-3:], device=pts.device, dtype=torch.float32)
        nT = len(self.o_t); T = torch.tensor((0.0,) + tuple(self.o_t) + tuple(-t for t in self.o_t), device=pts.device)
        ang = torch.arange(self.o_m, device=pts.device, dtype=torch.float32) * (2 * math.pi / self.o_m)
        ring = torch.cat([torch.zeros(1, 2, device=pts.device)] + [torch.stack([r * torch.cos(ang), r * torch.sin(ang)], -1) for r in self.o_r])   # (1 + R M, 2)
        R_ = len(self.o_r); out = []
        for a in range(0, pts.shape[0], chunk):
            p_, u_ = pts[a:a + chunk].float(), Fn.normalize(u[a:a + chunk].float(), dim=-1); n = p_.shape[0]
            ref = torch.where(u_[:, :1].abs() < 0.9, torch.tensor([1.0, 0.0, 0.0], device=p_.device).expand(n, 3), torch.tensor([0.0, 1.0, 0.0], device=p_.device).expand(n, 3))
            e1 = Fn.normalize(torch.cross(u_, ref, dim=-1), dim=-1); e2 = torch.cross(u_, e1, dim=-1)
            pos = (p_[:, None, None] + T[None, :, None, None] * u_[:, None, None]
                   + ring[None, None, :, :1] * e1[:, None, None] + ring[None, None, :, 1:] * e2[:, None, None])   # (n, nt, 1 + R M, 3)
            g = (pos / (size - 1).clamp_min(1) * 2 - 1).flip(-1).reshape(1, -1, 1, 1, 3)
            v = Fn.grid_sample(S, g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, :, 0, 0]
            v = v.view(C, n, T.numel(), -1).permute(1, 0, 2, 3)                                    # (n, C, nt, 1 + R M)
            c = torch.fft.rfft(v[..., 1:].reshape(n, C, T.numel(), R_, self.o_m), dim=-1) / self.o_m
            fe = torch.cat([v[..., :1], torch.stack([c[..., 0].real, c[..., 1].abs(), c[..., 2].abs()], -1).flatten(-2)], -1)   # (n, C, nt, 1 + 3R)
            fold = torch.cat([fe[:, :, :1], fe[:, :, 1:1 + nT] + fe[:, :, 1 + nT:], (fe[:, :, 1:1 + nT] - fe[:, :, 1 + nT:]).abs()], 2)
            out.append(Fn.silu(self.o_net(fold.flatten(1).to(self.o_net[0].weight.dtype))))
        return torch.cat(out)

    def forward(self, feats, points: torch.Tensor, sigma: torch.Tensor, dirs: torch.Tensor, context: torch.Tensor | None = None) -> dict:
        """``points (B, Q, 3)`` input voxels, ``sigma (B, Q)`` voxels, ``dirs (B, Q, D, 3)`` unit (z, y, x)."""
        from .bingham import rot6d_to_matrix
        from .sh import real_sh
        p = self.encode(feats, points, sigma, context)
        D = dirs.shape[2]
        Y = real_sh(dirs.reshape(-1, 3).float(), self.lmax).view(*dirs.shape[:3], -1)
        r = None
        if getattr(self, "ray", False):
            B, Q = points.shape[:2]
            r = torch.stack([self.ray_pairs(feats, points[b][:, None].expand(-1, D, -1).reshape(-1, 3), dirs[b].reshape(-1, 3), b).view(Q, D, -1)
                             for b in range(B)])
        o = None
        if getattr(self, "oprobe", False):
            B, Q = points.shape[:2]
            o = torch.stack([self.oprobe_feats(feats, points[b][:, None].expand(-1, D, -1).reshape(-1, 3), dirs[b].reshape(-1, 3), b).view(Q, D, -1)
                             for b in range(B)])
        out = self.decode(p[:, :, None].expand(-1, -1, D, -1), Y, r, o)
        if self.mass is not None:
            out["m"] = self.mass_of(p)
        return out

    def reset_kappa(self):
        """The concentration outputs restarted at kappa ~1 (log space), e.g. after loading a ``softplus`` checkpoint."""
        e = self.emb; last = self.dir[-1]
        with torch.no_grad():
            last.weight[7 + e:9 + e] *= 0.01; last.bias[7 + e:9 + e] = 0.0
            if self.dsh is not None:
                from .sh import n_coef
                nc = n_coef(self.dir_sh); w = self.dsh.weight.view(self.n_out - 1, nc, -1); b = self.dsh.bias.view(self.n_out - 1, nc)
                w[6 + e:8 + e] = 0.0; b[6 + e:8 + e] = 0.0

    def mass_of(self, p: torch.Tensor) -> torch.Tensor:
        """The mass (visible volume fraction of oriented material in the window) from point encodings ``p (..., hidden)``:
        a scalar per point, apart from the directional density (which then carries only the angular shape)."""
        return Fn.softplus(self.mass(p)[..., 0].float())

    def decode(self, p: torch.Tensor, Y: torch.Tensor, r: torch.Tensor | None = None, o: torch.Tensor | None = None) -> dict:
        """The outputs from point encodings ``p (..., hidden)`` and direction bases ``Y (..., n_coef)`` (same leading
        shape): any set of (point, direction) pairs -- inference evaluates only where the total ODF has density."""
        from .bingham import rot6d_to_matrix
        parts = [p, Y.to(p.dtype)]
        if getattr(self, "ray", False):
            parts.append(torch.zeros(*p.shape[:-1], 128, device=p.device, dtype=p.dtype) if r is None else r.to(p.dtype))
        if getattr(self, "oprobe", False):
            o = torch.zeros(*p.shape[:-1], 64, device=p.device, dtype=p.dtype) if o is None else o.to(p.dtype)
            parts.append(o)
        y = self.dir(torch.cat(parts, -1)).float()
        if getattr(self, "dsh", None) is not None:
            from .sh import n_coef
            nc = n_coef(self.dir_sh)
            c = self.dsh(p).float().view(*p.shape[:-1], self.n_out - 1, nc)
            y = torch.cat([y[..., :1], y[..., 1:] + (c * Y[..., None, :nc].float()).sum(-1)], -1)
        e = self.emb
        k = torch.exp(5.0 * torch.tanh(y[..., 7 + e:9 + e] / 5.0)) if getattr(self, "kappa_log", False) else Fn.softplus(y[..., 7 + e:9 + e]) * 10.0
        k = torch.stack([k.amax(-1), k.amin(-1)], -1)
        y0 = y[..., 0] + (self.dens(p).float() * Y.float()).sum(-1) if self.dens is not None else y[..., 0]
        if getattr(self, "oprobe_only", False):
            y0 = self.odens(o).float()[..., 0]
        elif getattr(self, "oprobe", False):
            y0 = y0 + self.odens(o).float()[..., 0]
        R = rot6d_to_matrix(y[..., 1 + e:7 + e]); zr = y[..., 1:1 + e]
        if getattr(self, "zv", None) is not None:
            a, c = R[..., :, 0].detach(), R[..., :, 2].detach()
            ax6 = lambda v: torch.stack([v[..., 0] ** 2, v[..., 1] ** 2, v[..., 2] ** 2, v[..., 0] * v[..., 1], v[..., 0] * v[..., 2], v[..., 1] * v[..., 2]], -1)
            zr = zr + self.zv(torch.cat([ax6(c), ax6(a)], -1) * 3.0)
        f = torch.exp(y0.clamp(-20.0, 12.0)) if getattr(self, "dens_log", False) else Fn.softplus(y0)
        out = {"f": f, "z": poincare(zr), "R": R, "kappa": k, "desc": y[..., 9 + e:14 + e]}
        if self.unc:                                                                              # the vote's uncertainty: log variance of its error
            out["log_s2"] = y[..., 14 + e].clamp(-8.0, 8.0)
        return out


class PopulationNet(nn.Module):
    """Image ``(B, 1, Z, Y, X)`` (grey levels scaled to 0-1) -> a dict of lattice outputs at stride 4 (``Z/4``...)."""

    def __init__(self, K: int = 6, widths=(16, 32, 64), embed: int = 8, n_kinds: int = 4, n_classes: int = 5, sh_lmax: int = 0,
                 lcn: float = 0.0, query: bool = False, desc_lmax: int = 0, axis_head: bool = False, head: str = "slots", J: int = 4,
                 context_level: bool = False, voxel_cond: bool = False, identity_dim: int = 0, points: bool = False, nd: int = 5,
                 free_sh: bool = False, dir_lmax: int = 8, vote_unc: bool = False, mass_head: bool = False, region_head: bool = False,
                 dens_sh: bool = False, dens_mlp: bool = False, mass_region: bool = False, region_dec: bool = False, z_vote: bool = False,
                 dir_sh: int = 0, ray: bool = False, region_density: bool = False, kappa_log: bool = False, oprobe: bool = False,
                 dens_log: bool = False, oprobe_only: bool = False, bg_head: bool = False):
        super().__init__()
        self.region_density = bool(region_density)                                              # the region map a density (sum of the
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
        if head == "entities":                                                                   # the entity field (notes/plan_entity_field_model.md)
            from .sh import n_coef as _nc
            self.total = nn.Conv3d(c2, _nc(sh_lmax or 8), 1)                                     # the overall ODF per lattice cell (free SH)
            self.frac = nn.Conv3d(c2, 1, 1)
            if self.has_points:
                self.points = nn.Conv3d(c2, 2, 1); nn.init.constant_(self.points.bias, -4.0)
            self.ent = EntityDecoder(c0, c1, c2, emb=max(int(embed), 2), hidden=256, lmax=int(dir_lmax), unc=bool(vote_unc),
                                     mass_head=bool(mass_head) and not mass_region, dens_sh=bool(dens_sh), dens_mlp=bool(dens_mlp),
                                     z_vote=bool(z_vote), dir_sh=int(dir_sh), ray=bool(ray), kappa_log=bool(kappa_log),
                                     oprobe=bool(oprobe), dens_log=bool(dens_log), oprobe_only=bool(oprobe_only))
            self.mass_region = bool(mass_region) and bool(region_head)                            # the mass: the region map in the query's window
            self.region = nn.Sequential(nn.Conv3d(c0 + c1 + c2, 32, 1), nn.SiLU(), nn.Conv3d(32, 1, 1)) if region_head else None
            # the background's occupancy (the isotropic remainder) at every voxel: the density's floor is background plus
            # isotropic objects (blobs, cells); this map tells them apart by place, as the floor cannot by direction
            self.bgh = nn.Sequential(nn.Conv3d(c0 + c1 + c2, 32, 1), nn.SiLU(), nn.Conv3d(32, 1, 1)) if bg_head else None
            self.region_dec = None
            if region_head and region_dec:                                                      # a small decoder: lattice -> stride 2 -> full,
                self.region_dec = nn.ModuleDict({                                                # 3x3x3 convolutions at each (shapes, not only
                    "r2": nn.Conv3d(c2, 32, 1), "r1": nn.Conv3d(c1, 32, 1), "c1": nn.Conv3d(64, 32, 3, padding=1),   # a per-voxel
                    "r0": nn.Conv3d(c0, 16, 1), "c0": nn.Conv3d(48, 32, 3, padding=1), "out": nn.Conv3d(32, 1, 1)})   # readout)
                nn.init.zeros_(self.region_dec["out"].weight); nn.init.zeros_(self.region_dec["out"].bias)   # starts as the plain head
            return
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
                   identity_dim=int(args.get("identity_dim", 0) or 0), points=bool(args.get("points", False)), free_sh=bool(args.get("free_sh", False)),
                   embed=int(args.get("embed", 8) or 8), dir_lmax=int(args.get("dir_lmax", 8) or 8), vote_unc=bool(args.get("vote_unc", False)),
                   mass_head=bool(args.get("mass_head", False)), region_head=bool(args.get("region_head", False)),
                   dens_sh=bool(args.get("dens_sh", False)), dens_mlp=bool(args.get("dens_mlp", False)),
                   mass_region=bool(args.get("mass_region", False)), region_dec=bool(args.get("region_dec", False)),
                   z_vote=bool(args.get("z_vote", False)), dir_sh=int(args.get("dir_sh", 0) or 0), ray=bool(args.get("ray", False)),
                   region_density=bool(args.get("region_density", False)), kappa_log=bool(args.get("kappa_log", False)),
                   oprobe=bool(args.get("oprobe", False)), dens_log=bool(args.get("dens_log", False)),
                   oprobe_only=bool(args.get("oprobe_only", False)), bg_head=bool(args.get("bg_head", False))) \
            ._with_args(args)

    def _with_args(self, args: dict) -> "PopulationNet":
        self._args = dict(args); return self

    def scalar_fields(self, x: torch.Tensor) -> torch.Tensor:
        """The image's rotation-invariant scalar fields for the oriented probe: its local contrast (the image itself
        without ``lcn``) and that blurred at sigma 1 and 2 (isotropic windows: rotating the image rotates the fields)."""
        from ..filters import gaussian
        lc = (self.local_contrast(x) if self.lcn else x).float()
        return torch.cat([lc, gaussian(lc, 1.0), gaussian(lc, 2.0)], 1)

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

    def region_value(self, logit: torch.Tensor) -> torch.Tensor:
        """The region map from its logits: a probability (the union of the populations), or with ``region_density`` a
        density (the sum of the populations' occupancies: two crossing populations count twice)."""
        return Fn.softplus(logit.float()) if getattr(self, "region_density", False) else torch.sigmoid(logit.float())

    def region_map(self, h0, h1, h) -> torch.Tensor:
        """Logits ``(B, Z, Y, X)`` of the visible occupancy of oriented material (the populations' union) at every voxel,
        from the three levels brought to full resolution (trilinear: the stride-2 and lattice centres land where they lie)
        -- a segmentation-like objective that teaches the trunk regions (inside / outside), not only their edges."""
        sz = h0.shape[2:]
        up = lambda t, size=sz: Fn.interpolate(t, size=size, mode="trilinear", align_corners=False)
        if getattr(self, "region_dec", None) is not None:
            d = self.region_dec; a = Fn.silu(d["c1"](torch.cat([up(Fn.silu(d["r2"](h)), h1.shape[2:]), Fn.silu(d["r1"](h1))], 1)))
            a = Fn.silu(d["c0"](torch.cat([up(a), Fn.silu(d["r0"](h0))], 1)))
            return (d["out"](a) + self.region(torch.cat([h0, up(h1), up(h)], 1)))[:, 0]
        return self.region(torch.cat([h0, up(h1), up(h)], 1))[:, 0]

    def bg_map(self, h0, h1, h) -> torch.Tensor:
        """The background's occupancy ``(B, Z, Y, X)`` in 0-1 (``bg_head``)."""
        sz = h0.shape[2:]
        up = lambda t: Fn.interpolate(t, size=sz, mode="trilinear", align_corners=False)
        return torch.sigmoid(self.bgh(torch.cat([h0, up(h1), up(h)], 1))[:, 0].float())

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
                context: torch.Tensor | None = None, voxel: torch.Tensor | None = None, e_points: torch.Tensor | None = None,
                e_sigma: torch.Tensor | None = None, e_dirs: torch.Tensor | None = None) -> dict:
        """The lattice outputs; with ``points (B, Q, 3)`` (input voxel coordinates) and ``sigma (B, Q)`` (voxels) and a
        query decoder, also ``q_sh (B, Q, n_coef)``: the populations' density at those points and scales. ``voxel``
        (B,): log2 of the voxel size in µm (NaN: unknown) for a voxel-conditioned model."""
        h0, h1, h = self.features(x)
        h = self.condition(h, voxel)
        B = x.shape[0]; cs = h.shape[2:]
        if self.head == "entities":
            out = {"total_sh": self.total(h), "fraction_logit": self.frac(h)[:, 0]}
            if self.has_points:
                out["points"] = Fn.softplus(self.points(h))
            if e_points is not None:
                fe = with_scal((h0, h1, h), self.scalar_fields(x)) if getattr(self.ent, "oprobe", False) else (h0, h1, h)
                out["ent"] = self.ent(fe, e_points.float(), e_sigma.float(), e_dirs, None if context is None else context.float())
            if self.region is not None:                                                         # the population region, every voxel
                out["region_logit"] = self.region_map(h0, h1, h); out["region"] = self.region_value(out["region_logit"])
                if e_points is not None and getattr(self, "mass_region", False):                 # where, once: the mass is the region's
                    out["ent"]["m"] = window_mean(out["region"], e_points.float(), e_sigma.float())   # window mean
            if getattr(self, "bgh", None) is not None:
                out["bg"] = self.bg_map(h0, h1, h)
            out["_feats"] = (h0, h1, h)
            return out
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


def receptive_field(model: PopulationNet, size: int | None = None) -> int:
    """The receptive field (voxels, along one axis) of a lattice output, measured: the extent of the input that changes
    the centre cell's output. Measured on a narrow copy of the architecture (the extent depends on the topology, not the
    widths), on an input grown until the extent no longer reaches its faces (a 96³ input clipped a 146-voxel field)."""
    import copy
    if size is None:
        args = getattr(model, "_args", None)
        probe = PopulationNet.from_args({**args, "widths": (4, 4, 4)}).cpu() if args else model
        s_ = 128
        while True:
            rf = receptive_field(probe, s_)
            if rf <= s_ - 16 or s_ >= 512:
                return rf
            s_ *= 2
    m = copy.deepcopy(model).eval(); dev = next(m.parameters()).device
    for p in m.modules():
        if isinstance(p, Block):
            p.g.data.fill_(1.0)                                                                  # measure with the residual branches open
    x = torch.randn(1, 1, size, size, size, device=dev, requires_grad=True)
    out = m(x); c = size // 8
    head = out["slots"][0, :, :, c, c, c].sum() if "slots" in out else out["total_sh"][0, :, c, c, c].sum() if "total_sh" in out else out["lobes_raw"][0, :, c, c, c].sum()
    (head + out["fraction_logit"][0, c, c, c]).backward()
    gnz = (x.grad[0, 0].abs() > 0).nonzero()
    return int(gnz[:, 0].max() - gnz[:, 0].min() + 1)
