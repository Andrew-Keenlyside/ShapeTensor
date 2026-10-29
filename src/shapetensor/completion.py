"""Population completion across crossings: each population carried along its own direction, from its arms on both
sides, into the region where its windows lost it (an overlap, a crossing).

After stage 2 (fibre sets) and a first stage 3 every set ``k`` has an activation layer ``A_k`` and a direction field;
the direction it is carried along, ``d_k(x)``, is that field extrapolated from the set's *pure* voxels (where it is alone:
inside an overlap its own field is read from crossing-confused dyads, near the bisector) -- :func:`steering`. For each voxel
the set is sampled along the line ``x + t d_k(x)``, ``t = 1..half`` forward and backward separately (trilinear):

* ``bridge_k = min(max forward A_k, max backward A_k)``: the set holds on *both* sides within ``half`` voxels along
  its own axis -- a crossing has the set's arms on both sides, a fascicle that ends or turns at the junction has them
  on one side only, so a free end is never extended (the fibre sets' concatenation rule, per voxel).

Three ways to use it (``mode``):

``"image"``       ``A'_k = max(A_k, bridge_k)``; directions from the readout's dyads of the block's features, as before.
``"image_cont"``  the same activation; in the bridged voxels each set's direction is its continuation ``d_k`` (the
                  arms'), not re-measured (tensor voting's stick field, straight).
``"shape"``       in shape space: the features averaged along ``d_k`` on each side, ``F^±_k``. Population ``k``'s fibres do
                  not change along ``d_k``, so its signature survives the average while a population crossing at an
                  angle averages towards background; the readout's mass within ``cone_deg`` of ``d_k`` on each side,
                  relative to its level where the set is present, confirms the set (``e_k``, the smaller side), and
                  ``A'_k = max(A_k, e_k bridge_k)``; each set's directions are read from the readout's dyads of its own
                  averaged features ``(F^+_k + F^-_k) / 2`` (a population-steered matched filter in feature space).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn


def _sampler(shape, device):
    Z, Y, X = shape
    base = torch.stack(torch.meshgrid(torch.arange(Z, device=device, dtype=torch.float32), torch.arange(Y, device=device, dtype=torch.float32),
                                      torch.arange(X, device=device, dtype=torch.float32), indexing="ij"))          # (3, Z, Y, X) in (z, y, x)
    scale = torch.tensor([2.0 / max(Z - 1, 1), 2.0 / max(Y - 1, 1), 2.0 / max(X - 1, 1)], device=device).view(3, 1, 1, 1)

    def sample(vol: torch.Tensor, d: torch.Tensor, t: float) -> torch.Tensor:
        """``vol (C, Z, Y, X)`` at ``x + t d(x)`` (zeros outside the block)."""
        p = (base + t * d) * scale - 1.0
        g = torch.stack([p[2], p[1], p[0]], -1)[None]                                                # grid_sample wants (x, y, z)
        return Fn.grid_sample(vol[None].float(), g, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
    return sample


def _cone_mass(readout, F: torch.Tensor, d: torch.Tensor, cos_cone: float) -> torch.Tensor:
    """The readout's probability within the cone around ``d (3, Z, Y, X)`` of averaged features ``F`` ``(Z, Y, X)``."""
    u = readout.grid                                                                                 # (n, 3) (z, y, x)
    out = torch.empty(tuple(F.shape[1:]), device=F.device)
    for z, shp, P in readout._chunks(F, None, True):                                                 # P (V, n)
        dz = d[:, z:z + shp[0]].reshape(3, -1)                                                       # (3, V)
        m = ((u @ dz).abs() >= cos_cone).T.float()                                                   # (V, n)
        out[z:z + shp[0]] = (P * m).sum(1).reshape(shp)
    return out


def _dyad_products(d: torch.Tensor) -> torch.Tensor:
    """``tr(M) I - M`` of the unit dyad ``M = d d^T`` in gradient-product order (see ``tensors.readout_products``)."""
    dz, dy, dx = d[0], d[1], d[2]
    M = torch.stack([dx * dx, dy * dy, dz * dz, dx * dy, dx * dz, dy * dz])
    tr = M[0] + M[1] + M[2]
    S = -M; S[0] += tr; S[1] += tr; S[2] += tr
    return S


def steering(L: torch.Tensor, dirs: torch.Tensor, k: int, sigma: float = 8.0, truncate: float = 3.0, dominance: float = 2.0) -> torch.Tensor:
    """Set ``k``'s direction extrapolated from where it is alone: its dyads on its *pure* voxels (present, ``L_k >= 1``,
    and at least ``dominance`` times every other set) spread by normalised convolution (Gaussian ``sigma``) and their
    principal axis taken. Inside an overlap the set's own first-pass direction is read from crossing-confused dyads
    (close to the bisector); this is its arms' direction instead. ``(3, Z, Y, X)``; where no pure voxel is within
    reach, the first-pass direction."""
    from .filters import gaussian
    from .tensors import direction_and_gap
    others = torch.cat([L[:k], L[k + 1:]]).max(0).values if L.shape[0] > 1 else torch.zeros_like(L[k])
    w = ((L[k] >= 1.0) & (L[k] >= dominance * others)).float() * L[k]
    d = dirs[k].float()
    D = _dyad_products(d) * w[None]
    num = torch.stack([gaussian(D[c], sigma, truncate=truncate) for c in range(6)])
    den = gaussian(w, sigma, truncate=truncate)
    reach = den > 1e-3 * float(den.max().clamp_min(1e-12))
    S = num / den.clamp_min(1e-12)[None]
    ds, _ = direction_and_gap(S)
    return torch.where(reach[None], ds, d)


def complete(cal, Fp: torch.Tensor, L: torch.Tensor, dirs: torch.Tensor, *, mode: str = "shape", half: int = 16, cone_deg: float = 25.0,
             base_products: torch.Tensor | None = None, steer: bool = True, steer_sigma: float = 8.0, reduce: str = "max"):
    """``Fp (C, Z, Y, X)`` the readers' averaged features, ``L (N, Z, Y, X)`` the sets' layers (multiples of the null),
    ``dirs (N, 3, Z, Y, X)`` their first-pass directions. ``steer``: sample along each set's direction extrapolated
    from its pure voxels (:func:`steering`) rather than its first-pass direction. ``reduce``: each side's evidence is
    the set's strongest activation within ``half`` voxels (``"max"``: its arm's level across the whole overlap) or the
    mean over them (``"mean"``: diluted by the overlap itself). Returns the completed layers
    ``(N, Z, Y, X)``, each set's direction products ``[(6, Z, Y, X)] * N`` for stage 3, and a report."""
    if mode not in ("image", "image_cont", "shape"):
        raise ValueError(f"mode must be 'image', 'image_cont' or 'shape', got {mode!r}")
    N = L.shape[0]
    sample = _sampler(tuple(L.shape[1:]), L.device)
    cos_cone = math.cos(math.radians(cone_deg))
    Lc, prods, rep = L.clone(), [], {"mode": mode, "half": half, "bridged_voxels": [], "evidence_ref": []}
    for k in range(N):
        d = steering(L, dirs, k, steer_sigma) if steer else dirs[k].float()
        af = torch.zeros_like(L[k]); ab = torch.zeros_like(L[k])
        Ff = torch.zeros_like(Fp, dtype=torch.float32) if mode == "shape" else None
        Fb = torch.zeros_like(Fp, dtype=torch.float32) if mode == "shape" else None
        for t in range(1, half + 1):
            sf, sb = sample(L[k][None], d, float(t))[0], sample(L[k][None], d, -float(t))[0]
            if reduce == "max":
                af = torch.maximum(af, sf); ab = torch.maximum(ab, sb)
            else:
                af += sf; ab += sb
            if mode == "shape":
                Ff += sample(Fp, d, float(t)); Fb += sample(Fp, d, -float(t))
        if reduce != "max":
            af /= half; ab /= half
        bridge = torch.minimum(af, ab)
        if mode == "shape":
            Ff /= half; Fb /= half
            ef, eb = _cone_mass(cal.readout, Ff, d, cos_cone), _cone_mass(cal.readout, Fb, d, cos_cone)
            e = torch.minimum(ef, eb)
            present = L[k] >= 1.0
            ref = float(e[present].median()) if bool(present.any()) else float(e.max())
            gate = (e / max(ref, 1e-6)).clamp(0, 1)
            new = torch.maximum(L[k], gate * bridge)
            Fl = 0.5 * (Ff + Fb); del Ff, Fb
            prods.append(cal.readout.dyad_products(Fl, prepped=True)); del Fl
            rep["evidence_ref"].append(round(ref, 4))
        else:
            new = torch.maximum(L[k], bridge)
            if mode == "image_cont":
                bridged = new > L[k] + 1e-6
                P = base_products.clone() if base_products is not None else cal.readout.dyad_products(Fp, prepped=True)
                P[:, bridged] = _dyad_products(d)[:, bridged] * P[:3, bridged].sum(0, keepdim=True).clamp_min(1e-6) / 2.0
                prods.append(P)
            else:
                prods.append(base_products if base_products is not None else cal.readout.dyad_products(Fp, prepped=True))
        rep["bridged_voxels"].append(int((new > L[k] + 1e-6).sum()))
        Lc[k] = new
    return Lc, prods, rep
