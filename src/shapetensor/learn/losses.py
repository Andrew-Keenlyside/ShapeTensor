"""The population model's loss, and the cube-group augmentation that keeps targets exact.

Per lattice cell the predicted ``K`` slots are matched to the target's ``K`` slots by the permutation of least cost
(all ``K!`` = 24 permutations, exactly), the cost of a pair being:

* **shape**: the Frobenius distance² between the trace-1 tensors, weighted by the target's kind (a sheet's tensor
  moves half as far per degree as a line's, so planar kinds weigh 4×; the term is the same per degree for every kind)
  and by ``min(1, sqrt(m / m_full))`` (populations from ``m_full`` up count fully, fainter ones by their √mass);
* **mass**: ``(√m̂ - √m)²`` -- √ so a faint population's error is not a rounding error of a strong one's; an empty
  target slot asks for zero mass;
* **kind**: cross-entropy of the kind logits (present target slots only).

Plus, per cell: the class shares (soft-label cross-entropy), the fraction (binary cross-entropy against the soft
target), the log size (L1 where defined), a **Hellinger** distance between the rendered densities of the predicted and
target slots on a direction grid (the chain's view: lobes, rings, flat), and an **identity** term: a contrastive loss
between the embeddings of matched slots in neighbouring cells, the slot of the same layer the positive. Only cells whose
receptive field lies inside the crop (``margin`` cells from the edge) count.
"""
from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as Fn

from .targets import render_odf

WF = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
KIND_WEIGHT = torch.tensor([1.0, 4.0, 1.0, 4.0])          # line, sheet, blob, interface
WEIGHTS = {"shape": 1.0, "mass": 1.0, "kind": 0.1, "classes": 0.5, "fraction": 0.5, "size": 0.1, "hellinger": 1.0, "identity": 0.1}


def _cells(x: torch.Tensor, margin: int) -> torch.Tensor:
    """Crop the lattice dims (the last three) by ``margin`` cells."""
    if margin <= 0:
        return x
    return x[..., margin:-margin, margin:-margin, margin:-margin]


def match(pred: dict, tgt: dict, m_full: float = 0.02):
    """The least-cost assignment of predicted to target slots per cell. Returns the per-cell matched cost ``(B, z, y,
    x)``, its parts, and ``perm (B, K, z, y, x)``: the predicted slot matched to each target slot."""
    S, rm, kl = pred["slots"].float(), pred["root_mass"].float(), pred["kind"].float()                  # (B,K,6,c), (B,K,c), (B,K,nk,c)
    St, mt, kt = tgt["slots"].float(), tgt["mass"].float(), tgt["kind"]
    B, K = rm.shape[:2]; dev = rm.device
    present = mt > 0
    d = S[:, :, None] - St[:, None]                                                                       # (B, Kp, Kt, 6, c)
    dS = (d ** 2 * WF.to(dev).view(1, 1, 1, 6, 1, 1, 1)).sum(3)                                           # (B, Kp, Kt, c)
    kw = KIND_WEIGHT.to(dev)[kt.clamp_min(0)] * (mt / m_full).clamp(max=1).sqrt() * present               # (B, Kt, c)
    shape = dS * kw[:, None]
    mass = (rm[:, :, None] - mt.clamp_min(0).sqrt()[:, None]) ** 2
    logp = Fn.log_softmax(kl, 2)                                                                          # (B, Kp, nk, c)
    kind = -torch.gather(logp[:, :, None].expand(-1, -1, K, -1, -1, -1, -1), 3, kt.clamp_min(0)[:, None, :, None].expand(-1, K, -1, -1, -1, -1, -1)).squeeze(3) * present[:, None]
    C = WEIGHTS["shape"] * shape + WEIGHTS["mass"] * mass + WEIGHTS["kind"] * kind                         # (B, Kp, Kt, c)
    perms = torch.tensor(list(itertools.permutations(range(K))), device=dev)                               # (P, K): target slot k <- predicted perms[p, k]
    kt_idx = torch.arange(K, device=dev)
    costs = C[:, perms, kt_idx[None].expand_as(perms)].sum(2)                                             # (B, P, c)
    best = costs.argmin(1)                                                                                # (B, c)
    perm = perms[best].movedim(-1, 1)                                                                     # (B, K, c)
    sel = lambda X: torch.gather(X, 1, perm.unsqueeze(1)).squeeze(1)                                      # X (B, Kp, Kt, c) -> (B, Kt, c): X[b, perm[b, t], t]
    parts = {"shape": sel(shape).sum(1), "mass": sel(mass).sum(1), "kind": sel(kind).sum(1)}
    return costs.min(1).values, parts, perm


def gather_slots(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Predicted per-slot outputs ``x (B, K, [C,] c...)`` reordered to the target slots by ``perm (B, K, c...)``."""
    if x.dim() == perm.dim():
        return torch.gather(x, 1, perm)
    return torch.gather(x, 1, perm[:, :, None].expand(-1, -1, x.shape[2], *perm.shape[2:]))


def identity_loss(embed: torch.Tensor, perm: torch.Tensor, tgt: dict, tau: float = 0.1, m_min: float = 0.02, axis: int = 0) -> torch.Tensor:
    """Contrastive identity: for each target slot with mass > ``m_min``, its matched embedding against the matched
    embeddings of the neighbouring cell (one step along ``axis``) -- the slot with the same layer the positive, the
    others (present ones) the negatives. Only cells where the neighbour has the layer and at least two slots count."""
    e = gather_slots(embed.float(), perm)                                                                  # (B, K, E, c)
    lay, m = tgt["layer"], tgt["mass"]
    dim = 3 + axis
    e0, e1 = e.narrow(dim, 0, e.shape[dim] - 1), e.narrow(dim, 1, e.shape[dim] - 1)
    a = 2 + axis
    l0, l1 = lay.narrow(a, 0, lay.shape[a] - 1), lay.narrow(a, 1, lay.shape[a] - 1)
    m0, m1 = m.narrow(a, 0, m.shape[a] - 1), m.narrow(a, 1, m.shape[a] - 1)
    logits = torch.einsum("bie...,bje...->bij...", e0, e1) / tau                                          # (B, K, K, c)
    cand = (m1 > m_min)[:, None]                                                                          # (B, 1, K, c)
    logits = logits.masked_fill(~cand, -1e4)
    pos = (l0[:, :, None] == l1[:, None]) & cand & (l0[:, :, None] >= 0)
    has = pos.any(2) & (m0 > m_min) & (cand.sum(2) >= 2)                                                   # (B, K, c)
    if not bool(has.any()):
        return embed.sum() * 0.0
    lse = torch.logsumexp(logits, 2)
    lpos = torch.logsumexp(logits.masked_fill(~pos, -1e4), 2)
    return (lse - lpos)[has].mean()


def loss(pred: dict, tgt: dict, grid: torch.Tensor, margin: int = 3, kappa: float = 20.0, axis: int = 0) -> tuple[torch.Tensor, dict]:
    """The total loss and its parts (floats), over the cells ``margin`` from the crop's edge."""
    P = {k: _cells(v, margin) for k, v in pred.items()}
    T = {k: _cells(v, margin) if torch.is_tensor(v) and v.dim() >= 4 else v for k, v in tgt.items()}
    cost, parts, perm = match(P, T)
    out = {"shape": parts["shape"].mean(), "mass": parts["mass"].mean(), "kind": parts["kind"].mean()}
    logq = Fn.log_softmax(P["class_logits"].float(), 1)
    out["classes"] = -(T["classes"].float() * logq).sum(1).mean()
    out["fraction"] = Fn.binary_cross_entropy_with_logits(P["fraction_logit"].float(), T["fraction"].float())
    sm = T["size_mask"]
    out["size"] = (P["log_size"].float() - T["size"].float().clamp_min(0.3).log()).abs()[sm].mean() if bool(sm.any()) else cost.sum() * 0.0
    f_pred = torch.stack([render_odf(P["slots"][b].float(), P["mass"][b].float(), grid, kappa) for b in range(P["mass"].shape[0])])
    f_tgt = torch.stack([render_odf(T["slots"][b].float(), T["mass"][b].float(), grid, kappa) for b in range(T["mass"].shape[0])])
    out["hellinger"] = ((f_pred.clamp_min(0).sqrt() - f_tgt.clamp_min(0).sqrt()) ** 2).mean()
    out["identity"] = identity_loss(P["embed"], perm, T, axis=axis)
    total = sum(WEIGHTS[k] * v for k, v in out.items())
    return total, {k: float(v) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------------------------
# cube-group augmentation: exact on the lattice, tensors rotated with the volume
# ---------------------------------------------------------------------------------------------------------------

def cube_transform(g: torch.Generator | None = None, device="cpu"):
    """A random element of the cube group (48: axis permutations × flips): ``(perm, flips)``."""
    perm = torch.randperm(3, generator=g, device=device).tolist()
    flips = [bool(b) for b in (torch.rand(3, generator=g, device=device) < 0.5).tolist()]
    return perm, flips


def apply_cube(x: torch.Tensor, perm, flips, tensor6: bool = False) -> torch.Tensor:
    """Apply a cube-group element to a field whose last three dims are (z, y, x): new axis j = old axis ``perm[j]``,
    reversed where ``flips[j]``. With ``tensor6`` the field's dim -4 holds 6 tensor components, transformed as
    ``S'_jk = s_j s_k S_perm[j] perm[k]``."""
    n = x.dim()
    x = x.permute(*range(n - 3), *(n - 3 + p for p in perm))
    fd = [n - 3 + j for j in range(3) if flips[j]]
    if fd:
        x = x.flip(fd)
    if tensor6:
        from .generator import SYM
        sgn = [-1.0 if f else 1.0 for f in flips]
        comps = []
        idx = {(i, j): c for c, (i, j) in enumerate(SYM)}
        idx.update({(j, i): c for (i, j), c in list(idx.items())})
        for j, k in SYM:
            comps.append(x.select(n - 4, idx[(perm[j], perm[k])]) * (sgn[j] * sgn[k]))
        x = torch.stack(comps, n - 4)
    return x


def augment(image: torch.Tensor, tgt: dict, g: torch.Generator | None = None):
    """The same random cube-group element applied to an image ``(..., Z, Y, X)`` and its lattice target."""
    perm, flips = cube_transform(g)                                                             # drawn on the CPU
    out = {}
    for k, v in tgt.items():
        if torch.is_tensor(v) and v.dim() >= 3:
            out[k] = apply_cube(v, perm, flips, tensor6=(k == "slots"))
        else:
            out[k] = v
    return apply_cube(image, perm, flips), out
