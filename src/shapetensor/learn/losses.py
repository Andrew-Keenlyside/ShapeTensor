"""The population model's loss, and the cube-group augmentation that keeps targets exact.

Per lattice cell the predicted ``K`` slots are matched to the target's ``K`` slots by the permutation of least cost
(all ``K!`` = 24 permutations, exactly), the cost of a pair being:

* **shape**: the Frobenius distance² between the trace-1 tensors, weighted by the target's kind (a sheet's tensor
  moves half as far per degree as a line's, so planar kinds weigh 4×; the term is the same per degree for every kind)
  and by ``min(1, sqrt(m / m_full))`` (populations from ``m_full`` up count fully, fainter ones by their √mass);
* **mass**: ``(√m̂ - √m)²`` -- √ so a faint population's error is not a rounding error of a strong one's; an empty
  target slot asks for zero mass;
* **kind**: cross-entropy of the kind logits (present target slots only).

Each term is averaged over its own support (orientation and kind over present populations; mass and presence over
slots) -- averaged over all cells, the orientation term was diluted by the empty ones and the other terms swamped it.
Plus, per cell: the **mixture** tensor (the populations' mass-weighted mean tensor -- a permutation-free orientation
signal, which the trunk learns from first), the class shares (soft-label cross-entropy), the fraction (binary
cross-entropy against the soft target), the log size (L1 where defined), a **Hellinger** distance between the rendered
densities of the predicted and target slots on a direction grid (the chain's view: lobes, rings, flat), and an
**identity** term: a contrastive loss
between the embeddings of matched slots in neighbouring cells, the slot of the same layer the positive, batch-wide
negatives. Only cells ``margin`` cells from the crop's edge count (4 cells = 16 voxels: the receptive field reaches
±23.5 voxels, but 99% of a trained network's gradient mass lies within ±12).
"""
from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as Fn

from .targets import render_odf

WF = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
KIND_WEIGHT = torch.tensor([1.0, 4.0, 1.0, 4.0])          # line, sheet, blob, interface
WEIGHTS = {"shape": 1.0, "mass": 1.0, "presence": 0.5, "kind": 0.1, "classes": 0.5, "fraction": 0.5, "size": 0.1, "hellinger": 1.0, "identity": 0.1, "mixture": 2.0}
PRESENT = 0.005                    # a target slot with at least this mass is a population (presence 1); below, empty (presence 0)
TIE = 0.7                          # target slots whose masses are within this ratio may exchange their predicted slots


def _cells(x: torch.Tensor, margin: int) -> torch.Tensor:
    """Crop the lattice dims (the last three) by ``margin`` cells."""
    if margin <= 0:
        return x
    return x[..., margin:-margin, margin:-margin, margin:-margin]


def match(pred: dict, tgt: dict, m_full: float = 0.02):
    """The least-cost assignment of predicted to target slots per cell. Returns the per-cell matched cost ``(B, z, y,
    x)``, its parts, and ``perm (B, K, z, y, x)``: the predicted slot matched to each target slot."""
    S, rm, kl, pl = pred["slots"].float(), pred["root_mass"].float(), pred["kind"].float(), pred["presence_logit"].float()   # (B,K,6,c), (B,K,c), (B,K,nk,c), (B,K,c)
    St, mt, kt = tgt["slots"].float(), tgt["mass"].float(), tgt["kind"]
    B, K = rm.shape[:2]; dev = rm.device
    present = mt > PRESENT
    d = S[:, :, None] - St[:, None]                                                                       # (B, Kp, Kt, 6, c)
    dS = (d ** 2 * WF.to(dev).view(1, 1, 1, 6, 1, 1, 1)).sum(3)                                           # (B, Kp, Kt, c)
    kw = KIND_WEIGHT.to(dev)[kt.clamp_min(0)] * (mt / m_full).clamp(max=1).sqrt() * present               # (B, Kt, c)
    shape = dS * kw[:, None]
    mass = (rm[:, :, None] - mt.clamp_min(0).sqrt()[:, None]) ** 2
    logp = Fn.log_softmax(kl, 2)                                                                          # (B, Kp, nk, c)
    kind = -torch.gather(logp[:, :, None].expand(-1, -1, K, -1, -1, -1, -1), 3, kt.clamp_min(0)[:, None, :, None].expand(-1, K, -1, -1, -1, -1, -1)).squeeze(3) * present[:, None]
    pres = Fn.binary_cross_entropy_with_logits(pl[:, :, None].expand(-1, -1, K, *pl.shape[2:]), present[:, None].expand(-1, K, -1, *present.shape[2:]).float(), reduction="none")
    C = WEIGHTS["shape"] * shape + WEIGHTS["mass"] * mass + WEIGHTS["presence"] * pres + WEIGHTS["kind"] * kind   # (B, Kp, Kt, c)
    perms = torch.tensor(list(itertools.permutations(range(K))), device=dev)                               # (P, K): target slot k <- predicted perms[p, k]
    kt_idx = torch.arange(K, device=dev)
    costs = C[:, perms, kt_idx[None].expand_as(perms)].sum(2)                                             # (B, P, c)
    # rank-restricted: predicted slot k is the k-th heaviest population (the targets are sorted by mass); a permutation
    # may only exchange slots whose target masses tie (within TIE of each other) or are both empty -- free matching let
    # the assignment wander from cell to cell and step to step, and no slot learned orientation
    mk, mj = mt[:, :, None], mt[:, None, :]                                                               # (B, Kt, 1, c), (B, 1, Kt, c)
    ratio = torch.minimum(mk, mj) / torch.maximum(mk, mj).clamp_min(1e-9)
    pk, pj = present[:, :, None], present[:, None, :]
    allowed = torch.eye(K, dtype=torch.bool, device=dev).view(1, K, K, *([1] * (mt.dim() - 2))) | ((ratio >= TIE) & pk & pj) | (~pk & ~pj)   # (B, Kt, Kp, c)
    valid = allowed[:, kt_idx[None].expand_as(perms), perms].all(2)                                      # (B, P, c)
    costs = torch.where(valid, costs, torch.full_like(costs, 1e9))
    best = costs.argmin(1)                                                                                # (B, c)
    perm = perms[best].movedim(-1, 1)                                                                     # (B, K, c)
    sel = lambda X: torch.gather(X, 1, perm.unsqueeze(1)).squeeze(1)                                      # X (B, Kp, Kt, c) -> (B, Kt, c): X[b, perm[b, t], t]
    parts = {"shape": sel(shape).sum(1), "mass": sel(mass).sum(1), "presence": sel(pres).sum(1), "kind": sel(kind).sum(1)}
    return costs.min(1).values, parts, perm


def gather_slots(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Predicted per-slot outputs ``x (B, K, [C,] c...)`` reordered to the target slots by ``perm (B, K, c...)``."""
    if x.dim() == perm.dim():
        return torch.gather(x, 1, perm)
    return torch.gather(x, 1, perm[:, :, None].expand(-1, -1, x.shape[2], *perm.shape[2:]))


def identity_loss(embed: torch.Tensor, perm: torch.Tensor, tgt: dict, tau: float = 0.1, m_min: float = 0.02, axis: int = 0, n_neg: int = 8,
                  g: torch.Generator | None = None) -> torch.Tensor:
    """Contrastive identity: for each target slot with mass > ``m_min``, its matched embedding against the matched
    embeddings of the neighbouring cell (one step along ``axis``) -- the slot with the same layer the positive, the
    neighbour's other present slots and ``n_neg`` present slots drawn from anywhere in the batch (other cells, other
    samples, other layers) the negatives. The batch-wide negatives rule out a trivial solution (a constant embedding
    per slot index separates the slots of one cell but not populations across the batch)."""
    e = gather_slots(embed.float(), perm)                                                                  # (B, K, E, c)
    lay, m = tgt["layer"], tgt["mass"]
    B, K, E = e.shape[:3]
    dim = 3 + axis; a = 2 + axis
    e0, e1 = e.narrow(dim, 0, e.shape[dim] - 1), e.narrow(dim, 1, e.shape[dim] - 1)
    l0, l1 = lay.narrow(a, 0, lay.shape[a] - 1), lay.narrow(a, 1, lay.shape[a] - 1)
    m0, m1 = m.narrow(a, 0, m.shape[a] - 1), m.narrow(a, 1, m.shape[a] - 1)
    logits = torch.einsum("bie...,bje...->bij...", e0, e1) / tau                                          # (B, K, K, c)
    cand = (m1 > m_min)[:, None]
    logits = logits.masked_fill(~cand, -1e4)
    pos = (l0[:, :, None] == l1[:, None]) & cand & (l0[:, :, None] >= 0)
    has = pos.any(2) & (m0 > m_min)
    if not bool(has.any()):
        return embed.sum() * 0.0
    # batch-wide negatives: present slots of other layers or other samples
    flat_e = e.movedim(2, -1).reshape(-1, E); flat_l = lay.reshape(-1); flat_b = torch.arange(B, device=e.device).view(B, 1, *([1] * (lay.dim() - 2))).expand_as(lay).reshape(-1)
    ok = m.reshape(-1) > m_min
    pool_e, pool_l, pool_b = flat_e[ok], flat_l[ok], flat_b[ok]
    anchors = e0.movedim(2, -1)[has]                                                                       # (n, E)
    a_l = l0[has]; a_b = torch.arange(B, device=e.device).view(B, 1, *([1] * (l0.dim() - 2))).expand_as(l0)[has]
    idx = torch.randint(0, pool_e.shape[0], (anchors.shape[0], n_neg), generator=g).to(e.device)          # drawn on the CPU generator
    neg = pool_e[idx]; neg_ok = (pool_l[idx] != a_l[:, None]) | (pool_b[idx] != a_b[:, None])             # a different population
    neg_logits = torch.einsum("ne,nke->nk", anchors, neg) / tau
    neg_logits = neg_logits.masked_fill(~neg_ok, -1e4)
    loc = logits.movedim(2, -1)[has]                                                                       # (n, K)
    locpos = pos.movedim(2, -1)[has]
    lse = torch.logsumexp(torch.cat([loc, neg_logits], 1), 1)
    lpos = torch.logsumexp(loc.masked_fill(~locpos, -1e4), 1)
    return (lse - lpos).mean()


def loss(pred: dict, tgt: dict, grid: torch.Tensor, margin: int = 4, kappa: float = 20.0, axis: int = 0, weights: dict | None = None,
         g: torch.Generator | None = None) -> tuple[torch.Tensor, dict]:
    """The total loss and its parts (floats), over the cells ``margin`` from the crop's edge; ``weights`` overrides
    :data:`WEIGHTS` for the per-cell terms (a curriculum)."""
    P = {k: _cells(v, margin) for k, v in pred.items()}
    T = {k: _cells(v, margin) if torch.is_tensor(v) and v.dim() >= 4 else v for k, v in tgt.items()}
    cost, parts, perm = match(P, T)
    n_pop = (T["mass"].float() > PRESENT).sum().clamp_min(1)                                       # each term over its own support: orientation and kind
    n_slot = T["mass"].numel() / T["mass"].shape[1] * 1.0                                           # per present population; mass and presence per cell
    out = {"shape": parts["shape"].sum() / n_pop, "mass": parts["mass"].sum() / n_slot, "presence": parts["presence"].sum() / n_slot, "kind": parts["kind"].sum() / n_pop}
    logq = Fn.log_softmax(P["class_logits"].float(), 1)
    out["classes"] = -(T["classes"].float() * logq).sum(1).mean()
    out["fraction"] = Fn.binary_cross_entropy_with_logits(P["fraction_logit"].float(), T["fraction"].float())
    sm = T["size_mask"]
    out["size"] = (P["log_size"].float() - T["size"].float().clamp_min(0.3).log()).abs()[sm].mean() if bool(sm.any()) else cost.sum() * 0.0
    f_pred = torch.stack([render_odf(P["slots"][b].float(), P["mass"][b].float(), grid, kappa) for b in range(P["mass"].shape[0])])
    f_tgt = torch.stack([render_odf(T["slots"][b].float(), T["mass"][b].float(), grid, kappa) for b in range(T["mass"].shape[0])])
    out["hellinger"] = (((f_pred + 1e-6).sqrt() - (f_tgt + 1e-6).sqrt()) ** 2).mean()          # eps: a finite gradient at zero density
    out["identity"] = identity_loss(P["embed"], perm, T, axis=axis, g=g)
    # permutation-free orientation: the cell's mass-weighted mean tensor of its populations (no matching needed)
    mt = T["mass"].float(); St = T["slots"].float(); tot = mt.sum(1)
    mix = (St * mt[:, :, None]).sum(1) / tot.clamp_min(1e-9)[:, None]
    wmix = (tot / 0.05).clamp(max=1)
    out["mixture"] = ((((P["mixture"].float() - mix) ** 2) * WF.to(mix.device).view(1, 6, 1, 1, 1)).sum(1) * wmix).sum() / wmix.sum().clamp_min(1)
    W = {**WEIGHTS, **(weights or {})}
    total = sum(W[k] * v for k, v in out.items())
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
