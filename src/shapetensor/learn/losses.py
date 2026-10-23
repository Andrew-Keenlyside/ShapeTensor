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

from .generator import eig3
from .targets import render_odf

WF = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
KIND_WEIGHT = torch.tensor([1.0, 4.0, 1.0, 4.0])          # line, sheet, blob, interface
WEIGHTS = {"shape": 1.0, "mass": 1.0, "presence": 0.5, "kind": 0.1, "classes": 0.5, "fraction": 0.5, "size": 0.1, "hellinger": 1.0, "identity": 0.1, "mixture": 2.0,
           "sh": 2.0, "query": 2.0, "descriptors": 0.5}
PRESENT = 0.005                    # a target slot with at least this mass is a population (presence 1); below, empty (presence 0)
TIE = 0.7                          # target slots whose masses are within this ratio may exchange their predicted slots
PRESENCE_BAND = None               # None: presence trained on every slot; (lo, hi): slots whose target mass lies between
                                   # lo and hi (faint populations near the threshold, a guess) do not train presence
DISTINCT = None                    # None: rank order among all present slots; a float: kept only among slots whose
                                   # trace-1 eigenvalue spectra lie within this distance (ideal tensors: line-blob 0.82,
                                   # line-sheet 0.71, sheet-blob 0.41, two lines 0; window pooling shortens them)


DESC_NC = 0                        # coefficients per descriptor density (0: none), set by train()
SHAPE_LOSS = "frobenius"           # "frobenius" | "devcos": the tensor cost of slots and mixture (see dev_cost)
SH_LOSS = "batchnorm"              # "batchnorm" (per-order error over the batch's energy) | "decoupled" (see sh_loss_decoupled)
_SH_EMA: dict = {}                 # running per-cell energy per order (decoupled loss)
SH_LMAX_ON = None                  # decoupled loss: highest order trained (None: all); raised by train() once l = 2 aligns


def sh_loss_decoupled(pred: torch.Tensor, tgt: torch.Tensor, key: str = "sh", lmax_on: int | None = None) -> torch.Tensor:
    """The density loss with magnitude and direction apart, per order: order 0 (mass) its own squared error; each order
    l >= 2: (|p_l| - |t_l|)^2 + 2 |t_l|^2 (1 - cos_l) -- the angular part weighted by the target's energy, so a flat
    (round) prediction gains nothing -- each over a running mean of the per-cell target energy of that order (fixed
    across batches: no down-weighting of line-rich batches). Orders above ``lmax_on`` are left out (l = 2 first).
    ``pred``, ``tgt``: ``(B, c, ...)`` with orders on dim 1."""
    from .sh import bands
    lmax = 2 * (len(bands_for(pred.shape[1])) - 1)
    total, n = 0.0, 0
    for i, b in enumerate(bands(lmax)):
        l = 2 * i
        if lmax_on is not None and l > lmax_on:
            break
        p_, t_ = pred[:, b], tgt[:, b]
        e = float((t_ ** 2).sum(1).mean())
        k = (key, l); _SH_EMA[k] = e if k not in _SH_EMA else 0.99 * _SH_EMA[k] + 0.01 * e
        norm = max(_SH_EMA[k], 1e-6)
        if l == 0:
            term = ((p_ - t_) ** 2).sum(1)
        else:
            pn, tn = p_.norm(dim=1), t_.norm(dim=1)
            cos = (p_ * t_).sum(1) / (pn * tn).clamp_min(1e-9)
            term = (pn - tn) ** 2 + 2 * tn ** 2 * (1 - cos)
        total = total + term.mean() / norm; n += 1
    return total / n


def dev_cost(P: torch.Tensor, T: torch.Tensor, axis: int) -> torch.Tensor:
    """A tensor cost that does not reward hedging: with D = S - I/3 (the deviator; trace-1 tensors), the squared
    Frobenius distance is (|Dp| - |Dt|)^2 + 2 |Dp| |Dt| (1 - cos): its angular part is weighted by the PREDICTED
    anisotropy, so a rounder prediction pays less for a wrong axis -- under uncertainty the minimum is round. Here
    the angular part is weighted by the TARGET's anisotropy instead: (|Dp| - |Dt|)^2 + 2 |Dt|^2 (1 - cos) -- the
    magnitude and the orientation of the shape learned separately; any kind (lines, sheets, solids), no categories.
    ``P``, ``T``: tensors with 6 components on ``axis`` (broadcastable)."""
    iso = torch.tensor([1, 1, 1, 0, 0, 0.0], device=P.device) / 3
    wf = WF.to(P.device)
    shp = [1] * P.dim(); shp[axis] = 6
    Dp, Dt = P - iso.view(shp), T - iso.view(shp)
    w = wf.view(shp)
    npn = (Dp * Dp * w).sum(axis).clamp_min(1e-12).sqrt(); ntn = (Dt * Dt * w).sum(axis).clamp_min(1e-12).sqrt()
    cos = (Dp * Dt * w).sum(axis) / (npn * ntn)
    return (npn - ntn) ** 2 + 2 * ntn ** 2 * (1 - cos)


def desc_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Descriptor densities ``(B, 3 n, ...)``: each descriptor's squared error over its target energy (floored at a
    tenth of the mean over the three plus the energy of mass 0.05 with descriptor 1 in every cell), averaged."""
    n = pred.shape[1] // 3; e = [(tgt[:, j * n:(j + 1) * n] ** 2).sum() for j in range(3)]
    fl = 0.1 * sum(e) / 3 + tgt[:, 0].numel() * (0.05 * math.sqrt(4 * math.pi)) ** 2               # + mass 0.05 with descriptor 1 everywhere
    return sum(((pred[:, j * n:(j + 1) * n] - tgt[:, j * n:(j + 1) * n]) ** 2).sum() / (e[j] + fl) for j in range(3)) / 3


def sh_loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """SH coefficients ``(B, n_coef, ...)``: each even order's squared error over the batch's energy in that order
    (floored: a tenth of the order-0 energy plus that of mass 0.05 everywhere), averaged over the orders."""
    from .sh import bands
    lmax = 2 * (len(bands_for(pred.shape[1])) - 1)
    floor = 0.1 * (tgt[:, 0] ** 2).sum() + tgt[:, 0].numel() * (0.05 * math.sqrt(4 * math.pi)) ** 2
    d2 = (pred - tgt) ** 2
    return sum(d2[:, b].sum() / ((tgt[:, b] ** 2).sum() + floor) for b in bands(lmax)) / len(bands(lmax))


def bands_for(n: int) -> list:
    """The even SH orders held by ``n`` coefficients."""
    out, l, c = [], 0, 0
    while c < n:
        out.append(l); c += 2 * l + 1; l += 2
    return out


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
    if SHAPE_LOSS == "devcos":
        dS = dev_cost(S[:, :, None], St[:, None], 3)                                                      # (B, Kp, Kt, c)
    else:
        d = S[:, :, None] - St[:, None]                                                                   # (B, Kp, Kt, 6, c)
        dS = (d ** 2 * WF.to(dev).view(1, 1, 1, 6, 1, 1, 1)).sum(3)                                       # (B, Kp, Kt, c)
    kw = KIND_WEIGHT.to(dev)[kt.clamp_min(0)] * (mt / m_full).clamp(max=1).sqrt() * present               # (B, Kt, c)
    shape = dS * kw[:, None]
    mass = (rm[:, :, None] - mt.clamp_min(0).sqrt()[:, None]) ** 2
    logp = Fn.log_softmax(kl, 2)                                                                          # (B, Kp, nk, c)
    kind = -torch.gather(logp[:, :, None].expand(-1, -1, K, -1, -1, -1, -1), 3, kt.clamp_min(0)[:, None, :, None].expand(-1, K, -1, -1, -1, -1, -1)).squeeze(3) * present[:, None]
    pres = Fn.binary_cross_entropy_with_logits(pl[:, :, None].expand(-1, -1, K, *pl.shape[2:]), present[:, None].expand(-1, K, -1, *present.shape[2:]).float(), reduction="none")
    if PRESENCE_BAND is not None:
        lo, hi = PRESENCE_BAND
        pres = pres * ((mt <= lo) | (mt >= hi))[:, None].float()
    C = WEIGHTS["shape"] * shape + WEIGHTS["mass"] * mass + WEIGHTS["presence"] * pres + WEIGHTS["kind"] * kind   # (B, Kp, Kt, c)
    perms = torch.tensor(list(itertools.permutations(range(K))), device=dev)                               # (P, K): target slot k <- predicted perms[p, k]
    kt_idx = torch.arange(K, device=dev)
    costs = C[:, perms, kt_idx[None].expand_as(perms)].sum(2)                                             # (B, P, c)
    # rank-restricted: predicted slot k is the k-th heaviest population (the targets are sorted by mass); a permutation
    # may only exchange slots whose target masses tie (within TIE of each other) or are both empty -- free matching let
    # the assignment wander from cell to cell and step to step, and no slot learned orientation
    from . import targets as _T
    key = mt if _T.ORDER == "mass" else _T.salience(St, mt, 2) + 1e-3 * mt                                # ties judged on the slots' order key
    mk, mj = key[:, :, None], key[:, None, :]                                                             # (B, Kt, 1, c), (B, 1, Kt, c)
    ratio = torch.minimum(mk, mj) / torch.maximum(mk, mj).clamp_min(1e-9)
    pk, pj = present[:, :, None], present[:, None, :]
    if DISTINCT is None:
        allowed = torch.eye(K, dtype=torch.bool, device=dev).view(1, K, K, *([1] * (mt.dim() - 2))) | ((ratio >= TIE) & pk & pj) | (~pk & ~pj)   # (B, Kt, Kp, c)
        valid = allowed[:, kt_idx[None].expand_as(perms), perms].all(2)                                  # (B, P, c)
    else:
        # order kept only among populations of similar shape: the rank rule exists to break the symmetry between
        # interchangeable slots (two crossing lines); populations whose eigenvalue spectra differ (a line and a blob,
        # rotation-invariant, no categories) are told apart by their tensors and may take any present slot
        ev, _ = eig3(St.movedim(2, 0))                                                                    # (3, B, K, c), ascending
        ev = ev / ev.sum(0, keepdim=True).clamp_min(1e-9)
        similar = (ev[:, :, :, None] - ev[:, :, None]).norm(dim=0) <= DISTINCT                            # (B, Kt, Kt, c)
        need = similar & (ratio < TIE) & pk & pj                                                          # pairs whose order must be kept
        valid = (present[:, perms] == present[:, None]).all(2)                                            # presents onto present ranks, empties onto empty
        for k in range(K):
            for j in range(k + 1, K):
                ok = (perms[:, k] < perms[:, j]).view(1, -1, *([1] * (mt.dim() - 2)))                     # (1, P, 1...)
                valid = valid & (~need[:, k, j][:, None] | ok)
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
    P = {k: (v if k.startswith("q_") else _cells(v, margin)) for k, v in pred.items()}
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
    if SHAPE_LOSS == "devcos":
        out["mixture"] = (dev_cost(P["mixture"].float(), mix, 1) * wmix).sum() / wmix.sum().clamp_min(1)
    else:
        out["mixture"] = ((((P["mixture"].float() - mix) ** 2) * WF.to(mix.device).view(1, 6, 1, 1, 1)).sum(1) * wmix).sum() / wmix.sum().clamp_min(1)
    if "q_sh" in P and "q_sh" in T:                                                                # the density at query points and scales
        qp, qt = P["q_sh"].float().transpose(1, 2), T["q_sh"].float().transpose(1, 2)
        nc = qt.shape[1] - 3 * DESC_NC
        out["query"] = sh_loss_decoupled(qp[:, :nc], qt[:, :nc], "query", SH_LMAX_ON) if SH_LOSS == "decoupled" else sh_loss(qp[:, :nc], qt[:, :nc])
        if DESC_NC:
            out["descriptors"] = desc_loss(qp[:, nc:], qt[:, nc:])
    if "sh" in P:                                                                                    # the populations' density in SH, no matching:
        from .sh import bands, density_sh                                                           # each order's squared error over the batch's
        nsh = P["sh"].shape[1] - 3 * DESC_NC
        lmax = 2 * (len(bands_for(nsh)) - 1)                                                        # energy in that order, averaged over the orders
        ct = T["sh_t"].float()[:, :nsh] if "sh_t" in T else density_sh(T["slots"].float(), T["mass"].float(), lmax, kappa)
        if DESC_NC and "sh_t" in T:
            out["descriptors"] = out.get("descriptors", 0.0) + desc_loss(P["sh"].float()[:, nsh:], T["sh_t"].float()[:, nsh:])
        if SH_LOSS == "decoupled":
            out["sh"] = sh_loss_decoupled(P["sh"].float()[:, :nsh], ct, "sh", SH_LMAX_ON)
        d2 = (P["sh"].float()[:, :nsh] - ct) ** 2
        # floor: a tenth of the batch's order-0 energy plus that of mass 0.05 in every cell (order 0 = sqrt(4 pi) m) --
        # a batch without anisotropy, or without populations, does not divide by ~0
        floor = 0.1 * (ct[:, 0] ** 2).sum() + ct[:, 0].numel() * (0.05 * math.sqrt(4 * math.pi)) ** 2
        out["sh"] = out["sh"] if SH_LOSS == "decoupled" else sum(d2[:, b].sum() / ((ct[:, b] ** 2).sum() + floor) for b in bands(lmax)) / len(bands(lmax))
    W = {**WEIGHTS, **(weights or {})}
    total = sum(W[k] * v for k, v in out.items())
    parts = {k: float(v) for k, v in out.items()}
    if "sh" in P:                                                                                    # the order-2 alignment (mass-weighted cosine)
        p2, t2 = P["sh"].float()[:, 1:6], ct[:, 1:6]
        tn2 = t2.norm(dim=1)
        parts["sh_l2cos"] = float(((p2 * t2).sum(1) / (p2.norm(dim=1) * tn2).clamp_min(1e-9) * tn2).sum() / tn2.sum().clamp_min(1e-9))
    iso = torch.tensor([1, 1, 1, 0, 0, 0.0], device=mix.device).view(1, 6, 1, 1, 1) / 3              # the mixture term of a round prediction:
    parts["mixture_floor"] = float(((((iso - mix) ** 2) * WF.to(mix.device).view(1, 6, 1, 1, 1)).sum(1) * wmix).sum() / wmix.sum().clamp_min(1))
    return total, parts


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


def _sh_block_rotation(perm, flips, n: int, device) -> torch.Tensor:
    """The rotation of ``n`` SH channels: the density (its order from ``n - 3 DESC_NC``) and, after it, three descriptor
    densities of ``DESC_NC`` coefficients each (rotations are block-diagonal by order, so theirs is the leading block)."""
    from .query import sh_rotation
    nsh = n - 3 * DESC_NC
    R = sh_rotation(perm, flips, 2 * (len(bands_for(nsh)) - 1), device)
    if not DESC_NC:
        return R
    Rd = R[:DESC_NC, :DESC_NC]
    return torch.block_diag(R, Rd, Rd, Rd)


def augment(image: torch.Tensor, tgt: dict, g: torch.Generator | None = None):
    """The same random cube-group element applied to an image ``(..., Z, Y, X)`` and its lattice target."""
    perm, flips = cube_transform(g)                                                             # drawn on the CPU
    out = {}
    for k, v in tgt.items():
        if k == "q_points":
            from .query import cube_points
            out[k] = cube_points(v, perm, flips, image.shape[-3:])
        elif k == "q_sh":
            out[k] = (v.float() @ _sh_block_rotation(perm, flips, v.shape[-1], v.device).T).to(v.dtype)
        elif k == "q_sigma":
            out[k] = v
        elif k == "sh_t":                                                                           # an SH field: move the cells, rotate the coefficients
            R = _sh_block_rotation(perm, flips, v.shape[-4], v.device)
            out[k] = torch.einsum("ij,j...->i...", R, apply_cube(v, perm, flips).float()).to(v.dtype)
        elif torch.is_tensor(v) and v.dim() >= 3:
            out[k] = apply_cube(v, perm, flips, tensor6=(k == "slots"))
        else:
            out[k] = v
    return apply_cube(image, perm, flips), out
