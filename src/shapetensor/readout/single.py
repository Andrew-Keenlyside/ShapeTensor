"""Single-direction read-outs of ``K``.

``eigen``: the smallest eigenvector of the channel-summed tensor (for one channel, BRIDGE's structure tensor).
``tca``: CP of the per-site array at rank R (default 3); the fibre is the null direction of the pieces' gradient-axis
Gram ``sum_r lam_r^2 d_r d_r^T``. Nothing is dropped. The pieces also give the boundary candidate: the strongest
piece's direction is the putative interface normal and its dominance (share of the strongest minus the second) a
label-free boundary strength, scored against the phantoms' geometry in the suite alongside Westin's ``cl`` of the
intensity tensor and the coarse gradient magnitude.
"""
from __future__ import annotations

import torch

from . import common
from .multi import cp_als


def eigen(K: torch.Tensor) -> dict:
    """``K (N, 3C, 3C)`` -> ``dir (N, 3)``, ``vals (N, 3)`` descending, Westin measures."""
    T = common.channel_sum(K)
    w, V = common.eigh_desc(T)
    return {"dir": common.canonical_sign(V[..., :, 2]), "vals": w, **common.westin(w)}


def tca(K: torch.Tensor, rank: int = 3, starts: int = 3, iters: int = 150, seed: int = 0) -> dict:
    """TCA read-out; also returns the pieces (``D (N, 3, R)`` directions, ``share (N, R)``, ``B`` signatures) and the
    boundary candidate (``bdir``, ``bshare``, ``bdom``)."""
    Y = common.cholesky_factor(K)                                       # (N, P, C, I)
    T = common.channel_sum(common.normalise_trace(K.double())).float()
    Cg = common.channel_gram(common.normalise_trace(K.double())).float()
    init = (torch.linalg.eigh(Cg)[1].flip(-1), torch.linalg.eigh(T)[1].flip(-1))
    A, B, D, err = cp_als(Y, rank, init, starts=starts, iters=iters, seed=seed)
    W = (A.mT @ A) * (B.mT @ B)                                         # (N, R, R)
    gram = D @ W @ D.mT                                                 # (N, 3, 3)
    d = common.canonical_sign(common.null_dir(gram))
    wd = W.diagonal(dim1=1, dim2=2)
    share = wd / wd.sum(1, keepdim=True).clamp_min(1e-30)
    srt = share.sort(1, descending=True)
    k_s = srt.indices[:, 0]
    idx = torch.arange(K.shape[0], device=K.device)
    bdir = D[idx, :, k_s]
    # how "unpaired" the strongest piece is: its smallest sine to any other piece (1 = perpendicular to all)
    sines = torch.linalg.norm(torch.cross(D.mT[:, :, None, :], D.mT[:, None, :, :], dim=-1), dim=-1)  # (N, R, R)
    sines = sines + torch.eye(rank, device=K.device)[None] * 2
    unpaired = sines[idx, k_s].min(1).values
    gw, _ = common.eigh_desc(gram)
    return {"dir": d, "D": D, "B": B, "share": share, "fit_err": err, "gram": gram,
            "bdir": common.canonical_sign(bdir), "bshare": srt.values[:, 0],
            "bdom": srt.values[:, 0] - srt.values[:, 1], "unpaired": unpaired,
            **{k: v for k, v in common.westin(gw).items() if k in ("gap", "cl", "cp")}}


def detectors(K_int: torch.Tensor, tca_out: dict | None = None, coarse_gm: torch.Tensor | None = None) -> dict:
    """Label-free boundary-strength candidates per site, each in [0, 1] or positive:
    ``cl`` Westin linear of the INTENSITY tensor; ``tca_dom`` the TCA strongest-piece dominance; ``tca_share``;
    ``gm`` the coarse gradient magnitude (fix A's detector) if given."""
    out = {}
    e = eigen(K_int)
    out["cl"] = e["cl"]
    out["n_cl"] = common.eigh_desc(common.channel_sum(K_int))[1][..., :, 0]       # e1: the sheet normal
    if tca_out is not None:
        out["tca_dom"] = tca_out["bdom"]
        out["tca_share"] = tca_out["bshare"]
        out["tca_unpaired"] = tca_out["unpaired"]
        out["n_tca"] = tca_out["bdir"]
    if coarse_gm is not None:
        out["gm"] = coarse_gm
    return out
