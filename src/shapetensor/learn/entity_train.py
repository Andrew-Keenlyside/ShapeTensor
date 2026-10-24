"""The entity-field model's training (notes/plan_entity_field_model.md): directions per query, the losses, evaluation.

Per query point (``e_points``, scale ``e_sigma``) the targets hold up to E entities: node (the crop's local tree),
mass and visibility, Bingham frame ``e_R`` and concentrations ``e_kappa`` (moment-matched to the entity's shape tensor),
descriptors. The decoder is asked at D directions per query -- half drawn from the entities' Bingham ODFs (by visible
mass), half uniform -- and the losses compare, per direction:

* **density**: ``f`` against the target ``rho(u) = sum_e m_e v_e p_e(u)`` (log1p L1);
* **vote**: on directions one entity owns (its share of rho > 1/2): the Bingham frame (axis-invariant, :func:`~shapetensor.
  learn.bingham.frame_loss`), the concentrations (log1p L1) and the descriptors;
* **tree**: hyperbolic distortion between owned directions' embeddings, ``(d_P(z_i, z_j) - t_ij)^2 / (1 + t_ij)`` with
  ``t_ij`` the tree distance of their entities (0 within an entity) -- any depth, no fixed levels;
* **overall**: the overall ODF head (free SH per lattice cell) against the exact pooled density; **fraction** and
  **points** as before.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn

from . import bingham as BG

ENT_WEIGHTS = {"density": 1.0, "vote": 1.0, "kappa": 0.2, "desc": 0.2, "tree": 0.5, "overall": 1.0, "fraction": 0.2, "points": 0.5, "contrast": 0.5}
N_DIRS = 32
TREE_PAIRS = 4096
OVERALL_FLOOR = 0.01                # the overall ODF's relative error: energy floor per lattice cell (~ occupancy 0.03)


def sample_dirs(T: dict, D: int = N_DIRS, frac_entity: float = 0.5, g: torch.Generator | None = None) -> torch.Tensor:
    """``(B, Q, D, 3)`` directions per query: a share drawn near the entities' Bingham ODFs (entity chosen by visible
    mass; mode axis spread by 1/sqrt(2 kappa + 1) along the frame's other axes), the rest uniform."""
    m = (T["e_mass"] * T["e_vis"]).clamp_min(0) * (T["e_node"] >= 0)
    B, Q, E = m.shape; dev = m.device
    De = int(round(D * frac_entity)); Du = D - De
    uni = Fn.normalize(torch.randn(B, Q, D, 3, generator=g, device=dev), dim=-1)
    if De == 0:
        return uni
    has = m.sum(-1) > 0
    pe = torch.where(has[..., None], m / m.sum(-1, keepdim=True).clamp_min(1e-12), torch.full_like(m, 1.0 / E))
    comp = torch.multinomial(pe.view(-1, E), De, replacement=True, generator=g).view(B, Q, De)
    R = torch.gather(T["e_R"], 2, comp[..., None, None].expand(-1, -1, -1, 3, 3))
    k = torch.gather(T["e_kappa"], 2, comp[..., None].expand(-1, -1, -1, 2))
    n = torch.randn(B, Q, De, 2, generator=g, device=dev)
    u = R[..., :, 2] + n[..., :1] * R[..., :, 0] / torch.sqrt(2 * k[..., :1] + 1) + n[..., 1:] * R[..., :, 1] / torch.sqrt(2 * k[..., 1:] + 1)
    u = torch.where(has[..., None, None], Fn.normalize(u, dim=-1), uni[:, :, :De])
    return torch.cat([uni[:, :, De:], u], 2)


def tree_distance(anc: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    """Tree distances between local nodes ``i`` and ``j`` (``(P,)``) of one crop's ancestor table ``anc (U, Dm)`` (each
    row the chain from the node to the root, -1 padded): depth_i + depth_j - 2 depth(lca)."""
    ai, aj = anc[i], anc[j]; vi, vj = ai >= 0, aj >= 0
    di = vi.sum(1) - 1; dj = vj.sum(1) - 1
    eq = (ai[:, :, None] == aj[:, None, :]) & vi[:, :, None] & vj[:, None, :]
    hit = eq.any(2)                                                                              # (P, Dm): which of i's ancestors j shares
    k = torch.where(hit.any(1), hit.float().argmax(1), di + 1)                                    # the first shared one, counted up from i
    return (di + dj - 2 * (di - k)).float()


def entity_loss(P: dict, T: dict, dirs: torch.Tensor, W: dict | None = None, margin: int = 4, g: torch.Generator | None = None):
    """The entity losses for one batch: ``P`` the model's outputs (``P["ent"]`` at ``dirs``), ``T`` the targets."""
    W = {**ENT_WEIGHTS, **(W or {})}
    ent = P["ent"]; dev = dirs.device; parts = {}
    mass = (T["e_mass"] * T["e_vis"]).clamp_min(0) * (T["e_node"] >= 0)                         # (B, Q, E)
    pe = BG.pdf(dirs[..., None, :], T["e_R"][:, :, None], T["e_kappa"][:, :, None])                # (B, Q, D, E)
    contrib = mass[:, :, None] * pe; rho = contrib.sum(-1)
    parts["density"] = (torch.log1p(ent["f"]) - torch.log1p(rho)).abs().mean()
    share = contrib / rho.clamp_min(1e-9)[..., None]; best, owner = share.max(-1)
    owned = (best > 0.5) & (rho > 0.02)
    total = W["density"] * parts["density"]
    if bool(owned.any()):
        R_t = torch.gather(T["e_R"][:, :, None].expand(-1, -1, dirs.shape[2], -1, -1, -1), 3, owner[..., None, None, None].expand(-1, -1, -1, 1, 3, 3))[..., 0, :, :]
        k_t = torch.gather(T["e_kappa"][:, :, None].expand(-1, -1, dirs.shape[2], -1, -1), 3, owner[..., None, None].expand(-1, -1, -1, 1, 2))[..., 0, :]
        d_t = torch.gather(T["e_desc"][:, :, None].expand(-1, -1, dirs.shape[2], -1, -1), 3, owner[..., None, None].expand(-1, -1, -1, 1, 5))[..., 0, :]
        parts["vote"] = BG.frame_loss(ent["R"][owned], R_t[owned], k_t[owned]).mean()
        parts["kappa"] = (torch.log1p(ent["kappa"][owned]) - torch.log1p(k_t[owned])).abs().mean()
        dt = torch.cat([torch.log2(1 + d_t[..., :1].clamp_min(0)), (d_t[..., 1:2] / d_t[..., :1].clamp_min(1e-3)).clamp(0, 1), d_t[..., 2:]], -1)
        parts["desc"] = (ent["desc"][owned] - dt[owned]).abs().mean()
        total = total + W["vote"] * parts["vote"] + W["kappa"] * parts["kappa"] + W["desc"] * parts["desc"]
        # the tree: pairs of owned directions within each crop
        node = torch.gather(T["e_node"], 2, owner)                                                  # (B, Q, D) local node of the owner
        lt = []
        for b in range(dirs.shape[0]):
            idx = torch.nonzero(owned[b].flatten())[:, 0]
            if idx.numel() < 2:
                continue
            n_ = min(TREE_PAIRS, idx.numel() * 4)
            pi = idx[torch.randint(0, idx.numel(), (n_,), generator=g, device=dev)]; pj = idx[torch.randint(0, idx.numel(), (n_,), generator=g, device=dev)]
            zb = ent["z"][b].reshape(-1, ent["z"].shape[-1]); nb = node[b].flatten()
            t = tree_distance(T["e_anc"][b], nb[pi], nb[pj])
            from .model import poincare_dist
            dz = poincare_dist(zb[pi], zb[pj])
            lt.append(((dz - t) ** 2 / (1 + t)).mean())
        if lt:
            parts["tree"] = torch.stack(lt).mean(); total = total + W["tree"] * parts["tree"]
        parts["owned_share"] = owned.float().mean().detach()
    # the overall ODF, fraction, points on the lattice (cells away from the faces)
    c = lambda x: x[..., margin:x.shape[-3] - margin, margin:x.shape[-2] - margin, margin:x.shape[-1] - margin]
    if "sh_t" in T and "total_sh" in P:
        nsh = P["total_sh"].shape[1]; tt = c(T["sh_t"][:, :nsh].float()); pt = c(P["total_sh"].float())
        floor = OVERALL_FLOOR * tt[:, 0].numel()                                                    # near-empty crops: no blow-up
        parts["overall"] = ((pt - tt) ** 2).sum() / ((tt ** 2).sum() + floor); total = total + W["overall"] * parts["overall"]
    if "fraction" in T and "fraction_logit" in P:
        parts["fraction"] = Fn.binary_cross_entropy_with_logits(c(P["fraction_logit"].float()), c(T["fraction"].float())); total = total + W["fraction"] * parts["fraction"]
    if "points" in P and "points_t" in T:
        from . import lobes as LB
        parts["points"] = LB.points_loss(c(P["points"]), c(T["points_t"])); total = total + W["points"] * parts["points"]
    return total, parts


@torch.no_grad()
def axis_error_deg(P: dict, T: dict, dirs: torch.Tensor) -> dict:
    """On owned directions: the error of the voted mode axis (peaked entities, kappa2 > 4) and of the a-axis (girdles,
    kappa1 - kappa2 > 4), in degrees (medians)."""
    ent = P["ent"]
    mass = (T["e_mass"] * T["e_vis"]).clamp_min(0) * (T["e_node"] >= 0)
    pe = BG.pdf(dirs[..., None, :], T["e_R"][:, :, None], T["e_kappa"][:, :, None]); contrib = mass[:, :, None] * pe
    share = contrib / contrib.sum(-1, keepdim=True).clamp_min(1e-9); best, owner = share.max(-1); owned = (best > 0.5) & (contrib.sum(-1) > 0.02)
    out = {"owned_share": float(owned.float().mean())}
    if not bool(owned.any()):
        return out
    D = dirs.shape[2]
    R_t = torch.gather(T["e_R"][:, :, None].expand(-1, -1, D, -1, -1, -1), 3, owner[..., None, None, None].expand(-1, -1, -1, 1, 3, 3))[..., 0, :, :][owned]
    k_t = torch.gather(T["e_kappa"][:, :, None].expand(-1, -1, D, -1, -1), 3, owner[..., None, None].expand(-1, -1, -1, 1, 2))[..., 0, :][owned]
    Rp = ent["R"][owned]
    ang = lambda a, b: torch.rad2deg(torch.acos((a * b).sum(-1).abs().clamp(max=1)))
    pk = k_t[:, 1] > 4; gd = (k_t[:, 0] - k_t[:, 1]) > 4
    if bool(pk.any()):
        out["mode_axis_deg"] = float(ang(Rp[pk][:, :, 2], R_t[pk][:, :, 2]).median())
    if bool(gd.any()):
        out["a_axis_deg"] = float(ang(Rp[gd][:, :, 0], R_t[gd][:, :, 0]).median())
    return out
