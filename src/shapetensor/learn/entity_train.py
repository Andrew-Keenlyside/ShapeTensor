"""The entity-field model's training (notes/plan_entity_field_model.md): directions per query, the losses, evaluation.

Per query point (``e_points``, scale ``e_sigma``) the targets hold up to E entities: node (the crop's local tree),
mass and visibility, Bingham frame ``e_R`` and concentrations ``e_kappa`` (moment-matched to the entity's shape tensor),
descriptors. The decoder is asked at D directions per query -- half drawn from the entities' Bingham ODFs (by visible
mass), half uniform -- and the losses compare, per direction:

* **density**: ``f`` against the target ``rho(u) = sum_e m_e v_e p_e(u)``: L1 of the logs with a floor relative to the
  query's mean density -- scale-free, so the angular shape is learned at any mass (log1p L1 on raw densities of 0.01-0.3
  was nearly linear and left the decoder flat);
* **vote**: on directions one entity owns (its share of rho > 1/2, rho above ``OWN_REL`` of the query's mean): the Bingham frame (axis-invariant, :func:`~shapetensor.
  learn.bingham.frame_loss`), the concentrations (log1p L1) and the descriptors;
* **tree**: hyperbolic distortion between owned directions' embeddings, ``(d_P(z_i, z_j) - t_ij)^2 / (1 + t_ij)`` with
  ``t_ij`` the tree distance of their entities (0 within an entity) -- any depth, no fixed levels -- and the gauge: each
  embedding's distance from the centre equals its entity's depth (the root at the centre, ancestors toward it);
* **overall**: the overall ODF head (free SH per lattice cell) against the exact pooled density; **fraction** and
  **points** as before.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn

from . import bingham as BG

ENT_WEIGHTS = {"density": 1.0, "vote": 1.0, "kappa": 0.2, "desc": 0.2, "tree": 0.5, "overall": 1.0, "fraction": 0.2, "points": 0.5, "contrast": 0.5, "cross": 0.5}
N_DIRS = 32
TREE_PAIRS = 4096
TREE_EDGE = 0.6                     # hyperbolic length of a tree edge: depth ~16 fits the ball (radius cap 1e-4 -> 9.9 from the centre)
DENS_FLOOR = 0.05                   # the density loss's floor: this share of the query's mean density (scale-free)
DENS_EPS = 1e-3                     # ... and at least this (empty queries: pushed to low density, not to -inf)
OWN_REL = 0.25                      # owned directions: density above this share of the query's mean (any mass)
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
    rm = rho.mean(-1, keepdim=True)                                                                # each query's mean density (its mass)
    eps = (DENS_FLOOR * rm).clamp_min(DENS_EPS)
    parts["density"] = (torch.log(ent["f"].float() + eps) - torch.log(rho + eps)).abs().mean()     # scale-free: shape at any mass
    share = contrib / rho.clamp_min(1e-9)[..., None]; best, owner = share.max(-1)
    owned = (best > 0.5) & (rho > OWN_REL * rm) & (rm > DENS_EPS)
    total = W["density"] * parts["density"]
    if bool(owned.any()):
        R_t = torch.gather(T["e_R"][:, :, None].expand(-1, -1, dirs.shape[2], -1, -1, -1), 3, owner[..., None, None, None].expand(-1, -1, -1, 1, 3, 3))[..., 0, :, :]
        k_t = torch.gather(T["e_kappa"][:, :, None].expand(-1, -1, dirs.shape[2], -1, -1), 3, owner[..., None, None].expand(-1, -1, -1, 1, 2))[..., 0, :]
        d_t = torch.gather(T["e_desc"][:, :, None].expand(-1, -1, dirs.shape[2], -1, -1), 3, owner[..., None, None].expand(-1, -1, -1, 1, 5))[..., 0, :]
        fl = BG.frame_loss(ent["R"][owned], R_t[owned], k_t[owned])
        parts["vote"] = fl.mean()
        if "log_s2" in ent:                                                                            # the vote's uncertainty: a readout of its error
            ls = ent["log_s2"][owned]                                                                  # (error / s^2 + log s^2 on the detached error:
            parts["vote_unc"] = 0.5 * ((fl.detach() + 1e-3) * torch.exp(-ls) + ls).mean()             # the vote itself keeps its plain loss)
            total = total + W.get("vote_unc", 0.2) * parts["vote_unc"]
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
            t = TREE_EDGE * tree_distance(T["e_anc"][b], nb[pi], nb[pj])
            from .model import poincare_dist
            dz = poincare_dist(zb[pi], zb[pj])
            # the gauge: each entity's distance from the ball's centre is its depth in the tree (the root at the centre,
            # ancestors toward it) -- depth readable at inference, embeddings comparable across windows and volumes
            dep = TREE_EDGE * ((T["e_anc"][b][nb[idx]] >= 0).sum(1) - 1).float(); r0 = 2 * torch.atanh(zb[idx].norm(dim=-1).clamp(max=1 - 1e-5))
            lt.append(((dz - t) ** 2 / (1 + t)).mean() + ((r0 - dep) ** 2 / (1 + dep)).mean())
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
    rho = contrib.sum(-1); rm = rho.mean(-1, keepdim=True)
    share = contrib / rho.clamp_min(1e-9)[..., None]; best, owner = share.max(-1); owned = (best > 0.5) & (rho > OWN_REL * rm) & (rm > DENS_EPS)
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


# ---------------------------------------------------------------------------------------------------------------
# across windows: overlapping crops of one scene must agree on the entities' places in the tree
# ---------------------------------------------------------------------------------------------------------------

def owners(T: dict, dirs: torch.Tensor):
    """The entity owning each direction (its share of the target density > 1/2, density above OWN_REL of the query's mean): ``(owner, owned)``,
    both ``(B, Q, D)``."""
    mass = (T["e_mass"] * T["e_vis"]).clamp_min(0) * (T["e_node"] >= 0)
    contrib = mass[:, :, None] * BG.pdf(dirs[..., None, :], T["e_R"][:, :, None], T["e_kappa"][:, :, None])
    rho = contrib.sum(-1); best, owner = (contrib / rho.clamp_min(1e-9)[..., None]).max(-1)
    rm = rho.mean(-1, keepdim=True)
    return owner, (best > 0.5) & (rho > OWN_REL * rm) & (rm > DENS_EPS)


def cross_distances(tree, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
    """Tree distances between the entities of two windows at the same queries: ``g1``, ``g2`` ``(Q, E)`` nodes of the
    scene's tree (-1: none; ``>= tree.n``: dense layers, children of the root) -> ``(Q, E, E)`` (-1 where either is
    none)."""
    Q, E = g1.shape; N = tree.n
    def chains(gg):                                                                               # nodes past the tree (dense layers): under the root
        gf = gg.flatten(); A = tree.ancestors(gf.clamp(0, N - 1))
        ext = gf >= N
        if bool(ext.any()):
            A[ext] = -1; A[ext, 0] = gf[ext]; A[ext, 1] = 0
        return A.view(Q, E, -1)
    A1, A2 = chains(g1), chains(g2)
    v1, v2 = A1 >= 0, A2 >= 0
    d1 = (v1.sum(-1) - 1)[:, :, None]; d2 = (v2.sum(-1) - 1)[:, None, :]
    eq = (A1[:, :, None, :, None] == A2[:, None, :, None, :]) & v1[:, :, None, :, None] & v2[:, None, :, None, :]
    hit = eq.any(-1)                                                                              # (Q, E, E, D): i's ancestors shared with j
    k = torch.where(hit.any(-1), hit.float().argmax(-1), (d1 + 1).expand(-1, -1, E))
    t = (d1 + d2 - 2 * (d1 - k)).float()
    return torch.where((g1[:, :, None] >= 0) & (g2[:, None, :] >= 0), t, torch.full_like(t, -1.0))


def cross_window_loss(P1: dict, P2: dict, T1: dict, T2: dict, X: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """At the same points and directions of two overlapping windows: the hyperbolic distance between the two windows'
    embeddings against the tree distance of their owners (0 for the same entity) -- one embedding across windows."""
    from .model import poincare_dist
    o1, w1 = owners(T1, dirs); o2, w2 = owners(T2, dirs)
    B, Q, D = o1.shape; E = X.shape[-1]
    t = torch.gather(X.view(B, Q, E * E), 2, (o1 * E + o2)).view(B, Q, D)
    t = torch.where(t >= 0, TREE_EDGE * t, t)
    ok = w1 & w2 & (t >= 0)
    if not bool(ok.any()):
        return dirs.new_zeros(())
    dz = poincare_dist(P1["ent"]["z"][ok].float(), P2["ent"]["z"][ok].float())
    return ((dz - t[ok]) ** 2 / (1 + t[ok])).mean()


class EntityPairBuffer:
    """Overlapping windows of one scene (``shift``: offset by multiples of 8 voxels, the same acquisition or another;
    ``modality``: the same window under another acquisition), their entity targets at the same queries (in the overlap)
    and the tree distances between the two windows' entities; drawn on demand into a small rotating buffer."""

    def __init__(self, n: int, size: int, device, gen: dict, seed0: int = 60_000_000, kinds=("shift", "modality"), n_points: int = 192):
        self.size, self.device, self.gen, self.next, self.kinds, self.n_points = size, device, dict(gen or {}), seed0, kinds, n_points
        self.items = [self._make() for _ in range(n)]

    def _make(self):
        from . import train as TR
        from .entities import _vis_at
        from .scene import draw_pair
        kind = self.kinds[self.next % len(self.kinds)]; seed = self.next; self.next += 1
        kw = {k: v for k, v in self.gen.items() if k not in ("crops_per_scene", "p_centre")}
        kw["canvas"] = max(kw.get("canvas", 128), 2 * self.size)
        (s1, d1), (s2, d2), info = draw_pair((self.size,) * 3, seed=seed, device=self.device, kind=kind, crops_per_scene=4, entity_ctx=True, **kw)
        sh = torch.tensor(info.get("shift", [0, 0, 0]), device=self.device, dtype=torch.float32)
        m = TR.ENTITY["margin"]; n = self.n_points; dev = self.device
        lo = torch.clamp(sh, min=0) + m; hi = torch.clamp(sh + self.size, max=self.size) - 1 - m           # the overlap, crop 1's voxels
        g = torch.Generator(device=dev).manual_seed(seed + 77)
        pts = lo + torch.rand(4 * n, 3, generator=g, device=dev) * (hi - lo)
        kl = [k for k, L in enumerate(s1.layers) if L.role != "isotropic"]                               # most queries on the entities' material
        occ = d1["occupancy"][kl].float().sum(0) if kl else torch.zeros_like(d1["occupancy"][0]).float()
        on = _vis_at(occ[None], pts)[:, 0] > 0.05
        pts = torch.cat([pts[on], pts[~on]])[:n]
        lo_s, hi_s = TR.ENTITY["sigma"]; sig = torch.exp(math.log(lo_s) + torch.rand(n, generator=g, device=dev) * math.log(hi_s / lo_s))
        t1 = TR.entity_queries(s1, d1, self.size, seed, dev, points=pts, sigma=sig)
        t2 = TR.entity_queries(s2, d2, self.size, seed, dev, points=pts - sh, sigma=sig)
        X = cross_distances(d1["entity_ctx"]["tree"], t1["e_gnode"], t2["e_gnode"]).half()
        vox = lambda d: torch.tensor(math.log2(d["voxel_um"]), device=dev)
        t1["voxel"], t2["voxel"] = vox(d1), vox(d2)
        return d1["image"].to(torch.uint8), d2["image"].to(torch.uint8), t1, t2, X, info

    def refresh(self, k: int = 1):
        for _ in range(k):
            i = self.next % len(self.items); self.items[i] = self._make()

    def batch(self, B: int, g: torch.Generator):
        idx = torch.randint(0, len(self.items), (B,), generator=g).tolist(); sel = [self.items[i] for i in idx]
        f = lambda k, v: v.float() if k in ("e_mass", "e_vis", "e_T6", "e_desc") else (v.long() if k in ("e_node", "e_anc", "e_gnode") else v)
        st = lambda j: {k: torch.stack([f(k, it[j][k]) for it in sel]) for k in sel[0][j]}
        x1 = torch.stack([it[0].float() / 255 for it in sel])[:, None]; x2 = torch.stack([it[1].float() / 255 for it in sel])[:, None]
        return x1, x2, st(2), st(3), torch.stack([it[4].float() for it in sel])
