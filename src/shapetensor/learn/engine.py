"""Inference with the population model. Every output is computed from the network's output; the image is the network's
input and nothing else (no autocorrelation, no noise floor from the image, no intensity threshold).

    from shapetensor.learn.engine import PopulationEngine
    eng = PopulationEngine("model.pt")
    out = eng.run(volume)                                 # a block: lattice outputs and the voxel field
    eng.run_volume(src, dst, chunk=(512, 512, 512))       # a whole volume in chunks, lattice outputs to a zarr group

The steps, each on the previous one's output:

1. **intensity window** -- grey levels to 0-1 by one pair of quantiles for the whole volume (:func:`window`, from
   random blocks, measured once so every chunk sees the same mapping);
2. **network** -- per lattice cell (stride 4, centred at voxel 4i + 1.5): ``K`` slots (shape tensor, mass, presence,
   kind, identity embedding), the shape fraction, the class shares, the size;
3. **populations** -- a slot is a population where its presence probability passes 0.5 and its mass ``min_mass``;
   masses stay absolute (a faint population keeps its small mass: the N heatmap can say "many weak" or "few strong");
   line slots carry directions, sheet and interface slots give the sheet normal and mass, blob slots a blob mass;
4. **carry-forward, in the slots' space** -- each line population smoothed along its own axis with the matching slots
   of the cells along it (:func:`carry`), then **bridged** (:func:`bridge`): short walks from every population follow
   it cell by cell (round bends) within ``reach`` cells and mark where it is missing or weaker; where the same
   direction arrives from both sides, a matching slot is raised to the weaker side's mass, or the population is
   inserted if the network lost it there. Never across populations, never past a free end;
5. **reading** -- N, directions and masses straight from the slots (near-duplicates within ``min_sep_deg`` merged):
   no peak-picking on a sampled sphere (which loses close crossings); identities by linking aligned populations of
   neighbouring cells (:func:`~shapetensor.populations.label_populations`);
6. **voxel field** -- for each population of a voxel's nearest cell, its tensor blended trilinearly with the same
   population (best-aligned slot) in the 8 surrounding cells, the vector the blend's principal axis, the weight the
   blended mass: smooth between cells, fading where a population ends.

Also per slot: its axis, linearity and planarity (the FA-like coherence of each population). ``fodf`` -- the line
populations rendered as a density on the direction grid with their absolute masses -- is kept for display.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from ..lift import hemisphere
from ..populations import label_populations
from .generator import eig3, outer6
from .model import PopulationNet, receptive_field
from .targets import render_odf

KIND_LINE, KIND_SHEET, KIND_BLOB, KIND_IFACE = 0, 1, 2, 3


def window(volume, lo_q: float = 0.005, hi_q: float = 0.995, n: int = 2_000_000, blocks: int = 64, block: int = 32, seed: int = 0) -> tuple[float, float]:
    """The intensity window of a whole volume: its ``lo_q`` and ``hi_q`` quantiles, from the whole array when small,
    else from ``blocks`` random ``block``³ blocks (only those are read: cheap on a chunked zarr)."""
    shape = tuple(volume.shape)
    if int(np.prod(shape)) <= n:
        v = np.asarray(volume[...]).ravel().astype(np.float64)
    else:
        rng = np.random.default_rng(seed); parts = []
        for _ in range(blocks):
            o = [int(rng.integers(0, max(1, s - block + 1))) for s in shape]
            parts.append(np.asarray(volume[tuple(slice(a, a + block) for a in o)]).ravel())
        v = np.concatenate(parts).astype(np.float64)
    return float(np.quantile(v, lo_q)), float(np.quantile(v, hi_q))


def _cells(cs, dev):
    n = int(np.prod(cs))
    pos = torch.stack(torch.meshgrid(*(torch.arange(c, device=dev, dtype=torch.float32) for c in cs), indexing="ij")).reshape(3, n)
    return pos, n


def _step(pos, a, t, cs):
    """The cells ``t`` steps along unit vectors ``a (3, n)`` from ``pos``: flat index and whether inside the block."""
    q = (pos + t * a).round()
    lim = torch.tensor(cs, device=pos.device, dtype=torch.float32).view(3, 1) - 1
    inb = ((q >= 0) & (q <= lim)).all(0)
    qc = torch.minimum(torch.maximum(q, torch.zeros_like(lim)), lim).long()
    return (qc[0] * cs[1] + qc[1]) * cs[2] + qc[2], inb


def _walk(af, kf, mf, live, k, sgn, reach, cos_tol, cs, pos):
    """Walk from every slot ``k`` along its population, ``sgn`` = ±1, up to ``reach`` cells: at each step the
    population's slot in the next cell (line, axis within ``cos_tol`` of the heading, the best aligned), the heading
    then turned to it (following bends). Yields ``t, qi, inside, hit, j, heading-before-step``."""
    K, n = mf.shape
    lim = torch.tensor(cs, device=pos.device, dtype=torch.float32).view(3, 1) - 1
    p = pos.clone(); a = af[k] * sgn; alive = live[k].clone()
    for t in range(1, reach + 1):
        p = p + a
        q = p.round(); inb = ((q >= 0) & (q <= lim)).all(0); alive = alive & inb
        qc = torch.minimum(torch.maximum(q, torch.zeros_like(lim)), lim).long(); qi = (qc[0] * cs[1] + qc[1]) * cs[2] + qc[2]
        na = af[:, :, qi]
        cos = (na * a[None]).sum(1)
        ok = live[:, qi] & (kf[:, qi] == kf[k][None]) & (cos.abs() >= cos_tol)
        j = torch.where(ok, cos.abs(), torch.full_like(cos, -1.0)).argmax(0)
        hit = ok.gather(0, j[None])[0] & alive
        yield t, qi, alive, hit, j, a
        aj = na.gather(0, j[None, None].expand(1, 3, n))[0]
        aj = torch.where((aj * a).sum(0, keepdim=True) < 0, -aj, aj)
        a = torch.where(hit[None], aj, a)


def carry(S: torch.Tensor, m: torch.Tensor, kind: torch.Tensor, axis: torch.Tensor, *, reach: int = 6, sigma_along: float = 3.0,
          match_deg: float = 25.0, kinds=(KIND_LINE,)) -> tuple[torch.Tensor, torch.Tensor]:
    """Population-specific smoothing in the slots' own space: from every slot of a kind in ``kinds``, walks along its
    population both ways (:func:`_walk`: cell by cell, turning to the matched slot's axis -- within ``match_deg`` of
    the heading, the best aligned -- so it follows bends and never jumps to a crossing partner), up to ``reach`` cells
    or the first gap. The slot's tensor becomes the mass- and distance-weighted mean (Gaussian ``sigma_along`` cells)
    of itself and the slots met, its mass the weighted mean of their masses. ``S (K, 6, z, y, x)``, ``m``, ``kind``
    ``(K, z, y, x)``, ``axis (K, 3, z, y, x)``."""
    K = m.shape[0]; cs = tuple(m.shape[1:]); dev = m.device
    pos, n = _cells(cs, dev)
    Sf, mf, kf, af = S.reshape(K, 6, n), m.reshape(K, n), kind.reshape(K, n), axis.reshape(K, 3, n)
    cos_tol = math.cos(math.radians(match_deg))
    live = torch.zeros(K, n, dtype=torch.bool, device=dev)
    for k_ in kinds:
        live |= kf == k_
    live &= mf > 0
    accS = Sf * mf[:, None]; accM = mf.clone(); accW = torch.ones_like(mf)
    for k in range(K):
        for sgn in (1.0, -1.0):
            going = live[k].clone()
            for t, qi, alive, hit, j, _ in _walk(af, kf, mf, live, k, sgn, reach, cos_tol, cs, pos):
                going = going & hit                                                              # stop at the first gap
                w = math.exp(-0.5 * (t / sigma_along) ** 2)
                mm = mf[:, qi].gather(0, j[None])[0] * going
                SS = Sf[:, :, qi].gather(0, j[None, None].expand(1, 6, n))[0]
                accS[k] += w * mm[None] * SS; accM[k] += w * mm; accW[k] += w * going
    S2 = torch.where(live[:, None], accS / accS[:, :3].sum(1, keepdim=True).clamp_min(1e-12), Sf)
    m2 = torch.where(live, accM / accW, mf)
    return S2.reshape(S.shape), m2.reshape(m.shape)


def bridge(S, m, kind, axis, present, grid, *, reach: int = 6, match_deg: float = 20.0, min_sep_deg: float = 25.0, min_mass: float = 0.01):
    """Bridge line populations across gaps in the slots' space, by short walks through the slot field. From every
    present line slot a walker steps cell by cell along the population, forwards and backwards, up to ``reach`` cells:
    in each cell it takes the matching slot (line, axis within ``match_deg`` of its heading) and turns to its axis, so it
    follows bends; where there is no matching slot -- a gap -- it keeps its heading and leaves an arrival (its direction,
    its mass); where the match is weaker than the walker, it leaves an arrival too (a dip). A direction that arrives at a
    cell from both sides holds there at the weaker side's mass: a matching slot is raised to it; where none is, the
    population is inserted into a free slot (strongest first, ``min_sep_deg`` apart). On a bend the walkers find their
    population in every cell, so nothing is inserted; a population present on one side only (a free end) is never
    extended. Returns updated ``S, m, kind, axis, present``."""
    K = m.shape[0]; cs = tuple(m.shape[1:]); dev = m.device; nd = grid.shape[0]
    pos, n = _cells(cs, dev)
    Sf, mf, kf, af, pf = S.reshape(K, 6, n).clone(), m.reshape(K, n).clone(), kind.reshape(K, n).clone(), axis.reshape(K, 3, n).clone(), present.reshape(K, n).clone()
    live = pf & (kf == KIND_LINE) & (mf > 0)
    cosm = math.cos(math.radians(match_deg)); coss = math.cos(math.radians(min_sep_deg))
    cosw = coss                                                                                     # a walker follows its population while it turns less than min_sep per cell;
                                                                                                    # a turn beyond (a crossing partner) is another population
    fw = torch.zeros(nd * n, device=dev); bw = torch.zeros(nd * n, device=dev)                  # arrivals moving along +g (from the -g side) / along -g
    for k in range(K):
        wm = torch.where(live[k], mf[k], torch.zeros_like(mf[k]))
        for sgn in (1.0, -1.0):
            for t, qi, alive, hit, j, a in _walk(af, kf, mf, live, k, sgn, reach, cosw, cs, pos):
                mj = mf[:, qi].gather(0, j[None])[0]
                arrive = alive & (~hit | (mj < wm))                                             # a gap, or a dip
                proj = a.T @ grid.T; bins = proj.abs().argmax(1)
                along = proj.gather(1, bins[:, None])[:, 0] > 0                                 # the side it came from: by its heading against the bin's
                for side, acc in ((along, fw), (~along, bw)):                                   # canonical direction (a slot's own sign is arbitrary)
                    sel = arrive & side
                    acc.scatter_reduce_(0, (bins * n + qi)[sel], wm[sel], reduce="amax")
    fw, bw = fw.view(nd, n), bw.view(nd, n)
    nbr = (grid @ grid.T).abs() >= cosm
    fwd = torch.stack([fw[nbr[i]].amax(0) for i in range(nd)]); bwd = torch.stack([bw[nbr[i]].amax(0) for i in range(nd)])
    br = torch.minimum(fwd, bwd)                                                                   # (nd, n)
    b = (af.movedim(1, -1) @ grid.T).abs().argmax(-1)
    for k in range(K):                                                                             # raise matching present slots
        mf[k] = torch.where(live[k], torch.maximum(mf[k], br.gather(0, b[k][None])[0]), mf[k])
    covered = torch.zeros(nd, n, dtype=torch.bool, device=dev)
    for k in range(K):
        covered |= ((grid @ af[k]).abs() >= coss) & live[k][None]                               # within min_sep of a present population: not a new one
    cand = torch.where(covered, torch.zeros_like(br), br)
    for _ in range(K):                                                                             # insert lost populations into free slots
        val, bi = cand.max(0)
        free = ~pf; ins = (val >= min_mass) & free.any(0)
        if not bool(ins.any()):
            break
        slot = free.float().argmax(0)
        d = grid[bi].T                                                                             # (3, n)
        oh = Fn.one_hot(slot, K).T.bool() & ins[None]
        Sf = torch.where(oh[:, None], outer6(d.T).T[None], Sf); mf = torch.where(oh, val[None], mf)
        kf = torch.where(oh, torch.full_like(kf, KIND_LINE), kf); af = torch.where(oh[:, None], d[None], af); pf = pf | oh
        cand = torch.where((grid @ d).abs() >= coss, torch.zeros_like(cand), cand)
    rs = lambda x, lead: x.reshape(lead + cs)
    return rs(Sf, (K, 6)), rs(mf, (K,)), rs(kf, (K,)), rs(af, (K, 3)), rs(pf, (K,))


class PopulationEngine:
    def __init__(self, checkpoint: str | dict | PopulationNet | None = None, *, device: str = "cuda", nmax: int = 4, min_sep_deg: float = 25.0,
                 min_mass: float = 0.01, kappa: float = 20.0, momentum: bool = True, reach: int = 6, sigma_along: float = 3.0, match_deg: float = 20.0,
                 label_angle_deg: float = 20.0, min_fixels: int = 8, exact: bool = False):
        self.device = device
        if isinstance(checkpoint, PopulationNet):
            self.net = checkpoint.to(device).eval(); rf = None
        else:
            rf = None; ck = None
            if checkpoint is not None:
                ck = torch.load(checkpoint, map_location=device) if isinstance(checkpoint, (str, Path)) else checkpoint
            self.net = PopulationNet.from_args(ck["args"] if ck is not None else None).to(device).eval()
            if ck is not None:
                self.net.load_state_dict(ck["model"]); rf = ck.get("receptive_field")
        self.rf = rf if rf is not None else receptive_field(self.net)
        self.exact = exact                                                                          # fp32 without TF32 (chunks equal one pass bit for bit) vs bf16 (~10x faster)
        self.grid = hemisphere(160, device)
        self.nmax, self.min_sep_deg, self.min_mass, self.kappa = int(nmax), float(min_sep_deg), float(min_mass), float(kappa)
        self.momentum, self.reach, self.sigma_along, self.match_deg = bool(momentum), int(reach), float(sigma_along), float(match_deg)
        self.label_angle_deg, self.min_fixels = float(label_angle_deg), int(min_fixels)

    # -- the network -------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def network(self, v: torch.Tensor, win: tuple[float, float]) -> dict:
        """The network on a block ``v (Z, Y, X)`` (Z, Y, X multiples of 4) after the intensity window."""
        lo, hi = win
        x = ((v.to(self.device).float() - lo) / max(hi - lo, 1e-6)).clamp(0, 1)[None, None]
        tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = not self.exact
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(not self.exact) and x.is_cuda):
                out = self.net(x)
        finally:
            torch.backends.cudnn.allow_tf32 = tf32
        return {k: v_[0].float() for k, v_ in out.items()}

    # -- everything after it: on the network's output only ----------------------------------------------------------

    def populations(self, net: dict, labels: bool = True) -> dict:
        """Steps 3-5 on the network's lattice output (one block, no batch dim)."""
        S, m = net["slots"], net["mass"]                                                          # (K, 6, c), (K, c)
        K = m.shape[0]; cs = tuple(m.shape[1:])
        kind = net["kind"].argmax(1)
        present = (torch.sigmoid(net["presence_logit"]) > 0.5) & (m >= self.min_mass)
        m = m * present
        val, vec = eig3(S.movedim(1, 0))                                                           # (3, K, c), (3, 3, K, c)
        planar = (kind == KIND_SHEET) | (kind == KIND_IFACE)
        axis = torch.where(planar[None], vec[0], vec[2]).movedim(0, 1)                              # (K, 3, c)
        lin = (val[2] - val[1]) / val[2].clamp_min(1e-9); pla = (val[1] - val[0]) / val[2].clamp_min(1e-9)
        # step 4: the line populations carried forward and bridged, in the slots' space
        line = present & (kind == KIND_LINE)
        Sl, ml = S, m * line
        if self.momentum:
            Sl, ml = carry(S, ml, kind, axis, reach=self.reach, sigma_along=self.sigma_along, match_deg=self.min_sep_deg)
            _, vl = eig3(Sl.movedim(1, 0)); axl = torch.where(planar[None], vl[0], vl[2]).movedim(0, 1)
            Sl, ml, kl, axl, pl = bridge(Sl, ml, kind, axl, line, self.grid, reach=self.reach, match_deg=self.match_deg, min_sep_deg=self.min_sep_deg, min_mass=self.min_mass)
            line = pl & (kl == KIND_LINE)
        else:
            axl = axis
        ml = ml * line
        # step 5: merge near-duplicate line populations, sort by mass, keep nmax
        coss = math.cos(math.radians(self.min_sep_deg))
        order = ml.argsort(0, descending=True)
        Ls = torch.gather(Sl, 0, order[:, None].expand_as(Sl)); Lm = torch.gather(ml, 0, order); La = torch.gather(axl, 0, order[:, None].expand_as(axl))
        for i in range(K):
            for j in range(i + 1, K):
                dup = (Lm[i] > 0) & (Lm[j] > 0) & ((La[i] * La[j]).sum(0).abs() >= coss)
                Ls[i] = torch.where(dup[None], (Ls[i] * Lm[i] + Ls[j] * Lm[j]) / (Lm[i] + Lm[j]).clamp_min(1e-12), Ls[i])
                Lm[i] = torch.where(dup, Lm[i] + Lm[j], Lm[i]); Lm[j] = torch.where(dup, torch.zeros_like(Lm[j]), Lm[j])
        order = Lm.argsort(0, descending=True)
        Ls, Lm = torch.gather(Ls, 0, order[:, None].expand_as(Ls)), torch.gather(Lm, 0, order)
        _, lv = eig3(Ls.movedim(1, 0)); La = lv[2].movedim(0, 1)
        nm = min(self.nmax, K)
        dirs, mass, tens = La[:nm] * (Lm[:nm] > 0)[:, None], Lm[:nm], Ls[:nm]
        sheet_m = m * present * planar
        heavy = sheet_m.argmax(0)
        sheet_normal = torch.gather(axis, 0, heavy[None, None].expand(1, 3, *cs))[0] * (sheet_m.sum(0) > 0)[None]
        out = {"N": (mass > 0).sum(0), "dirs_cells": dirs, "mass": mass, "line_tensors": tens,
               "fodf": render_odf(Ls, Lm, self.grid, self.kappa), "fraction": torch.sigmoid(net["fraction_logit"]),
               "classes": torch.softmax(net["class_logits"], 0), "size": net["log_size"].exp(),
               "slot_axis": axis, "slot_mass": m, "slot_kind": kind, "slot_present": present, "slot_linearity": lin, "slot_planarity": pla,
               "slot_embed": net["embed"], "sheet_normal": sheet_normal, "sheet_mass": sheet_m.sum(0), "blob_mass": (m * present * (kind == KIND_BLOB)).sum(0)}
        if labels:
            out["labels_cells"], out["table"] = label_populations(dirs, mass, max_angle_deg=self.label_angle_deg, min_fixels=self.min_fixels)
        return out

    # -- step 6: the voxel field --------------------------------------------------------------------------------------

    def field(self, pop: dict, shape, labels: bool = True, slab_voxels: int = 1 << 19) -> dict:
        """The voxel field of a block of ``shape``: for each population of a voxel's nearest cell, its tensor blended
        trilinearly with the same population (best-aligned, within ``match_deg``) in the 8 surrounding cells; ``vecs
        (nmax, 3, Z, Y, X)``, ``weights`` (the blended masses), ``labels``, ``N``, ``vec``/``weight`` (the first)."""
        Z, Y, X = shape; dev = self.device
        D, M, T = pop["dirs_cells"], pop["mass"], pop["line_tensors"]                              # (R, 3, c), (R, c), (R, 6, c)
        R = M.shape[0]; cs = tuple(M.shape[1:])
        Lc = pop.get("labels_cells") if labels else None
        cosm = math.cos(math.radians(self.match_deg))
        vecs = torch.zeros((R, 3, Z, Y, X), device=dev); weights = torch.zeros((R, Z, Y, X), device=dev)
        labs = torch.zeros((R, Z, Y, X), dtype=torch.int64, device=dev) if Lc is not None else None
        zs = max(1, slab_voxels // (Y * X))
        cy = (torch.arange(Y, device=dev, dtype=torch.float32) - 1.5) / 4; cx = (torch.arange(X, device=dev, dtype=torch.float32) - 1.5) / 4
        lim = torch.tensor(cs, device=dev, dtype=torch.float32).view(3, 1) - 1
        flat = lambda q: (q[0] * cs[1] + q[1]) * cs[2] + q[2]
        Df, Mf, Tf = D.reshape(R, 3, -1), M.reshape(R, -1), T.reshape(R, 6, -1)
        for z0 in range(0, Z, zs):
            z1 = min(Z, z0 + zs)
            cz = (torch.arange(z0, z1, device=dev, dtype=torch.float32) - 1.5) / 4
            U = torch.stack(torch.meshgrid(cz, cy, cx, indexing="ij")).reshape(3, -1)            # continuous cell coordinates
            U = torch.minimum(torch.maximum(U, torch.zeros_like(lim)), lim)
            i0 = torch.minimum(U.floor(), (lim - 1).clamp_min(0)); fr = U - i0; i0 = i0.long()
            near = U.round().long(); ni = flat(near)
            dn = Df[:, :, ni]                                                                     # (R, 3, v): the nearest cell's populations
            accT = torch.zeros((R, 6) + ni.shape, device=dev); accM = torch.zeros((R,) + ni.shape, device=dev); accW = torch.zeros_like(accM)
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        off = torch.tensor([dz, dy, dx], device=dev).view(3, 1)
                        q = torch.minimum(i0 + off, lim.long()); qi = flat(q)
                        w = torch.where(off.bool(), fr, 1 - fr).prod(0)                            # (v,)
                        dc, mc, tc = Df[:, :, qi], Mf[:, qi], Tf[:, :, qi]                          # (R, 3, v), (R, v), (R, 6, v)
                        cos = torch.einsum("rcv,scv->rsv", dn, dc).abs() * (mc[None] > 0)          # (R near, R corner, v)
                        best = cos.argmax(1); ok = cos.gather(1, best[:, None])[:, 0] >= cosm       # (R, v)
                        mb = mc.gather(0, best) * ok; tb = tc.gather(0, best[:, None].expand(-1, 6, -1))
                        accT += (w * mb)[:, None] * tb; accM += w * mb; accW += w
            Tb = accT / accM.clamp_min(1e-12)[:, None]
            _, vv = eig3(Tb.movedim(1, 0).reshape(6, -1)); v3 = vv[2].reshape(3, R, -1).movedim(1, 0)
            v3 = torch.where((v3 * dn).sum(1, keepdim=True) < 0, -v3, v3)                          # the nearest cell's sense
            has = (Mf[:, ni] > 0)
            vecs[:, :, z0:z1] = (v3 * has[:, None]).reshape(R, 3, z1 - z0, Y, X)
            weights[:, z0:z1] = (accM / accW.clamp_min(1e-12) * has).reshape(R, z1 - z0, Y, X)
            if labs is not None:
                labs[:, z0:z1] = (Lc.reshape(R, -1)[:, ni] * has).reshape(R, z1 - z0, Y, X)
        out = {"vecs": vecs, "weights": weights, "vec": vecs[0], "weight": weights[0], "N": (weights > 0).sum(0)}
        if labs is not None:
            out["labels"] = labs
        return out

    def run(self, v, win: tuple[float, float] | None = None, native: bool = True, labels: bool = True) -> dict:
        """One block: the network, the populations, and (``native``) the voxel field. The block is edge-padded to whole
        cells. ``win``: the intensity window (None: this block's own quantiles -- for a whole volume use :func:`window`)."""
        v = torch.as_tensor(np.asarray(v)) if not torch.is_tensor(v) else v
        win = win if win is not None else window(v.cpu().numpy())
        shp = tuple(v.shape); pad = [(-s) % 4 for s in shp]
        vp = Fn.pad(v.float()[None, None], (0, pad[2], 0, pad[1], 0, pad[0]), mode="replicate")[0, 0] if any(pad) else v
        pop = self.populations(self.network(vp, win), labels=labels)
        if native:
            pop.update({f"voxel_{k}": x for k, x in self.field(pop, shp, labels=labels).items()})
        return pop

    # -- a whole volume ---------------------------------------------------------------------------------------------

    def post_halo_cells(self) -> int:
        """Cells of context the post-processing needs: the carry-forward then the bridge, each ``reach`` cells."""
        return (2 * self.reach + 1) if self.momentum else 0

    def halo(self) -> int:
        """Voxels of context a chunk needs for its cells to equal one pass: half the receptive field plus the
        post-processing's reach, rounded up to whole cells."""
        return int(4 * math.ceil((self.rf / 2 + 4 * self.post_halo_cells()) / 4))

    def run_volume(self, src, dst: str, *, chunk=(512, 512, 512), win: tuple[float, float] | None = None, min_fraction: float = 0.01, log=print):
        """A whole volume ``src`` (array-like ``(Z, Y, X)``; zarr arrays read chunk by chunk) in chunks with a halo of
        :meth:`halo` voxels; lattice outputs written to the zarr group ``dst``: ``N``, ``dirs (nmax, 3, ...)``, ``mass``,
        ``fraction``, ``classes``, ``sheet_normal``, ``sheet_mass``, ``blob_mass``. The network runs on the chunk plus
        its halo; the populations on the chunk plus the post-processing's halo only. A chunk whose fraction stays below
        ``min_fraction`` is written as empty after the network."""
        import zarr
        Z, Y, X = src.shape
        assert all(c % 4 == 0 for c in chunk), "chunk must be a multiple of 4"
        cz, cy, cx = (math.ceil(n / 4) for n in (Z, Y, X)); h = self.halo(); ph = self.post_halo_cells()
        win = win if win is not None else window(src)
        root = zarr.open_group(dst, mode="w")
        cc = tuple(c // 4 for c in chunk); R = self.nmax
        arr = {"N": root.create_array("N", shape=(cz, cy, cx), chunks=cc, dtype="uint8"),
               "dirs": root.create_array("dirs", shape=(R, 3, cz, cy, cx), chunks=(R, 3) + cc, dtype="float16"),
               "mass": root.create_array("mass", shape=(R, cz, cy, cx), chunks=(R,) + cc, dtype="float16"),
               "fraction": root.create_array("fraction", shape=(cz, cy, cx), chunks=cc, dtype="float16"),
               "classes": root.create_array("classes", shape=(5, cz, cy, cx), chunks=(5,) + cc, dtype="float16"),
               "sheet_normal": root.create_array("sheet_normal", shape=(3, cz, cy, cx), chunks=(3,) + cc, dtype="float16"),
               "sheet_mass": root.create_array("sheet_mass", shape=(cz, cy, cx), chunks=cc, dtype="float16"),
               "blob_mass": root.create_array("blob_mass", shape=(cz, cy, cx), chunks=cc, dtype="float16")}
        root.attrs.update({"window": list(win), "step": 4, "origin": 1.5, "halo": h, "receptive_field": self.rf, "chunk": list(chunk)})
        n_done = 0
        for z0 in range(0, Z, chunk[0]):
            for y0 in range(0, Y, chunk[1]):
                for x0 in range(0, X, chunk[2]):
                    a0 = (z0, y0, x0)
                    lo = [max(0, a - h) for a in a0]
                    hi = [min(n, a + c + h) for a, c, n in zip(a0, chunk, (Z, Y, X))]
                    pad_hi = [(-(b - a)) % 4 for a, b in zip(lo, hi)]
                    blk = np.asarray(src[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
                    if any(pad_hi):
                        blk = np.pad(blk, [(0, p) for p in pad_hi], mode="edge")
                    net = self.network(torch.as_tensor(blk), win)
                    ncell = [math.ceil(min(c, n - a) / 4) for a, c, n in zip(a0, chunk, (Z, Y, X))]
                    c0 = [(a - l) // 4 for a, l in zip(a0, lo)]                                     # the chunk's first cell in the block
                    sub = tuple(slice(max(0, s - ph), min(e, s + k + ph)) for s, k, e in zip(c0, ncell, net["mass"].shape[1:]))
                    core = tuple(slice(s - u.start, s - u.start + k) for s, k, u in zip(c0, ncell, sub))
                    dst_sl = tuple(slice(a // 4, a // 4 + k) for a, k in zip(a0, ncell))
                    net = {k: v[(slice(None),) * (v.dim() - 3) + sub] for k, v in net.items()}
                    frac = torch.sigmoid(net["fraction_logit"])
                    if float(frac[core].max()) < min_fraction:
                        arr["fraction"][dst_sl] = frac[core].half().cpu().numpy(); continue
                    pop = self.populations(net, labels=False)

                    def put(name, x, lead):
                        y = x[(slice(None),) * lead + core]
                        arr[name][(slice(None),) * lead + dst_sl] = y.half().cpu().numpy() if y.dtype.is_floating_point else y.cpu().numpy()
                    arr["N"][dst_sl] = pop["N"][core].to(torch.uint8).cpu().numpy()
                    put("dirs", pop["dirs_cells"], 2); put("mass", pop["mass"], 1); put("fraction", pop["fraction"], 0)
                    put("classes", pop["classes"], 1); put("sheet_normal", pop["sheet_normal"], 1); put("sheet_mass", pop["sheet_mass"], 0); put("blob_mass", pop["blob_mass"], 0)
                    n_done += 1
        log(f"run_volume: {n_done} chunks with content, halo {h} voxels, window {win}")
        return root
