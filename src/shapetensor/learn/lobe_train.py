"""Training the lobe-mixture model: its loss, consistency terms, difficulty widening and evaluation.

Used by :func:`~shapetensor.learn.train.train` with ``--head lobes``. Per cell the model's ``J`` lobes are rendered to
harmonics (:mod:`~shapetensor.learn.lobes`) and compared with the exact density of the orientations present (and the
descriptor densities); there is no slot matching and no kind or class head: a population is a lobe with mass. Terms:

* ``sh``: per-order squared error over a running energy per order;
* ``ce``: the direction cross-entropy (a missing lobe costs a log, not a square);
* ``desc``: the five descriptor densities (log size, relative wall, lumen fraction, flattening, angularity);
* ``fraction``: the shape volume fraction; ``points``: branch-point / endpoint maps; ``identity``: a discriminative
  loss on object ids at stride 2 (an object smaller than the resolution is identified by its parent);
* ``query``: the decoder's lobes at random points and scales against the exact windowed density;
* ``consistency``: a render at half the resolution against the fine prediction blurred and pooled (stop-gradient on
  the fine side), and the same crop under another acquisition;
* ``ssl``: on unlabelled crops, rotation equivariance (the prediction of a cube-rotated crop against the rotated
  prediction, stop-gradient) and invariance to an intensity / noise perturbation.

The gate (orientation first): until the order-2 error falls below ``gate`` x the order-2 target energy (EMA), only
``sh`` and ``ce`` train; then the rest ramps in.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as Fn

from . import lobes as LB
from .losses import _cells, apply_cube, cube_transform
from .sh import n_coef

LOBE_WEIGHTS = {"sh": 1.0, "ce": 0.5, "aux": 1.0, "desc": 0.5, "fraction": 0.2, "points": 0.5, "identity": 0.1, "query": 1.0,
                "consistency": 0.5, "ssl": 0.2, "contrast": 0.5}
ORIENT = ("sh", "ce", "aux")


def density(model, raw: torch.Tensor, lmax: int = 8, desc_lmax: int = 4):
    """The density SH, descriptor densities and lobes of a lobe model's raw head ``(B, C, ...)``.

    For a free-SH model (an ablation) the channels are the coefficients and there are no lobes."""
    if getattr(model, "free_sh", False):
        nsh = n_coef(lmax)
        return raw[:, :nsh].float(), (raw[:, nsh:].float() if desc_lmax and raw.shape[1] > nsh else None), None
    lob = LB.decode(raw, model.J, model.nd)
    f, fq = LB.render(lob, lmax, desc_lmax if model.nd else 0)
    return f, fq, lob


def lobe_loss(model, P: dict, T: dict, margin: int = 4, weights: dict | None = None, lmax: int = 8, desc_lmax: int = 4,
              update: bool = True) -> tuple[torch.Tensor, dict, dict]:
    """The lobe model's loss over the cells ``margin`` from the crop's edge.

    Returns the total, the parts (detached) and the rendered fields (for the consistency terms)."""
    W = {**LOBE_WEIGHTS, **(weights or {})}
    f, fq, lob = density(model, P["lobes_raw"], lmax, desc_lmax if "sh_t" in T and T["sh_t"].shape[1] > n_coef(lmax) else 0)
    c = lambda x: _cells(x, margin)
    nsh = n_coef(lmax)
    ct = T["sh_t"][:, :nsh].float()
    out = {"sh": LB.sh_mse_running(c(f), c(ct), "sh", update=update), "ce": LB.direction_ce(c(f), c(ct), lmax)}
    if fq is not None:
        out["desc"] = LB.desc_mse(c(fq), c(T["sh_t"][:, nsh:].float()), model.nd)
    if "fraction" in T:
        out["fraction"] = Fn.binary_cross_entropy_with_logits(c(P["fraction_logit"].float()), c(T["fraction"].float()))
    if "points" in P and "points_t" in T:
        out["points"] = LB.points_loss(c(P["points"]), c(T["points_t"]))
    if "identity" in P and "identity_t" in T:
        out["identity"] = LB.identity_loss(_cells(P["identity"], 2 * margin), _cells(T["identity_t"], 2 * margin))
    if "q_lobes_raw" in P and "q_sh" in T:
        qf, _, _ = density(model, P["q_lobes_raw"].transpose(1, 2), lmax, 0)
        out["query"] = LB.sh_mse_running(qf, T["q_sh"].float().transpose(1, 2)[:, :nsh], "query", update=update)
    if "aux_sh" in P:                               # the auxiliary free-harmonics head
        out["aux"] = LB.sh_mse_running(c(P["aux_sh"].float()), c(ct), "aux", update=update)
    total = sum(W[k] * v for k, v in out.items())
    parts = {k: v.detach() for k, v in out.items()}                 # converted to floats once per log, not per term
    t2 = c(ct)[:, 1:6]
    # the gate's signal: the order-2 error against the order-2 energy
    p2 = c(P["aux_sh"].float())[:, 1:6] if "aux_sh" in P else c(f)[:, 1:6]
    parts["gate_num"] = ((p2 - t2) ** 2).sum().detach(); parts["gate_den"] = (t2 ** 2).sum().detach()
    # the lobes' own order-2 error (pooled, not a mean of ratios)
    parts["lobe_l2_num"] = ((c(f).detach()[:, 1:6] - t2) ** 2).sum()
    return total, parts, {"f": f, "lob": lob}


# ---------------------------------------------------------------------------------------------------------------
# consistency: paired renders and unlabelled real crops
# ---------------------------------------------------------------------------------------------------------------

def zoom_consistency(model, fine: dict, coarse: dict, info: dict, lmax: int = 8, sigma_w: float = 5.0) -> torch.Tensor:
    """The coarse render's density against the fine one's, blurred to the coarse window and pooled 2³ (stop-gradient
    on the fine side).

    At voxel 2v the target window is sigma_w coarse voxels = 2 sigma_w fine voxels, so the fine lattice field needs a
    further sqrt((2 sigma_w)^2 - sigma_w^2) / 4 cells of blur."""
    f_f = density(model, fine["lobes_raw"], lmax, 0)[0].detach()
    f_c = density(model, coarse["lobes_raw"], lmax, 0)[0]
    pooled = LB.pool_sh(f_f, math.sqrt((2 * sigma_w) ** 2 - sigma_w ** 2) / 4, 2)                 # (B, c, n/2...)
    oz, oy, ox = info["offset_cells"]; n = pooled.shape[2:]
    reg = f_c[:, :, oz:oz + n[0], oy:oy + n[1], ox:ox + n[2]]
    m = 2                                           # leave out cells whose blur reaches the crop edge
    return LB.sh_mse_running(_cells(reg, m), _cells(pooled[:, :, :reg.shape[2], :reg.shape[3], :reg.shape[4]], m), "cons_zoom")


def modality_consistency(model, a: dict, b: dict, lmax: int = 8, margin: int = 4) -> torch.Tensor:
    """The same crop under two acquisitions: their densities agree (symmetric, each side against the other's
    stop-gradient)."""
    fa = density(model, a["lobes_raw"], lmax, 0)[0]
    fb = density(model, b["lobes_raw"], lmax, 0)[0]
    return 0.5 * (LB.sh_mse_running(_cells(fa, margin), _cells(fb.detach(), margin), "cons_mod")
                  + LB.sh_mse_running(_cells(fb, margin), _cells(fa.detach(), margin), "cons_mod", update=False))


def contrast_consistency(model, f_a: torch.Tensor, P_b: dict, lmax: int = 8, margin: int = 4) -> torch.Tensor:
    """The same geometry through two contrast routes: the second route's density against the first's (stop-gradient
    on the first, which also has its supervised loss), so invariance to the route is trained directly."""
    f_b = density(model, P_b["lobes_raw"], lmax, 0)[0]
    return LB.sh_mse_running(_cells(f_b, margin), _cells(f_a.detach(), margin), "cons_contrast")


def ssl_consistency(model, x: torch.Tensor, g: torch.Generator, lmax: int = 8, margin: int = 4, voxel=None) -> torch.Tensor:
    """Unlabelled crops ``x (B, 1, Z, Y, X)``: the prediction on a cube-rotated, intensity-perturbed copy against the
    rotated prediction on the original (stop-gradient), for equivariance and invariance without labels."""
    from .losses import _sh_block_rotation
    from . import losses as _L
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        f0 = density(model, model(x, voxel=voxel)["lobes_raw"], lmax, 0)[0].float()
    perm, flips = cube_transform(g)
    xr = apply_cube(x, perm, flips)
    a = 0.7 + 0.6 * float(torch.rand(1, generator=g)); gam = math.exp(0.4 * (float(torch.rand(1, generator=g)) - 0.5))
    xr = (xr.clamp(0, 1) ** gam * a + 0.03 * torch.randn(xr.shape, generator=g).to(x.device)).clamp(0, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        f1 = density(model, model(xr, voxel=voxel)["lobes_raw"], lmax, 0)[0]
    saved = _L.DESC_NC; _L.DESC_NC = 0
    try:
        R = _sh_block_rotation(perm, flips, f0.shape[1], f0.device)
    finally:
        _L.DESC_NC = saved
    tgt = torch.einsum("ij,bj...->bi...", R, apply_cube(f0, perm, flips))
    return LB.sh_mse_running(_cells(f1, margin), _cells(tgt, margin), "ssl")


class PairBuffer:
    """Paired renders (zoom or modality) of scenes, drawn on demand and kept in a small rotating buffer."""

    def __init__(self, n: int, size: int, device, gen: dict, seed0: int = 50_000_000, kinds=("zoom", "modality")):
        self.size, self.device, self.gen, self.next, self.kinds = size, device, dict(gen or {}), seed0, kinds
        self.items = [self._make() for _ in range(n)]

    def _make(self):
        from .scene import draw_pair
        kind = self.kinds[self.next % len(self.kinds)]
        kw = {k: v for k, v in self.gen.items() if k not in ("crops_per_scene",)}
        kw["canvas"] = max(kw.get("canvas", 128), 2 * self.size)
        (s1, d1), (s2, d2), info = draw_pair((self.size,) * 3, seed=self.next, device=self.device, kind=kind, crops_per_scene=4, **kw)
        self.next += 1
        prep = lambda d: (d["image"].to(torch.uint8), math.log2(d["voxel_um"]))
        return prep(d1), prep(d2), info

    def refresh(self, k: int = 1):
        for _ in range(k):
            i = self.next % len(self.items); self.items[i] = self._make()

    def batch(self, B: int, g: torch.Generator, kind: str):
        pool = [it for it in self.items if it[2]["kind"] == kind]
        if not pool:
            return None
        idx = torch.randint(0, len(pool), (B,), generator=g).tolist()
        sel = [pool[i] for i in idx]
        if kind == "zoom":                          # one offset per batch (the first)
            off = sel[0][2]["offset_cells"]; sel = [s_ for s_ in sel if s_[2]["offset_cells"] == off]
        xa = torch.stack([s_[0][0].float() / 255 for s_ in sel])[:, None]; xb = torch.stack([s_[1][0].float() / 255 for s_ in sel])[:, None]
        va = torch.tensor([s_[0][1] for s_ in sel], device=xa.device); vb = torch.tensor([s_[1][1] for s_ in sel], device=xa.device)
        return xa, xb, va, vb, sel[0][2]


class RealBuffer:
    """Random crops of unlabelled volumes (OME-Zarr levels, 3-D arrays or ``.npz`` crops) for the self-supervised
    consistency terms. Crops hold at least ``min_tissue`` tissue (background of either polarity masked) and are
    normalised by the tissue's robust range (the medium clipped or kept, at random)."""

    def __init__(self, sources: list, size: int, device, n: int = 64, seed: int = 0, min_tissue: float = 0.5):
        import zarr
        self.size, self.device = size, device
        rng = np.random.default_rng(seed); self.items = []
        arrs = []
        for src in sources:
            path, _, level = src.partition("@")
            if path.endswith(".npz"):               # pre-extracted crops
                z = np.load(path)
                crops = z["crops"]
                for i in rng.permutation(crops.shape[0])[:n]:
                    c = crops[i]
                    if c.shape[0] < size:
                        continue
                    o = [int(rng.integers(0, c.shape[k] - size + 1)) for k in range(3)]
                    self.items.append(torch.tensor(c[o[0]:o[0] + size, o[1]:o[1] + size, o[2]:o[2] + size], dtype=torch.uint8, device=device))
                continue
            a = zarr.open(path, mode="r")
            if hasattr(a, "keys") and not hasattr(a, "shape"):
                a = a[level or "0"]
            arrs.append(a)
        tries = 0
        while arrs and len(self.items) < n and tries < 20 * n:
            tries += 1
            a = arrs[rng.integers(0, len(arrs))]
            o = [int(rng.integers(0, max(1, s - size))) for s in a.shape[-3:]]
            x = np.asarray(a[tuple(slice(p, p + size) for p in o)], dtype=np.float32)
            if x.shape != (size,) * 3 or float(x.std()) < 1e-6:
                continue
            from .realdata import normalise
            y, share = normalise(torch.tensor(x, device=device), clip_medium=bool(rng.random() < 0.5))
            if share < min_tissue:                  # mostly background, dark or bright
                continue
            self.items.append((y * 255).round().to(torch.uint8))

    def batch(self, B: int, g: torch.Generator):
        idx = torch.randint(0, len(self.items), (B,), generator=g).tolist()
        return torch.stack([self.items[i].float() / 255 for i in idx])[:, None]


# ---------------------------------------------------------------------------------------------------------------
# difficulty widening
# ---------------------------------------------------------------------------------------------------------------

EASY = {"bundle": 0.5, "sheet": 0.15, "tree": 0.15, "cells": 0.1, "plywood": 0.05, "fascicles": 0.05}


def difficulty_gen(level: float, base: dict) -> dict:
    """Scene-generator settings at a difficulty ``level`` in [0, 1]: from line-rich scenes of one or two populations
    and mostly additive images, to every family, up to five populations and every imaging physics."""
    from .scene import Scene
    lv = float(min(max(level, 0.0), 1.0))
    g = dict(base or {})
    fw = g.pop("family_w", None) or {}              # per-family factors (e.g. {"compose": 0}: off)
    fam = {k: ((1 - lv) * EASY.get(k, 0.0) + lv * v) * float(fw.get(k, 1.0)) for k, v in Scene.FAMILIES.items()}
    # packed tissue is allowed at every level (it has its own scene-level gate: Scene.build, P_TISSUE)
    fam["tissue"] = float(fw.get("tissue", 1.0))
    g.update({"families": fam, "n_groups": [1, 1 + int(round(1 + 3 * lv))], "p_physics": 0.25 + 0.6 * lv})
    return g


# ---------------------------------------------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------------------------------------------

def sh_peaks(coef: torch.Tensor, lmax: int):
    """Calibrated peaks of an SH field ``(c, z, y, x)``: ``dirs (3, 3, N)``, ``vals (3, N)``."""
    from .sh import peaks_gated
    D, V, _ = peaks_gated(coef.flatten(1).float(), lmax, nmax=3)
    return D, V


def peak_metrics(Dp, Vp, Dt, Vt, mass_t, m_min: float = 0.05, sep_deg: float = 30.0) -> dict:
    """Peak-based scores of predicted against target peaks (``(3, 3, N)`` / ``(3, N)``).

    On cells with one target peak: the top predicted peak's axis error and how often a second appears. On cells with
    two target peaks more than ``sep_deg`` apart: how often two are predicted and their mean axis error (best
    pairing)."""
    ang = lambda a, b: torch.rad2deg(torch.acos((a * b).sum(0).abs().clamp(max=1)))
    nt, npd = (Vt > 0).sum(0), (Vp > 0).sum(0)
    live = mass_t > m_min
    one = live & (nt == 1) & (Vp[0] > 0)
    two = live & (nt == 2) & (ang(Dt[0], Dt[1]) > sep_deg)
    out = {}
    if bool(one.any()):
        out["single_deg"] = float(ang(Dp[0][:, one], Dt[0][:, one]).median()); out["false_second"] = float((npd[one] >= 2).float().mean())
        out["n_single"] = int(one.sum())
    if bool(two.any()):
        det = two & (npd >= 2)
        out["second_detected"] = float(det.sum() / two.sum()); out["n_crossing"] = int(two.sum())
        if bool(det.any()):
            e = torch.minimum((ang(Dp[0][:, det], Dt[0][:, det]) + ang(Dp[1][:, det], Dt[1][:, det])) / 2,
                              (ang(Dp[0][:, det], Dt[1][:, det]) + ang(Dp[1][:, det], Dt[0][:, det])) / 2)
            out["crossing_deg"] = float(e.median())
        # the dominant population at crossings (any prediction): the best predicted peak against the target's top
        top = two & (Vp[0] > 0)
        if bool(top.any()):
            e1 = torch.stack([ang(Dp[j][:, top], Dt[0][:, top]) + 90 * (Vp[j][top] <= 0) for j in range(3)]).amin(0)
            out["crossing_top_deg"] = float(e1.median())
    return out


READOUT = "lobes"                  # "lobes" | "aux": which head of a lobe model predict_sh reads


@torch.no_grad()
def predict_sh(model, x: torch.Tensor, voxel=None, lmax: int = 8, desc_lmax: int = 4):
    """A model's density field (and descriptor densities, lobes) for images ``x (B, 1, ...)``: lobe models are
    rendered, SH-head models read directly."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        P = model(x, voxel=voxel) if getattr(model, "head", "slots") == "lobes" else model(x)
    # the same trunk's free-harmonics head
    if "lobes_raw" in P and READOUT == "aux" and "aux_sh" in P:
        return P["aux_sh"].float(), None, None, P
    if "lobes_raw" in P:
        f, fq, lob = density(model, P["lobes_raw"], lmax, desc_lmax)
        return f, fq, lob, P
    if "sh" in P:
        return P["sh"].float()[:, :n_coef(model.sh_lmax)], None, None, P
    from .sh import density_sh                      # a slots-only model: its present slots' density
    m = P["mass"].float() * (P["presence_logit"].float() > 0)
    return density_sh(P["slots"].float(), m, lmax), None, None, P


@torch.no_grad()
def evaluate_lobes(model, val: list, margin: int, device, lmax: int = 8, desc_lmax: int = 4) -> dict:
    """Scores on the held-out set.

    Loss parts; per-order unexplained energy of the density (order 2, orders 4+); peak scores from the rendered density
    and from the lobes; descriptors at the target's top peak; identity separation (an AUC over voxel pairs: the same
    object closer than different objects); points and fraction correlation."""
    from .train import stack
    model.eval()
    acc = {}; num2 = den2 = num4 = den4 = 0.0
    P_sh, P_lb, T_sh, M_t = [], [], [], []
    desc_err = []; id_auc = []; pts_c = []; fr = []
    crop = lambda v: _cells(v, margin)
    for img, t in val:
        x, T = stack([(img, t)], aug=False)
        v = T.get("voxel")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            P = model(x, T.get("q_points"), T.get("q_sigma"), voxel=v)
        _, parts, rend = lobe_loss(model, P, T, margin, update=False, lmax=lmax, desc_lmax=desc_lmax)
        for k, val_ in parts.items():
            acc[k] = acc.get(k, 0.0) + val_ / len(val)
        f = crop(rend["f"])[0]; nsh = n_coef(lmax); ct = crop(T["sh_t"])[0, :nsh].float()
        num2 += float(((f[1:6] - ct[1:6]) ** 2).sum()); den2 += float((ct[1:6] ** 2).sum())
        num4 += float(((f[6:] - ct[6:]) ** 2).sum()); den4 += float((ct[6:] ** 2).sum())
        P_sh.append(f.flatten(1)); T_sh.append(ct.flatten(1)); M_t.append(ct[0].flatten() / math.sqrt(4 * math.pi))
        if rend["lob"] is not None:
            dirs, mass = LB.lobe_peaks({k: crop(v_) for k, v_ in rend["lob"].items()})
            P_lb.append((dirs[0].flatten(2)[:3], mass[0].flatten(1)[:3]))
        if model.nd and T["sh_t"].shape[1] > nsh:   # descriptors at the target's top peak
            from .sh import peaks as _pk
            nd_ = n_coef(desc_lmax)
            ctf = crop(T["sh_t"])[0].float().flatten(1)
            strong = ctf[0] > 0.1 * math.sqrt(4 * math.pi)
            if bool(strong.any()):
                fq = density(model, P["lobes_raw"], lmax, desc_lmax)[1]
                cpf = torch.cat([f.flatten(1), crop(fq)[0].flatten(1)], 0)
                td_, _ = _pk(ctf[:nsh, strong], lmax, nmax=1, rel=0.3)
                Yd = LB.sh_poly(td_[0].T, desc_lmax)
                def at(cf):
                    f0 = (Yd * cf[:nd_, strong].T).sum(1).clamp_min(1e-6)
                    return torch.stack([(Yd * cf[nsh + j * nd_: nsh + (j + 1) * nd_, strong].T).sum(1) / f0 for j in range(model.nd)])
                desc_err.append(at(cpf) - at(ctf))
        if "identity" in P and "identity_t" in T:
            e = _cells(P["identity"], 2 * margin)[0].float().flatten(1); ids = _cells(T["identity_t"], 2 * margin)[0].flatten()
            ok = torch.nonzero(ids >= 0)[:, 0]
            if ok.numel() > 100:
                gi = torch.Generator(device="cpu").manual_seed(0)
                i = ok[torch.randint(0, ok.numel(), (4000,), generator=gi).to(ok.device)]; j = ok[torch.randint(0, ok.numel(), (4000,), generator=gi).to(ok.device)]
                same = ids[i] == ids[j]; dist = (e[:, i] - e[:, j]).norm(dim=0)
                if bool(same.any()) and bool((~same).any()):
                    ds, dd = dist[same], dist[~same]
                    id_auc.append(float((dd[None, :1000] > ds[:1000, None]).float().mean()))
        if "points" in P and "points_t" in T:
            a_, b_ = crop(P["points"]).float().flatten(), crop(T["points_t"]).float().flatten()
            if float(b_.std()) > 0:
                pts_c.append(float(torch.corrcoef(torch.stack([a_, b_]))[0, 1]))
        fr.append((torch.sigmoid(crop(P["fraction_logit"]).float()).flatten(), crop(T["fraction"]).float().flatten()))
    model.train()
    out = {f"val_{k}": float(v) for k, v in acc.items()}
    if "val_lobe_l2_num" in out:
        out["val_lobe_l2_ratio"] = out.pop("val_lobe_l2_num") / max(out["val_gate_den"], 1e-9)
    out["val_unexplained_l2"] = num2 / max(den2, 1e-9); out["val_unexplained_l4p"] = num4 / max(den4, 1e-9)
    Pc, Tc, Mc = torch.cat(P_sh, 1), torch.cat(T_sh, 1), torch.cat(M_t)
    Dp, Vp = sh_peaks(Pc, lmax); Dt, Vt = sh_peaks(Tc, lmax)
    out.update({f"val_sh_{k}": v for k, v in peak_metrics(Dp, Vp, Dt, Vt, Mc).items()})
    if P_lb:
        Dl = torch.cat([d for d, _ in P_lb], -1); Ml = torch.cat([m for _, m in P_lb], -1)
        out.update({f"val_lobe_{k}": v for k, v in peak_metrics(Dl, Ml, Dt, Vt, Mc).items()})
    if desc_err:
        de = torch.cat(desc_err, 1)
        for j, nm in enumerate(("logsize", "wall", "lumen", "flat", "angular")[:de.shape[0]]):
            out[f"val_desc_{nm}_mae"] = float(de[j].abs().median())
    if id_auc:
        out["val_identity_auc"] = float(np.mean(id_auc))
    if pts_c:
        out["val_points_corr"] = float(np.mean(pts_c))
    fp = torch.cat([a for a, _ in fr]); ft = torch.cat([b for _, b in fr])
    out["val_fraction_corr"] = float(torch.corrcoef(torch.stack([fp, ft]))[0, 1]) if float(ft.std()) > 0 else float("nan")
    return out
