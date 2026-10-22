"""Inference with the population model. Every output is computed from the network's output; the image is the network's
input and nothing else (no autocorrelation, no noise floor from the image, no intensity threshold).

    from shapetensor.learn.engine import PopulationEngine
    eng = PopulationEngine("model.pt")
    out = eng.run(volume)                                 # a block: lattice outputs and the voxel field
    eng.run_volume(src, dst, chunk=(256, 256, 256))       # a whole volume in chunks, lattice outputs to a zarr group

The steps, each on the previous one's output:

1. **intensity window** -- grey levels to 0-1 by one pair of percentiles for the whole volume (``window``; measured once,
   so every chunk sees the same mapping);
2. **network** -- per lattice cell (stride 4, centred at voxel 4i + 1.5): ``K`` slots (shape tensor, mass, kind,
   identity embedding), the shape fraction, the class shares, the size;
3. **split by kind** -- line slots carry directions; sheet and interface slots give the sheet normal and mass; blob
   slots a blob mass;
4. **carry-forward** -- each line population smoothed along its own axis in the slots' space and bridged across gaps
   no longer than ``reach`` (:func:`carry`: population by population, never across), then rendered as a density on the
   direction grid (each slot a Bingham lobe of its tensor weighted by its mass; the cell's total scaled to its weight
   ``min(1, fraction / full_fraction)``); a direction held on both sides of a cell within ``reach`` is lifted to that
   level there (the bridge of :func:`~shapetensor.populations.momentum`, without its smoothing): a population the
   network lost in a few cells comes back, a free end is never extended;
5. **populations** -- lobes of the carried density (:func:`~shapetensor.populations.peaks`; ``nmax``, ``min_sep_deg``,
   ``rel``, ``min_share``): directions, masses, ``N`` (the heatmap); identities by linking aligned lobes between
   neighbouring cells (:func:`~shapetensor.populations.label_populations`);
6. **voxel field** -- the carried density interpolated to every voxel and read there (the first vector everywhere,
   weighted; further vectors where the weight passes ``min_weight``), labels from the nearest cell.

Also per slot: its axis, linearity and planarity (the FA-like coherence of each population), straight from the tensor.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from ..field import Orientations
from ..populations import label_populations, momentum, peaks
from .generator import full3
from .model import PopulationNet
from .targets import render_odf

KIND_LINE, KIND_SHEET, KIND_BLOB, KIND_IFACE = 0, 1, 2, 3


def window(volume, lo_q: float = 0.005, hi_q: float = 0.995, n: int = 2_000_000) -> tuple[float, float]:
    """The intensity window of a whole volume: its ``lo_q`` and ``hi_q`` quantiles (on a strided subsample)."""
    v = np.asarray(volume).ravel()
    v = v[:: max(1, v.size // n)].astype(np.float64)
    return float(np.quantile(v, lo_q)), float(np.quantile(v, hi_q))


def carry(S: torch.Tensor, m: torch.Tensor, kind: torch.Tensor, axis: torch.Tensor, *, reach: int = 6, sigma_along: float = 3.0,
          match_deg: float = 20.0, kinds=(0,)) -> tuple[torch.Tensor, torch.Tensor]:
    """Population-specific carry-forward in the slots' own space. For each slot of a kind in ``kinds`` (lines by
    default) at each cell, the cells along its own axis at ±1..``reach`` cells are visited; in each, the matching slot
    (same kind, axis within ``match_deg``, the heaviest such) is taken. The slot's tensor becomes the mass- and
    distance-weighted mean (Gaussian ``sigma_along`` cells) of itself and its matches, its mass the weighted mean of
    the masses, raised to the weaker of the two sides' largest matched mass (a population present on both sides of a
    gap within ``reach`` is bridged across it, never extended past a free end). Slots never mix with other populations.
    ``S (K, 6, z, y, x)``, ``m``, ``kind`` ``(K, z, y, x)``, ``axis (K, 3, z, y, x)``; returns the new ``S`` and ``m``."""
    K = m.shape[0]; cs = m.shape[1:]; dev = m.device
    n = m[0].numel()
    pos = torch.stack(torch.meshgrid(*(torch.arange(c, device=dev, dtype=torch.float32) for c in cs), indexing="ij")).reshape(3, n)
    Sf, mf, kf, af = S.reshape(K, 6, n), m.reshape(K, n), kind.reshape(K, n), axis.reshape(K, 3, n)
    lim = torch.tensor(cs, device=dev).view(3, 1) - 1
    cosm = math.cos(math.radians(match_deg))
    live = torch.zeros(K, n, dtype=torch.bool, device=dev)
    for k_ in kinds:
        live |= kf == k_
    live &= mf > 0
    accS = Sf * mf[:, None]; accM = mf.clone(); accW = torch.ones_like(mf)
    side = [torch.zeros_like(mf), torch.zeros_like(mf)]
    for t in [i for i in range(-reach, reach + 1) if i != 0]:
        w = math.exp(-0.5 * (t / sigma_along) ** 2)
        for k in range(K):
            q = (pos + t * af[k]).round().clamp(min=torch.zeros_like(lim), max=lim).long()      # (3, n): the cell t steps along the slot's axis
            qi = (q[0] * cs[1] + q[1]) * cs[2] + q[2]
            inb = ((pos + t * af[k]).round() == q.float()).all(0)                               # inside the block
            nS, nm, nk, na = Sf[:, :, qi], mf[:, qi], kf[:, qi], af[:, :, qi]                    # the neighbour's K slots
            ok = (nk == kf[k][None]) & ((na * af[k][None]).sum(1).abs() >= cosm) & (nm > 0) & inb[None]
            j = torch.where(ok, nm, torch.full_like(nm, -1.0)).argmax(0)                         # the heaviest matching slot
            hit = ok.gather(0, j[None])[0] & live[k]
            mm = nm.gather(0, j[None])[0] * hit
            SS = nS.gather(0, j[None, None].expand(1, 6, n))[0]
            accS[k] += w * mm[None] * SS; accM[k] += w * mm; accW[k] += w * hit
            sd = 0 if t < 0 else 1
            side[sd][k] = torch.maximum(side[sd][k], mm)
    S2 = torch.where(live[:, None], accS / (accS[:, :3].sum(1, keepdim=True)).clamp_min(1e-12), Sf)   # trace 1 again
    m2 = torch.where(live, torch.maximum(accM / accW, torch.minimum(side[0], side[1])), mf)
    return S2.reshape(S.shape), m2.reshape(m.shape)


class PopulationEngine:
    def __init__(self, checkpoint: str | dict | PopulationNet | None = None, *, device: str = "cuda", nmax: int = 3, min_sep_deg: float = 25.0, rel: float = 0.2,
                 min_share: float = 0.1, full_fraction: float = 0.1, min_weight: float = 0.2, kappa: float = 20.0, momentum: bool = True,
                 reach: int = 6, sigma_along: float = 3.0, label_angle_deg: float = 20.0, min_fixels: int = 8, exact: bool = True):
        self.device = device
        if isinstance(checkpoint, PopulationNet):
            self.net = checkpoint.to(device).eval(); self.rf = None
        else:
            self.net = PopulationNet().to(device).eval(); self.rf = None
            if checkpoint is not None:
                ck = torch.load(checkpoint, map_location=device) if isinstance(checkpoint, (str, Path)) else checkpoint
                self.net.load_state_dict(ck["model"]); self.rf = ck.get("receptive_field")
        if self.rf is None:
            from .model import receptive_field
            self.rf = receptive_field(self.net)
        self.exact = exact                                                                      # TF32 off: a chunked run equals one pass to rounding
        # the chain's lattice stages, on the model's lattice (stride 4, cells centred at 4i + 1.5)
        self.o = Orientations(scale=10.0, step=4, nmax=nmax, min_sep_deg=min_sep_deg, rel=rel, min_share=min_share, momentum=momentum, reach=reach,
                              sigma_along=sigma_along, min_weight=min_weight, label_angle_deg=label_angle_deg, min_fixels=min_fixels, device=device)
        self.o.origins = (1.5, 1.5, 1.5)
        self.full_fraction, self.kappa = float(full_fraction), float(kappa)

    # -- the network -------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def network(self, v: torch.Tensor, win: tuple[float, float]) -> dict:
        """The network on a block ``v (Z, Y, X)`` (any dtype; Z, Y, X multiples of 4) after the intensity window."""
        lo, hi = win
        x = ((v.to(self.device).float() - lo) / max(hi - lo, 1e-6)).clamp(0, 1)[None, None]
        tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = not self.exact
        try:
            out = self.net(x)
        finally:
            torch.backends.cudnn.allow_tf32 = tf32
        return {k: v_[0].float() for k, v_ in out.items()}

    # -- everything after it: on the network's output only ----------------------------------------------------------

    def populations(self, net: dict, labels: bool = True) -> dict:
        """Steps 3-5 on the network's lattice output (``net``: one block's outputs without the batch dim)."""
        o = self.o; grid = o.grid
        S, m = net["slots"], net["mass"]                                                        # (K, 6, c), (K, c)
        kind = net["kind"].argmax(1)                                                            # (K, c)
        fraction = torch.sigmoid(net["fraction_logit"]); classes = torch.softmax(net["class_logits"], 0)
        # per slot: axis, linearity, planarity
        ev, V = torch.linalg.eigh(full3(S.movedim(1, -1)))                                       # ascending
        l1, l2, l3 = ev[..., 2], ev[..., 1], ev[..., 0]
        lin = (l1 - l2) / l1.clamp_min(1e-9); pla = (l2 - l3) / l1.clamp_min(1e-9)
        planar = (kind == KIND_SHEET) | (kind == KIND_IFACE)
        axis = torch.where(planar[..., None], V[..., 0], V[..., 2]).movedim(-1, 1)                # (K, 3, c)
        # step 3: split by kind
        line_m = m * (kind == KIND_LINE); sheet_m = m * planar; blob_m = m * (kind == KIND_BLOB)
        sheet_mass = sheet_m.sum(0)
        sheet_normal = torch.gather(axis, 0, sheet_m.argmax(0)[None, None].expand(1, 3, *m.shape[1:]))[0]
        # step 4: each line population carried forward along itself in the slots' space, then rendered as a density
        if o.momentum:
            S, line_m = carry(S, line_m, kind, axis, reach=o.reach, sigma_along=o.sigma_along)
        weight = (fraction / self.full_fraction).clamp(0, 1)
        f = render_odf(S, line_m, grid, self.kappa)                                              # (n, c)
        tot = f.sum(0)
        g = f / tot.clamp_min(1e-12)[None] * (weight * (tot > 1e-9))[None]
        if o.momentum:                                                                          # a population the network lost in a few cells, held on both sides: bridged
            g = momentum(g, grid, reach=o.reach, smooth=False)
        wt = g.sum(0)
        # step 5: lobes, N, identities
        dirs, share = peaks(g / wt.clamp_min(1e-12)[None], grid, nmax=o.nmax, rel=o.rel, min_mass=o.min_share, min_sep_deg=o.min_sep_deg)
        sheet = sheet_mass > line_m.sum(0)
        mass = share * wt[None] * (wt >= o.min_weight).float()                                   # no populations below the weight floor
        out = {"fodf": g, "weight": wt, "sheet": sheet, "dirs_cells": dirs, "mass": mass, "N": (mass > 0).sum(0),
               "fraction": fraction, "classes": classes, "size": net["log_size"].exp(),
               "slot_axis": axis, "slot_mass": m, "slot_kind": kind, "slot_linearity": lin, "slot_planarity": pla, "slot_embed": net["embed"],
               "sheet_normal": sheet_normal, "sheet_mass": sheet_mass, "blob_mass": blob_m.sum(0)}
        if labels:
            out["labels_cells"], out["table"] = label_populations(dirs, mass, max_angle_deg=o.label_angle_deg, min_fixels=o.min_fixels)
        return out

    def field(self, pop: dict, shape, labels: bool = True) -> dict:
        """Step 6: the voxel field of a block of ``shape``, from the carried density. Vectors are kept where a sheet
        dominates (the density holds line slots only, so a sheet adds no ring to mistake for populations, and a fibre
        crossing a boundary keeps its vector)."""
        return self.o.field({**pop, "sheet": torch.zeros_like(pop["sheet"])}, shape, labels=labels)

    def run(self, v, win: tuple[float, float] | None = None, native: bool = True, labels: bool = True) -> dict:
        """One block: the network, the populations, and (``native``) the voxel field. ``win``: the intensity window
        (None: this block's own percentiles -- for a whole volume use :func:`window` once)."""
        v = torch.as_tensor(np.asarray(v)) if not torch.is_tensor(v) else v
        win = win if win is not None else window(v.cpu().numpy())
        net = self.network(v, win)
        pop = self.populations(net, labels=labels)
        if native:
            pop.update({f"voxel_{k}": x for k, x in self.field(pop, tuple(v.shape), labels=labels).items()})
        return pop

    # -- a whole volume ---------------------------------------------------------------------------------------------

    def halo(self) -> int:
        """Voxels of context a chunk needs for its cells to equal one pass: half the receptive field plus the
        carry-forward's reach, rounded up to whole cells."""
        need = self.rf / 2 + 4 * (self.o.reach + 1 if self.o.momentum else 0)
        return int(4 * math.ceil(need / 4))

    def run_volume(self, src, dst: str, *, chunk=(256, 256, 256), win: tuple[float, float] | None = None, min_fraction: float = 0.01, log=print):
        """A whole volume ``src`` (an array-like ``(Z, Y, X)``; zarr arrays are read chunk by chunk) in chunks with a halo
        of :meth:`halo` voxels; lattice outputs written to the zarr group ``dst``: ``N``, ``dirs`` ``(nmax, 3, ...)``,
        ``mass``, ``fraction``, ``classes``, ``sheet_normal``, ``sheet_mass``, ``blob_mass``. A chunk whose fraction stays
        below ``min_fraction`` everywhere is written as empty after the network (no carry-forward, no peaks)."""
        import zarr
        Z, Y, X = src.shape
        assert all(c % 4 == 0 for c in chunk), "chunk must be a multiple of 4"
        cz, cy, cx = (math.ceil(n / 4) for n in (Z, Y, X)); h = self.halo(); o = self.o
        win = win if win is not None else window(src[:: max(1, Z // 64)])
        root = zarr.open_group(dst, mode="w")
        cc = tuple(c // 4 for c in chunk)
        arr = {"N": root.create_array("N", shape=(cz, cy, cx), chunks=cc, dtype="uint8"),
               "dirs": root.create_array("dirs", shape=(o.nmax, 3, cz, cy, cx), chunks=(o.nmax, 3) + cc, dtype="float16"),
               "mass": root.create_array("mass", shape=(o.nmax, cz, cy, cx), chunks=(o.nmax,) + cc, dtype="float16"),
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
                    lo = [max(0, a - h) for a in (z0, y0, x0)]
                    hi = [min(n, a + c + h) for a, c, n in zip((z0, y0, x0), chunk, (Z, Y, X))]
                    pad_hi = [(-(b - a)) % 4 for a, b in zip(lo, hi)]                               # whole cells
                    blk = np.asarray(src[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
                    if any(pad_hi):
                        blk = np.pad(blk, [(0, p) for p in pad_hi], mode="edge")
                    net = self.network(torch.as_tensor(blk), win)
                    core = tuple(slice((a - l) // 4, (a - l) // 4 + math.ceil(min(c, n - a) / 4)) for a, l, c, n in zip((z0, y0, x0), lo, chunk, (Z, Y, X)))
                    dst_sl = tuple(slice(a // 4, a // 4 + (s.stop - s.start)) for a, s in zip((z0, y0, x0), core))
                    frac = torch.sigmoid(net["fraction_logit"])
                    if float(frac[core].max()) < min_fraction:
                        arr["fraction"][dst_sl] = frac[core].half().cpu().numpy(); continue
                    pop = self.populations(net, labels=False)
                    put = lambda name, x, lead: arr[name].__setitem__((slice(None),) * lead + dst_sl, x[(slice(None),) * lead + core].half().cpu().numpy() if x.dtype.is_floating_point else x[(slice(None),) * lead + core].cpu().numpy())
                    arr["N"][dst_sl] = pop["N"][core].to(torch.uint8).cpu().numpy()
                    put("dirs", pop["dirs_cells"], 2); put("mass", pop["mass"], 1); put("fraction", pop["fraction"], 0)
                    put("classes", pop["classes"], 1); put("sheet_normal", pop["sheet_normal"], 1); put("sheet_mass", pop["sheet_mass"], 0); put("blob_mass", pop["blob_mass"], 0)
                    n_done += 1
        log(f"run_volume: {n_done} chunks with content, halo {h} voxels, window {win}")
        return root
