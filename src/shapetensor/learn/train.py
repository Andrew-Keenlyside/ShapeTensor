"""Training the population model on the synthesis, generated on the fly.

    python -m shapetensor.learn.train --out DIR [--steps 20000] [--size 128] [--batch 4] [--accum 4] [--buffer 128]

Samples are drawn (:func:`~shapetensor.learn.generator.draw`: a modality, matched content, its acquisition) and their
lattice targets computed on the GPU into a replay buffer of (image, target) pairs; each step takes ``accum``
micro-batches of ``batch`` from it (an effective batch of ``batch * accum``), applies one random cube-group element to
each pair (exact on the lattice) and a random intensity window jitter, and replaces ``refresh`` pairs with new draws.

A curriculum, against the collapse to constant outputs seen when everything is asked at once:

* **nuisance strength** ramps from ``strength0`` to 1 over the first half of training (clear images first);
* **orientation first** -- the slots' orientation and the mixture tensor from the start; mass, presence, kind and the
  Hellinger term ramped in from max(300, 5% of the steps) over 300 steps (present from the start, they swamp the
  orientation signal and the network never learns it); the cell scalars (classes, fraction, size) and the identity
  term off for the first 20% of steps, ramped in by 40%.

AdamW (``lr`` 3e-4), warm-up then cosine, bf16 autocast, clipping at 5. A fixed held-out set (seeds from 10^7, full
strength, ``strength1``) is scored every ``val_every`` steps with metrics that can see learning: the matched slot axes' error against
the untrained network's (the chance level of a best-of-K match), the spread and correlation of the predicted fraction
(a collapsed network is constant), and the engine's population count against the target's. ``--overfit N``: a fixed
set of N pairs and no refresh, scored on those pairs (a sanity check that the loss can be driven down).
"""
from __future__ import annotations

import argparse
import json
import os
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as Fn

from ..lift import hemisphere
from .generator import draw, eig3
from .losses import WEIGHTS, augment, gather_slots, loss, match
from .model import PopulationNet, receptive_field
from .targets import target

TKEYS = ("slots", "mass", "kind", "layer", "fraction", "classes", "size", "size_mask")
HALF = ("slots", "mass", "fraction", "classes", "size")
QKEYS = ("q_points", "q_sigma", "q_sh", "sh_t")
LKEYS = ("identity_t", "points_t", "voxel", "fraction")    # the lobe model's extra targets
EKEYS = ("e_points", "e_sigma", "e_level", "e_scale", "e_tot", "e_bgm", "e_node", "e_mass", "e_vis", "e_T6", "e_desc", "e_anc", "e_n", "e_R", "e_kappa", "e_gnode", "region_t", "iface_t", "bg_t", "e_bg",
         "ori_t", "iso_t", "e_ovis")   # the entity model's
HEAD = "slots"                     # "slots" | "lobes" (set by train())
DESC_KEY = "descriptors"           # the drawn descriptors the descriptor densities use ("descriptors5": the scenes' five)
SH_EXACT = 0                       # > 0: the SH head's lattice target from the exact per-voxel density to this order
DESC_LMAX = 0                      # > 0: also the descriptor densities (thickness, hollowness, angularity) to this order
CONTEXT = None                     # (lo, hi): also train the query decoder on copies whose input outside a random cube of
                                   # half-width lo..hi voxels around a query is replaced by the mean, with the visible
                                   # context per query as an input


def context_mask(x: torch.Tensor, points: torch.Tensor, half: torch.Tensor):
    """Copies of images ``x (B, 1, Z, Y, X)`` visible only inside a cube of half-width ``half (B,)`` around each
    sample's first query point, blended to the image's mean over a 4-voxel cosine taper inside the faces (a hard step
    would read as an interface); and each query's visible context (its distance to the cube's faces), ``(B, Q)``."""
    B = x.shape[0]; size = torch.tensor(x.shape[-3:], device=x.device, dtype=points.dtype)
    c = points[:, 0]                                                                                  # (B, 3)
    lo = (c - half[:, None]).clamp(min=0); hi = torch.minimum(c + half[:, None], size - 1)
    ar = [torch.arange(n, device=x.device, dtype=points.dtype) for n in x.shape[-3:]]
    depth = torch.full((B, *x.shape[-3:]), 1e9, device=x.device, dtype=points.dtype)                 # distance inside the cube (< 0 outside)
    for j in range(3):
        a = ar[j].view(*([1] * (j + 1)), -1, *([1] * (2 - j)))
        depth = torch.minimum(depth, torch.minimum(a - lo[:, j].view(B, 1, 1, 1), hi[:, j].view(B, 1, 1, 1) - a))
    w = (0.5 - 0.5 * torch.cos(math.pi * (depth / 4).clamp(0, 1)))                                   # a 4-voxel cosine taper: no hard step
    mean = x.mean(dim=(2, 3, 4), keepdim=True)
    xm = w[:, None] * x + (1 - w[:, None]) * mean
    ctx = torch.minimum(points - lo[:, None], hi[:, None] - points).amin(-1)                        # (B, Q); < 0: outside
    return xm, ctx
QUERY = None                       # {"n_points", "n_levels", "lmax"}: draw query targets per sample (set by train())
SCALARS = ("classes", "fraction", "size", "identity")
LATE = ("mass", "presence", "kind", "hellinger")                                                   # after orientation: they swamp it if present from the start


CACHE = None                       # a directory: training pairs cached on disk, shared by runs drawing the same pairs
_CODE_VERSION = None


def _code_version() -> str:
    """A hash of the sources that determine a pair (generator, targets, query, SH, filters): a changed generator never
    serves stale cached pairs."""
    global _CODE_VERSION
    if _CODE_VERSION is None:
        import hashlib
        from pathlib import Path as _P
        here = _P(__file__).parent
        h = hashlib.sha1()
        for f in (here / "generator.py", here / "targets.py", here / "query.py", here / "sh.py", here / "scene.py", here / "physics.py", here.parent / "filters.py"):
            h.update(f.read_bytes())
        _CODE_VERSION = h.hexdigest()[:12]
    return _CODE_VERSION


def make(seed: int, size: int, device, strength: float = 1.0, gen: dict | None = None) -> tuple[torch.Tensor, dict]:
    """One training pair (see :func:`_make_pair`); with :data:`CACHE` set, read from / written to a disk cache keyed by
    everything that determines the pair (atomic writes: concurrent runs share it)."""
    if CACHE is None:
        return _make_pair(seed, size, device, strength, gen)
    import hashlib
    from pathlib import Path as _P
    from . import targets as _tg
    key = json.dumps([seed, size, round(float(strength), 4), gen or {}, QUERY, SH_EXACT, DESC_LMAX, _tg.ORDER, HEAD, DESC_KEY, _code_version()], sort_keys=True, default=str)
    f = _P(CACHE) / (hashlib.sha1(key.encode()).hexdigest() + ".pt")
    if f.exists():
        try:
            img, out = torch.load(f, map_location=device, weights_only=False)
            return img, out
        except Exception:
            pass
    img, out = _make_pair(seed, size, device, strength, gen)
    tmp = f.with_suffix(f".{os.getpid()}.tmp")
    torch.save((img.cpu(), {k: v.cpu() for k, v in out.items()}), tmp); os.replace(tmp, f)
    return img, out


READ_MIN = 0.01                    # entities: a crop's visible material (population occupancy x visibility, away from the faces) below this
READ_KEEP = 0.25                   # share is (almost) unreadable: kept at this rate, else drawn again (about 40 % of draws were; 1.0: off)


def visible_material(s, d, margin: int = 8) -> float:
    """The visible material of a drawn crop: the population layers' occupancy x visibility, summed and clipped at 1,
    its mean away from the faces -- the population-region target's mean."""
    occ = d["occupancy"]; K = occ.shape[0]
    keep = [k for k, L in enumerate(s.layers) if L.role not in NOT_OBJECT]
    if not keep:
        return 0.0
    vis = d["visibility"].float(); vis = vis.view(-1, 1, 1, 1).expand(K, *occ.shape[1:]) if vis.dim() == 1 else vis
    m = slice(margin, -margin) if min(occ.shape[1:]) > 4 * margin else slice(None)
    return float((occ[keep][:, m, m, m].float() * vis[keep][:, m, m, m]).sum(0).clamp(0, 1).mean())


def _make_pair(seed: int, size: int, device, strength: float = 1.0, gen: dict | None = None) -> tuple[torch.Tensor, dict]:
    """One training pair: the 8-bit image and its lattice target (floats in half precision). ``gen``: arguments for
    :func:`~shapetensor.learn.generator.draw` (a modality, a background style, crossing groups, layer kinds, ...)."""
    if gen and "scene" in gen:                                                                         # the object-scene generator
        from .scene import draw_scene
        s, d = draw_scene((size,) * 3, seed=seed, device=device, strength=strength, entity_ctx=(HEAD == "entities"), **(gen["scene"] or {}))
        if HEAD == "entities" and READ_KEEP < 1.0:                                                     # the readability cap: a crop with (almost)
            keep = ((seed * 2654435761) % 4294967296) / 4294967296.0 < READ_KEEP                       # no visible material stays at READ_KEEP
            k = 0                                                                                      # (decided once: empty and unreadable crops
            while not keep and k < 4 and visible_material(s, d) < READ_MIN:                            # teach "nothing here"), else is drawn again:
                k += 1; cps = max(1, int((gen["scene"] or {}).get("crops_per_scene", 1) or 1))         # another crop of the same (cached) scene
                s, d = draw_scene((size,) * 3, seed=seed + k * 1_000_003, device=device, strength=strength, entity_ctx=True,   # first, then
                                  scene_seed=seed // cps if k == 1 else None, **(gen["scene"] or {}))                         # new scenes
                d.setdefault("record", {})["redrawn"] = k
    else:
        s, d = draw((size,) * 3, seed=seed, device=device, strength=strength, **(gen or {}))
    t = target(s, d)
    out = {k: (t[k].half() if k in HALF else t[k]) for k in TKEYS}
    F = None
    if SH_EXACT or (QUERY and QUERY.get("exact")):                                                     # the exact per-voxel density
        from .query import voxel_density
        F = voxel_density(s, d, SH_EXACT or QUERY["lmax"], desc_lmax=DESC_LMAX, desc_key=DESC_KEY if DESC_KEY in d else "descriptors",
                          roles_out=NOT_ENTITY if HEAD == "entities" else ("isotropic",))
    if HEAD == "lobes":                                                                                # identities (stride 2), points (lattice), voxel size
        if "identity" in d:
            out["identity_t"] = d["identity"][::2, ::2, ::2].to(torch.int32)
        if "points" in d:
            out["points_t"] = Fn.avg_pool3d(d["points"][None].float(), 4)[0].half()
        out["voxel"] = torch.tensor(math.log2(d["voxel_um"]) if d.get("voxel_um") else float("nan"), device=device)
    if SH_EXACT:                                                                                       # lattice target: window sigma 5, 4³ average
        from .targets import _pool
        out["sh_t"] = _pool(F, 5.0, 4).half()
    if QUERY:                                                                                          # the density at random points and scales
        from .query import draw_queries, layer_density, sample_density
        gq = torch.Generator(device=device).manual_seed(seed + 7919)
        pts, sig = draw_queries((size,) * 3, QUERY["n_points"], QUERY["n_levels"], gq, device)
        out["q_points"], out["q_sigma"] = pts, sig
        out["q_sh"] = (sample_density(F, pts, sig) if F is not None else layer_density(s, d, pts, sig, QUERY["lmax"])).half()
    if HEAD == "entities":                                                                             # entities at random points and scales
        out.update(entity_queries(s, d, size, seed, device))
        if "points" in d:                                                                              # branch and end points (the points head)
            out["points_t"] = Fn.avg_pool3d(d["points"][None].float(), 4)[0].half()
        K = d["occupancy"].shape[0]                                                                    # the population region: the visible
        vis = d["visibility"].float(); vis = vis.view(-1, 1, 1, 1).expand(K, *d["occupancy"].shape[1:]) if vis.dim() == 1 else vis   # occupancy of
        keep = [k for k, L in enumerate(s.layers) if L.role not in NOT_ENTITY]                         # every population layer, each voxel
        rsum = (d["occupancy"][keep].float() * vis[keep]).sum(0) if keep else torch.zeros_like(d["occupancy"][0]).float()
        out["region_t"] = (rsum if REGION_DENSITY else rsum.clamp(0, 1)).half()                       # a density: crossings count both
        ifc = [k for k, L in enumerate(s.layers) if L.role == "interface"]                           # boundaries between background regions:
        out["iface_t"] = (d["occupancy"][ifc].float().sum(0).clamp(0, 1) if ifc else torch.zeros_like(d["occupancy"][0]).float()).half()   # hard negatives
        bgl = [k for k, L in enumerate(s.layers) if L.role == "isotropic"]                           # the background's occupancy (the bg head)
        out["bg_t"] = ((d["occupancy"][bgl].float() * vis[bgl]).sum(0).clamp(0, 1) if bgl else torch.zeros_like(d["occupancy"][0]).float()).half()
        shp_l = [k for k, L in enumerate(s.layers) if L.role == "shape"]                             # the objects by kind (the split head, as the
        ori_k = [k for k in shp_l if getattr(s.layers[k], "kind", "line") in ("line", "sheet")]      # phantoms' targets): oriented material and
        iso_k = [k for k in shp_l if k not in ori_k]; z_ = torch.zeros_like(d["occupancy"][0]).float()   # isotropic objects (densities)
        out["ori_t"] = (d["occupancy"][ori_k].float().sum(0) if ori_k else z_).half()
        out["iso_t"] = (d["occupancy"][iso_k].float().sum(0) if iso_k else z_).half()
        out["voxel"] = torch.tensor(math.log2(d["voxel_um"]) if d.get("voxel_um") else float("nan"), device=device)
    out["kind"] = out["kind"].to(torch.int8); out["layer"] = out["layer"].to(torch.int16)
    return d["image"].to(torch.uint8), out


#: layer roles that are not a population for the entity model: the interfaces between background regions (a boundary is
#: not an object: as targets they taught the network that every intensity step is oriented material -- 63 % of the
#: target mass, edges traced like a structure tensor; fibres crossing a boundary stay entities, the step is only
#: context). The isotropic remainder -- the background, the space without fibres -- **is** a population (one entity,
#: isotropic: a uniform floor under the oriented lobes, inferred everywhere and dropped afterwards by its isotropy), so
#: a region of material whose orientation does not show and a background region are both what they look like, and no
#: target asks for the edge between them. ``set_background_entity(False)``: the isotropic remainder left out (before).
NOT_ENTITY = ("interface",)
#: entities: each region interface a boundary-sheet entity (planar, mass = the boundary's sharpness) -- rendered, so targeted
INTERFACE_SHEETS = True
#: roles that are not objects: what the readability cap and the query placement do not count as material
NOT_OBJECT = ("isotropic", "interface")
GROUND_AS_BG = True                 # scenes of objects: the ground (isotropic fine fibrils under everything) is the populations head's
                                    # background -- its share, no slot (its ODF is known: isotropic); else 55-60 % of every window
                                    # went to ground fibrils and the foreground barely registered in the K slots (30 Oct)
#: the region target a density (the sum of the populations' visible occupancies, crossings counting both) rather than
#: their union clipped at one -- with the targets' populations no longer sharing the voxel (generator.TARGETS_BUDGET)
REGION_DENSITY = True
OVIS_TARGETS = True                # entities: the orientation targets' visibility measured on the acquired image at each query
                                   # (``entities.orientation_visibility``; 28 Oct review): an erased lobe asks for a flat density
#: an optional global dispersion floor on the entity tensors, ``T' = c T + (1 - c)(I - T) / 2``. Off: every object's
#: own spread is drawn at construction, so the targets are exactly the window-summed per-object elliptical ODFs
ENTITY_DISPERSION = None


def set_background_entity(on: bool = True):
    global NOT_ENTITY
    NOT_ENTITY = ("interface",) if on else ("isotropic", "interface")

#: the entity targets' sampling: query points per crop, scales (crop voxels, log-uniform), entities kept per query, the
#: resolution rule (an object is its own entity when its radius >= max(r_min, c_sig * sigma)), padded tree sizes
P_OBJECT_SCALE = 0.25            # the share of queries at scale 0 (objects alone)
LEVELS = (0, 1, 2, 99)            # grouping levels drawn per query: objects, one or two steps up their groups, the population
ENTITY = {"n_points": 384, "sigma": (1.0, 12.0), "emax": 8, "r_min": 1.5, "c_sig": 0.5, "u_max": 4096, "d_max": 16, "margin": 2, "p_tissue": 0.7,
          "p_cross": 0.2, "p_line": 0.0, "levels": None, "scales": None}   # scales (29 Oct): None; "random" (each query a
                                             # scale: objects alone at P_OBJECT_SCALE, else log-uniform 1 voxel - the crop); a number   # levels (29 Oct): None, the resolution rule; "object", every query at its
                                             # objects; "random", each query at a grouping level drawn from LEVELS   # p_line (29 Oct): a share of the queries on line-kind layers' material (the rest of
                                             # p_tissue on any object's: blobs and cells took most of it)


def acquisition_blur(rec: dict) -> float:
    """The acquisition's own blur as one Gaussian sigma (output voxels): the PSF (rms over the axes), a resolution loss
    by a factor f (~0.4 f), thick slices along one axis (~0.4 f / sqrt 3, spread over the three)."""
    psf = rec.get("psf") or [0.0, 0.0, 0.0]
    s2 = sum(float(v) ** 2 for v in psf) / max(len(psf), 1)
    if rec.get("resolution"):
        s2 += (0.4 * float(rec["resolution"])) ** 2
    if rec.get("thick_slices"):
        s2 += (0.4 * float(rec["thick_slices"])) ** 2 / 3
    return math.sqrt(s2)


def entity_queries(s, d: dict, size: int, seed: int, device, points: torch.Tensor | None = None, sigma: torch.Tensor | None = None) -> dict:
    """Query points and scales drawn in the crop, their entities' targets, and the crop's local tree (the entities'
    nodes and their ancestors, re-indexed; ``e_anc (u_max, d_max)`` each node's chain to the root, -1 padded). The
    objects' entities come from the canvas (:func:`~shapetensor.learn.entities.entity_targets`); layers drawn without
    objects (underlays, region interfaces) are one population-level entity each, under the root
    (:func:`~shapetensor.learn.entities.dense_entities`) -- together everything the total density holds. ``points`` /
    ``sigma``: given queries instead of drawn ones. ``e_gnode`` the entities' nodes in the scene's tree (dense layers:
    ``tree.n + j``, children of the root; -1: pads) for losses across windows."""
    from .entities import dense_entities, entity_targets
    E = ENTITY; ctx = d["entity_ctx"]; g = torch.Generator(device=device).manual_seed(seed + 4243)
    n = E["n_points"]; m = E["margin"]; emax = E["emax"]
    gl_ = [k for k in ctx["group_layer"] if k >= 0]
    dense = [k for k, L in enumerate(s.layers) if (L.role not in NOT_ENTITY or (INTERFACE_SHEETS and L.role == "interface")) and k not in gl_]
    ent_layers = gl_ + dense
    pts = m + torch.rand(n, 3, generator=g, device=device) * (size - 1 - 2 * m)
    obj_layers = [k for k in ent_layers if s.layers[k].role not in NOT_OBJECT]                      # most queries on the objects' material
    if obj_layers:                                                                                  # (not on the background)
        occ = d["occupancy"][obj_layers].float().sum(0)
        inner = occ[m:size - m, m:size - m, m:size - m] > 0.05
        cand = torch.nonzero(inner).float() + m
        k = int(E.get("p_tissue", 0.7) * n)
        if cand.shape[0] > 0 and k > 0:
            pick = torch.randint(0, cand.shape[0], (k,), generator=g, device=device)
            pts[:k] = cand[pick] + torch.rand(k, 3, generator=g, device=device) - 0.5
        kl = min(int(E.get("p_line", 0.0) * n), k)                                                  # the first of them on lines' material
        line_l = [k_ for k_ in obj_layers if getattr(s.layers[k_], "kind", None) == "line"]
        if kl > 0 and line_l:
            lc = torch.nonzero(d["occupancy"][line_l].float().sum(0)[m:size - m, m:size - m, m:size - m] > 0.05).float() + m
            if lc.shape[0] > 0:
                pick = torch.randint(0, lc.shape[0], (kl,), generator=g, device=device)
                pts[:kl] = lc[pick] + torch.rand(kl, 3, generator=g, device=device) - 0.5
        kc = int(E.get("p_cross", 0.0) * n)                                                         # a share in crossing cores: where two
        if kc > 0 and len(obj_layers) >= 2:                                                         # objects' populations overlap
            from ..filters import gaussian as _gauss                                                  # (at window scale: interleaved fibres of
            oc = _gauss(d["occupancy"][obj_layers].float()[None], 2.0)[0][:, m:size - m, m:size - m, m:size - m]   # two populations rarely share a voxel)
            second = oc.topk(2, dim=0).values[1]
            cc = torch.nonzero(second > 0.2).float() + m
            if cc.shape[0] > 0:
                pick = torch.randint(0, cc.shape[0], (kc,), generator=g, device=device)
                pts[k:k + kc] = cc[pick] + torch.rand(kc, 3, generator=g, device=device) - 0.5
    lo, hi = E["sigma"]; sig = torch.exp(math.log(lo) + torch.rand(n, generator=g, device=device) * math.log(hi / lo))
    if points is not None:
        pts, sig = points.float(), sigma.float(); n = pts.shape[0]
    lv = None                                                                                       # the grouping level of each query
    if E.get("levels") == "object":
        lv = torch.zeros(n, dtype=torch.long, device=device)
    elif isinstance(E.get("levels"), int):                                                          # one level for every query
        lv = torch.full((n,), int(E["levels"]), dtype=torch.long, device=device)
    elif E.get("levels") == "random":
        lv = torch.tensor(LEVELS, device=device)[torch.randint(0, len(LEVELS), (n,), generator=g, device=device)]
    sc_q = None; cent = off = None                                                                 # the scale of collections of each query
    if E.get("scales") is not None:
        if E["scales"] == "random":
            u_ = torch.rand(n, generator=g, device=device)
            sc_q = torch.where(torch.rand(n, generator=g, device=device) < P_OBJECT_SCALE, torch.zeros(n, device=device), torch.exp(u_ * math.log(size)))
        else:
            sc_q = torch.full((n,), float(E["scales"]), device=device)
        from .entities import leaf_centroids
        if "_leaf_cent" not in ctx:                                                                # (once per crop: every object's centre)
            ctx["_leaf_cent"] = leaf_centroids(ctx["entries"], ctx["tree"], ctx["canvas_shape"])
            ctx["_coll_off"] = torch.rand(3, generator=g, device=device) * 256
        cent, off = ctx["_leaf_cent"], ctx["_coll_off"]
    vis, ov = d.get("visibility"), d.get("orientation_visibility")
    gr_bg = [g_ for g_, k_ in enumerate(ctx["group_layer"]) if GROUND_AS_BG and 0 <= k_ < len(s.layers)
             and getattr(s.layers[k_], "mrole", None) == "ground"]                                   # the ground: the populations' background
    K = d["occupancy"].shape[0]
    full = lambda x: None if x is None else (x.float().view(-1, 1, 1, 1).expand(K, *d["occupancy"].shape[1:]) if x.dim() == 1 else x.float())
    t = entity_targets(ctx["entries"], ctx["tree"], ctx["canvas_shape"], ctx["origin"], ctx["zoom"], (size,) * 3, pts, sig, grid=ctx["grid"],
                       vis=full(vis), group_layer=ctx["group_layer"], r_min=E["r_min"], c_sig=E["c_sig"], emax=emax,
                       mask=ctx.get("mask"), keep=ctx.get("keep"), ovis=full(ov), sig_acq=acquisition_blur(d.get("record", {})), comp=ctx.get("comp"), level=lv, scale=sc_q, centroids=cent, coll_offset=off,
                       bg_groups=torch.tensor(gr_bg, device=device, dtype=torch.long))
    tree = ctx["tree"]; node = t["node"]; live = node >= 0
    e_tot, e_bgm = t["tot"].clone(), t["bg"].clone()
    anc = tree.ancestors(torch.unique(node[live])) if bool(live.any()) else torch.zeros(0, 1, dtype=torch.long, device=device)
    U = torch.unique(torch.cat([anc[anc >= 0], torch.zeros(1, dtype=torch.long, device=device)]))   # the root always (the dense entities' parent)
    nd = len(dense)
    if U.numel() + nd > E["u_max"]:                                                                  # rare: the latest-made nodes beyond the pad dropped (masked)
        U = U[:E["u_max"] - nd]
    loc = torch.full((tree.n,), -1, dtype=torch.long, device=device); loc[U] = torch.arange(U.numel(), device=device)
    A = tree.ancestors(U)[:, :E["d_max"]]; A = torch.where(A >= 0, loc[A.clamp_min(0)], A)
    e_anc = torch.full((E["u_max"], E["d_max"]), -1, dtype=torch.int32, device=device); e_anc[:A.shape[0], :A.shape[1]] = A.int()
    e_node = torch.where(live, loc[node.clamp_min(0)], node); gnode = node.clone()
    mass, evis, T, desc = t["mass"], t["vis"], t["T"], t["desc"]
    if nd:                                                                                          # the dense layers' entities, merged by visible mass
        dl = [s.layers[k] for k in dense]
        dd = torch.tensor([[float(getattr(L, "radius", 0.0)), 0.0, 0.0, 1.0, 0.0] for L in dl], device=device)
        Td = d["tensors"][dense]
        iso_l = [j for j, L in enumerate(dl) if L.role == "isotropic"]                              # the background has no orientation: its
        if iso_l:                                                                                   # tensor stays I / 3 (a warp's Jacobian
            Td = Td.clone(); Td[iso_l] = torch.tensor([1, 1, 1, 0, 0, 0.0], device=device, dtype=Td.dtype).view(1, 6, 1, 1, 1) / 3   # would make it a weak line)
        occ_d = d["occupancy"][dense]
        if_l = [j for j, L in enumerate(dl) if L.role == "interface"]                                # a region's boundary: a sheet of material,
        if if_l:                                                                                    # normal along the region's gradient, its mass
            from ..filters import gaussian as _gs                                                   # the boundary's sharpness (gradient magnitude,
            occ_d = occ_d.float().clone(); Td = Td.float().clone()                                  # voxel units); the region's interior is plain
            for j in if_l:                                                                          # tissue (the other layers' business)
                gm = _gs(occ_d[j][None, None], 1.0)[0, 0]
                gz, gy, gx = torch.gradient(gm); gn = torch.sqrt(gz ** 2 + gy ** 2 + gx ** 2)
                nz, ny, nx = gz / gn.clamp_min(1e-6), gy / gn.clamp_min(1e-6), gx / gn.clamp_min(1e-6)
                Td[j] = torch.stack([1 - nz * nz, 1 - ny * ny, 1 - nx * nx, -nz * ny, -nz * nx, -ny * nx]) / 2
                occ_d[j] = gn.clamp(0, 1)
            dd = dd.clone(); dd[if_l, 3] = 0.0                                                      # (not a volume population)
        de = dense_entities(occ_d, Td, pts, sig, vis=full(vis)[dense] if vis is not None else None,
                            ovis=full(ov)[dense] if ov is not None else None, desc=dd)
        md = (de["mass"] * de["live"] * de["vis"]).clamp_min(0)                                     # the window's totals: the dense layers too
        e_tot = e_tot + md.sum(1)
        e_bgm = e_bgm + md[:, [j for j, L in enumerate(dl) if L.role == "isotropic"]].sum(1)
        ids = U.numel() + torch.arange(nd, device=device)
        e_anc[ids, 0] = ids.int(); e_anc[ids, 1] = int(loc[0])
        node_d = torch.where(de["live"], ids[None].expand(n, nd), torch.full((n, nd), -1, device=device, dtype=torch.long))
        e_node = torch.cat([e_node, node_d], 1); mass = torch.cat([mass, de["mass"] * de["live"]], 1); evis = torch.cat([evis, de["vis"]], 1)
        gd_ = tree.n + torch.arange(nd, device=device)                                                   # global: the scene's layers keep their order
        gnode = torch.cat([gnode, torch.where(de["live"], gd_[None].expand(n, nd), torch.full((n, nd), -1, device=device, dtype=torch.long))], 1)
        T = torch.cat([T, de["T"]], 1); desc = torch.cat([desc, de["desc"]], 1)
        score = torch.where(e_node >= 0, mass * evis, torch.full_like(mass, -1.0))
        o = torch.argsort(score, 1, descending=True)[:, :emax]
        e_node, mass, evis, gnode = e_node.gather(1, o), mass.gather(1, o), evis.gather(1, o), gnode.gather(1, o)
        T = T.gather(1, o[..., None, None].expand(-1, -1, 3, 3)); desc = desc.gather(1, o[..., None].expand(-1, -1, 5))
        mass = mass * (e_node >= 0); evis = evis * (e_node >= 0)
    live = e_node >= 0
    if ENTITY_DISPERSION is not None:                                                                # a dispersion floor: no object is drawn
        c_ = float(ENTITY_DISPERSION); I3 = torch.eye(3, device=device)                              # sharper than the image can show
        tr_ = T.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-9)[..., None, None]
        Tn = T / tr_
        T = torch.where(live[..., None, None], c_ * Tn + (1 - c_) * (I3 - Tn) / 2, T)
    T6 = torch.stack([T[..., 0, 0], T[..., 1, 1], T[..., 2, 2], T[..., 0, 1], T[..., 0, 2], T[..., 1, 2]], -1)
    from .bingham import fit as bfit                                                                   # the entities' Bingham ODFs (moment matched)
    R = torch.eye(3, device=device).expand(*e_node.shape, 3, 3).clone(); Kp = torch.zeros(*e_node.shape, 2, device=device)
    if bool(live.any()):
        Rf, Kf = bfit(T[live].double()); R[live], Kp[live] = Rf.float(), Kf.float()
    bg_g = torch.tensor([tree.n + j for j, k in enumerate(dense) if s.layers[k].role == "isotropic"] or [-2], device=device)
    e_bg = (gnode[..., None] == bg_g).any(-1) & (e_node >= 0)                                          # the background's entity (no object)
    if gr_bg:                                                                                          # and the ground's (scenes of objects)
        on_t = (gnode >= 0) & (gnode < tree.n)
        e_bg = e_bg | (on_t & torch.isin(tree.group[gnode.clamp(0, tree.n - 1)], torch.tensor(gr_bg, device=device)))
    out = {"e_points": pts, "e_sigma": sig, "e_node": e_node.int(), "e_mass": mass.half(), "e_vis": evis.half(),
           "e_T6": T6.half(), "e_desc": desc.half(), "e_anc": e_anc, "e_n": torch.tensor(U.numel() + nd, device=device),
           "e_R": R.float(), "e_kappa": Kp.float(), "e_gnode": torch.where(e_node >= 0, gnode, torch.full_like(gnode, -1)), "e_bg": e_bg,
           "e_level": lv if lv is not None else torch.full((pts.shape[0],), -1, dtype=torch.long, device=device),
           "e_scale": sc_q if sc_q is not None else torch.full((pts.shape[0],), -1.0, device=device),
           "e_tot": torch.maximum(e_tot, ((mass * evis).clamp_min(0) * live).sum(1)), "e_bgm": e_bgm}
    if OVIS_TARGETS and "image" in d:
        from .entities import orientation_visibility
        out["e_ovis"] = orientation_visibility(d["image"].float() / 255, pts, sig, R, Kp, (mass * evis).clamp_min(0) * live).half()
    return out


ROUTES = 0.0                       # > 0: batch-time contrast routes (learn/augment.py) at this strength, a fresh draw each use
VIEWS = 1                          # 2: also a second route of the same pairs (``_view2``) for the contrast-consistency loss


def stack(pairs, g=None, aug: bool = True):
    """A batch from (image, target) pairs: the image ``(B, 1, Z, Y, X)`` in 0-1, the targets stacked; with ``aug``
    each pair cube-transformed and its intensities jittered (a different window: ``a x + b``, clipped), and with
    :data:`ROUTES` sent through a random contrast route (:func:`~shapetensor.learn.augment.contrast_route`); with
    :data:`VIEWS` = 2 a second, independent route of the same geometry rides along as ``_view2``."""
    import numpy as _np
    from .augment import contrast_route
    imgs, imgs2, tgts = [], [], []
    for img, t in pairs:
        t = {k: (v.float() if k in HALF or k in ("q_sh", "sh_t", "points_t", "e_mass", "e_vis", "e_T6", "e_desc", "region_t", "iface_t", "bg_t", "ori_t", "iso_t", "e_ovis") else
                 v.long() if k in ("kind", "layer", "identity_t", "e_node", "e_anc") else v) for k, v in t.items()}
        x = img.float() / 255.0
        if aug:
            x, t = augment(x, t, g)
            a = 0.6 + 0.8 * float(torch.rand(1, generator=g)); b = 0.4 * (float(torch.rand(1, generator=g)) - 0.5)
            x = (a * (x - 0.5) + 0.5 + b).clamp(0, 1)
            if ROUTES > 0:
                base = x
                rng = _np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g)))
                x = contrast_route(base, rng, ROUTES)
                if VIEWS == 2:
                    imgs2.append(contrast_route(base, _np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g))), ROUTES))
        imgs.append(x); tgts.append(t)
    keys = TKEYS + tuple(k for k in QKEYS + LKEYS + EKEYS if k in tgts[0] and k not in TKEYS)
    T = {k: torch.stack([t[k] for t in tgts]) for k in keys}
    if imgs2:
        T["_view2"] = torch.stack(imgs2)[:, None]
    return torch.stack(imgs)[:, None], T


class Buffer:
    def __init__(self, n: int, size: int, device, seed0: int = 0, strength: float = 1.0, log=print, gen: dict | None = None,
                 simple: dict | None = None, p_simple: float = 0.0, n0: int | None = None):
        # ``n0``: pairs drawn at the start (default ``n``); new pairs then fill the buffer up to ``n`` before replacing any
        self.size, self.device, self.next, self.gen, self.simple = size, device, seed0, gen, simple
        t0 = time.time(); self.cap = int(n)
        self.items, self.simple_flag = [], []
        for _ in range(min(n, n0) if n0 else n):
            self.items.append(self._make(strength, p_simple)); self.simple_flag.append(self.last_simple)
        self.w = [anisotropy(t) for _, t in self.items]
        log(f"buffer: {len(self.items)} of {n} pairs at {size}^3 (strength {strength:.2f}, simple share {p_simple:.2f}) in {time.time() - t0:.0f} s")

    def _make(self, strength, p_simple: float = 0.0):
        import random
        is_simple = self.simple is not None and random.Random(self.next * 7 + 3).random() < p_simple
        item = make(self.next, self.size, self.device, strength, self.simple if is_simple else self.gen); self.next += 1
        self.last_simple = is_simple
        return item

    def refresh(self, k: int, g: torch.Generator, strength: float, p_simple: float = 0.0):
        for _ in range(k):
            if len(self.items) < self.cap:
                self.items.append(self._make(strength, p_simple)); self.w.append(anisotropy(self.items[-1][1])); self.simple_flag.append(self.last_simple)
                continue
            i = int(torch.randint(0, len(self.items), (1,), generator=g))
            self.items[i] = self._make(strength, p_simple); self.w[i] = anisotropy(self.items[i][1]); self.simple_flag[i] = self.last_simple

    def put(self, items: list, g: torch.Generator):
        """Replace random buffer slots with ``items`` (from a producer)."""
        for it in items:
            if len(self.items) < self.cap:
                self.items.append(it); self.w.append(anisotropy(it[1])); self.simple_flag.append(False)
                continue
            i = int(torch.randint(0, len(self.items), (1,), generator=g))
            self.items[i] = it; self.w[i] = anisotropy(it[1]); self.simple_flag[i] = False

    def batch(self, B: int, g: torch.Generator, alpha: float = 0.0):
        """``B`` pairs, uniformly or, with ``alpha > 0``, in proportion to their non-round target energy ** alpha."""
        if alpha > 0:
            p = torch.tensor(self.w, dtype=torch.float64).clamp_min(1e-12) ** alpha
            idx = torch.multinomial(p / p.sum(), B, replacement=True, generator=g).tolist()
        else:
            idx = torch.randint(0, len(self.items), (B,), generator=g).tolist()
        return stack([self.items[i] for i in idx], g)


def anisotropy(t: dict) -> float:
    """A pair's non-round target energy: the cells' mean of sum_k m_k |S_k - I/3|^2 (the populations' deviatoric
    energy, mass-weighted) -- category-free; what an orientation-learning batch is made of."""
    if "sh_t" in t:                                                                                   # the density's angular share (orders >= 2 over order 0),
        c = t["sh_t"].float(); occ = c[0] > 0.05 * math.sqrt(4 * math.pi)                          # per occupied cell -- scale-free (review 10: the
        if not bool(occ.any()):                                                                       # raw energy went as mass^2 and favoured hard draws)
            return 0.0
        return float(((c[1:] ** 2).sum(0)[occ] / (c[0][occ] ** 2)).mean())
    S = t["slots"].float(); m = t["mass"].float()
    iso = torch.tensor([1, 1, 1, 0, 0, 0.0], device=S.device).view(1, 6, 1, 1, 1) / 3
    wf = torch.tensor([1, 1, 1, 2, 2, 2.0], device=S.device).view(1, 6, 1, 1, 1)
    return float((m * ((S - iso) ** 2 * wf).sum(1)).sum(0).mean())


def _axes(S: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
    """Axes of tensors ``S (..., 6)``: a line's / blob's largest eigenvector, a sheet's / interface's normal."""
    _, V = eig3(S.movedim(-1, 0))
    planar = ((kind == 1) | (kind == 3))[None]
    return torch.where(planar, V[0], V[2]).movedim(0, -1)


@torch.no_grad()
def evaluate(model, val, grid, margin: int, device, engine=None) -> dict:
    """The held-out set: loss parts; the median axis error of matched line populations whose target is linear (mass >
    0.05, linearity > 0.5) and of the mixture head on cells with one line population; the predicted
    fraction's spread over the set and its correlation with the target's; the engine's population count (line
    populations, N) against the target's own (the same engine reading the target slots)."""
    from . import losses as _L
    saved = (_L.DISTINCT, _L.PRESENCE_BAND); _L.DISTINCT = None; _L.PRESENCE_BAND = None                  # one reference rule for every run
    try:
        return _evaluate(model, val, grid, margin, device, engine)
    finally:
        _L.DISTINCT, _L.PRESENCE_BAND = saved


def st_axis_cells(x: torch.Tensor) -> torch.Tensor:
    """A structure-tensor reference: the least-gradient direction (gradient sigma 1, window sigma 5) per 4³ cell of
    an image ``(Z, Y, X)``: ``(3, z, y, x)``."""
    from ..filters import gaussian
    xs = gaussian(x.float(), 1.0, truncate=3.0); gz, gy, gx = torch.gradient(xs)
    G = torch.stack([Fn.avg_pool3d(gaussian(c, 5.0, truncate=3.0)[None, None], 4)[0, 0] for c in (gz * gz, gy * gy, gx * gx, gz * gy, gz * gx, gy * gx)])
    full = torch.stack([torch.stack([G[0], G[3], G[4]]), torch.stack([G[3], G[1], G[5]]), torch.stack([G[4], G[5], G[2]])])
    _, V = torch.linalg.eigh(full.permute(2, 3, 4, 0, 1))
    return V[..., :, 0].permute(3, 0, 1, 2)


def _evaluate(model, val, grid, margin: int, device, engine=None) -> dict:
    model.eval(); parts, errs, mix_errs, fp, ft, nagree, ntot = {}, [], [], [], [], 0.0, 0.0
    sh_one, sh_false, sh_det, sh_err2 = [], [], [], []
    q_eval = []; ctx_eval = {}
    ref = {k: [] for k in ("slot_single", "slot_shared", "mix_single", "st_single", "st_shared", "sh_single", "sh_shared")}
    desc_err = []
    crop = lambda v: v[..., margin:-margin, margin:-margin, margin:-margin]
    for img, t in val:
        x, T = stack([(img, t)], aug=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            P = model(x, T.get("q_points"), T.get("q_sigma"))
        _, p = loss(P, T, grid, margin)
        for k, v in p.items():
            parts[k] = parts.get(k, 0.0) + v / len(val)
        Pc = {k: crop(v) for k, v in P.items() if not k.startswith("q_")}; Tc = {k: crop(v) for k, v in T.items() if not k.startswith("q_")}
        if "q_sh" in P:
            q_eval.append((P["q_sh"][0].float(), T["q_sh"][0].float(), T["q_sigma"][0].float()))
            if CONTEXT is not None:
                for hb in (8.0, 16.0):
                    xm, ctx = context_mask(x, T["q_points"], torch.tensor([hb], device=x.device))
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                        Pm = model(xm, T["q_points"], T["q_sigma"], ctx)
                    kq = ctx[0] >= torch.clamp(1.5 * T["q_sigma"][0], min=2.0)
                    if bool(kq.any()):
                        ctx_eval.setdefault(hb, []).append((Pm["q_sh"][0].float()[kq], T["q_sh"][0].float()[kq]))
        _, _, perm = match(Pc, Tc)
        Sp = gather_slots(Pc["slots"].float(), perm)
        kt = Tc["kind"]; mt = Tc["mass"].float()
        ang = torch.rad2deg(torch.acos((_axes(Sp.movedim(2, -1), kt) * _axes(Tc["slots"].float().movedim(2, -1), kt)).sum(-1).abs().clamp(max=1.0)))
        tv, _ = eig3(Tc["slots"].float().movedim(2, 0))                                                   # (3, 1, K, c): the target's own linearity
        lin = (tv[2] - tv[1]) / tv[2].clamp_min(1e-9)
        sel = (mt > 0.05) & (kt == 0) & (lin > 0.5)                                                       # cells where the target is a line with an axis
        if bool(sel.any()):
            errs.append(ang[sel])
        # single cells (the line is the only anisotropic population) vs shared cells, under the reference rule
        an = (tv[2] - tv[0]) / tv.sum(0).clamp_min(1e-9)                                                 # (1, K, c)
        single_cell = (sel.sum(1) == 1) & (((mt > 0.05) & (an > 0.25)).sum(1) == 1)                      # (1, c)
        single, shared = sel & single_cell[:, None], sel & ~single_cell[:, None]
        ref["slot_single"].append(ang[single]); ref["slot_shared"].append(ang[shared])
        t_ax = _axes(Tc["slots"].float().movedim(2, -1), kt)                                            # (1, K, c..., 3)
        st_ax = crop(st_axis_cells(x[0, 0]))                                                            # (3, c...)
        angv = lambda a_, b_: torch.rad2deg(torch.acos((a_ * b_).sum(-1).abs().clamp(max=1)))
        e_st = angv(st_ax.movedim(0, -1)[None, None], t_ax)                                              # (1, K, c)
        ref["st_single"].append(e_st[single]); ref["st_shared"].append(e_st[shared])
        _, Vmx = eig3(Pc["mixture"][0].float())
        e_mx = angv(Vmx[2].movedim(0, -1)[None, None], t_ax)
        ref["mix_single"].append(e_mx[single])
        if "sh" in Pc:
            from .sh import peaks as _peaks
            from .sh import peaks_gated as _pkg
            pdd, pvv, _ = _pkg(Pc["sh"][0].float().flatten(1), model.sh_lmax, nmax=3)                    # calibrated (review 5)
            cs_ = Pc["sh"].shape[2:]
            pk = pdd.movedim(1, -1).reshape(3, *cs_, 3)                                                  # (3 peaks, c..., 3)
            pkv = pvv.reshape(3, *cs_)
            e_pk = torch.stack([torch.where(pkv[j][None, None] > 0, angv(pk[j][None, None], t_ax), torch.full_like(e_st, 90.0)) for j in range(3)]).amin(0)
            ref["sh_single"].append(e_pk[single]); ref["sh_shared"].append(e_pk[shared])
        if "sh_t" in Tc and getattr(model, "desc_lmax", 0):                                              # descriptors at the target's top peak
            from .sh import n_coef as _nc, peaks as _pk, real_sh as _rs
            nsh_ = _nc(model.sh_lmax); nd_ = _nc(model.desc_lmax)
            ctf = Tc["sh_t"][0].float().flatten(1); cpf = Pc["sh"][0].float().flatten(1)
            strong = ctf[0] > 0.1 * math.sqrt(4 * math.pi)
            if bool(strong.any()):
                td_, _ = _pk(ctf[:, strong], model.sh_lmax, nmax=1, rel=0.3)
                u_ = td_[0].T                                                                            # (n, 3)
                Yd = _rs(u_, model.desc_lmax)                                                            # (n, nd)
                def at(cf):
                    f0 = (Yd * cf[:nd_, strong].T).sum(1).clamp_min(1e-6)
                    return torch.stack([(Yd * cf[nsh_ + j * nd_: nsh_ + (j + 1) * nd_, strong].T).sum(1) / f0 for j in range(3)])
                desc_err.append(at(cpf) - at(ctf))
        # the permutation-free mixture head against the target mixture, where the cell holds one line population
        one = sel[:, 0] & ~(mt[:, 1:] > 0.05).any(1)
        if bool(one.any()):
            _, Vm = eig3(Pc["mixture"][0].float()); _, Vt = eig3(Tc["slots"][0, 0].float())
            mix_errs.append(torch.rad2deg(torch.acos((Vm[2] * Vt[2]).sum(0).abs().clamp(max=1.0)))[one[0]])
        if "sh" in Pc:                                                                                    # the SH density's peaks, no matching
            from .sh import peaks
            from .sh import peaks_gated
            pd, pv, _ = peaks_gated(Pc["sh"][0].float().flatten(1), model.sh_lmax, nmax=3)                # calibrated (review 5)
            tvv, tVv = eig3(Tc["slots"][0].float().movedim(1, 0)); ta = tVv[2].flatten(2)                # (3, K, N)
            aniso = ((tvv[2] - tvv[0]) / tvv.sum(0).clamp_min(1e-9)).flatten(1)                          # (K, N)
            mk = Tc["mass"][0].float().flatten(1)
            tl = sel[0].flatten(1)                                                                       # (K, N) line targets with an axis
            angd = lambda a_, b_: torch.rad2deg(torch.acos((a_ * b_).sum(0).abs().clamp(max=1)))
            others = (mk > 0.05) & (aniso > 0.25); others[0] = False                                    # anisotropic populations besides the line
            o1 = tl[0] & (tl.sum(0) == 1) & ~others.any(0)                                               # one line, the rest near-isotropic
            if bool(o1.any()):
                sh_one.append(angd(pd[0], ta[:, 0])[o1]); sh_false.append((pv[1] > 0)[o1].float())
            two = tl[0] & tl[1] & (angd(ta[:, 0], ta[:, 1]) > 30)
            if bool(two.any()):
                a1, a2, p1, p2 = ta[:, 0][:, two], ta[:, 1][:, two], pd[0][:, two], pd[1][:, two]
                e_ = torch.minimum((angd(p1, a1) + angd(p2, a2)) / 2, (angd(p1, a2) + angd(p2, a1)) / 2)
                det = pv[1][two] > 0
                sh_det.append(det.float()); sh_err2.append(e_[det])
        fp.append(torch.sigmoid(Pc["fraction_logit"].float()).flatten()); ft.append(Tc["fraction"].float().flatten())
        if engine is not None:
            pe = engine.populations({k: v[0].float() for k, v in P.items()}, labels=False)
            to = {"slots": T["slots"][0], "mass": T["mass"][0], "root_mass": T["mass"][0].sqrt(), "presence_logit": torch.where(T["mass"][0] > 0.005, 10.0, -10.0),
                  "kind": 30.0 * torch.nn.functional.one_hot(T["kind"][0].clamp_min(0), 4).movedim(-1, 1).float(), "embed": P["embed"][0].float(),
                  "fraction_logit": P["fraction_logit"][0].float(), "class_logits": P["class_logits"][0].float(), "log_size": P["log_size"][0].float()}
            pt = engine.populations(to, labels=False)
            Np, Nt = crop(pe["N"]), crop(pt["N"])
            nagree += float((Np == Nt).float().sum()); ntot += Np.numel()
    model.train()
    e = torch.cat(errs) if errs else torch.zeros(1)
    fp, ft = torch.cat(fp), torch.cat(ft)
    corr = float(((fp - fp.mean()) * (ft - ft.mean())).mean() / (fp.std() * ft.std()).clamp_min(1e-9))
    em = torch.cat(mix_errs) if mix_errs else torch.zeros(1)
    out = {**{f"val_{k}": v for k, v in parts.items()}, "val_axis_deg_median": float(e.median()), "val_axis_deg_p90": float(e.quantile(0.9)),
           "val_mixture_axis_deg_median": float(em.median()),
           "val_fraction_std": float(fp.std()), "val_fraction_corr": corr, "val_fraction_mae": float((fp - ft).abs().mean())}
    if engine is not None:
        out["val_N_agree"] = nagree / max(ntot, 1)
    cat = lambda L: torch.cat(L) if L else torch.zeros(0)
    if desc_err:
        de = torch.cat(desc_err, 1)                                                                        # (3, n)
        for j, nm in enumerate(("thickness", "hollowness", "angularity")):
            out[f"ref_desc_{nm}_mae"] = float(de[j].abs().median())
    for k_, L_ in ref.items():
        v_ = cat(L_)
        if v_.numel():
            out[f"ref_{k_}"] = float(v_.median())
    if q_eval:
        out.update(query_metrics(q_eval, model.sh_lmax))
    for hb, L_ in ctx_eval.items():                                                                   # the top peak with limited context
        from .sh import peaks as _pk
        Pq = torch.cat([p_ for p_, _ in L_]); Tq = torch.cat([t_ for _, t_ in L_])
        pd_, _ = _pk(Pq.T, model.sh_lmax, rel=0.3); td_, tv_ = _pk(Tq.T, model.sh_lmax, rel=0.3)
        one_ = (tv_[0] > 0) & (tv_[1] == 0) & (Tq[:, 0] > 0.05 * math.sqrt(4 * math.pi))
        if bool(one_.any()):
            out[f"val_q_axis_ctx{int(hb)}"] = float(torch.rad2deg(torch.acos((pd_[0][:, one_] * td_[0][:, one_]).sum(0).abs().clamp(max=1))).median())
    if sh_one or sh_det:
        out["val_sh_axis_deg_median"] = float(cat(sh_one).median()) if sh_one else float("nan")                # top peak, one-line cells
        out["val_sh_false_second"] = float(cat(sh_false).mean()) if sh_false else float("nan")                 # a second peak where one line
        out["val_sh_second_detected"] = float(cat(sh_det).mean()) if sh_det else float("nan")                  # crossings (> 30 deg): 2 peaks
        e2 = cat(sh_err2); out["val_sh_crossing_deg_median"] = float(e2.median()) if e2.numel() else float("nan")   # their mean axis error
    return out


def query_metrics(q_eval, lmax: int) -> dict:
    """Queries by scale band (sigma < 3, 3-7, > 7 voxels): the top density peak's axis error where the target density
    has one clear peak; where it has two (> 25 deg apart, the second >= 0.3 of the first), how often the prediction has
    two, and their mean axis error."""
    from .sh import peaks
    P = torch.cat([p for p, _, _ in q_eval]); T = torch.cat([t for _, t, _ in q_eval]); S = torch.cat([s for _, _, s in q_eval])
    from .sh import peaks_gated
    pd, pv, _ = peaks_gated(P.T, lmax); td, tv, _ = peaks_gated(T.T, lmax)                              # calibrated (review 5)
    nt, npred = (tv > 0).sum(0), (pv > 0).sum(0)
    ang = lambda a, b: torch.rad2deg(torch.acos((a * b).sum(0).abs().clamp(max=1)))
    out = {}
    for name, sel in (("fine", S < 3), ("mid", (S >= 3) & (S <= 7)), ("coarse", S > 7)):
        one = sel & (nt == 1) & (T[:, 0] > 0.05 * math.sqrt(4 * math.pi))
        two = sel & (nt == 2)
        if bool(one.any()):
            out[f"val_q_axis_{name}"] = float(ang(pd[0][:, one], td[0][:, one]).median())
        if bool(two.any()):
            det = (npred >= 2) & two
            out[f"val_q_second_{name}"] = float(det.sum() / two.sum())
            if bool(det.any()):
                e = torch.minimum((ang(pd[0][:, det], td[0][:, det]) + ang(pd[1][:, det], td[1][:, det])) / 2,
                                  (ang(pd[0][:, det], td[1][:, det]) + ang(pd[1][:, det], td[0][:, det])) / 2)
                out[f"val_q_crossing_{name}"] = float(e.median())
    return out


def train(out: str, steps: int = 20000, size: int = 128, batch: int = 4, accum: int = 4, buffer: int = 128, refresh: int = 1, lr: float = 3e-4,
          warmup: int = 500, margin: int = 4, val_n: int = 32, val_every: int = 500, log_every: int = 25, seed: int = 0, device: str = "cuda",
          strength0: float = 0.15, strength1: float = 1.0, overfit: int = 0, gen: dict | None = None, resume: str | None = None,
          widths: tuple = (32, 64, 128), distinct: float | None = None, sh_lmax: int = 0, lcn: float = 0.0, presence_band=None, weights: dict | None = None, query: int = 0,
          simple: dict | None = None, p_simple=(0.0, 0.0), init: str | None = None, order: str = "mass", exact: bool = False, desc_lmax: int = 0,
          context=None, late_start: int | None = None, scalar_window=None, cache: str | None = None,
          gate: float | None = None, aniso=None, shape_loss: str = "frobenius", simple_window: int | None = None,
          sh_loss: str = "batchnorm", axis_head: bool = False, gate_max: int | None = None):
    from .engine import PopulationEngine
    from . import losses as _losses
    _losses.PRESENCE_BAND = tuple(presence_band) if presence_band else None
    _losses.SHAPE_LOSS = shape_loss
    _losses.SH_LOSS = sh_loss
    _losses.SH_LMAX_ON = 2 if sh_loss == "decoupled" else None                                          # order 2 first, all orders once it aligns
    from . import targets as _targets
    _targets.ORDER = order                                                                               # slot order of the targets (and of ties)
    global QUERY
    QUERY = {"n_points": query, "n_levels": 4, "lmax": sh_lmax, "exact": exact} if query else None
    global SH_EXACT, DESC_LMAX, CONTEXT, CACHE
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True); CACHE = cache
    CONTEXT = tuple(context) if context else None
    assert not context or query, "--context needs --query"
    SH_EXACT = sh_lmax if (exact and sh_lmax) else 0
    assert not desc_lmax or exact, "--desc-lmax needs --exact"
    DESC_LMAX = desc_lmax
    from .sh import n_coef as _nc
    _losses.DESC_NC = _nc(desc_lmax) if desc_lmax else 0
    assert not query or sh_lmax, "--query needs --sh-lmax"
    if weights:
        _losses.WEIGHTS.update(weights)                                                                   # loss weights for this run
    _losses.DISTINCT = distinct                                                                          # slot order kept only among similar shapes
    outd = Path(out); outd.mkdir(parents=True, exist_ok=True)
    logf = open(outd / "log.jsonl", "a")
    def log(msg):
        print(msg, flush=True)
    args = dict(steps=steps, size=size, batch=batch, accum=accum, buffer=buffer, refresh=refresh, lr=lr, warmup=warmup, margin=margin, val_n=val_n,
                seed=seed, strength0=strength0, strength1=strength1, overfit=overfit, gen=gen, widths=list(widths), distinct=distinct, sh_lmax=sh_lmax, lcn=lcn,
                presence_band=list(presence_band) if presence_band else None, weights=weights, query=query,
                simple=simple, p_simple=list(p_simple), init=init, order=order, exact=exact, desc_lmax=desc_lmax,
                context=list(context) if context else None, late_start=late_start, scalar_window=list(scalar_window) if scalar_window else None,
                gate=gate, aniso=list(aniso) if aniso else None, shape_loss=shape_loss, simple_window=simple_window,
                sh_loss=sh_loss, axis_head=axis_head, gate_max=gate_max)
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    model = PopulationNet.from_args(args).to(device)
    rf = receptive_field(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    start = 0
    if init:                                                                                           # warm start: weights only (new heads stay fresh)
        sd = torch.load(init, map_location=device)["model"]; own = model.state_dict()
        sd = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(sd, strict=False); log(f"init: {len(sd)}/{len(own)} tensors from {init}")
    if resume:
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["step"]
    sched = lambda s: min(1.0, (s + 1) / warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    strength = lambda s: strength1 if overfit else strength0 + (strength1 - strength0) * min(1.0, s / (0.5 * steps))
    sw = scalar_window or (0.2 * steps, 0.4 * steps)                                                  # the cell scalars: 0 until 20%, full by 40% (default)
    ramp = lambda s: min(1.0, max(0.0, (s - sw[0]) / max(sw[1] - sw[0], 1)))
    o0 = late_start if late_start is not None else max(300, int(0.05 * steps))
    oramp = lambda s: min(1.0, max(0.0, (s - o0) / 300))                                               # mass, presence, kind, Hellinger: after orientation (o0 steps)
    grid = hemisphere(160, device)
    log(f"model: {sum(p.numel() for p in model.parameters())} parameters, receptive field {rf} voxels; args {args}")
    sw_ = simple_window or 0.5 * steps                                                                # the simple share: p0 -> p1 over this many steps
    psimple = lambda s_: p_simple[0] + (p_simple[1] - p_simple[0]) * min(1.0, s_ / sw_)
    buf = Buffer(overfit or buffer, size, device, seed0=seed * 1_000_000 + start * refresh, strength=strength(start), log=log, gen=gen,
                 simple=simple, p_simple=psimple(start))
    def rotated(i):                                                                                  # each held-out draw under a fixed cube element
        img, t = make(10_000_000 + i, size, device, strength1, gen)
        return augment(img, t, torch.Generator().manual_seed(i))
    val = buf.items[:min(val_n, overfit)] if overfit else [rotated(i) for i in range(val_n)]   # overfit: score the training pairs
    engine = PopulationEngine(model, device=device)
    v0 = evaluate(model, val, grid, margin, device, engine)
    logf.write(json.dumps({"step": start, "baseline": True, **v0}) + "\n"); logf.flush()
    log(f"  val {start} (baseline): " + " ".join(f"{k} {x:.4f}" for k, x in v0.items()))
    t0 = time.time(); run = {}
    gstate = {"open": gate is None, "at": None, "ema": None}                                          # the gate: orientation only until it breaks away
    alpha = lambda s_: aniso[0] * max(0.0, 1 - s_ / aniso[1]) if aniso else 0.0
    for step in range(start, steps):
        if gate is not None and not gstate["open"]:
            lr_now = lr * min(1.0, (step + 1) / warmup); r = 0.0; ro = 0.0                                # constant rate, orientation terms only
        elif gate is not None:
            s0 = gstate["at"]; prog = (step - s0) / max(steps - s0, 1)
            lr_now = lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog)))
            ro = min(1.0, (step - s0) / 300); r = min(1.0, max(0.0, (step - s0 - 300) / 300))
        else:
            lr_now = lr * sched(step); r = ramp(step); ro = oramp(step)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        W = {**{k: WEIGHTS[k] * r for k in SCALARS}, **{k: WEIGHTS[k] * ro for k in LATE}}
        opt.zero_grad(set_to_none=True); tot_v = 0.0; parts_acc = {}
        for _ in range(accum):
            x, T = buf.batch(batch, g, alpha(step))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                P = model(x, T.get("q_points"), T.get("q_sigma"))
            total, parts = loss(P, T, grid, margin, axis=int(torch.randint(0, 3, (1,), generator=g)), weights=W, g=g)
            if CONTEXT is not None:                                                                   # the decoder with limited context
                from .losses import DESC_NC, desc_loss, sh_loss
                half = CONTEXT[0] + (CONTEXT[1] - CONTEXT[0]) * torch.rand(x.shape[0], generator=g).to(x.device)
                xm, ctx = context_mask(x, T["q_points"], half)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                    Pm = model(xm, T["q_points"], T["q_sigma"], ctx)
                keep = ctx >= torch.clamp(1.5 * T["q_sigma"], min=2.0)                                # queries whose scale the visible context covers
                if bool(keep.any()):
                    qp_, qt_ = Pm["q_sh"].float()[keep].T[None], T["q_sh"].float()[keep].T[None]       # (1, c, n)
                    nc_ = qt_.shape[1] - 3 * DESC_NC
                    lc_ = sh_loss(qp_[:, :nc_], qt_[:, :nc_])
                    total = total + WEIGHTS["query"] * lc_; parts["query_ctx"] = float(lc_)
            (total / accum).backward(); tot_v += float(total) / accum
            for k, v in parts.items():
                parts_acc[k] = parts_acc.get(k, 0.0) + v / accum
        if gate is not None and not gstate["open"]:                                                   # the mixture term against its round floor
            ratio = parts_acc.get("mixture", 0.0) / max(parts_acc.get("mixture_floor", 1e-9), 1e-9)
            gstate["ema"] = ratio if gstate["ema"] is None else 0.95 * gstate["ema"] + 0.05 * ratio
            if (step - start >= 50 and gstate["ema"] < gate) or (gate_max is not None and step + 1 >= gate_max):
                gstate["open"], gstate["at"] = True, step + 1
                log(f"gate open at step {step + 1}: mixture / round floor {gstate['ema']:.3f}")
                logf.write(json.dumps({"step": step + 1, "gate_open": True, "ratio": gstate["ema"]}) + "\n"); logf.flush()
        parts_acc["gate_ratio"] = gstate["ema"] if gstate["ema"] is not None else float("nan")
        if _losses.SH_LOSS == "decoupled" and _losses.SH_LMAX_ON is not None and "sh_l2cos" in parts_acc:
            l2c = gstate.setdefault("l2c", 0.0); gstate["l2c"] = 0.95 * l2c + 0.05 * parts_acc["sh_l2cos"]
            if step - start >= 50 and gstate["l2c"] > 0.3:
                _losses.SH_LMAX_ON = None; log(f"all SH orders on at step {step + 1}: order-2 cosine {gstate['l2c']:.2f}")
                logf.write(json.dumps({"step": step + 1, "sh_all_orders": True}) + "\n"); logf.flush()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        opt.step()
        if not overfit:
            buf.refresh(refresh, g, strength(step), psimple(step))
        for k, v in {**parts_acc, "total": tot_v, "grad_norm": gn, "buffer_simple": sum(buf.simple_flag) / len(buf.simple_flag)}.items():
            run[k] = run.get(k, 0.0) + v
        if (step + 1) % log_every == 0:
            rec = {"step": step + 1, "time_s": round(time.time() - t0, 1), "lr": lr_now, "strength": strength(step), "scalar_weight": r,
                   **{k: v / log_every for k, v in run.items()}}
            logf.write(json.dumps(rec) + "\n"); logf.flush(); run = {}
            log(f"step {step + 1} {rec['time_s']}s s={rec['strength']:.2f} total {rec['total']:.4f} shape {rec['shape']:.4f} mass {rec['mass']:.4f} "
                f"pres {rec['presence']:.4f} hell {rec['hellinger']:.4f} cls {rec['classes']:.3f} frac {rec['fraction']:.3f}")
        if (step + 1) % val_every == 0 or step + 1 == steps:
            v = evaluate(model, val, grid, margin, device, engine)
            logf.write(json.dumps({"step": step + 1, **v}) + "\n"); logf.flush()
            log(f"  val {step + 1}: " + " ".join(f"{k} {x:.4f}" for k, x in v.items()))
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1, "args": args, "receptive_field": rf}, outd / "model.pt")
    return model


def train_lobes(out: str, steps: int = 6000, size: int = 128, batch: int = 4, accum: int = 4, buffer: int = 128, refresh: int = 1, lr: float = 3e-4,
                warmup: int = 300, margin: int = 4, val_n: int = 24, val_every: int = 500, log_every: int = 25, seed: int = 0, device: str = "cuda",
                strength0: float = 0.15, strength1: float = 0.8, gen: dict | None = None, widths: tuple = (32, 64, 128), sh_lmax: int = 8, desc_lmax: int = 4,
                lobes_J: int = 4, context_level: bool = True, voxel_cond: bool = True, identity_dim: int = 8, points: bool = True, query: int = 0,
                gate: float | None = 0.85, gate_max: int | None = 1500, difficulty: str = "auto", level0: float = 0.3, pairs: int = 0, pair_every: int = 4,
                real: list | None = None, ssl_every: int = 4, voxel_dropout: float = 0.3, weights: dict | None = None, cache: str | None = None,
                init: str | None = None, resume: str | None = None, aniso=None, free_sh: bool = False, routes: float = 1.0, contrast_views: bool = True,
                producer: bool = True, cpu_workers: int = 0, cpu_threads: int = 4, channels_last: bool = False, compile_model: bool = False,
                contrast_every: int = 2, ckpt_minutes: float = 20.0, benchmark: int = 0):
    """The lobe-mixture model's training (see :mod:`~shapetensor.learn.lobe_train`): scene generator, exact density
    and descriptor targets, the gate (orientation first), difficulty widening (``auto``: a level that rises when the
    order-2 error is low), paired-render consistency (``pairs`` > 0), self-supervised consistency on ``real`` crops.

    Built to run on a cluster (:mod:`~shapetensor.learn.dataflow`): pairs made by a background thread on the training
    GPU (``producer``) and by ``cpu_workers`` CPU-only processes while the GPU trains; data parallel over the GPUs of a
    node under ``torchrun`` (rank 0 validates, logs, saves); cuDNN autotuning and TF32, optionally channels-last and
    ``torch.compile``; the second contrast view every ``contrast_every`` steps; a checkpoint every ``ckpt_minutes``
    and on the scheduler's warning signal (exit 99: resubmit), ``resume='auto'`` to continue; ``benchmark`` > 0 times
    that many steps (data, forward / backward, extra terms, optimiser) and exits."""
    from . import dataflow as DF
    from . import lobe_train as LT
    from . import losses as _losses
    global HEAD, DESC_KEY, SH_EXACT, DESC_LMAX, QUERY, CACHE, ROUTES, VIEWS, READ_KEEP
    READ_KEEP = float(read_keep); set_background_entity(background_entity)
    global REGION_DENSITY
    REGION_DENSITY = bool(region_density); ET.REGION_DENSITY = REGION_DENSITY
    rank, world, local = DF.init_distributed(device)
    if str(device).startswith("cuda"):
        device = f"cuda:{local}"
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True                                                       # fixed shapes: pick the fastest convolution algorithms
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    HEAD, DESC_KEY, SH_EXACT, DESC_LMAX = "lobes", "descriptors5", sh_lmax, desc_lmax
    VIEWS = 2 if (routes and contrast_views) else 1
    _losses.DESC_K = 5; _losses.DESC_NC = n_coef_(desc_lmax) if desc_lmax else 0
    QUERY = {"n_points": query, "n_levels": 4, "lmax": sh_lmax, "exact": True} if query else None
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True); CACHE = cache
    if weights:
        LT.LOBE_WEIGHTS.update(weights)
    base = dict((gen or {}).get("scene") or {"canvas": 2 * size, "crops_per_scene": 8})
    level = {"v": level0 if difficulty == "auto" else 1.0, "since": 0, "ema": None}
    cur_gen = lambda: {"scene": LT.difficulty_gen(level["v"], base) if difficulty == "auto" else base}
    def set_routes():                                                                              # batch-time routes widen with the level
        global ROUTES
        ROUTES = routes * (0.4 + 0.6 * level["v"]) if routes else 0.0
    main = rank == 0
    outd = Path(out); outd.mkdir(parents=True, exist_ok=True)
    logf = open(outd / "log.jsonl", "a") if main else None
    def log(m):
        if main:
            print(m, flush=True)
    def logj(d):
        if main:
            logf.write(json.dumps(d) + "\n"); logf.flush()
    args = dict(head="lobes", steps=steps, size=size, batch=batch, accum=accum, buffer=buffer, refresh=refresh, lr=lr, warmup=warmup, margin=margin, val_n=val_n,
                seed=seed, strength0=strength0, strength1=strength1, gen={"scene": base}, widths=list(widths), sh_lmax=sh_lmax, desc_lmax=desc_lmax, lobes_J=lobes_J,
                context_level=context_level, voxel_cond=voxel_cond, identity_dim=identity_dim, points=points, query=query, gate=gate, gate_max=gate_max,
                difficulty=difficulty, level0=level0, pairs=pairs, pair_every=pair_every, real=real, ssl_every=ssl_every, voxel_dropout=voxel_dropout,
                weights=weights, aniso=list(aniso) if aniso else None, free_sh=free_sh, routes=routes, contrast_views=contrast_views, world=world,
                contrast_every=contrast_every, channels_last=channels_last, compile_model=compile_model)
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed * 1000 + rank)
    model = PopulationNet.from_args(args).to(device)
    rf = receptive_field(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    start = 0
    gst = {"open": gate is None, "at": None, "ema": None}
    if resume == "auto":
        resume = str(outd / "model.pt") if (outd / "model.pt").exists() else None
    if init and not resume:                                                                        # warm start (weights only; grown input channels filled)
        DF.load_partial(model, torch.load(init, map_location=device)["model"], log)
    if resume:
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["step"]; level["v"] = ck.get("level", level["v"])
        gst.update(ck.get("gate", {})); level["since"] = start
        for k_, v_ in ck.get("ema", {}).items():
            from . import lobes as _LB
            _LB._EMA[tuple(k_.split("|")[:1]) + (int(k_.split("|")[1]),)] = v_.to(device)
        log(f"resumed at step {start} (level {level['v']:.1f}, gate open {gst['open']})")
    if channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    fwd = torch.compile(model) if compile_model else model
    strength = lambda s_: strength0 + (strength1 - strength0) * min(1.0, s_ / (0.5 * steps))
    log(f"model: {sum(p.numel() for p in model.parameters())} parameters, receptive field {rf} voxels, {world} GPU(s); args {args}")
    seed0 = seed * 1_000_000_000 + rank * 100_000_000 + start * refresh * 4                         # each rank its own stream of pairs
    buf = Buffer(buffer, size, device, seed0=seed0, strength=strength(start), log=log, gen=cur_gen())
    cfg = {"strength": strength(start), "gen": cur_gen()}
    prod = DF.Producer(make, size, device, buf.next + 50_000_000, cfg) if producer else None
    cpu = None
    if cpu_workers > 0:
        from . import lobes as _LB
        state = {"HEAD": HEAD, "DESC_KEY": DESC_KEY, "SH_EXACT": SH_EXACT, "DESC_LMAX": DESC_LMAX, "QUERY": QUERY, "CACHE": None,
                 "L.DESC_K": _losses.DESC_K, "L.DESC_NC": _losses.DESC_NC}
        cpu = DF.CpuWorkers(cpu_workers, size, buf.next + 90_000_000, cfg, state, threads=cpu_threads)
        log(f"cpu workers: {cpu_workers} x {cpu_threads} threads")
    val = []
    if main and not benchmark:
        def rotated(i):
            img, t = make(10_000_000 + i, size, device, strength1, {"scene": {**base, "crops_per_scene": 1}})
            return augment(img, t, torch.Generator().manual_seed(i))
        val = [rotated(i) for i in range(val_n)]
    pbuf = LT.PairBuffer(pairs, 64, device, base) if pairs else None
    rbuf = None; rv = float("nan")
    if real:
        srcs = [r_.split("=")[0] for r_ in real]; rvox = [float(r_.split("=")[1]) if "=" in r_ else float("nan") for r_ in real]
        rbuf = LT.RealBuffer(srcs, size, device, n=64, seed=seed + rank); rv = math.log2(rvox[0]) if math.isfinite(rvox[0]) else float("nan")
        log(f"real buffer: {len(rbuf.items)} crops from {srcs}")
    if main and not benchmark:
        v0 = LT.evaluate_lobes(model, val, margin, device, sh_lmax, desc_lmax)
        logj({"step": start, "baseline": True, **v0})
        log(f"  val {start} (baseline): " + " ".join(f"{k} {x:.4f}" for k, x in v0.items()))
    alpha = lambda s_: aniso[0] * max(0.0, 1 - s_ / aniso[1]) if aniso else 0.0
    ckp = DF.Checkpointer(ckpt_minutes)
    def save(step_):
        if not main:
            return
        from . import lobes as _LB
        ema = {f"{k[0]}|{k[1]}": v.detach().cpu() for k, v in _LB._EMA.items() if torch.is_tensor(v)}
        tmp = outd / "model.pt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step_, "args": args, "receptive_field": rf, "level": level["v"],
                    "gate": dict(gst), "ema": ema}, tmp)
        os.replace(tmp, outd / "model.pt")
    tick = (lambda: torch.cuda.synchronize(device)) if str(device).startswith("cuda") else (lambda: None)
    bench = {k: [] for k in ("data", "main", "extra", "opt", "step")} if benchmark else None
    t0 = time.time(); run = {}; got_pairs = 0
    set_routes()
    WHICH = ("density", "vote", "vote_unc", "kappa", "desc", "tree")
    ramp = lambda s_: 1.0 if where_first <= 0 else min(1.0, max(0.0, (s_ - where_first) / where_first))
    def which_w(s_):                                                                                 # the "which way" losses, ramped
        a_ = ramp(s_)
        return None if a_ >= 1.0 else {k: ET.ENT_WEIGHTS.get(k, 0.2) * a_ for k in WHICH}
    end = min(steps, start + benchmark) if benchmark else steps
    for step in range(start, end):
        if bench is not None:
            tick(); ts = time.perf_counter()
        if gate is not None and not gst["open"]:
            lr_now = lr * min(1.0, (step + 1) / warmup); ro = r2 = 0.0
        else:
            s0 = gst["at"] if gst["at"] is not None else 0; prog = (step - s0) / max(steps - s0, 1)
            lr_now = lr * min(1.0, (step + 1) / warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog)))
            ro = min(1.0, (step - s0) / 300); r2 = min(1.0, max(0.0, (step - s0 - 300) / 300))
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        W = {k: (LT.LOBE_WEIGHTS[k] if k in LT.ORIENT else LT.LOBE_WEIGHTS[k] * (r2 if k in ("consistency", "ssl", "contrast") else ro)) for k in LT.LOBE_WEIGHTS}
        opt.zero_grad(set_to_none=True); pa = {}; tot_v = 0.0; t_data = 0.0
        for mb in range(accum):
            if bench is not None:
                tick(); td = time.perf_counter()
            x, T = buf.batch(batch, g, alpha(step))
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            v = T["voxel"].clone()
            v[torch.rand(v.shape[0], generator=g).to(v.device) < voxel_dropout] = float("nan")       # the model must also work without it
            if bench is not None:
                tick(); t_data += time.perf_counter() - td
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                P = fwd(x, T.get("q_points"), T.get("q_sigma"), voxel=v)
            total, parts, rend = LT.lobe_loss(model, P, T, margin, W, sh_lmax, desc_lmax)
            if "_view2" in T and W.get("contrast", 0) > 0 and mb == 0 and step % max(1, contrast_every) == 0:   # the same geometry by another route
                x2 = T["_view2"].contiguous(memory_format=torch.channels_last_3d) if channels_last else T["_view2"]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                    P2 = fwd(x2, voxel=v)
                lcon = LT.contrast_consistency(model, rend["f"], P2, sh_lmax, margin)
                total = total + W["contrast"] * lcon; parts["contrast"] = lcon.detach()
            (total / accum).backward(); tot_v = tot_v + total.detach() / accum
            for k, v_ in parts.items():
                pa[k] = pa.get(k, 0.0) + v_ / accum
        if bench is not None:
            tick(); tm = time.perf_counter()
        if pbuf is not None and W["consistency"] > 0 and step % pair_every == 0:                    # paired renders
            kind = ("zoom", "modality")[(step // pair_every) % 2]
            pb = pbuf.batch(2, g, kind)
            if pb is not None:
                xa, xb, va, vb, info = pb
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                    Pa = fwd(xa, voxel=va); Pb = fwd(xb, voxel=vb)
                lc = LT.zoom_consistency(model, Pa, Pb, info, sh_lmax) if kind == "zoom" else LT.modality_consistency(model, Pa, Pb, sh_lmax)
                (W["consistency"] * lc).backward(); pa[f"cons_{kind}"] = lc.detach()
            pbuf.refresh(1)
        if rbuf is not None and W["ssl"] > 0 and step % ssl_every == 0:                             # unlabelled real crops
            xr = rbuf.batch(2, g)
            ls = LT.ssl_consistency(model, xr, g, sh_lmax, margin, voxel=torch.full((xr.shape[0],), rv, device=device))
            (W["ssl"] * ls).backward(); pa["ssl"] = ls.detach()
        if bench is not None:
            tick(); te = time.perf_counter()
        DF.allreduce_grads(model, world)
        gsum = DF.allreduce_mean(torch.stack([pa["gate_num"], pa["gate_den"]]), world)              # every rank reads the same gate signal
        gn_, gd_ = float(gsum[0]), float(gsum[1])
        # running sums of the error and of the energy, their ratio the signal: a step whose crops hold almost no
        # orientation energy cannot blow it up (a running mean of per-step ratios could)
        gst["num"] = gn_ if gst.get("num") is None else 0.95 * gst["num"] + 0.05 * gn_
        gst["den"] = gd_ if gst.get("den") is None else 0.95 * gst["den"] + 0.05 * gd_
        gst["ema"] = gst["num"] / max(gst["den"], 1e-9); ratio = gn_ / max(gd_, 1e-9)
        if gate is not None and not gst["open"] and ((step - start >= 50 and gst["ema"] < gate) or (gate_max is not None and step + 1 >= gate_max)):
            gst["open"], gst["at"] = True, step + 1
            log(f"gate open at step {step + 1}: order-2 error / energy {gst['ema']:.3f}")
            logj({"step": step + 1, "gate_open": True, "ratio": gst["ema"]})
        if difficulty == "auto" and gst["open"]:                                                    # widen when the current level is learned
            level["num"] = gn_ if level.get("num") is None else 0.98 * level["num"] + 0.02 * gn_
            level["den"] = gd_ if level.get("den") is None else 0.98 * level["den"] + 0.02 * gd_
            level["ema"] = level["num"] / max(level["den"], 1e-9)
            if level["v"] < 1.0 and step - level["since"] >= 300 and level["ema"] < 0.6:
                level["v"] = min(1.0, level["v"] + 0.1); level["since"] = step; buf.gen = cur_gen(); level["ema"] = level["num"] = level["den"] = None; set_routes()
                cfg["gen"] = cur_gen()
                if cpu is not None:
                    cpu.update(gen=cfg["gen"])
                log(f"difficulty -> {level['v']:.1f} at step {step + 1}")
                logj({"step": step + 1, "difficulty": level["v"]})
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        cfg["strength"] = strength(step)
        if cpu is not None and step % 25 == 0:
            cpu.update(strength=cfg["strength"])
        new = (prod.take(refresh) if prod is not None else []) + (cpu.take(refresh, device) if cpu is not None else [])
        if prod is None and cpu is None:
            buf.refresh(refresh, g, strength(step))
        else:
            buf.put(new, g); got_pairs += len(new)
        if bench is not None:
            tick(); tend = time.perf_counter()
            bench["data"].append(t_data); bench["main"].append(tm - ts - t_data); bench["extra"].append(te - tm); bench["opt"].append(tend - te); bench["step"].append(tend - ts)
        for k, v_ in {**pa, "total": tot_v, "grad_norm": gn.detach(), "gate_ratio": gst["ema"], "level": level["v"], "_n": 1}.items():
            run[k] = run.get(k, 0.0) + v_
        if (step + 1) % log_every == 0:
            nw = max(int(run.pop("_n", 1)), 1)                                                      # the steps in this window (a resume starts mid-window)
            rec = {"step": step + 1, "time_s": round(time.time() - t0, 1), "lr": lr_now, "strength": strength(step),
                   **{k: float(v_) / nw for k, v_ in run.items() if k not in ("lobe_l2_num",)}, "new_pairs_per_step": got_pairs / nw,
                   "producer_pairs_s": prod.rate() if prod is not None else 0.0, "cpu_pairs_s": cpu.rate() if cpu is not None else 0.0}
            if "lobe_l2_num" in run and "gate_den" in run:
                rec["lobe_l2_ratio"] = float(run["lobe_l2_num"]) / max(float(run["gate_den"]), 1e-9)
            logj(rec); run = {}; got_pairs = 0
            log(f"step {step + 1} {rec['time_s']}s s={rec['strength']:.2f} lv={rec['level']:.1f} total {rec['total']:.4f} sh {rec['sh']:.4f} ce {rec['ce']:.4f} "
                f"aux {rec.get('aux', 0):.4f} desc {rec.get('desc', 0):.4f} id {rec.get('identity', 0):.4f} gate {rec['gate_ratio']:.3f} "
                f"lobe_l2 {rec.get('lobe_l2_ratio', float('nan')):.3f} new/step {rec['new_pairs_per_step']:.2f}")
        if not benchmark and ((step + 1) % val_every == 0 or step + 1 == steps):
            if main:
                v_ = LT.evaluate_lobes(model, val, margin, device, sh_lmax, desc_lmax)
                logj({"step": step + 1, **v_})
                log(f"  val {step + 1}: " + " ".join(f"{k} {x:.4f}" for k, x in v_.items()))
            save(step + 1); ckp.last = time.time()
        elif not benchmark and ckp.due():
            save(step + 1); log(f"checkpoint at step {step + 1}")
        stop = torch.tensor([1.0 if ckp.stop else 0.0], device=device)
        if float(DF.allreduce_mean(stop, world)) > 0:                                               # the scheduler warned: save and leave for a resubmit
            save(step + 1); log(f"stopping at step {step + 1} on the scheduler's signal (exit 99: resubmit)")
            if prod is not None:
                prod.close()
            if cpu is not None:
                cpu.close()
            raise SystemExit(99)
    if bench is not None and main:
        import numpy as _np
        k0 = max(1, len(bench["step"]) // 5)                                                        # warm-up steps (autotuning, compile) left out
        rep = {k: round(float(_np.median(v[k0:])), 4) for k, v in bench.items() if len(v) > k0}
        rep.update({"samples_per_s_per_gpu": round(batch * accum / max(rep.get("step", 1e9), 1e-9), 2), "gpus": world,
                    "producer_pairs_s": round(prod.rate(), 3) if prod is not None else 0.0, "cpu_pairs_s": round(cpu.rate(), 3) if cpu is not None else 0.0,
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2) if str(device).startswith("cuda") else 0.0,
                    "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "cpu", "batch": batch, "accum": accum, "size": size,
                    "channels_last": channels_last, "compile": compile_model})
        log("BENCHMARK " + json.dumps(rep)); logj({"benchmark": rep})
    if prod is not None:
        prod.close()
    if cpu is not None:
        cpu.close()
    DF.finish_distributed(world)
    return model


def train_entities(out: str, steps: int = 6000, size: int = 128, batch: int = 4, accum: int = 4, buffer: int = 128, refresh: int = 1, lr: float = 3e-4,
                   warmup: int = 300, margin: int = 4, val_n: int = 16, val_every: int = 500, seed: int = 0, strength0: float = 0.15, strength1: float = 1.0,
                   gen: dict | None = None, widths=(32, 64, 128), embed: int = 16, context_level: bool = True, voxel_cond: bool = True, points: bool = True,
                   level0: float = 0.3, level_steps: float = 0.5, voxel_dropout: float = 0.3, weights: dict | None = None, init: str | None = None,
                   resume: str | None = None, routes: float = 1.0, contrast_views: bool = True, contrast_every: int = 2, producer: bool = True,
                   cpu_workers: int = 0, cpu_threads: int = 4, channels_last: bool = False, compile_model: bool = False, ckpt_minutes: float = 20.0,
                   benchmark: int = 0, device: str = "cuda", log_every: int = 25, n_dirs: int = 32, pairs: int = 16, pair_every: int = 4,
                   dir_lmax: int = 8, vote_unc: bool = False, e_sigma: tuple | None = None, deep_tree: bool = False, lcn: float = 0.0,
                   dens_sh: bool = True, read_keep: float = READ_KEEP, dens_mlp: bool = True, mass_region: bool = True,
                   region_dec: bool = False, where_first: int = 0, z_vote: bool = False, dir_sh: int = 0, background_entity: bool = True,
                   ray: bool = False, e_cross: float = 0.0, e_line: float = 0.0, region_density: bool = True, kappa_log: bool = False, vote_objects: bool = False,
                   dens_same_norm: bool = False, own_objects: bool = False, bg_head: bool = False, orient_from_texture: bool = True,
                   obj_head: bool = False, mc_fibres: bool = False, oriented: int = 0, geometry_targets: bool = False, block_norm: int = 1,
                   odirect: int = 0, od_sigmas: list | None = None, split_head: bool = False, mass_vote: bool = False, overfit: int = 0,
                   init_reset: str = "", ovis_targets: bool = True, od_norm: bool = False, dens_vis_w: float = 0.0, dens_gate: bool = False,
                   freeze_trunk: int = 0, trunk_lr: float = 1.0, init_fresh: str = "", st_path: bool = False, iso_head: bool = False, dens_share: bool = False, freeze_extra: str = "", reset_dens_logit: bool = False, dens_readout: bool = False, readout_lobe: bool = False, gate_share: bool = False, dens_nobg: bool = False, strength_mix: bool = False, dens_ce: bool = False, ce_no_st: bool = False, pop_k: int = 0, pop_dir: int = 0, oprobe: bool = False, oprobe_learned: int = 0, pop_seq: int = 0, pop_match_share: bool = False, od_widths: list | None = None, probe_scale: bool = False, pop_tie: float = 0.1, pop_scale: bool = False, scales: str | None = None,
                   matrix_p: float = 0.0, vis_boost: float | None = None, make_pool: int = 0, pool: str = ""):
    """The entity-field model's training (notes/plan_entity_field_model.md; :mod:`~shapetensor.learn.entity_train`):
    the scene generator with entity targets at random points and scales, the overall ODF on the lattice, directions
    sampled per query, the difficulty widened linearly from ``level0`` to 1 over the first ``level_steps`` of the run,
    batch-time contrast routes with a second view of the same geometry (consistency every ``contrast_every`` steps),
    the voxel size dropped at ``voxel_dropout``; every ``pair_every`` steps two overlapping windows of one scene (from a
    buffer of ``pairs``) trained to agree on their entities' places in the tree. ``dir_lmax``: the decoder's direction
    basis order; ``vote_unc``: a vote uncertainty output; ``e_sigma``: the query scales (voxels); ``deep_tree``:
    fascicles and branches in the generator's hierarchies; ``dens_sh``: the density's shape as harmonics of the point
    code (``dens_mlp``: through a small MLP); ``read_keep``: the share of (almost) unreadable crops kept (:data:`READ_KEEP`);
    ``mass_region``: the mass is the region map's mean in the query's window (where, once), not a separate output;
    ``where_first``: steps of the "where" losses alone (region, mass, overall ODF, fraction, points), the "which way"
    losses (density shape, vote, concentration, descriptors, tree, cross-window, contrast) then ramped in over as many. The cluster machinery of :func:`train_lobes`: producer thread, CPU
    workers, data parallel, checkpoints and ``--resume auto``, benchmark mode. ``overfit`` N: N fixed pairs at full
    difficulty and strength (no new pairs, no window pairs), scored on themselves -- can the model fit them at all?"""
    from . import dataflow as DF
    from . import entity_train as ET
    if overfit:
        buffer, refresh, producer, cpu_workers, pairs, level0, strength0 = int(overfit), 0, False, 0, 0, 1.0, strength1
    from . import lobe_train as LT
    from . import losses as _losses
    from .model import poincare_dist
    global HEAD, DESC_KEY, SH_EXACT, DESC_LMAX, QUERY, CACHE, ROUTES, VIEWS, READ_KEEP, REGION_DENSITY, OVIS_TARGETS
    READ_KEEP = float(read_keep); set_background_entity(background_entity); OVIS_TARGETS = bool(ovis_targets)
    REGION_DENSITY = bool(region_density); ET.REGION_DENSITY = REGION_DENSITY                          # (28 Oct review: was set in train_lobes only, so
                                                                                                      # the region, a density target, was trained with BCE)
    rank, world, local = DF.init_distributed(device)
    if str(device).startswith("cuda"):
        device = f"cuda:{local}"; torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        torch.set_num_threads(min(torch.get_num_threads(), 6))                                       # the trainer's own CPU work (pair draws)
                                                                                                     # beside the CPU workers: not every core
    HEAD, DESC_KEY, SH_EXACT, DESC_LMAX, QUERY = "entities", "descriptors5", 8, 0, None
    VIEWS = 2 if (routes and contrast_views) else 1
    _losses.DESC_K = 5; _losses.DESC_NC = 0
    if weights:
        ET.ENT_WEIGHTS.update(weights)
    base = dict((gen or {}).get("scene") or {"canvas": 2 * size, "crops_per_scene": 8})
    from . import augment as _au; _au.PROFILE = base.get("profile")                                # the batch routes follow the scene profile
    from .scene import FILL_QUOTA
    base.setdefault("cross_p", 0.5); base.setdefault("p_centre", 0.5)                                 # crossing scenes, crops centred on crossings
    base.setdefault("fill_quota", FILL_QUOTA)                                                        # the rest stratified by oriented fill
    base.setdefault("deep_tree", bool(deep_tree))
    if e_sigma:
        ENTITY["sigma"] = (float(e_sigma[0]), float(e_sigma[1]))
    ENTITY["p_cross"] = float(e_cross)                                                                # queries in crossing cores
    ENTITY["p_line"] = float(e_line)                                                                  # ... and on lines' material
    if scales:                                                                                        # targets as collections at a scale
        ENTITY["scales"] = scales if scales == "random" else float(scales)
    from . import entity_train as _et
    _et.VOTE_OBJECTS = bool(vote_objects)                                                              # votes only for oriented owners
    _et.DENS_SAME_NORM = bool(dens_same_norm)                                                          # the density's shape: both sides' means alike
    _et.DENS_VIS_WEIGHT = float(dens_vis_w)                                                            # ... weighted toward visible orientations
    _et.DENS_READOUT_LOBE = bool(readout_lobe)
    _et.DENS_NOBG = bool(dens_nobg)
    _et.DENS_CE = bool(dens_ce)
    _et.POP_MATCH_AS_LOSS = not bool(pop_match_share)
    _et.POP_TIE = float(pop_tie)                                                                      # (>= 1: the matching free)
    from . import scene as _sc
    _sc.MATRIX_P = float(matrix_p)                                                                    # scenes built of objects (ground, matrix,
    if vis_boost is not None:                                                                         # tissue layers): their share of scenes
        _sc.VIS_GUARANTEE_P = float(vis_boost)                                                        # (and the visibility boost's probability)
    _et.POP_BG = int(pop_seq) >= 2                                                                     # the background apart (pop_seq 2)                                                  # the populations' matching cost
    _et.DENS_READOUT = bool(dens_readout)                                                              # the two-part gate, the readout trained
    _et.DENS_SHARE = bool(dens_share)                                                                  # the gate = the visible oriented share
    _et.OWN_OBJECTS = bool(own_objects)                                                                # directions owned among oriented entities
    from . import generator as _gn
    _gn.ORIENT_FROM_TEXTURE = bool(orient_from_texture)                                                # orientation visibility from oriented texture
    _gn.ORIENT_VIS = not bool(ovis_targets)                                                            # (not needed with the image-measured
                                                                                                       # visibility: half the generator's visibility cost)
    _gn.TARGETS_FROM_GEOMETRY = bool(geometry_targets)                                                 # no visibility editing of the targets
    from . import fibres as _fb
    _fb.PACKING, _fb.RELAX = ("random", True) if mc_fibres else ("hex", False)                        # Monte Carlo random fibres (collisions resolved)
    level = {"v": level0}
    cur_gen = lambda: {"scene": LT.difficulty_gen(level["v"], base)}
    def set_routes():
        global ROUTES
        ramp_ = 1.0 if level0 >= 1.0 else min(1.0, max(0.0, (level["v"] - level0) / (1.0 - level0)))   # from none at the start (clean images
        ROUTES = routes * ramp_ if routes else 0.0                                                         # first: the density formed only then,
                                                                                                           # 27 Oct) to full with the difficulty
    main = rank == 0
    outd = Path(out); outd.mkdir(parents=True, exist_ok=True)
    logf = open(outd / "log.jsonl", "a") if main else None
    log = (lambda m: print(m, flush=True)) if main else (lambda m: None)
    def logj(d):
        if main:
            logf.write(json.dumps(d) + "\n"); logf.flush()
    args = dict(head="entities", steps=steps, size=size, batch=batch, accum=accum, buffer=buffer, refresh=refresh, lr=lr, warmup=warmup, margin=margin,
                val_n=val_n, seed=seed, strength0=strength0, strength1=strength1, gen={"scene": base}, widths=list(widths), sh_lmax=8, embed=embed,
                context_level=context_level, voxel_cond=voxel_cond, points=points, level0=level0, level_steps=level_steps, voxel_dropout=voxel_dropout,
                weights=weights, routes=routes, contrast_views=contrast_views, world=world, contrast_every=contrast_every, n_dirs=n_dirs,
                channels_last=channels_last, compile_model=compile_model, entity=dict(ENTITY), pairs=pairs, pair_every=pair_every,
                dir_lmax=dir_lmax, vote_unc=vote_unc, lcn=lcn, mass_head=True, region_head=True, dens_sh=bool(dens_sh), read_keep=READ_KEEP,
                dens_mlp=bool(dens_mlp), mass_region=bool(mass_region), region_dec=bool(region_dec), where_first=int(where_first),
                z_vote=bool(z_vote), dir_sh=int(dir_sh), background_entity=bool(background_entity), ray=bool(ray),
                region_density=bool(region_density), kappa_log=bool(kappa_log), vote_objects=bool(vote_objects), dens_same_norm=bool(dens_same_norm), own_objects=bool(own_objects), bg_head=bool(bg_head), orient_from_texture=bool(orient_from_texture), obj_head=bool(obj_head),
                block_norm=int(block_norm), odirect=int(odirect), od_sigmas=list(od_sigmas) if od_sigmas else None, od_widths=list(od_widths) if od_widths else None, split_head=bool(split_head),
                mass_vote=bool(mass_vote), od_norm=bool(od_norm), ovis_targets=bool(ovis_targets), dens_vis_w=float(dens_vis_w), dens_gate=bool(dens_gate), gate2=bool(dens_readout), dens_readout=bool(dens_readout), readout_lobe=bool(readout_lobe), gate_share=bool(gate_share), dens_nobg=bool(dens_nobg), strength_mix=bool(strength_mix), dens_ce=bool(dens_ce), ce_no_st=bool(ce_no_st), pop_k=int(pop_k), pop_dir=int(pop_dir), oprobe=bool(oprobe) or bool(oprobe_learned), oprobe_learned=int(oprobe_learned), pop_seq=int(pop_seq), pop_match_share=bool(pop_match_share), probe_scale=bool(probe_scale), pop_tie=float(pop_tie), pop_scale=bool(pop_scale), matrix_p=float(matrix_p), vis_boost=float(_sc.VIS_GUARANTEE_P),
                freeze_trunk=int(freeze_trunk), trunk_lr=float(trunk_lr), init_fresh=init_fresh, st_path=bool(st_path), iso_head=bool(iso_head), dens_share=bool(dens_share), freeze_extra=freeze_extra, reset_dens_logit=bool(reset_dens_logit),
                mc_fibres=bool(mc_fibres), oriented=int(oriented), geometry_targets=bool(geometry_targets))
    skipped = {}                                                                                   # non-finite steps skipped
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed * 1000 + rank)
    gd = torch.Generator(device=device).manual_seed(seed * 1000 + rank + 7) if str(device).startswith("cuda") else g
    model = PopulationNet.from_args(args).to(device)
    rf = receptive_field(model)
    TRUNK = ("stem", "l0", "l1", "l2", "l3", "d1", "d2", "d3", "skip", "mix", "film")                  # the convolutional trunk (not the heads)
    extra = [p_.strip() for p_ in (freeze_extra or "").split(",") if p_.strip()]                    # modules frozen with the trunk (e.g. ent.dir:
    is_gate = lambda n_: ".gate." in n_ or n_.endswith(".gcal") or n_.endswith(".ce_gain") or ".ce_floor." in n_                                                              # the isolated harness collapsed to flat within
    is_trunk = lambda n_: n_.split(".")[0] in TRUNK or any(n_.startswith(p_ + ".") or n_ == p_ for p_ in extra)   # 50 steps with the direction MLP trainable)
    gate_p = [p_ for n_, p_ in model.named_parameters() if is_gate(n_)]
    trunk_p = [p_ for n_, p_ in model.named_parameters() if is_trunk(n_) and not is_gate(n_)]
    rest_p = [p_ for n_, p_ in model.named_parameters() if not is_trunk(n_) and not is_gate(n_)]
    groups = [{"params": rest_p}, {"params": trunk_p, "lr_mult": float(trunk_lr), "trunk": True}]   # the trunk: frozen for ``freeze_trunk``
    if gate_p:                                                                                      # steps on a warm start, then at ``trunk_lr``
        groups.append({"params": gate_p, "lr_mult": 10.0})                                         # x the rate (29 Oct: the other heads' gradients
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=1e-4)                                       # eroded the pilot's density reading); the
                                                                                                    # visibility gate at 10x (its BCE moved 0.693
                                                                                                    # -> 0.685 in 2000 steps at 1e-4)
    start = 0
    if resume == "auto":
        resume = str(outd / "model.pt") if (outd / "model.pt").exists() else None
    if init and not resume:
        ick = torch.load(init, map_location=device, weights_only=False)
        isd = ick["model"]
        hit_ = lambda pre: (lambda k: any(k == p or k.startswith(p + ".") or ("." + p + ".") in k for p in pre))
        if init_fresh:                                                                               # heads left at their fresh initialisation
            pre = [p.strip() for p in init_fresh.split(",") if p.strip()]; fresh = [k for k in isd if hit_(pre)(k)]   # (e.g. the density path:
            isd = {k: v for k, v in isd.items() if k not in fresh}                                   # the pilot's lobes are wrong on scene crops
            log(f"init: {len(fresh)} tensors fresh ({', '.join(fresh[:6])}{', ...' if len(fresh) > 6 else ''})")   # and were unlearnt first, 29 Oct)
        if init_reset:                                                                               # heads at zero
            pre = [p.strip() for p in init_reset.split(",") if p.strip()]                            # (28 Oct review: the pilot's lattice ODF head
            hit = hit_(pre)                                                                          # was 93 % of the warm start's gradient)
            drop = [k for k in isd if hit(k)]; isd = {k: v for k, v in isd.items() if k not in drop}
            own = model.state_dict()
            with torch.no_grad():
                for k in drop:                                                                       # at zero: the head predicts nothing, its
                    if k in own:                                                                     # loss starts at its floor (a random head's
                        own[k].zero_()                                                               # lattice ODF loss was 10-19 in a dry run)
            log(f"init: {len(drop)} tensors at zero ({', '.join(drop[:6])}{', ...' if len(drop) > 6 else ''})")
        DF.load_partial(model, isd, log)
        if reset_dens_logit:                                                                         # the pilot's density logit row of the direction
            with torch.no_grad():                                                                    # MLP at zero: its lobes (wrong on scene crops)
                model.ent.dir[-1].weight[0].zero_(); model.ent.dir[-1].bias[0].zero_()                # no longer multiply the fresh density path
            log("init: the direction MLP's density logit at zero")
        if kappa_log and not (ick.get("args") or {}).get("kappa_log"):                               # a softplus checkpoint's kappa rows
            model.ent.reset_kappa(); log("kappa outputs restarted in log space")
    if resume:
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["step"]; level["v"] = ck.get("level", level["v"])
        log(f"resumed at step {start} (level {level['v']:.2f}){'; --init ignored' if init else ''}")
    if channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    fwd = torch.compile(model) if compile_model else model
    strength = lambda s_: strength0 + (strength1 - strength0) * min(1.0, s_ / (0.5 * steps))
    lvl = lambda s_: level0 + (1.0 - level0) * min(1.0, s_ / max(1.0, level_steps * steps))
    if strength_mix:                                                                               # 28 Oct (J7: a ramp that ends at full strength
        import random as _random                                                                   # taught the gate to stay off everywhere): each
        u_ = lambda s_: _random.Random(seed * 7919 + s_).random()                                  # step's new crops at one draw between the clear
        strength = lambda s_: strength0 + (strength1 - strength0) * u_(s_)                         # and the full end, strength and difficulty
        lvl = lambda s_: level0 + (1.0 - level0) * u_(s_)                                          # together
    level["v"] = lvl(start)
    log(f"model: {sum(p.numel() for p in model.parameters())} parameters, receptive field {rf} voxels, {world} GPU(s); args {args}")
    if make_pool:                                                                                  # pre-generate crops with the whole GPU, then stop
        import random as _random                                                                   # (30 Oct: one card cannot train and generate
        pd = Path(pool) if pool else outd / "pool"; pd.mkdir(parents=True, exist_ok=True)         # scenes of objects at once: 0.1-0.2 new crops
        cps = int(base.get("crops_per_scene", 1) or 1); shard, k_, t0p = [], 0, time.time()       # per step)
        for i in range(int(make_pool)):
            u = _random.Random(seed * 7919 + 1_000_003 + i // cps).random()                       # as strength_mix: one draw per scene, strength
            st_, lv_ = strength0 + (strength1 - strength0) * u, level0 + (1.0 - level0) * u       # and difficulty together
            x_, t_ = make(seed * 1_000_000_000 + 700_000_000 + i, size, device, st_, {"scene": LT.difficulty_gen(lv_, base)})
            shard.append((x_.cpu(), {kk: (vv.cpu() if torch.is_tensor(vv) else vv) for kk, vv in t_.items()}))
            if len(shard) == 16 or i == int(make_pool) - 1:                                        # (small shards: a deadline loses little)
                torch.save(shard, pd / f"shard_{k_:04d}.pt"); k_ += 1; shard = []
                log(f"pool: {i + 1} of {make_pool} crops in {time.time() - t0p:.0f} s -> {pd}")
        return
    poolsrc = DF.PoolSource(pool, seed, hold=val_n) if pool else None
    if poolsrc is not None:
        producer, cpu_workers = False, 0; log(f"pool: {len(poolsrc)} crops from {pool}, {val_n} more held out for validation (no producer, no CPU workers)")
    seed0 = seed * 1_000_000_000 + rank * 100_000_000 + start * refresh * 4
    buf = Buffer(buffer, size, device, seed0=seed0, strength=strength(start), log=log, gen=cur_gen(),
                 n0=(1 if poolsrc is not None else min(buffer, max(16, 2 * batch))) if (producer or cpu_workers > 0 or poolsrc is not None) else None)   # the rest from the producers, as they come
    if poolsrc is not None:
        buf.put(poolsrc.take(buffer - 1, device), torch.Generator().manual_seed(seed))                # (the buffer full from the start)
    cfg = {"strength": strength(start), "gen": cur_gen()}
    prod = DF.Producer(make, size, device, buf.next + 50_000_000, cfg) if producer else None
    cpu = None
    if cpu_workers > 0:
        state = {"HEAD": HEAD, "DESC_KEY": DESC_KEY, "SH_EXACT": SH_EXACT, "DESC_LMAX": DESC_LMAX, "QUERY": QUERY, "CACHE": None,
                 "L.DESC_K": _losses.DESC_K, "L.DESC_NC": _losses.DESC_NC, "ENTITY": dict(ENTITY), "READ_KEEP": READ_KEEP, "NOT_ENTITY": NOT_ENTITY, "REGION_DENSITY": REGION_DENSITY, "OVIS_TARGETS": OVIS_TARGETS, "GN.ORIENT_VIS": _gn.ORIENT_VIS,
                 "GN.ORIENT_FROM_TEXTURE": bool(orient_from_texture), "GN.TARGETS_FROM_GEOMETRY": bool(geometry_targets), "ENTITY_DISPERSION": ENTITY_DISPERSION,
                 "FB.PACKING": _fb.PACKING, "FB.RELAX": _fb.RELAX, "SC.MATRIX_P": _sc.MATRIX_P, "SC.VIS_GUARANTEE_P": _sc.VIS_GUARANTEE_P}
        cpu = DF.CpuWorkers(cpu_workers, size, buf.next + 90_000_000, cfg, state, threads=cpu_threads)
    pbuf = ET.EntityPairBuffer(pairs, size, "cpu", LT.difficulty_gen(level["v"], base), seed0=60_000_000 + rank * 1_000_000 + start,
                               out_device=device, n0=2, strength=strength(start)).start_async() if pairs else None   # drawn on the CPU, in a thread
    val = []                                                                                         # (see EntityPairBuffer)
    def make_val(i):                                                                                 # on a shared card an OOM here lost the run
        vg = {"scene": {**LT.difficulty_gen(1.0, base), "crops_per_scene": 1}}                       # before its first step (28 Oct): retried
        for t_ in range(3):                                                                          # once, then drawn on the CPU
            try:
                x_, t_v = make(10_000_000 + i, size, device if t_ < 2 else "cpu", strength1, vg)
                return x_.to(device), {k: v.to(device) if torch.is_tensor(v) else v for k, v in t_v.items()}
            except torch.OutOfMemoryError:
                if t_ == 2:
                    raise
                torch.cuda.empty_cache(); log(f"val crop {i}: GPU out of memory, {'retrying in 30 s' if t_ == 0 else 'drawing it on the CPU'}"); time.sleep(30 if t_ == 0 else 0)
    if overfit and main:
        val = list(buf.items[:val_n]); torch.save([(x_.cpu() if torch.is_tensor(x_) else x_, {k: v.cpu() if torch.is_tensor(v) else v for k, v in t_.items()}) for x_, t_ in buf.items], outd / "overfit_items.pt")
    elif main and not benchmark and poolsrc is not None:                                           # from the pool: its last crops, held out
        val = [augment(*it, torch.Generator().manual_seed(i)) for i, it in enumerate(poolsrc.holdout(val_n, device))]
    elif main and not benchmark:
        val = [augment(*make_val(i), torch.Generator().manual_seed(i)) for i in range(val_n)]
    def evaluate():
        model.eval(); acc = {}
        gv = torch.Generator(device=device).manual_seed(123) if str(device).startswith("cuda") else torch.Generator().manual_seed(123)
        with torch.no_grad():
            for i in range(0, len(val), 2):
                x, T = stack(val[i:i + 2], aug=False)
                dirs = ET.sample_dirs(T, n_dirs, g=gv)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                    P = model(x, voxel=T["voxel"], e_points=T["e_points"], e_sigma=T["e_sigma"], e_dirs=dirs, e_scale=T.get("e_scale"))
                _, parts = ET.entity_loss(P, T, dirs, margin=margin, g=gv)
                for k, v_ in {**{k: float(v) for k, v in parts.items()}, **ET.axis_error_deg(P, T, dirs), **ET.where_skill(P, T, dirs, margin)}.items():
                    acc.setdefault(k, []).append(v_)
        model.train()
        return {f"val_{k}": float(np.mean(v)) for k, v in acc.items()}
    import numpy as np
    if main and not benchmark:
        v0 = evaluate(); logj({"step": start, "baseline": True, **v0}); log(f"  val {start} (baseline): " + " ".join(f"{k} {x:.4f}" for k, x in v0.items()))
    ckp = DF.Checkpointer(ckpt_minutes)
    def save(step_):
        if main:
            tmp = outd / "model.pt.tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step_, "args": args, "receptive_field": rf, "level": level["v"]}, tmp)
            os.replace(tmp, outd / "model.pt")
    tick = (lambda: torch.cuda.synchronize(device)) if str(device).startswith("cuda") else (lambda: None)
    bench = {k: [] for k in ("data", "main", "opt", "step")} if benchmark else None
    t0 = time.time(); run = {}; got_pairs = 0
    set_routes()
    WHICH = ("density", "vote", "vote_unc", "kappa", "desc", "tree")
    ramp = lambda s_: 1.0 if where_first <= 0 else min(1.0, max(0.0, (s_ - where_first) / where_first))
    def which_w(s_):                                                                                 # the "which way" losses, ramped
        a_ = ramp(s_)
        w_ = {} if a_ >= 1.0 else {k: ET.ENT_WEIGHTS.get(k, 0.2) * a_ for k in WHICH}
        if init and s_ < warmup:                                                                     # a warm start: the lattice ODF loss, far off on
            w_["overall"] = ET.ENT_WEIGHTS["overall"] * s_ / max(warmup, 1)                          # new targets, ramped in so it cannot own the
        return w_ or None                                                                            # clipped gradient (28 Oct review: 93 % of it)
    end = min(steps, start + benchmark) if benchmark else steps
    for step in range(start, end):
        if bench is not None:
            tick(); ts = time.perf_counter()
        prog = step / max(steps, 1)
        lr_now = lr * min(1.0, (step + 1) / warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog)))
        for pg in opt.param_groups:
            pg["lr"] = lr_now * pg.get("lr_mult", 1.0) * (0.0 if (pg.get("trunk") and step < freeze_trunk) else 1.0)
        opt.zero_grad(set_to_none=True); pa = {}; tot_v = 0.0; t_data = 0.0
        for mb in range(accum):
            if bench is not None:
                tick(); td = time.perf_counter()
            x, T = buf.batch(batch, g, 0.0)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            v = T["voxel"].clone(); v[torch.rand(v.shape[0], generator=g).to(v.device) < voxel_dropout] = float("nan")
            dirs = ET.sample_dirs(T, n_dirs, g=gd)
            if bench is not None:
                tick(); t_data += time.perf_counter() - td
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                P = fwd(x, voxel=v, e_points=T["e_points"], e_sigma=T["e_sigma"], e_dirs=dirs, e_scale=T.get("e_scale"))
            total, parts = ET.entity_loss(P, T, dirs, W=which_w(step), margin=margin, g=gd)
            if "_view2" in T and ET.ENT_WEIGHTS.get("contrast", 0) * ramp(step) > 0 and mb == 0 and step % max(1, contrast_every) == 0:   # another route, same geometry
                x2 = T["_view2"].contiguous(memory_format=torch.channels_last_3d) if channels_last else T["_view2"]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                    P2 = fwd(x2, voxel=v, e_points=T["e_points"], e_sigma=T["e_sigma"], e_dirs=dirs, e_scale=T.get("e_scale"))
                e1, e2 = P["ent"], P2["ent"]
                lcon = (torch.log1p(e1["f"]) - torch.log1p(e2["f"])).abs().mean() + 0.1 * poincare_dist(e1["z"], e2["z"]).mean() \
                    + ((P["total_sh"].float() - P2["total_sh"].float()) ** 2).mean() / (P["total_sh"].float() ** 2).mean().clamp_min(1e-6)
                total = total + ET.ENT_WEIGHTS["contrast"] * ramp(step) * lcon; parts["contrast"] = lcon.detach()
            (total / accum).backward(); tot_v = tot_v + total.detach() / accum
            for k, v_ in parts.items():
                pa[k] = pa.get(k, 0.0) + v_ / accum
        if pbuf is not None and ET.ENT_WEIGHTS.get("cross", 0) * ramp(step) > 0 and step % max(1, pair_every) == 0:   # overlapping windows
            x1, x2, T1, T2, X = pbuf.batch(2, g)
            dp = ET.sample_dirs(T1, n_dirs, g=gd)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                P1 = fwd(x1, voxel=T1["voxel"], e_points=T1["e_points"], e_sigma=T1["e_sigma"], e_dirs=dp)
                P2 = fwd(x2, voxel=T2["voxel"], e_points=T2["e_points"], e_sigma=T2["e_sigma"], e_dirs=dp)
            l1, _ = ET.entity_loss(P1, T1, dp, W=which_w(step), margin=margin, g=gd); l2, _ = ET.entity_loss(P2, T2, dp, W=which_w(step), margin=margin, g=gd)
            lx = ET.cross_window_loss(P1, P2, T1, T2, X, dp)
            (0.25 * (l1 + l2) + ET.ENT_WEIGHTS["cross"] * ramp(step) * lx).backward(); pa["cross"] = lx.detach()
            pbuf.refresh(1)                                                                           # a new pair when one is ready (drawn in a thread)
        if bench is not None:
            tick(); tm = time.perf_counter()
        DF.allreduce_grads(model, world)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if torch.isfinite(gn):
            opt.step()
        else:                                                                                         # a non-finite gradient: the step skipped
            opt.zero_grad(set_to_none=True); skipped["n"] = skipped.get("n", 0) + 1
            if skipped["n"] % 10 == 1:
                print(f"step {step}: non-finite gradient, skipped ({skipped['n']} so far)", flush=True)
        if (level["v"] < 1.0 or strength_mix) and abs(lvl(step + 1) - level["v"]) >= 0.05:                              # the difficulty, widened in steps of 0.05
            level["v"] = lvl(step + 1); buf.gen = cur_gen(); cfg["gen"] = cur_gen(); set_routes()
            if pbuf is not None:
                pbuf.gen = dict(cur_gen()["scene"])
            if cpu is not None:
                cpu.update(gen=cfg["gen"])
        cfg["strength"] = strength(step)
        if cpu is not None and step % 25 == 0:
            cpu.update(strength=cfg["strength"])
        if pbuf is not None:
            pbuf.strength = cfg["strength"]                                                           # the pairs follow the curriculum too
        new = (prod.take(refresh) if prod is not None else []) + (cpu.take(refresh, device) if cpu is not None else []) \
            + (poolsrc.take(refresh, device) if poolsrc is not None else [])
        if prod is None and cpu is None and poolsrc is None:
            buf.refresh(refresh, g, strength(step))
        else:
            buf.put(new, g); got_pairs += len(new)
        if bench is not None:
            tick(); tend = time.perf_counter()
            bench["data"].append(t_data); bench["main"].append(tm - ts - t_data); bench["opt"].append(tend - tm); bench["step"].append(tend - ts)
        for k, v_ in {**pa, "total": tot_v, "grad_norm": gn.detach(), "level": level["v"], "_n": 1}.items():
            run[k] = run.get(k, 0.0) + v_
        if (step + 1) % log_every == 0:
            nw = max(int(run.pop("_n", 1)), 1)
            rec = {"step": step + 1, "time_s": round(time.time() - t0, 1), "lr": lr_now, "strength": strength(step),
                   **{k: float(v_) / nw for k, v_ in run.items()}, "new_pairs_per_step": got_pairs / nw}
            logj(rec); run = {}; got_pairs = 0
            log(f"step {step + 1} {rec['time_s']}s s={rec['strength']:.2f} lv={rec['level']:.2f} total {rec['total']:.4f} dens {rec.get('density', 0):.4f} "
                f"vote {rec.get('vote', 0):.4f} kappa {rec.get('kappa', 0):.4f} tree {rec.get('tree', 0):.4f} overall {rec.get('overall', 0):.4f} "
                f"cross {rec.get('cross', 0):.4f} owned {rec.get('owned_share', 0):.2f} new/step {rec['new_pairs_per_step']:.2f}")
        if not benchmark and ((step + 1) % val_every == 0 or step + 1 == steps):
            if main:
                v_ = evaluate(); logj({"step": step + 1, **v_}); log(f"  val {step + 1}: " + " ".join(f"{k} {x:.4f}" for k, x in v_.items()))
            save(step + 1); ckp.last = time.time()
        elif not benchmark and ckp.due():
            save(step + 1); log(f"checkpoint at step {step + 1}")
        stop = torch.tensor([1.0 if ckp.stop else 0.0], device=device)
        if float(DF.allreduce_mean(stop, world)) > 0:
            save(step + 1); log(f"stopping at step {step + 1} on the scheduler's signal (exit 99: resubmit)")
            if prod is not None:
                prod.close()
            if cpu is not None:
                cpu.close()
            if pbuf is not None:
                pbuf.stop()
            raise SystemExit(99)
    if bench is not None and main:
        k0 = max(1, len(bench["step"]) // 5)
        rep = {k: round(float(np.median(v[k0:])), 4) for k, v in bench.items() if len(v) > k0}
        rep.update({"samples_per_s_per_gpu": round(batch * accum / max(rep.get("step", 1e9), 1e-9), 2), "gpus": world,
                    "producer_pairs_s": round(prod.rate(), 3) if prod is not None else 0.0,
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2) if str(device).startswith("cuda") else 0.0,
                    "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "cpu", "batch": batch, "accum": accum, "size": size})
        log("BENCHMARK " + json.dumps(rep)); logj({"benchmark": rep})
    if prod is not None:
        prod.close()
    if cpu is not None:
        cpu.close()
    if pbuf is not None:                                                                              # (the pair thread, before the interpreter
        pbuf.stop()                                                                                    # exits: inside torch it would abort)
    DF.finish_distributed(world)
    return model


def n_coef_(lmax: int) -> int:
    from .sh import n_coef
    return n_coef(lmax)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--size", type=int, default=128, help="training crop side (voxels), whatever the voxel size; the scene canvas defaults to twice it")
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--accum", type=int, default=4); ap.add_argument("--buffer", type=int, default=128)
    ap.add_argument("--refresh", type=int, default=1); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--margin", type=int, default=4)
    ap.add_argument("--val-n", type=int, default=32); ap.add_argument("--val-every", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strength0", type=float, default=0.15); ap.add_argument("--overfit", type=int, default=0); ap.add_argument("--resume")
    ap.add_argument("--init-reset", default="", help="warm start: comma-separated heads put at zero, e.g. total (the lattice ODF head)")
    ap.add_argument("--init-fresh", default="", help="warm start: comma-separated modules left at their fresh initialisation, e.g. ent.dens,ent.od_net,ent.od_dens")
    ap.add_argument("--warmup", type=int, default=500); ap.add_argument("--strength1", type=float, default=1.0)
    ap.add_argument("--axis-head", action="store_true", help="a committed axis output per slot and for the mixture (axis cost: target anisotropy x (1 - cos^2); review 8)")
    ap.add_argument("--sh-loss", default="batchnorm", choices=("batchnorm", "decoupled"), help="density loss: per-order error over the batch's energy, or magnitude + target-weighted angle per order over running energies (order 2 first)")
    ap.add_argument("--shape-loss", default="frobenius", choices=("frobenius", "devcos"), help="tensor cost of slots and mixture: Frobenius, or deviator magnitude + target-weighted angle (does not reward round predictions)")
    ap.add_argument("--simple-window", type=int, default=None, help="steps over which the simple share falls from p0 to p1 (default half the run)")
    ap.add_argument("--gate-max", type=int, default=None, help="open the gate at this step at the latest (a fallback if it never breaks away)")
    ap.add_argument("--gate", type=float, default=None, help="orientation terms only, constant rate, until the mixture loss falls below this fraction of its round floor (EMA, e.g. 0.85); then the late terms and scalars ramp in and the rate decays")
    ap.add_argument("--aniso", type=float, nargs=2, default=None, help="sample buffer pairs in proportion to their non-round target energy ** alpha, alpha annealed from A0 to 0 over STEPS (e.g. 1 600)")
    ap.add_argument("--cache", help="a directory caching the training and validation pairs on disk (shared by runs drawing the same pairs)")
    ap.add_argument("--late-start", type=int, default=None, help="step at which mass, presence, kind and Hellinger start ramping in (300 over 300 steps); default max(300, 5%% of steps)")
    ap.add_argument("--scalar-window", type=int, nargs=2, default=None, help="steps over which the cell scalars (classes, fraction, size, identity) ramp from 0 to full; default 20%%-40%% of steps")
    ap.add_argument("--context", type=float, nargs=2, default=None, help="train the query decoder also with the input visible only within a random half-width in [lo, hi] voxels (e.g. 6 32)")
    ap.add_argument("--desc-lmax", type=int, default=0, help="also predict the descriptor densities (thickness, hollowness, angularity) to this SH order (needs --exact; e.g. 4)")
    ap.add_argument("--exact", action="store_true", help="SH head and query targets from the exact per-voxel density (orientations present in the window) instead of the slots / per-layer mean tensors")
    ap.add_argument("--order", default="mass", choices=("mass", "salience"), help="target slot order: heaviest or most anisotropic mass first")
    ap.add_argument("--init", help="warm start: load the weights (not the optimiser or step) of this checkpoint")
    ap.add_argument("--simple", type=json.loads, default=None, help="a curriculum: draw() arguments of the simple regime (JSON), mixed in with --p-simple")
    ap.add_argument("--p-simple", type=float, nargs=2, default=[0.0, 0.0], help="share of simple samples at the start and from half-way (e.g. 0.8 0.2)")
    ap.add_argument("--query", type=int, default=0, help="query points per sample: train the decoder of the density at any point and scale (needs --sh-lmax)")
    ap.add_argument("--presence-band", type=float, nargs=2, default=None, help="target masses between these two do not train presence (e.g. 0.002 0.02)")
    ap.add_argument("--weights", type=json.loads, default=None, help='loss weight overrides as JSON, e.g. {"presence": 0.2}')
    ap.add_argument("--lcn", type=float, default=0.0, help="add a local-contrast input channel ((x - mean) / sd in a Gaussian window of this sigma, voxels; e.g. 3)")
    ap.add_argument("--sh-lmax", type=int, default=0, help="also predict the populations' density in even SH to this order (permutation-free; e.g. 8)")
    ap.add_argument("--distinct", type=float, default=None, help="slot order kept only among populations whose eigenvalue spectra lie within this distance (e.g. 0.2); default: among all")
    ap.add_argument("--widths", type=int, nargs=3, default=[32, 64, 128], help="channels at strides 1, 2, 4")
    ap.add_argument("--gen", type=json.loads, default=None, help='draw() arguments as JSON, e.g. {"style": "flat", "crossing": false, "k_layers": 1, "kinds": ["line"]}')
    ap.add_argument("--head", default="slots", choices=("slots", "lobes", "entities"), help="output: slots (+ SH head), the lobe mixture, or the entity field (notes/plan_entity_field_model.md)")
    ap.add_argument("--embed", type=int, default=16, help="entities: Poincare embedding size")
    ap.add_argument("--n-dirs", type=int, default=32, help="entities: directions asked per query point")
    ap.add_argument("--dir-lmax", type=int, default=8, help="entities: the decoder's direction basis order (16: sharper switches between close entities)")
    ap.add_argument("--vote-unc", action="store_true", help="entities: a vote uncertainty output (a readout of the vote's expected error)")
    ap.add_argument("--e-sigma", type=float, nargs=2, default=None, help="entities: query scales, voxels (default 1 12: the overnight pilot's 1-6 left wider windows untrained)")
    ap.add_argument("--deep-tree", action="store_true", help="entities: fascicles and branches in the generator's hierarchies")
    ap.add_argument("--no-dens-sh", action="store_true", help="entities: the density from the direction MLP only (no harmonics of the point code)")
    ap.add_argument("--read-keep", type=float, default=READ_KEEP, help="entities: the share of (almost) unreadable crops kept (1: no cap)")
    ap.add_argument("--where-first", type=int, default=0, help="entities: steps of the where-losses alone (region, mass, overall), the which-way losses then ramped in over as many")
    ap.add_argument("--no-region-density", action="store_true", help="entities: the region map the populations' union (a probability), not their density")
    ap.add_argument("--e-cross", type=float, default=0.2, help="entities: the share of query points placed in crossing cores (two populations overlapping at window scale)")
    ap.add_argument("--e-line", type=float, default=0.0, help="entities: the share of query points placed on line-kind layers' material (within the tissue share)")
    ap.add_argument("--ray", action="store_true", help="entities: the decoder reads the features along each queried direction (the ray probe)")
    ap.add_argument("--kappa-log", action="store_true", help="entities: the vote's concentrations in log space (the softplus head collapses)")
    ap.add_argument("--dens-same-norm", action="store_true", help="entities: the density target's shape normalised over the prediction's uniform directions "
                                                                   "(against the exact mean, the lobes were trained ~20 %% low)")
    ap.add_argument("--orient-from-texture", action="store_true", help="(the default now) entities: orientation visibility from the layers' oriented texture")
    ap.add_argument("--orient-from-contrast", action="store_true", help="entities: orientation visibility from the regions' contrast t (before 2026-10-27)")
    ap.add_argument("--obj-head", action="store_true", help="entities: an objects'-occupancy head (populations' mass predicted directly, relative loss)")
    ap.add_argument("--block-norm", type=int, default=1, help="entities: the trunk's residual blocks read per-voxel normalised features (1; 2 also a normalised update; "
                                                               "0: off -- the from-scratch trunk blew up without it, 27 Oct)")
    ap.add_argument("--odirect", type=int, default=0, help="entities: the decoder's direct oriented path with this many bank directions (e.g. 256)")
    ap.add_argument("--od-norm", action="store_true", help="entities: the direct path also sees each level's energy along u relative to the strongest direction at the point")
    ap.add_argument("--no-ovis-targets", action="store_true", help="entities: the density targets at the generator's visibility, not the image's (entities.orientation_visibility)")
    ap.add_argument("--dens-vis-w", type=float, default=0.0, help="entities: the density loss weighted per query by 1 + this x the orientation's visibility")
    ap.add_argument("--dens-gate", action="store_true", help="entities: a visibility gate per query (trained on the image's orientation visibility); the lobe trained only where it shows")
    ap.add_argument("--reset-dens-logit", action="store_true", help="warm start: the direction MLP's density logit row at zero (its lobes no longer multiply the density path)")
    ap.add_argument("--freeze-extra", default="", help="entities: comma-separated modules frozen with the trunk (--freeze-trunk), e.g. ent.dir")
    ap.add_argument("--dens-readout", action="store_true", help="entities (with --dens-gate): a two-part gate (visibility x amplitude), the visibility on the image-measured v, the gated readout trained end to end")
    ap.add_argument("--pop-k", type=int, default=0, help="entities: the populations head -- K slots per query, each an existence, a share and an elliptical ODF (set prediction, no direction grid)")
    ap.add_argument("--pop-dir", type=int, default=0, help="entities: each population's ellipse the second moment of learned weights over this many directions (e.g. 96; needs --odirect)")
    ap.add_argument("--oprobe", action="store_true", help="entities: the oriented probe (a learned code of the image sampled along and around each direction); with --pop-dir also read by the populations")
    ap.add_argument("--oprobe-learned", type=int, default=0, help="entities: the oriented probe also samples this many learned fields (the finest trunk level projected; implies --oprobe)")
    ap.add_argument("--pop-seq", type=int, nargs="?", const=1, default=0, help="entities (--pop-dir): the slots in turn; 1: each direction scored with the earlier slots' u^T T u; 2: with their smoothed direction weights, the background its own share and the shares by stick breaking; 3: as 2, a slot existing when it holds a share")
    ap.add_argument("--pop-tie", type=float, default=0.1, help="entities (--pop-seq >= 2): slot k matched to the k-th largest population, swaps only within this share; >= 1: free matching")
    ap.add_argument("--pop-scale", action="store_true", help="populations: the model is given each query's collection scale (on with --scales)")
    ap.add_argument("--make-pool", type=int, default=0, help="entities: make this many crops (the run's own settings; strength and difficulty drawn per scene as --strength-mix) into --pool (default OUT/pool) and stop")
    ap.add_argument("--pool", default="", help="entities: train from a pool of pre-made crops (--make-pool): the buffer filled and refreshed from it")
    ap.add_argument("--matrix-p", type=float, default=0.0, help="entities: share of scenes built of objects (ground, matrix, tissue layers: scene.MATRIX_P); 0: the older scenes")
    ap.add_argument("--vis-boost", type=float, default=None, help="entities: in scenes of objects, the probability a barely visible foreground population is raised (scene.VIS_GUARANTEE_P; default its value)")
    ap.add_argument("--scales", default=None, help="entities: targets as collections of objects at a scale per query ('random', or a number of voxels; 0: objects); gives the model the scale (pop_scale)")
    ap.add_argument("--od-widths", type=float, nargs="+", default=None, help="entities: the direct bank's cross-section widths in voxels (default 0.6; e.g. 0.6 1.5 3)")
    ap.add_argument("--probe-scale", action="store_true", help="entities: the oriented probe's offsets scaled by the query's window (window / 3, within 0.5-2.5)")
    ap.add_argument("--pop-match-share", action="store_true", help="entities: match slots to populations with ellipse costs weighted by share (before 29 Oct 02:20) rather than as the loss weighs them")
    ap.add_argument("--ce-no-st", action="store_true", help="entities (with --dens-ce): no structure tensor in the density (neither its term nor its cues)")
    ap.add_argument("--dens-ce", action="store_true", help="entities: the density as a normalised distribution (log-SH + bank + structure tensor over a uniform floor) trained by cross-entropy against the non-background ODF")
    ap.add_argument("--strength-mix", action="store_true", help="entities: each step's crops at a random strength and difficulty between strength0/level0 and strength1/1 (a mix, not a ramp)")
    ap.add_argument("--dens-nobg", action="store_true", help="entities (with --dens-readout): the readout's target without the background entity (the bg head's)")
    ap.add_argument("--gate-share", action="store_true", help="entities (with --dens-readout): the gate's second output regresses the line share; a 3-scalar calibrator maps both to the gate")
    ap.add_argument("--readout-lobe", action="store_true", help="entities (with --dens-readout): the lobe trained through the readout's loss too")
    ap.add_argument("--dens-share", action="store_true", help="entities (with --dens-gate): the gate's target is the visible oriented share; the lobe trained on the oriented part alone")
    ap.add_argument("--iso-head", action="store_true", help="entities: a per-point isotropic pedestal added to the density")
    ap.add_argument("--st-path", action="store_true", help="entities: the structure tensor in the query's window read along each direction into the density (classical init)")
    ap.add_argument("--freeze-trunk", type=int, default=0, help="entities: the convolutional trunk frozen for this many steps (a warm start: the heads adapt first)")
    ap.add_argument("--trunk-lr", type=float, default=1.0, help="entities: the trunk's learning rate as a multiple of --lr (after --freeze-trunk)")
    ap.add_argument("--od-sigmas", type=float, nargs="+", default=None, help="entities: the direct path's pooling levels (e.g. 2 4 6 8)")
    ap.add_argument("--split-head", action="store_true", help="entities: a head for the objects' occupancy by kind (oriented / isotropic; targets ori_t / iso_t)")
    ap.add_argument("--mass-vote", action="store_true", help="entities: a per-direction vote for the owner's mass")
    ap.add_argument("--bg-head", action="store_true", help="entities: a background-occupancy head (isotropic objects told from the background by place)")
    ap.add_argument("--own-objects", action="store_true", help="entities: directions owned among the oriented entities only (the background owns none)")
    ap.add_argument("--vote-objects", action="store_true", help="entities: vote / kappa / descriptor losses only where an oriented entity owns the direction")
    ap.add_argument("--mc-fibres", action="store_true", help="entities: Monte Carlo random fibres in the scene generator (random packing, collision relaxation: learn/fibres.py)")
    ap.add_argument("--oriented", type=int, default=0, help="entities: oriented line energies along this many directions as extra input channels (0: off; e.g. 24)")
    ap.add_argument("--geometry-targets", action="store_true", help="entities: targets from the true geometry (no visibility editing: generator.TARGETS_FROM_GEOMETRY)")
    ap.add_argument("--dir-sh", type=int, default=0, help="entities: the per-direction outputs also as harmonics of the point code to this order (0: off; e.g. 6)")
    ap.add_argument("--no-background-entity", action="store_true", help="entities: the isotropic background not a population (before)")
    ap.add_argument("--z-vote", action="store_true", help="entities: the tree embedding sees the vote's axes (detached)")
    ap.add_argument("--region-dec", action="store_true", help="entities: a small decoder (lattice -> stride 2 -> full, 3x3x3) for the region map")
    ap.add_argument("--no-dens-mlp", action="store_true", help="entities: the density's harmonics linear in the point code")
    ap.add_argument("--no-mass-region", action="store_true", help="entities: the mass from its own output on the point code, not the region map's window")
    ap.add_argument("--level-steps", type=float, default=0.5, help="entities: the share of the run over which the difficulty widens to 1")
    ap.add_argument("--lobes-J", type=int, default=6, help="lobes per cell (6: direct fits recover two crossing axes ~90 %% of the time from random starts, 4: ~75 %%)")
    ap.add_argument("--no-context-level", action="store_true", help="lobes: no stride-8 trunk level")
    ap.add_argument("--no-voxel-cond", action="store_true", help="lobes: no voxel-size conditioning")
    ap.add_argument("--identity-dim", type=int, default=8, help="lobes: identity embedding size (0: none)")
    ap.add_argument("--no-points", action="store_true", help="lobes: no branch / endpoint maps")
    ap.add_argument("--difficulty", default="auto", choices=("auto", "fixed"), help="lobes: widen the generator's difficulty as the model learns, or the full mix from the start")
    ap.add_argument("--level0", type=float, default=0.3, help="lobes: the starting difficulty level")
    ap.add_argument("--pairs", type=int, default=None, help="paired renders kept for consistency (lobes: default 0, none; entities: overlapping windows, default 16)")
    ap.add_argument("--pair-every", type=int, default=4)
    ap.add_argument("--real", nargs="*", default=None, help="lobes: unlabelled real volumes for self-supervised consistency, PATH[@LEVEL][=VOXEL_UM]")
    ap.add_argument("--ssl-every", type=int, default=4)
    ap.add_argument("--voxel-dropout", type=float, default=0.3)
    ap.add_argument("--free-sh", action="store_true", help="lobes ablation: free harmonics instead of lobes (same trunk, targets, losses)")
    ap.add_argument("--routes", type=float, default=1.0, help="lobes: batch-time contrast routes (transfer functions, filters, fields, noise, missing lines / blocks) at this strength (scaled by the difficulty level); 0: off")
    ap.add_argument("--no-contrast-views", action="store_true", help="lobes: no second route per pair (no contrast-consistency loss)")
    ap.add_argument("--contrast-every", type=int, default=2, help="lobes: the second contrast view every this many steps")
    ap.add_argument("--no-producer", action="store_true", help="lobes: make pairs in the training loop (no background producer thread)")
    ap.add_argument("--cpu-workers", type=int, default=0, help="lobes: CPU-only pair-making processes (cluster: spare cores / --cpu-threads)")
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--channels-last", action="store_true", help="lobes: channels-last 3-D memory layout (tensor cores; benchmark first)")
    ap.add_argument("--compile", action="store_true", help="lobes: torch.compile the network (benchmark first)")
    ap.add_argument("--ckpt-minutes", type=float, default=20.0, help="lobes: save every this many minutes (and on SIGUSR1/2, SIGTERM)")
    ap.add_argument("--benchmark", type=int, default=0, help="lobes: time this many steps (data / forward-backward / extras / optimiser) and exit")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--device", default="cuda", help="lobes: cuda (each torchrun rank its local GPU) or cpu (tests)")
    a = ap.parse_args()
    if a.head == "entities":
        train_entities(a.out, a.steps, a.size, a.batch, a.accum, a.buffer, a.refresh, a.lr, warmup=min(a.warmup, 300), margin=a.margin, val_n=a.val_n,
                       val_every=a.val_every, seed=a.seed, strength0=a.strength0, strength1=a.strength1, gen=a.gen, widths=tuple(a.widths), embed=a.embed,
                       context_level=not a.no_context_level, voxel_cond=not a.no_voxel_cond, points=not a.no_points, level0=a.level0,
                       voxel_dropout=a.voxel_dropout, weights=a.weights, init=a.init, resume=a.resume, routes=a.routes, contrast_views=not a.no_contrast_views,
                       contrast_every=a.contrast_every, producer=not a.no_producer, cpu_workers=a.cpu_workers, cpu_threads=a.cpu_threads,
                       channels_last=a.channels_last, compile_model=a.compile, ckpt_minutes=a.ckpt_minutes, benchmark=a.benchmark, device=a.device,
                       log_every=a.log_every, n_dirs=a.n_dirs, pairs=16 if a.pairs is None else a.pairs, pair_every=a.pair_every,
                       dir_lmax=a.dir_lmax, vote_unc=a.vote_unc, e_sigma=a.e_sigma, deep_tree=a.deep_tree, lcn=a.lcn, dens_sh=not a.no_dens_sh,
                       read_keep=a.read_keep, level_steps=a.level_steps, dens_mlp=not a.no_dens_mlp, mass_region=not a.no_mass_region,
                       region_dec=a.region_dec, where_first=a.where_first, z_vote=a.z_vote, dir_sh=a.dir_sh,
                       background_entity=not a.no_background_entity, ray=a.ray, e_cross=a.e_cross, e_line=a.e_line, region_density=not a.no_region_density,
                       kappa_log=a.kappa_log, vote_objects=a.vote_objects, dens_same_norm=a.dens_same_norm, own_objects=a.own_objects, bg_head=a.bg_head, orient_from_texture=not a.orient_from_contrast, obj_head=a.obj_head,
                       block_norm=a.block_norm, odirect=a.odirect, od_sigmas=a.od_sigmas, split_head=a.split_head, mass_vote=a.mass_vote, od_norm=a.od_norm, ovis_targets=not a.no_ovis_targets, dens_vis_w=a.dens_vis_w, dens_gate=a.dens_gate, freeze_trunk=a.freeze_trunk, trunk_lr=a.trunk_lr, init_fresh=a.init_fresh, st_path=a.st_path, iso_head=a.iso_head, dens_share=a.dens_share, freeze_extra=a.freeze_extra, reset_dens_logit=a.reset_dens_logit, dens_readout=a.dens_readout, readout_lobe=a.readout_lobe, gate_share=a.gate_share, dens_nobg=a.dens_nobg, strength_mix=a.strength_mix, dens_ce=a.dens_ce, ce_no_st=a.ce_no_st, pop_k=a.pop_k, pop_dir=a.pop_dir, oprobe=a.oprobe, oprobe_learned=a.oprobe_learned, pop_seq=a.pop_seq, pop_match_share=a.pop_match_share, od_widths=a.od_widths, probe_scale=a.probe_scale, pop_tie=a.pop_tie, pop_scale=a.pop_scale or bool(a.scales), scales=a.scales, matrix_p=a.matrix_p, vis_boost=a.vis_boost, make_pool=a.make_pool, pool=a.pool,
                       mc_fibres=a.mc_fibres, oriented=a.oriented, geometry_targets=a.geometry_targets, overfit=a.overfit, init_reset=a.init_reset)
        return
    if a.head == "lobes":
        train_lobes(a.out, a.steps, a.size, a.batch, a.accum, a.buffer, a.refresh, a.lr, warmup=min(a.warmup, 300), margin=a.margin, val_n=a.val_n,
                    val_every=a.val_every, seed=a.seed, strength0=a.strength0, strength1=a.strength1, gen=a.gen, widths=tuple(a.widths),
                    sh_lmax=a.sh_lmax or 8, desc_lmax=a.desc_lmax or 4, lobes_J=a.lobes_J, context_level=not a.no_context_level, voxel_cond=not a.no_voxel_cond,
                    identity_dim=a.identity_dim, points=not a.no_points, query=a.query, gate=a.gate, gate_max=a.gate_max, difficulty=a.difficulty,
                    level0=a.level0, pairs=a.pairs or 0, pair_every=a.pair_every, real=a.real, ssl_every=a.ssl_every, voxel_dropout=a.voxel_dropout,
                    weights=a.weights, cache=a.cache, init=a.init, resume=a.resume, aniso=a.aniso, free_sh=a.free_sh, routes=a.routes,
                    contrast_views=not a.no_contrast_views, producer=not a.no_producer, cpu_workers=a.cpu_workers, cpu_threads=a.cpu_threads,
                    channels_last=a.channels_last, compile_model=a.compile, contrast_every=a.contrast_every, ckpt_minutes=a.ckpt_minutes,
                    benchmark=a.benchmark, log_every=a.log_every, device=a.device)
        return
    train(a.out, a.steps, a.size, a.batch, a.accum, a.buffer, a.refresh, a.lr, warmup=a.warmup, margin=a.margin, val_n=a.val_n, val_every=a.val_every,
          seed=a.seed, strength0=a.strength0, strength1=a.strength1, overfit=a.overfit, gen=a.gen, resume=a.resume, widths=tuple(a.widths), distinct=a.distinct, sh_lmax=a.sh_lmax, lcn=a.lcn, presence_band=a.presence_band, weights=a.weights, query=a.query, simple=a.simple, p_simple=tuple(a.p_simple), init=a.init, order=a.order, exact=a.exact, desc_lmax=a.desc_lmax, context=a.context, late_start=a.late_start, scalar_window=a.scalar_window, cache=a.cache, gate=a.gate, aniso=a.aniso, shape_loss=a.shape_loss, simple_window=a.simple_window, sh_loss=a.sh_loss, axis_head=a.axis_head, gate_max=a.gate_max)


if __name__ == "__main__":
    main()
