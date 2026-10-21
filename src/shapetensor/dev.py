"""The dev pipeline: anatomix-dev-vit features, sliceTCA smoothing, fibre-set populations, directions read from the shape.

Stages (the released pipeline in ``shapetensor.pipeline`` is unchanged and remains the reference):

1. **Features** -- ``devvit_slicetca``: anatomix-dev-vit at full resolution, patch-grid shift-averaged
   (``shapetensor.features.devvit``), then sliceTCA smoothing (``shapetensor.smoothing``), which replaces the 3³ box.
   ``engine="anatomix_devvit"`` with ``smooth=3`` (the box) validates within a few degrees of it at a fifth of the
   calibration time.
2. **Populations** -- fibre sets (``shapetensor.fibre_sets``): the released stage 2 per window, joined by continuity
   along the fibres, split where two groups cross, pruned below one window's length, re-unmixed; thresholds from
   synthetic references read through the block's own readers. The readers (signature dictionary, orientation readout)
   are calibrated as in the released pipeline (self-supervised by the excised structure tensor), plus a model of
   synthetic fibre-free texture (``presence_texture``): dev-vit's readout is confident enough to give texture a
   direction (and its patch grid puts it on the axes), so a window holds fibres when its features lie further from
   texture than any fibre-free texture window's (``FibreSetConfig.presence="distance"``, a Gaussian texture model; a
   linear fibres-vs-texture probe, ``"probe"``, misses dense crossings), not when its orientation density is peaked.
   dev-vit's readers must be calibrated in-domain (the block, or the volume it belongs to): fitted on straight
   bundles alone they do not carry over to other geometry and tissue.
3. **Direction per population** -- read from the shape space (``direction="readout"``): the orientation readout's
   per-voxel distributions as dyads ``sum_u P_u u u^T`` (``OrientationReadout.dyad_products``), weighted by the
   population's layer (each layer relative to the largest at the voxel, as the fibre-set prototypes were scored) and
   integrated over ``rho``; the population's direction is their principal axis. ``"shape"`` uses the features'
   gradients instead (``shapetensor.tensors.feature_products``; with dev-vit they follow the patch grid, see the
   results log) and ``"intensity"`` the released stage 3 (the image's excised structure tensor).
4. **Merging** -- unchanged (``shapetensor.merge``): the single field from the activation-weighted unit-trace tensors,
   and every population's vector ordered by activation.

Status: validated on the phantom suite against the released pipeline and the windowed fibre sets on encoder 2
(results log section 32, ``scripts/validate_dev.py`` with ``CALIBRATION=pool``) and for resilience to streak and ring
artefacts (``scripts/artefacts.py``, ``scripts/artefacts_real.py``). Crossings are found but not yet resolved into two
vectors (as with every method so far).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import torch

from . import st
from .fibre_sets import FibreSetConfig, References, fibre_sets, iso_textures
from .merge import multi_field, single_field
from .pipeline import Calibration, calibrate, run_volume
from .tensors import feature_products, population_tensor, shape_tensor


@dataclass
class DevConfig:
    """Everything the dev pipeline needs, serialisable (written with every field it produces)."""
    engine: str = "devvit_slicetca"
    engine_kwargs: dict = field(default_factory=lambda: {"core": 64, "shifts": [[0, 0, 0], [4, 4, 4]], "ranks": [2, 2, 2], "spacing": 3})
    sigma: float = 1.0
    rho: float = 10.0
    truncate: float = 3.0
    smooth: int = 1                                   # the readers' box average: sliceTCA already smooths
    seed: int = 0
    fibre_sets: FibreSetConfig = field(default_factory=FibreSetConfig)
    direction: str = "readout"                        # "readout" (the readout's distributions) | "shape" (feature gradients) | "intensity" (released stage 3)
    relative_layers: bool = True
    excise: bool = True
    tensor_floor: float = 0.01                        # in nulls: the layers are in multiples of the null
    merge_floor: float = 0.1
    vox_mm: float = 0.02
    reference_setting: str = "calibrated"
    presence_texture: bool = True                     # fit the presence probe on synthetic fibre-free texture
    iso_readout: bool = False                         # also a no-orientation output in the readout (it takes crossings for texture)

    def to_dict(self) -> dict:
        d = asdict(self); d["fibre_sets"] = self.fibre_sets.to_dict(); return d

    @classmethod
    def from_dict(cls, d: dict) -> "DevConfig":
        d = dict(d); fs = FibreSetConfig(**d.pop("fibre_sets", {}))
        return cls(fibre_sets=fs, **d)

    def calibration_kwargs(self, device: str = "cuda", iso: bool = True) -> dict:
        """The keyword arguments of :func:`shapetensor.pipeline.calibrate` (with the fibre-free texture when
        ``presence_texture`` and ``iso``)."""
        kw = dict(self.engine_kwargs)
        for k in ("shifts", "ranks"):
            if k in kw:
                kw[k] = [tuple(x) for x in kw[k]] if k == "shifts" else tuple(kw[k])
        out = {"engine": self.engine, "engine_kwargs": kw, "sigma": self.sigma, "rho": self.rho, "truncate": self.truncate,
               "smooth": self.smooth, "seed": self.seed}
        if iso and (self.presence_texture or self.iso_readout):
            out["iso"] = iso_textures(self.reference_setting, device); out["iso_readout"] = self.iso_readout
        return out


def calibrate_dev(volume, tissue=None, cfg: DevConfig | None = None, *, features=None, spacing=None, device: str = "cuda") -> Calibration:
    """The readers on a block with the dev features (the released :func:`shapetensor.pipeline.calibrate`)."""
    cfg = cfg or DevConfig()
    return calibrate(volume, tissue, features=features, spacing=spacing, device=device, **cfg.calibration_kwargs(device))


def apply_dev(cal: Calibration, volume, tissue=None, *, cfg: DevConfig | None = None, references: References | None = None,
              features=None, region=None, spacing=None, nmax: int | None = None, timings: dict | None = None, device: str = "cuda", log=None,
              **_ignored) -> dict:
    """Stages 2-4 of the dev pipeline on ``volume (Z, Y, X)`` with a calibration. Returns the keys the released
    :func:`shapetensor.pipeline.apply` returns where they apply (vectors in (z, y, x), no masks applied):
    ``activation (N, Z, Y, X)`` the sets' layers in multiples of the null; ``dirs``, ``gap`` each set's direction and
    uniqueness; ``vec``, ``vec_gap``, ``weight`` (the summed layers x ``tau``, so ``weight / tau`` >= 1 where a set is
    present, as in the released field) and ``tensor`` the single field; ``vecs``, ``weights`` (x ``tau``) the
    multi-vector field, the ``nmax`` strongest at each voxel when given; ``n_populations``; ``nulls`` and ``report`` from
    stage 2; ``tau``; ``confidence``."""
    cfg = cfg or DevConfig()
    t0 = time.perf_counter(); marks = {}
    v = st._to_torch(volume, device, torch.float32)
    tis = torch.ones_like(v, dtype=torch.bool) if tissue is None else st._to_torch(tissue, device) > 0.5
    F = cal.features(v) if features is None else features
    marks["features"] = time.perf_counter() - t0
    if region is None:
        m = cal.margin
        region = tuple(slice(m, max(m + 1, n - m)) for n in v.shape)
    if references is None:
        references = References(tuple(s.stop - s.start for s in region), cfg.reference_setting, device)
    nulls = references.nulls(cal, cfg.fibre_sets); marks["nulls"] = time.perf_counter() - t0
    L, rep = fibre_sets(cal, v, tis, F, nulls, cfg.fibre_sets, region=region, vox_mm=cfg.vox_mm, device=device, log=log)
    rep.pop("labels", None); marks["populations"] = time.perf_counter() - t0
    if L.shape[0] == 0:
        L = torch.zeros((1,) + tuple(v.shape), device=v.device)                       # no set: one empty layer (the field falls back)
    A = L / L.max(0).values.clamp_min(1.0)[None] if cfg.relative_layers else L
    w_exc = st.excision_weights(v, mask=tis if tissue is not None else None, spacing=st.spacing_ratio(spacing)) if cfg.excise else None
    if cfg.direction == "shape":
        P = feature_products(F, cfg.sigma, cfg.truncate, spacing=spacing)
    elif cfg.direction == "readout":
        P = cal.readout.dyad_products(cal.dictionary.prep(F), prepped=True)
    elif cfg.direction == "intensity":
        P = st.gradient_products(v, cfg.sigma, cfg.truncate, spacing=spacing, device=v.device)
    else:
        raise ValueError(f"direction must be 'readout', 'shape' or 'intensity', got {cfg.direction!r}")
    S, dirs, gaps = [], [], []
    for k in range(A.shape[0]):
        if cfg.direction in ("shape", "readout"):
            Sk, dk, gk = shape_tensor(P, A[k], cfg.rho, cfg.truncate, w_exc, cfg.tensor_floor, spacing)
        else:
            Sk, dk, gk = population_tensor(v, A[k], cfg.sigma, cfg.rho, cfg.truncate, w_exc, cfg.tensor_floor, spacing, P)
        S.append(Sk); dirs.append(dk); gaps.append(gk)
    S, dirs, gaps = torch.stack(S), torch.stack(dirs), torch.stack(gaps); del P
    vec, vec_gap, _, T = single_field(S, A, floor=cfg.merge_floor, return_tensor=True); del S
    order_w = L if cfg.relative_layers else A                                          # the multi-vector order and weights: the layers in nulls
    vecs, weights = multi_field(dirs, order_w)
    if nmax is not None and vecs.shape[0] > nmax:                                     # ordered by activation at each voxel
        vecs, weights = vecs[:nmax], weights[:nmax]
    tau = float(cal.tau)
    weight = L.clamp_min(0).sum(0) * tau
    marks["tensors_merge"] = time.perf_counter() - t0
    if timings is not None:
        timings.update({k: round(x, 3) for k, x in marks.items()})
    tissue_above = float(((L.max(0).values >= 1) & tis).float().sum() / tis.float().sum().clamp_min(1))
    return {"n_populations": int(L.shape[0]) if float(L.abs().sum()) > 0 else 0, "activation": L, "dirs": dirs, "gap": gaps,
            "vec": vec, "vec_gap": vec_gap, "weight": weight, "tensor": T, "vecs": vecs, "weights": weights * tau, "tau": tau,
            "nulls": {k: (v_ if k != "response" else None) for k, v_ in nulls.items()}, "report": rep,
            "confidence": {"readout_overlap": round(float(cal.readout_overlap), 3), "tissue_above_null": round(tissue_above, 3)}}


def population_field_dev(volume, tissue=None, cfg: DevConfig | None = None, *, device: str = "cuda", timings: dict | None = None, log=None) -> dict:
    """The dev pipeline on one block: calibrate on it, then stages 2-4."""
    cfg = cfg or DevConfig()
    v = st._to_torch(volume, device, torch.float32)
    t0 = time.perf_counter()
    cal = calibrate_dev(v, tissue, cfg, device=device)
    F = cal.features(v)
    if timings is not None:
        timings["calibration"] = round(time.perf_counter() - t0, 3)
    out = apply_dev(cal, v, tissue, cfg=cfg, features=F, device=device, timings=timings, log=log)
    out["calibration"] = cal; out["config"] = cfg.to_dict()
    return out


def run_volume_dev(src: str, dst: str, cfg: DevConfig | None = None, *, level: str | None = None, chunk=(192, 192, 192), nmax: int = 3,
                   tissue_threshold: float | None = None, calibration_size: int = 256, calibration_origin=None, calibrate_chunks: bool = True,
                   min_bins: int = 12, blend: int = 32, multi: bool = True, spacing=None, rank: int = 0, world: int = 1,
                   device: str = "cuda", log=print) -> str:
    """The dev pipeline over a whole zarr volume through the released chunked driver (:func:`shapetensor.pipeline.run_volume`:
    per-chunk calibration with the volume's as fallback, overlap blending of the merged tensors, the same output layout).
    The references are built once for the largest analysed region and their features computed once."""
    cfg = cfg or DevConfig()
    reach = int(cfg.truncate * (cfg.sigma + cfg.rho) + 0.5)
    from .features import make_engine
    eng_margin = int(getattr(make_engine(cfg.engine, device=device, **cfg.calibration_kwargs(iso=False)["engine_kwargs"]), "margin", 24))
    refs = References(tuple(int(c) + 2 * blend for c in chunk), cfg.reference_setting, device)
    cal_kw = cfg.calibration_kwargs(device)                                              # the no-orientation texture made once

    def cal_fn(vol, tissue=None, *, features=None, spacing=None, device=device, **_):
        return calibrate(vol, tissue, features=features, spacing=spacing, device=device, **cal_kw)

    def apply_fn(cal, blk, tis=None, *, nmax=nmax, spacing=None, features=None, device=device):
        return apply_dev(cal, blk, tis, cfg=cfg, references=refs, features=features, spacing=spacing, nmax=nmax, device=device)

    out = run_volume(src, dst, level=level, chunk=chunk, nmax=nmax, tissue_threshold=tissue_threshold, calibration_size=calibration_size,
                     calibration_origin=calibration_origin, calibrate_chunks=calibrate_chunks, min_bins=min_bins, blend=blend, multi=multi,
                     spacing=spacing, rank=rank, world=world, device=device, log=log, calibrate_fn=cal_fn, apply_fn=apply_fn,
                     engine_margin=eng_margin, pipeline="dev", sigma=cfg.sigma, rho=cfg.rho, truncate=cfg.truncate)
    import zarr
    zarr.open_group(dst, mode="a").attrs.update({"dev_config": json.loads(json.dumps(cfg.to_dict())), "reach": reach})
    return out
