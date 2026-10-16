"""The validation suite driver.

Pass 1, :func:`build_caches`: per (phantom, seed, engine) the features and gradients once, then per rho the
channel-resolved tensor ``K`` (and, for the multi-population phantoms, the gradient-direction ODF) at the sites of
the mid-y row, subsampled to ``stride`` voxels, saved under ``DATA_ROOT/cache/<tag>/``.
Pass 2, :func:`run_single` / :func:`run_multi` / :func:`run_hybrid`: the read-outs on a cache and their scores,
saved as JSON under ``DATA_ROOT/results/<tag>/`` with the per-site arrays beside them for the figures.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .. import DATA_ROOT
from ..features import make_engine
from ..filters import gaussian
from ..odf import fit_ring_width, ring_profile, zonal_kernel
from ..pca import PCA
from ..phantoms import make
from ..readout import common, hybrid, multi, single
from ..tensor import gradients, k_sites, odf_sites
from . import score

CACHE_DIR = Path(DATA_ROOT) / "cache"
RESULTS_DIR = Path(DATA_ROOT) / "results"
MULTI_PIDS = ("P4", "P5")
NULL_PIDS = ("P3-0", "P8", "P9")


@dataclass
class Spec:
    pids: tuple
    seeds: tuple = (0, 1, 2)
    engines: tuple = ("intensity", "jet", "anatomix16", "anatomix48")
    rhos: tuple = (4.0, 8.0, 16.0, 32.0)
    sigma: float = 1.0
    truncate: float = 3.0
    stride: int = 4
    rho_max: float = 32.0
    vox: float = 0.02
    n_pcs: int = 16
    lmax: int = 8
    odf: bool = True                       # ODF caches for the multi-population phantoms
    tex: dict = field(default_factory=dict)
    engine_kwargs: dict = field(default_factory=dict)
    tag: str = "main"
    device: str = "cuda"

    def texture(self) -> make.TextureParams:
        return make.TextureParams(**self.tex)


def cache_path(spec: Spec, pid: str, seed: int, engine: str, rho: float) -> Path:
    return CACHE_DIR / spec.tag / f"{pid}_s{seed}" / f"{engine}_rho{rho:g}.npz"


def result_path(spec: Spec, kind: str, pid: str, seed: int, engine: str, rho: float) -> Path:
    return RESULTS_DIR / spec.tag / kind / f"{pid}_s{seed}_{engine}_rho{rho:g}.json"


def ensure_phantom(spec: Spec, pid: str, seed: int, log=print):
    p = make.path_for(pid, seed)
    if not p.exists():
        t0 = time.time()
        grid = make.grid_for(pid, vox=spec.vox, rho_max=spec.rho_max)
        img, tr, meta = make.build(pid, seed, spec.texture(), grid, device=spec.device)
        make.save(pid, seed, img, tr, meta)
        log(f"built {pid} s{seed} {img.shape} in {time.time() - t0:.0f} s")
    return make.load(pid, seed)


def _is_multi(pid: str) -> bool:
    return pid.startswith(MULTI_PIDS)


def build_caches(spec: Spec, pid: str, seed: int, log=print):
    img, tr, meta, grid = ensure_phantom(spec, pid, seed, log)
    dev = spec.device
    vol = None
    for engine in spec.engines:
        missing = [r for r in spec.rhos if not cache_path(spec, pid, seed, engine, r).exists()]
        if not missing:
            continue
        if vol is None:
            vol = torch.as_tensor(img, device=dev).float()
        t0 = time.time()
        eng = make_engine(engine, **spec.engine_kwargs.get(engine, {}))
        F = eng(vol)
        desc = eng.describe()
        if F.shape[0] > spec.n_pcs:
            pca = PCA.fit(F, spec.n_pcs, seed=seed)
            F = pca(F)
            desc["suite_pca_explained"] = [round(float(x), 4) for x in pca.explained.tolist()]
        t_feat = time.time() - t0
        G = gradients(F, spec.sigma, spec.truncate)
        del F
        Gm = None
        if engine == "intensity":
            gm = sum(gaussian(vol, 3.0, order=o, truncate=spec.truncate) ** 2 for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))).sqrt()
            Gm = gm[None].half()
            del gm
        for rho in missing:
            t1 = time.time()
            b = max(1, int(rho // 4))
            s = max(1, spec.stride // b)
            K, sites = k_sites(G, rho, spec.truncate, b, [grid.y_mid], grid.core_zx)
            out = {"K": K[0, ::s, ::s].cpu().numpy(), "z": sites.z[::s], "x": sites.x[::s], "rho": rho, "reduce": b,
                   "stride": s * b, "engine": json.dumps(desc), "sigma": spec.sigma, "truncate": spec.truncate}
            if Gm is not None:
                gm2, _ = k_sites(Gm, rho, spec.truncate, b, [grid.y_mid], grid.core_zx, pairs=(np.array([0]), np.array([0])))
                out["gm"] = gm2[0, ::s, ::s, 0].clamp_min(0).sqrt().cpu().numpy()
            if spec.odf and (_is_multi(pid) or make.split_pid(pid)[0] in ("P8", "P9")):
                O, _ = odf_sites(G, rho, spec.truncate, b, [grid.y_mid], grid.core_zx, lmax=spec.lmax)
                out["odf"] = O[0, ::s, ::s].cpu().numpy()
            p = cache_path(spec, pid, seed, engine, rho)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, **out)
            log(f"  {pid} s{seed} {engine:10s} rho {rho:g}: K {out['K'].shape} in {time.time() - t1:.0f} s")
        del G, Gm
        torch.cuda.empty_cache()
        log(f"{pid} s{seed} {engine}: features {t_feat:.0f} s, total {time.time() - t0:.0f} s")


def load_cache(spec: Spec, pid: str, seed: int, engine: str, rho: float) -> dict:
    with np.load(cache_path(spec, pid, seed, engine, rho)) as z:
        d = {k: z[k] for k in z.files}
    d["engine"] = json.loads(str(d["engine"]))
    return d


def _K_tensor(c: dict, device) -> torch.Tensor:
    Kp = torch.as_tensor(c["K"].reshape(-1, c["K"].shape[-1]), device=device)
    n3 = common.nc3_from_packed(Kp.shape[1])
    return common.unpack(Kp, n3)


def _truth_at(spec: Spec, pid: str, seed: int, c: dict):
    _, tr, meta, grid = make.load(pid, seed)
    t = score.sample_truth(tr, c["z"], c["x"])
    return t, grid


def _save_result(path: Path, summary: dict, arrays: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1))
    if arrays:
        np.savez_compressed(path.with_suffix(".npz"), **arrays)


def run_single(spec: Spec, pid: str, seed: int, engine: str, rho: float, log=print, force: bool = False) -> dict:
    """Eigen and TCA read-outs, detectors, scores; for the intensity engine the eigen read-out only."""
    rp = result_path(spec, "single", pid, seed, engine, rho)
    if rp.exists() and not force:
        return json.loads(rp.read_text())
    t0 = time.time()
    c = load_cache(spec, pid, seed, engine, rho)
    K = _K_tensor(c, spec.device)
    t, grid = _truth_at(spec, pid, seed, c)
    res = {"pid": pid, "seed": seed, "engine": engine, "rho": rho, "n_sites": int(K.shape[0]), "engine_desc": c["engine"]}
    arrays = {"z": c["z"], "x": c["x"]}
    e = single.eigen(K)
    outs = {"eigen": e["dir"]}
    # ``trace``: the window's total gradient energy of the engine's channels -- for anatomix16 the 16-D output's own
    # boundary strength |df/dx| (what that layer was trained to see), for the intensity engine the gradient energy
    dets = {"cl": e["cl"], "gap": e["gap"], "trace": e["vals"].sum(-1)}
    if K.shape[-1] > 3:
        tc = single.tca(K, rank=3)
        outs["tca"] = tc["dir"]
        dets.update({"tca_dom": tc["bdom"], "tca_share": tc["bshare"], "tca_unpaired": tc["unpaired"], "tca_gap": tc["gap"]})
        arrays["tca_bdir"] = tc["bdir"].cpu().numpy()
        arrays["tca_gram"] = tc["gram"].cpu().numpy()
        res["tca_fit_err_median"] = float(tc["fit_err"].median())
    if "gm" in c:
        dets["gm"] = torch.as_tensor(c["gm"].reshape(-1), device=spec.device)
    arrays["T"] = common.channel_sum(K).cpu().numpy()
    res["scores"] = {}
    for name, d in outs.items():
        dn = d.cpu().numpy()
        arrays[f"dir_{name}"] = dn
        err = score.angle_err(dn, t["dir1"])
        arrays[f"err_{name}"] = err
        res["scores"][name] = {"bands": score.band_summary(err, t["dist"], t["region"]),
                               "parallel": score.parallel_share(dn, t["bnorm"], t["dist"], t["region"])}
    res["parallel_truth"] = score.parallel_share(t["dir1"], t["bnorm"], t["dist"], t["region"])
    res["detectors"] = {}
    for name, s in dets.items():
        sn = s.detach().cpu().numpy().astype(np.float32).reshape(-1)
        arrays[f"det_{name}"] = sn
        if np.isfinite(t["dist"]).any():
            res["detectors"][name] = score.detector_pr(sn, t["dist"], t["region"])
        wm = t["region"] == 2
        res["detectors"].setdefault(name, {})["quantiles_wm"] = {q: float(np.nanquantile(sn[wm], q / 100)) for q in (50, 90, 95, 99)}
    res["elapsed_s"] = time.time() - t0
    _save_result(rp, res, arrays)
    log(f"single {pid} s{seed} {engine} rho {rho:g}: " + "  ".join(
        f"{k} <0.1mm {v['bands'].get('0-0.1mm', {}).get('median', float('nan')):.1f} 0.25-0.5 {v['bands'].get('0.25-0.5mm', {}).get('median', float('nan')):.1f} >2mm {v['bands'].get('>2mm', {}).get('median', float('nan')):.1f} noif {v['bands'].get('nointerface', {}).get('median', float('nan')):.1f}"
        for k, v in res["scores"].items()) + f"  ({res['elapsed_s']:.0f} s)")
    return res


def csd_kernel(spec: Spec, engine: str, rho: float, seed: int, log=print) -> tuple[torch.Tensor, float]:
    """The single-population ring response for this engine / rho, fitted on the single-population line phantom."""
    pid = "P4-lines-single"
    if not cache_path(spec, pid, seed, engine, rho).exists():
        build_caches(spec, pid, seed, log)
    c = load_cache(spec, pid, seed, engine, rho)
    t, _ = _truth_at(spec, pid, seed, c)
    O = torch.as_tensor(c["odf"].reshape(-1, c["odf"].shape[-1]), device=spec.device)
    width = fit_ring_width(O, torch.as_tensor(t["dir1"], device=spec.device), spec.lmax)
    return zonal_kernel(spec.lmax, ring_profile(width)), width


def run_multi(spec: Spec, pid: str, seed: int, engine: str, rho: float, log=print, force: bool = False,
              methods=("bic", "overcomplete", "frt", "csd", "csd_refined", "csd_stable"), stride: int = 2) -> dict:
    """The multi-population read-outs on every ``stride``-th site in z and x (the CP fits dominate the cost and
    3k sites per seed are plenty for the shares and medians)."""
    rp = result_path(spec, "multi", pid, seed, engine, rho)
    if rp.exists() and not force:
        return json.loads(rp.read_text())
    t0 = time.time()
    c = load_cache(spec, pid, seed, engine, rho)
    if stride > 1:
        c = dict(c)
        c["K"], c["z"], c["x"] = c["K"][::stride, ::stride], c["z"][::stride], c["x"][::stride]
        if "odf" in c:
            c["odf"] = c["odf"][::stride, ::stride]
    K = _K_tensor(c, spec.device)
    t, grid = _truth_at(spec, pid, seed, c)
    res = {"pid": pid, "seed": seed, "engine": engine, "rho": rho, "n_sites": int(K.shape[0]), "methods": {}}
    arrays = {"z": c["z"], "x": c["x"]}
    outs = {}
    C = K.shape[-1] // 3
    if "bic" in methods and C > 1:
        outs["bic"] = multi.n_by_bic(K, starts=2, iters=150)
    if "overcomplete" in methods and C > 1:
        outs["overcomplete"] = multi.overcomplete_group(K, rank=6, starts=2, iters=150)
    if "odf" in c and any(m in methods for m in ("frt", "csd", "csd_refined", "csd_stable")):
        O = torch.as_tensor(c["odf"].reshape(-1, c["odf"].shape[-1]), device=spec.device)
        if "frt" in methods:
            outs["frt"] = multi.n_by_peaks(O, spec.lmax, mode="frt", rel=0.5)
        if any(m in methods for m in ("csd", "csd_refined", "csd_stable")):
            kern, width = csd_kernel(spec, engine, rho, seed, log)
            res["csd_ring_width_deg"] = float(np.degrees(width))
            cs = multi.n_by_peaks(O, spec.lmax, mode="csd", kernel=kern, rel=0.3)
            outs["csd"] = cs
            if C > 1 and ("csd_refined" in methods or "csd_stable" in methods):
                rf = multi.refine_fixed_n(K, cs["fibres"], cs["count"])
                outs["csd_refined"] = rf
                if "csd_stable" in methods:
                    st = multi.stability(K, rf["fibres"], rf["count"], n_boot=3)
                    keep = st["keep"]
                    fib = rf["fibres"].clone(); frac = rf["fracs"].clone()
                    frac[~keep] = 0
                    frac = frac / frac.sum(1, keepdim=True).clamp_min(1e-30)
                    fib[~keep] = 0
                    # compact: stable populations first
                    order = torch.argsort(frac, 1, descending=True)
                    fib = torch.gather(fib, 1, order[..., None].expand(-1, -1, 3)); frac = torch.gather(frac, 1, order)
                    outs["csd_stable"] = {"fibres": fib, "fracs": frac, "count": (frac > 0).sum(1), "spread": st["spread"]}
    for name, o in outs.items():
        fib, frac, cnt = (o[k].cpu().numpy() for k in ("fibres", "fracs", "count"))
        arrays[f"{name}_fibres"], arrays[f"{name}_fracs"], arrays[f"{name}_count"] = fib, frac, cnt
        res["methods"][name] = score.multi_scores(fib, frac, cnt, t["dir1"], t["dir2"], t["npop"], t["frac1"],
                                                  t.get("cross_deg"), t["region"])
    res["elapsed_s"] = time.time() - t0
    _save_result(rp, res, arrays)
    msg = []
    for k, v in res["methods"].items():
        if "two" in v:
            msg.append(f"{k}: N2 {v['two']['found2_share']:.0%} both<15 {v['two']['both_within_15_share']:.0%} e {v['two']['err1'].get('median', float('nan')):.0f}/{v['two']['err2'].get('median', float('nan')):.0f}")
        elif "single" in v:
            msg.append(f"{k}: e {v['single'].get('median', float('nan')):.1f} false2 {v['single']['false_second_share']:.0%}")
        elif "null" in v:
            msg.append(f"{k}: false_any {v['null']['false_any_share']:.0%}")
    log(f"multi {pid} s{seed} {engine} rho {rho:g}: " + " | ".join(msg) + f"  ({res['elapsed_s']:.0f} s)")
    return res


def null_pids_for(pid: str | None) -> tuple:
    """The no-interface phantoms at the same texture level as ``pid`` (its ``-t<f>`` suffix)."""
    if pid is None:
        return NULL_PIDS
    base, t = make.split_pid(pid)
    suf = "" if t == 1.0 else f"-t{t:g}"
    return (f"P3-0{suf}", f"P8{suf}", "P9")


def null_s0(spec: Spec, engine: str, rho: float, seed: int, detector: str, q: float = 0.99, log=print,
            pid: str | None = None) -> float:
    """The null-calibrated threshold of a detector: its q-quantile over WM sites of the no-interface phantoms
    (at the texture level of ``pid``)."""
    vals = []
    for pid in null_pids_for(pid):
        rp = result_path(spec, "single", pid, seed, engine, rho)
        if not rp.exists():
            if not cache_path(spec, pid, seed, engine, rho).exists():
                build_caches(spec, pid, seed, log)
            run_single(spec, pid, seed, engine, rho, log)
        with np.load(rp.with_suffix(".npz")) as z:
            if f"det_{detector}" not in z.files:
                continue
            s = z[f"det_{detector}"]
        _, tr, _, _ = make.load(pid, seed)
        c = load_cache(spec, pid, seed, engine, rho)
        t = score.sample_truth({"region": tr["region"]}, c["z"], c["x"])
        vals.append(s[t["region"] == 2])
    return float(np.nanquantile(np.concatenate(vals), q))


def run_hybrid(spec: Spec, pid: str, seed: int, engine: str, rho: float, detectors=("cl", "tca_dom", "gm"),
               log=print, force: bool = False) -> dict:
    """H1 (blend / switch) of the intensity eigen direction and the shape TCA direction, switched by each detector
    at its null-calibrated s0; scored like the single read-outs."""
    rp = result_path(spec, "hybrid", pid, seed, engine, rho)
    if rp.exists() and not force:
        return json.loads(rp.read_text())
    t0 = time.time()
    for eng in ("intensity", engine):
        if not result_path(spec, "single", pid, seed, eng, rho).exists():
            run_single(spec, pid, seed, eng, rho, log)
    with np.load(result_path(spec, "single", pid, seed, "intensity", rho).with_suffix(".npz")) as z:
        T_int = torch.as_tensor(z["T"], device=spec.device); d_int = torch.as_tensor(z["dir_eigen"], device=spec.device)
        det_int = {k[4:]: torch.as_tensor(z[k], device=spec.device) for k in z.files if k.startswith("det_")}
    with np.load(result_path(spec, "single", pid, seed, engine, rho).with_suffix(".npz")) as z:
        G_sh = torch.as_tensor(z["tca_gram"], device=spec.device); d_sh = torch.as_tensor(z["dir_tca"], device=spec.device)
        det_sh = {k[4:]: torch.as_tensor(z[k], device=spec.device) for k in z.files if k.startswith("det_")}
    c = load_cache(spec, pid, seed, engine, rho)
    t, grid = _truth_at(spec, pid, seed, c)
    res = {"pid": pid, "seed": seed, "engine": engine, "rho": rho, "scores": {}, "s0": {}}
    arrays = {"z": c["z"], "x": c["x"]}
    for det in detectors:
        src = det_int if det in ("cl", "gm") else det_sh
        if det not in src:
            continue
        s0 = null_s0(spec, "intensity" if det in ("cl", "gm") else engine, rho, seed, det, log=log, pid=pid)
        res["s0"][det] = s0
        for mode in ("blend", "switch"):
            if mode == "blend":
                h = hybrid.blend(T_int, G_sh, src[det], s0)
            else:
                h = hybrid.switch(d_int, d_sh, src[det], s0)
            dn = h["dir"].cpu().numpy()
            err = score.angle_err(dn, t["dir1"])
            key = f"{mode}_{det}"
            arrays[f"dir_{key}"] = dn; arrays[f"err_{key}"] = err; arrays[f"w_{key}"] = h["w"].cpu().numpy()
            res["scores"][key] = {"bands": score.band_summary(err, t["dist"], t["region"]),
                                  "parallel": score.parallel_share(dn, t["bnorm"], t["dist"], t["region"]),
                                  "switched_share_wm": float(h["w"].cpu().numpy()[t["region"] == 2].mean())}
    res["elapsed_s"] = time.time() - t0
    _save_result(rp, res, arrays)
    log(f"hybrid {pid} s{seed} {engine} rho {rho:g}: " + "  ".join(
        f"{k} <0.1 {v['bands'].get('0-0.1mm', {}).get('median', float('nan')):.1f} >2 {v['bands'].get('>2mm', {}).get('median', float('nan')):.1f} sw {v['switched_share_wm']:.0%}"
        for k, v in res["scores"].items()))
    return res


def _gate_to_block(gate_sites: np.ndarray, c: dict, grid, block_shape) -> torch.Tensor:
    """A per-site map ``(nz', nx')`` on the mid-y row -> a block ``(Z, Y, X)`` gate by nearest site (constant in y)."""
    stride = int(c["stride"])
    z0, x0 = int(c["z"][0] - (stride - 1) / 2), int(c["x"][0] - (stride - 1) / 2)
    Z, Y, X = block_shape
    iz = np.clip((np.arange(Z) - z0) // stride, 0, gate_sites.shape[0] - 1)
    ix = np.clip((np.arange(X) - x0) // stride, 0, gate_sites.shape[1] - 1)
    g2 = gate_sites[iz[:, None], ix[None, :]].astype(np.float32)
    return torch.as_tensor(np.broadcast_to(g2[:, None, :], (Z, Y, X)).copy())


def run_excised(spec: Spec, pid: str, seed: int, engine: str, rho: float, detectors=("cl", "tca_dom"), factor: float = 3.0,
                log=print, force: bool = False) -> dict:
    """H2: the intensity structure tensor with fix A's gradient excision applied only where a detector (from the
    shape engine's single read-out, or the intensity tensor's own ``cl``) exceeds its null-calibrated ``s0``
    (``excise_<det>``), plus the ungated excision everywhere (``excise_all``, fix A as it was)."""
    from ..tensor import excision_weights
    rp = result_path(spec, "hybrid", pid, seed, f"excised-{engine}", rho)
    if rp.exists() and not force:
        return json.loads(rp.read_text())
    t0 = time.time()
    img, tr, meta, grid = make.load(pid, seed)
    c_int = load_cache(spec, pid, seed, "intensity", rho)
    t = score.sample_truth(tr, c_int["z"], c_int["x"])
    dev = spec.device
    vol = torch.as_tensor(img, device=dev).float()
    G = gradients(vol[None], spec.sigma, spec.truncate)
    b, s = int(c_int["reduce"]), int(c_int["stride"]) // int(c_int["reduce"])
    gates = {"all": None}
    for det in detectors:
        src_engine = "intensity" if det in ("cl", "gm") else engine
        rs = result_path(spec, "single", pid, seed, src_engine, rho).with_suffix(".npz")
        if not rs.exists():
            run_single(spec, pid, seed, src_engine, rho, log)
        with np.load(rs) as z:
            if f"det_{det}" not in z.files:
                continue
            sd = z[f"det_{det}"].reshape(len(z["z"]), len(z["x"]))
        s0 = null_s0(spec, src_engine, rho, seed, det, log=log, pid=pid)
        gate = 1.0 / (1.0 + np.exp(-(sd - s0) / max(s0 / 4.0, 1e-6)))
        gates[det] = (_gate_to_block(gate, c_int, grid, img.shape).to(dev), s0)
    res = {"pid": pid, "seed": seed, "engine": engine, "rho": rho, "scores": {}, "s0": {}}
    arrays = {"z": c_int["z"], "x": c_int["x"]}
    for name, g in gates.items():
        gate = None if g is None else g[0]
        w = excision_weights(vol, sigma_coarse=3.0, factor=factor, truncate=spec.truncate, gate=gate)
        K, sites = k_sites(G, rho, spec.truncate, b, [grid.y_mid], grid.core_zx, weights=w.half())
        Kt = common.unpack(K[0, ::s, ::s].reshape(-1, 6), 3)
        d = single.eigen(Kt)["dir"].cpu().numpy()
        err = score.angle_err(d, t["dir1"])
        key = f"excise_{name}"
        arrays[f"dir_{key}"], arrays[f"err_{key}"] = d, err
        res["scores"][key] = {"bands": score.band_summary(err, t["dist"], t["region"]),
                              "parallel": score.parallel_share(d, t["bnorm"], t["dist"], t["region"]),
                              "excised_share": float(1.0 - w.mean())}
        if g is not None:
            res["s0"][name] = g[1]
        del w, K
    del G, vol
    torch.cuda.empty_cache()
    res["elapsed_s"] = time.time() - t0
    _save_result(rp, res, arrays)
    log(f"excised {pid} s{seed} gate:{engine} rho {rho:g}: " + "  ".join(
        f"{k} <0.1 {v['bands'].get('0-0.1mm', {}).get('median', float('nan')):.1f} >2 {v['bands'].get('>2mm', {}).get('median', float('nan')):.1f} ex {v['excised_share']:.0%}"
        for k, v in res["scores"].items()))
    return res


def run_all(spec: Spec, log=print, hybrid_engines=("anatomix48", "anatomix16", "jet")):
    """Caches, singles, multis and hybrids for the whole spec."""
    for pid in spec.pids:
        for seed in spec.seeds:
            build_caches(spec, pid, seed, log)
            for engine in spec.engines:
                for rho in spec.rhos:
                    run_single(spec, pid, seed, engine, rho, log)
                    if _is_multi(pid):
                        run_multi(spec, pid, seed, engine, rho, log)
    for pid in spec.pids:
        if make.split_pid(pid)[0] in NULL_PIDS or _is_multi(pid):
            continue
        for seed in spec.seeds:
            for engine in hybrid_engines:
                if engine not in spec.engines:
                    continue
                for rho in spec.rhos:
                    run_hybrid(spec, pid, seed, engine, rho, log=log)
                    if engine == hybrid_engines[0]:
                        run_excised(spec, pid, seed, engine, rho, log=log)
