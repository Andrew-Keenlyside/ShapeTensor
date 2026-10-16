"""False populations on sheets: the CSD N-population read-out (order 8, the suite's kernel and the null-calibrated
threshold) on phantoms that have ONE population and a sheet-like structure -- the laminar slab P7, the gyrus P1, the
step P3-40, the vessel P6 -- reported as the share of sites with N >= 2 by distance band, and the same after
removing the sites the null-calibrated cl gate flags (cl > s0). Seed 0, rho 16 / 32, intensity and anatomix48."""
import json, time
import numpy as np, torch
from shapetensor.validate import suite, score
from shapetensor.features import make_engine
from shapetensor.pca import PCA
from shapetensor.tensor import gradients, odf_sites, k_sites
from shapetensor.readout import common, single, multi

tex = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/tex_calibrated.json"))["texture"]
tex["widths"] = [tuple(w) for w in tex["widths"]]
spec = suite.Spec(pids=(), tex=tex, tag="main", engine_kwargs={"anatomix48": {"nyquist": "none"}})
calib = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/csd_calibrated.json"))
BANDS = {"0-0.1mm": (0, 0.1), "0.1-0.25mm": (0.1, 0.25), "0.25-0.5mm": (0.25, 0.5), ">2mm": (2.0, 1e9)}
t0 = time.time(); out = {}


def rel_for(engine, rho, pid):
    lvl = "t-t0.5" if "t0." in pid else "t1"                     # t0.25 phantoms use the t0.5 calibration (none at t0.25)
    c = calib.get(f"{engine}|rho{rho:g}|{lvl}") or calib.get(f"{engine}|rho{rho:g}|t1")
    return float(c["rel_star"])


for pid in ("P3-40-t0.5", "P1-t0.5", "P7-t0.25", "P6-t0.5"):
    img, tr, meta, grid = suite.ensure_phantom(spec, pid, 0)
    vol = torch.as_tensor(img, device="cuda").float()
    cl_by_rho = {}
    for engine in ("intensity", "anatomix48"):
        eng = make_engine(engine, **spec.engine_kwargs.get(engine, {}))
        F = eng(vol)
        if F.shape[0] > spec.n_pcs:
            F = PCA.fit(F, spec.n_pcs, seed=0)(F)
        G = gradients(F, spec.sigma, spec.truncate); del F
        for rho in (16.0, 32.0):
            b = max(1, int(rho // 4)); s = max(1, spec.stride // b)
            O, sites = odf_sites(G, rho, spec.truncate, b, [grid.y_mid], grid.core_zx, lmax=spec.lmax)
            O = O[0, ::s * 2, ::s * 2]; zs, xs = sites.z[::s * 2], sites.x[::s * 2]
            t = score.sample_truth(tr, zs, xs)
            Of = O.reshape(-1, O.shape[-1])
            kern, width = suite.csd_kernel(spec, engine, rho, 0, log=lambda *a: None)
            rel0 = rel_for(engine, rho, pid)
            # gates at the ODF's own sites: intensity cl and gm (null thresholds from the no-interface phantoms)
            if engine == "intensity":
                K, _ = k_sites(G, rho, spec.truncate, b, [grid.y_mid], grid.core_zx)
                K = K[0, ::s * 2, ::s * 2].reshape(-1, K.shape[-1])
                cl_by_rho[rho] = single.eigen(common.unpack(K, 3))["cl"].cpu().numpy(); del K
                from shapetensor.filters import gaussian
                gmv = sum(gaussian(vol, 3.0, order=o, truncate=spec.truncate) ** 2 for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))).sqrt()
                Kg, _ = k_sites(gmv[None].half(), rho, spec.truncate, b, [grid.y_mid], grid.core_zx)
                cl_by_rho[("gm", rho)] = Kg[0, ::s * 2, ::s * 2, 0].float().clamp_min(0).sqrt().cpu().numpy().reshape(-1); del Kg, gmv
            cl = cl_by_rho[rho]; gm = cl_by_rho[("gm", rho)]
            s0 = suite.null_s0(spec, "intensity", rho, 0, "cl", pid=pid, log=lambda *a: None)
            g0 = suite.null_s0(spec, "intensity", rho, 0, "gm", pid=pid, log=lambda *a: None)
            gates = {"none": np.ones(len(cl), bool), "cl": cl <= s0, "gm": gm <= g0, "cl+gm": (cl <= s0) & (gm <= g0)}
            res = {"rel0": rel0, "ring_width_deg": float(np.degrees(width)), "cl_s0": float(s0), "gm_s0": float(g0), "share_gated": {k: float(1 - v.mean()) for k, v in gates.items()}}
            for rel in (rel0, rel0 + 0.1, rel0 + 0.2):
                r = multi.n_by_peaks(Of, spec.lmax, mode="csd", kernel=kern, rel=rel)
                cnt = (r["count"].cpu().numpy() if hasattr(r["count"], "cpu") else np.asarray(r["count"])).reshape(-1)
                for gname, keep in gates.items():
                    row = {}
                    for band, (lo, hi) in BANDS.items():
                        m = (t["dist"] >= lo) & (t["dist"] < hi) & (t["region"] == 2) & keep
                        if m.sum() < 20: continue
                        row[band] = float((cnt[m] >= 2).mean())
                    res[f"rel{rel:.1f}|{gname}"] = row
                    print(f"[{time.time() - t0:4.0f}s] {pid:12s} {engine:10s} rho {rho:2g} rel {rel:.1f} gate {gname:5s}: " + " ".join(f"{bd} {v:.0%}" for bd, v in row.items()), flush=True)
            out[f"{pid}|{engine}|rho{rho:g}"] = res
            del O, Of; torch.cuda.empty_cache()
        del G; torch.cuda.empty_cache()
json.dump(out, open("/home/andrew/bridge_wmcov/shapetensor/results/sheet_null_gates.json", "w"), indent=1)
print("SHEET NULL GATES DONE", flush=True)
