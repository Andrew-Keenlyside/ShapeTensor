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


for pid in ("P7-t0.5", "P7-t0.25", "P1-t0.5", "P3-40-t0.5", "P6-t0.5", "P4-lines-single-t0.5"):
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
            rel = rel_for(engine, rho, pid)
            r = multi.n_by_peaks(Of, spec.lmax, mode="csd", kernel=kern, rel=rel)
            cnt = (r["count"].cpu().numpy() if hasattr(r["count"], "cpu") else np.asarray(r["count"])).reshape(-1)
            # the intensity cl at the same sites (from the intensity gradients, the first engine), and its null threshold
            if engine == "intensity":
                K, _ = k_sites(G, rho, spec.truncate, b, [grid.y_mid], grid.core_zx)
                K = K[0, ::s * 2, ::s * 2].reshape(-1, K.shape[-1])
                cl_by_rho[rho] = single.eigen(common.unpack(K, 3))["cl"].cpu().numpy(); del K
            cl = cl_by_rho[rho]
            s0 = suite.null_s0(spec, "intensity", rho, 0, "cl", pid=pid, log=lambda *a: None)
            keep = cl <= s0
            res = {"rel": rel, "ring_width_deg": float(np.degrees(width)), "cl_s0": float(s0), "share_gated_out": float(1 - keep.mean())}
            for band, (lo, hi) in BANDS.items():
                m = (t["dist"] >= lo) & (t["dist"] < hi) & (t["region"] == 2) if "dist" in t else np.ones(len(cnt), bool)
                if m.sum() < 20: continue
                res[band] = {"n_ge2": float((cnt[m] >= 2).mean()), "n_ge2_after_cl_gate": float((cnt[m & keep] >= 2).mean()) if (m & keep).sum() else float("nan"), "sites": int(m.sum())}
            out[f"{pid}|{engine}|rho{rho:g}"] = res
            msg = " ".join(f"{band} {v['n_ge2']:.0%}->{v['n_ge2_after_cl_gate']:.0%}" for band, v in res.items() if isinstance(v, dict))
            print(f"[{time.time() - t0:4.0f}s] {pid:22s} {engine:10s} rho {rho:2g} rel {rel}: N>=2 by band (raw->cl-gated): {msg}; gated out {res['share_gated_out']:.0%}", flush=True)
            del O, Of; torch.cuda.empty_cache()
        del G; torch.cuda.empty_cache()
json.dump(out, open("/home/andrew/bridge_wmcov/shapetensor/results/sheet_null.json", "w"), indent=1)
print("SHEET NULL DONE", flush=True)
