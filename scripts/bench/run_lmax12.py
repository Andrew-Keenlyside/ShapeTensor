"""Follow-up: angular resolution of the CSD N-population read-out at SH order 12 on the 45 and 30 degree crossings
(intensity and anatomix48 ODFs recomputed at lmax 12 from the cached features' gradients: the K caches have no
ODF at order 12, so this recomputes the gradients from the phantom image per engine)."""
import json, time
import numpy as np, torch
from shapetensor.validate import suite, score
from shapetensor.phantoms import make
from shapetensor.features import make_engine
from shapetensor.pca import PCA
from shapetensor.tensor import gradients, odf_sites
from shapetensor.odf import fit_ring_width, ring_profile, zonal_kernel
from shapetensor.readout import common, multi

LMAX = 12
tex = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/tex_calibrated.json"))["texture"]
tex["widths"] = [tuple(w) for w in tex["widths"]]
spec = suite.Spec(pids=(), tex=tex, tag="main", engine_kwargs={"anatomix48": {"nyquist": "none"}})
t0 = time.time()
out = {}


def odf_for(pid, seed, engine, rho):
    img, tr, meta, grid = suite.ensure_phantom(spec, pid, seed)
    vol = torch.as_tensor(img, device="cuda").float()
    eng = make_engine(engine, **spec.engine_kwargs.get(engine, {}))
    F = eng(vol)
    if F.shape[0] > spec.n_pcs:
        F = PCA.fit(F, spec.n_pcs, seed=seed)(F)
    G = gradients(F, spec.sigma, spec.truncate)
    del F
    b = max(1, int(rho // 4)); s = max(1, spec.stride // b)
    O, sites = odf_sites(G, rho, spec.truncate, b, [grid.y_mid], grid.core_zx, lmax=LMAX)
    del G
    torch.cuda.empty_cache()
    O = O[0, ::s * 2, ::s * 2]
    t = score.sample_truth(tr, sites.z[::s * 2], sites.x[::s * 2])
    return O.reshape(-1, O.shape[-1]), t


for engine in ("intensity", "anatomix48"):
    for rho in (16.0, 32.0):
        Os, ts = odf_for("P4-lines-single", 0, engine, rho)
        width = fit_ring_width(Os, torch.as_tensor(ts["dir1"], device="cuda"), LMAX)
        kern = zonal_kernel(LMAX, ring_profile(width))
        for pid in ("P4-lines-c90-f1", "P4-lines-c60-f1", "P4-lines-c45-f1", "P4-lines-c30-f1", "P4-lines-single"):
            O, t = odf_for(pid, 0, engine, rho)
            for rel in (0.3, 0.2):
                r = multi.n_by_peaks(O, LMAX, mode="csd", kernel=kern, rel=rel)
                sc = score.multi_scores(*(r[k].cpu().numpy() for k in ("fibres", "fracs", "count")), t["dir1"], t["dir2"], t["npop"], t["frac1"], t.get("cross_deg"), t["region"])
                key = f"{pid}|{engine}|rho{rho:g}|rel{rel}"
                out[key] = {"ring_width_deg": float(np.degrees(width)), **sc}
                msg = (f"N2 {sc['two']['found2_share']:.0%} both<15 {sc['two']['both_within_15_share']:.0%} e {sc['two']['err1'].get('median', float('nan')):.0f}/{sc['two']['err2'].get('median', float('nan')):.0f}"
                       if "two" in sc else f"e {sc['single'].get('median', float('nan')):.1f} false2 {sc['single']['false_second_share']:.0%}")
                print(f"[{time.time() - t0:5.0f}s] lmax{LMAX} {pid:18s} {engine:10s} rho {rho:2g} rel {rel}: {msg}", flush=True)
json.dump(out, open("/home/andrew/bridge_wmcov/shapetensor/results/lmax12_crossings.json", "w"), indent=1)
print("LMAX12 DONE", flush=True)
