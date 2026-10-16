"""Label-free prevalence of the sheet-dominated (failure) regime in the real 20 um crop: Westin cl and the orientation
gap of the intensity structure tensor at rho 8 / 16 / 32, over bright (WM-like) tissue, against the phantom nulls."""
import json, numpy as np, torch, zarr
from shapetensor.tensor import gradients, k_dense
from shapetensor.readout import common, single
from shapetensor.filters import gaussian
z0 = zarr.open("/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr", mode="r")["0"]
vol = torch.as_tensor(np.asarray(z0[:]).astype(np.float32), device="cuda")
bright = gaussian(vol, 2.0) >= 142.5
G = gradients(vol[None], 1.0, 3.0)
out = {}
for rho in (8.0, 16.0, 32.0):
    b = 8
    m = int(3 * rho + 0.5); m = int(np.ceil(m / b) * b)
    K, sites = k_dense(G, rho, 3.0, b, ((m, 768 - m), (m, 768 - m), (m, 768 - m)))
    e = single.eigen(common.unpack(K.reshape(-1, 6), 3))
    br = bright[m:768 - m:b, m:768 - m:b, m:768 - m:b].reshape(-1)
    # take the bright mask at the site centres (offset b//2)
    br = bright[m + b // 2:768 - m:b, m + b // 2:768 - m:b, m + b // 2:768 - m:b].reshape(-1)[:e["cl"].numel()]
    cl, gap = e["cl"].cpu().numpy(), e["gap"].cpu().numpy()
    sel = br.cpu().numpy()[:len(cl)]
    q = lambda a, p: float(np.quantile(a[sel], p))
    out[f"rho{rho:g}"] = {"n_bright_sites": int(sel.sum()), "cl_q50": q(cl, .5), "cl_q90": q(cl, .9), "cl_q99": q(cl, .99),
                          "gap_q10": q(gap, .1), "gap_q50": q(gap, .5), "gap_q90": q(gap, .9),
                          "share_cl_gt_0.3": float((cl[sel] > 0.3).mean()), "share_cl_gt_0.5": float((cl[sel] > 0.5).mean()),
                          "share_gap_lt_0.2": float((gap[sel] < 0.2).mean())}
    print(rho, json.dumps(out[f"rho{rho:g}"]), flush=True)
json.dump(out, open("/home/andrew/bridge_wmcov/shapetensor/results/real_cl_S2146.json", "w"), indent=1)
# the phantom nulls for comparison: cl / gap quantiles from the pilot2 singles (intensity engine)
from shapetensor.validate.report import load
for r in load("pilot2", "single") + load("pilot", "single"):
    if r["engine"] == "intensity" and r["pid"] in ("P3-0", "P3-0-t0.25", "P3-0-t0.1", "P3-40-t0.25", "P3-40"):
        d = r["detectors"]
        print(r["pid"], r["rho"], "cl q50/q99", round(d["cl"]["quantiles_wm"]["50"], 3), round(d["cl"]["quantiles_wm"]["99"], 3),
              "gap q50", round(d["gap"]["quantiles_wm"]["50"], 3))
