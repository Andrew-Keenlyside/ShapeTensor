"""The parameter page's figures: every parameter of the pipeline varied one at a time on the HiP-CT block, the rest at
their defaults, with the excised structure tensor at the same scales where the parameter is a scale.

    python scripts/parameters.py             -> docs/img/params_<name>.png, results/parameters.json

Each run reports the populations found, the null, and the median angle of its single field to the default run's
(and to the excised structure tensor at the same σ / ρ) over the inner tissue.
"""
import json, sys, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import BRIGHT, RHO, SIGMA, _hipct_blocks, dev                                   # noqa: E402
from shapetensor import st                                                                     # noqa: E402
from shapetensor.pipeline import apply, calibrate                                             # noqa: E402
from shapetensor.validate import angle_deg                                                     # noqa: E402
from shapetensor.viz import dec_rgb                                                            # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
RES = Path(__file__).resolve().parents[1] / "results"
CAL = {"sigma": [0.5, 1.0, 2.0], "rho": [5.0, 10.0, 20.0], "smooth": [1, 3, 5]}                 # recalibrated: the scales
APP = {"nmax": [1, 2, 3], "min_sep_deg": [15.0, 25.0, 40.0], "window": [0, 64, 96], "tensor_floor": [0.0, 0.01, 0.1], "iterations": [1, 2, 4]}
DEFAULT = {"sigma": SIGMA, "rho": RHO, "smooth": 3, "nmax": 3, "min_sep_deg": 25.0, "window": 64, "tensor_floor": 0.01, "iterations": 2, "tissue": "threshold"}
M = 24


def main():
    OUT.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
    vc, v = _hipct_blocks(); tc, t = vc >= BRIGHT, v >= BRIGHT
    sel = torch.zeros_like(t); sel[M:-M, M:-M, M:-M] = t[M:-M, M:-M, M:-M]
    sl = (slice(M, 192 - M), 96, slice(M, 192 - M))
    cal0 = calibrate(vc, tc, sigma=SIGMA, rho=RHO); Fc = cal0.features(vc); F = cal0.features(v)
    cals = {}                                                                                  # (sigma, rho, smooth, tissue) -> calibration

    def calibration(sigma=SIGMA, rho=RHO, smooth=3, tissue="threshold"):
        key = (sigma, rho, smooth, tissue)
        if key not in cals:
            cals[key] = cal0 if key == (SIGMA, RHO, 3, "threshold") else calibrate(vc, tc if tissue == "threshold" else None, sigma=sigma, rho=rho, smooth=smooth, features=Fc)
        return cals[key]

    runs, rec = {}, []

    def run(name, value):
        p = dict(DEFAULT); p[name] = value
        cal = calibration(p["sigma"], p["rho"], p["smooth"], p["tissue"])
        t0 = time.perf_counter()
        out = apply(cal, v, t if p["tissue"] == "threshold" else None, nmax=p["nmax"], min_sep_deg=p["min_sep_deg"], window=p["window"],
                    tensor_floor=p["tensor_floor"], iterations=p["iterations"], features=F)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        d_st = st.structure_tensor(v, sigma=p["sigma"], rho=p["rho"], mask=t, order="zyx")
        N = out["modes"].shape[0]
        r = {"parameter": name, "value": value, "n_populations": N, "tau": round(out["tau"], 3), "seconds": round(dt, 1),
             "to_structure_tensor_deg": round(float(angle_deg(d_st, out["vec"])[sel].median()), 1), "tissue_above_null": out["confidence"]["tissue_above_null"]}
        keep = {"vec": out["vec"][(slice(None),) + sl].cpu().numpy(), "weight": out["weight"][sl].cpu().numpy(), "st": d_st[(slice(None),) + sl].cpu().numpy(), "N": N,
                "vec2": out["vecs"][1][(slice(None),) + sl].cpu().numpy() if N > 1 else None, "w2": out["weights"][1][sl].cpu().numpy() if N > 1 else None, "full": out["vec"]}
        runs[(name, value)] = (r, keep); print(json.dumps(r), flush=True)
        return r, keep

    r0, k0 = run("default", None); wref = float(np.percentile(k0["weight"], 99))
    for name, values in list(CAL.items()) + list(APP.items()) + [("tissue", ["threshold", "none"])]:
        for val in values:
            if val == DEFAULT[name]:
                runs[(name, val)] = runs[("default", None)]; continue
            run(name, val)
    for (name, val), (r, k) in runs.items():
        r = dict(r); r["parameter"], r["value"] = name, val
        r["to_default_deg"] = round(float(angle_deg(k0["full"], k["full"])[sel].median()), 1); rec.append(r)
    json.dump(rec, open(RES / "parameters.json", "w"), indent=1)
    # the figures: a strip per parameter; two rows where the structure tensor (scales) or the second vector (detection) says something
    for name, values in list(CAL.items()) + list(APP.items()) + [("tissue", ["threshold", "none"])]:
        two = name in ("sigma", "rho", "nmax", "min_sep_deg", "window")
        fig, axs = plt.subplots(2 if two else 1, len(values), figsize=(4.5 * len(values), 9.2 if two else 4.8), dpi=100, squeeze=False)
        for j, val in enumerate(values):
            r, k = runs[(name, val)]; d = next(x for x in rec if x["parameter"] == name and x["value"] == val)
            top = axs[0, j]; top.imshow(dec_rgb(k["vec"]), origin="lower", interpolation="nearest")
            lab = f"{name} = {val}" + ("  (default)" if val == DEFAULT[name] else "")
            top.set_title(f"{lab}\n{k['N']} population{'s' if k['N'] != 1 else ''}, {d['to_default_deg']}° from the default", fontsize=10)
            if two and name in ("sigma", "rho"):
                axs[1, j].imshow(dec_rgb(k["st"]), origin="lower", interpolation="nearest"); axs[1, j].set_title(f"excised structure tensor at the same σ, ρ ({d['to_structure_tensor_deg']}° apart)", fontsize=10)
            elif two:
                if k["vec2"] is not None:
                    axs[1, j].imshow(dec_rgb(k["vec2"]), origin="lower", interpolation="nearest"); axs[1, j].set_title("second vector (DEC)", fontsize=10)
                else:
                    axs[1, j].imshow(np.zeros(k["vec"].shape[1:] + (3,)), origin="lower"); axs[1, j].set_title("no second population", fontsize=10)
        for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{name}: the shape tensor's single field (DEC at every voxel)", x=0.01, ha="left", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / f"params_{name}.png"); plt.close(fig); print("wrote", f"params_{name}.png", flush=True)


if __name__ == "__main__":
    main()
