"""Calibrate fibre dispersion (and re-check amplitude) against the real crop's label-free orientation-gap statistics:
the structure tensor's (l2 - l3) / l2 at rho 8 / 16 / 32 on the homogeneous phantom P8 vs the S-21-46 bright tissue."""
import itertools, json, numpy as np, torch
from shapetensor.phantoms import make, stats
from shapetensor.tensor import gradients, k_dense
from shapetensor.readout import common, single

real = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/real_cl_S2146.json"))
target = {r: real[f"rho{r:g}"] for r in (8.0, 16.0, 32.0)}
print("real gap q10/q50/q90:", {r: (round(v["gap_q10"], 3), round(v["gap_q50"], 3), round(v["gap_q90"], 3)) for r, v in target.items()},
      " cl q50:", {r: round(v["cl_q50"], 3) for r, v in target.items()})
base = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/tex_calibrated.json"))["texture"]
base["widths"] = [tuple(w) for w in base["widths"]]
g = make.Grid(vox=0.02, core=(256, 64, 256), margin=(96, 96, 96))


def gaps(tex):
    img, _, _ = make.build("P8", 0, tex, g)
    v = torch.as_tensor(img, device="cuda").float()
    G = gradients(v[None], 1.0, 3.0)
    out = {}
    for rho in (8.0, 16.0, 32.0):
        b = 8
        K, _ = k_dense(G, rho, 3.0, b, ((96, 352), (96, 160), (96, 352)))
        e = single.eigen(common.unpack(K.reshape(-1, 6), 3))
        gp, cl = e["gap"].cpu().numpy(), e["cl"].cpu().numpy()
        out[rho] = (float(np.quantile(gp, .1)), float(np.quantile(gp, .5)), float(np.quantile(gp, .9)), float(np.median(cl)))
    p = stats.profile(img[96:-96, 96:-96, 96:-96].astype(np.float32))
    tb = {k: v["rms_texture"] for k, v in p["dog_rms"].items()}
    return out, tb


rows = []
for disp, pitch, amp in itertools.product((0.0, 10.0, 20.0, 30.0), (0.0, 10.0), (1.0, 0.6)):
    tex = make.TextureParams(**{**base, "amp": base["amp"] * amp, "dispersion_deg": disp, "pitch_deg": pitch})
    out, tb = gaps(tex)
    cost = sum((np.log(out[r][1]) - np.log(target[r]["gap_q50"])) ** 2 for r in out)
    rows.append((cost, disp, pitch, amp, out, tb))
    print(f"disp {disp:4.0f} pitch {pitch:4.0f} amp x{amp:.1f}: gap q50 " + " ".join(f"rho{r:g} {out[r][1]:.3f}" for r in out)
          + f" | q10/q90 rho32 {out[32.0][0]:.3f}/{out[32.0][2]:.3f} | cl q50 rho32 {out[32.0][3]:.3f} | tex bands "
          + " ".join(f"{v:.1f}" for v in tb.values()) + f" | cost {cost:.3f}", flush=True)
rows.sort(key=lambda r: r[0])
c, disp, pitch, amp, out, tb = rows[0]
print("BEST", dict(dispersion_deg=disp, pitch_deg=pitch, amp=base["amp"] * amp), {r: round(out[r][1], 3) for r in out}, "cost", round(c, 3))
json.dump({"best": dict(dispersion_deg=disp, pitch_deg=pitch, amp=base["amp"] * amp), "gap_q50": {str(r): out[r][1] for r in out},
           "rows": [dict(cost=r[0], disp=r[1], pitch=r[2], amp=r[3], gaps={str(k): v for k, v in r[4].items()}, tex=r[5]) for r in rows]},
          open("/home/andrew/bridge_wmcov/shapetensor/results/calib_dispersion.json", "w"), indent=1)
