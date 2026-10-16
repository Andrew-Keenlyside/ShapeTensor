"""Calibrate the phantom noise (correlated) and fibre texture against the real 20 um crop's band profile."""
import json, itertools, numpy as np
from shapetensor.phantoms import make, stats

real = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/real_stats_S2146.json"))["bright"]
tn = {k: v["noise"] for k, v in real["dog_rms"].items()}
tt = {k: v["rms_texture"] for k, v in real["dog_rms"].items()}
print("target noise bands", {k: round(v, 2) for k, v in tn.items()}, "resid noise_sd", round(real["noise_sd"], 2))
print("target texture bands", {k: round(v, 2) for k, v in tt.items()})
g = make.Grid(vox=0.02, core=(192, 64, 192), margin=(32, 32, 32))
LV = {"wm": 164.0, "gm": 126.0, "csf": 60.0}


def prof(pid, tex):
    img, _, _ = make.build(pid, 0, tex, g)
    return stats.profile(img[32:-32, 32:-32, 32:-32].astype(np.float32))


def lcost(d, t):
    return sum((np.log(max(d[k], 0.05)) - np.log(max(t[k], 0.05))) ** 2 for k in t)


# A. noise: correlation length and amplitude on the noise-only phantom
best_n = None
for sn in (0.0, 0.5, 0.7, 0.9, 1.1, 1.4):
    p = prof("P9", make.TextureParams(amp=0, noise_sd=10.0, noise_sigma=sn, levels=LV))
    nb = {k: v["noise"] for k, v in p["dog_rms"].items()}
    # best amplitude scale in log space
    k = np.exp(np.mean([np.log(tn[b]) - np.log(max(nb[b], 1e-3)) for b in tn]))
    c = lcost({b: nb[b] * k for b in nb}, tn)
    print(f"noise sigma {sn:.1f}: sd_white {10 * k:5.1f}  bands x k " + " ".join(f"{b}:{nb[b] * k:.2f}" for b in nb) + f"  cost {c:.3f}", flush=True)
    if best_n is None or c < best_n[0]:
        best_n = (c, sn, 10 * k)
_, SN, SD = best_n
print("NOISE:", "sigma", SN, "sd_white", round(SD, 2))

# B. texture on the homogeneous phantom with the chosen noise
rows = []
W3 = ((1.0, 0.5), (2.0, 0.35), (3.0, 0.15))
W4 = ((1.0, 0.45), (2.0, 0.3), (3.0, 0.15), (6.0, 0.1))
W5 = ((0.7, 0.4), (1.0, 0.3), (2.0, 0.2), (4.0, 0.1))
for widths, density, amp in itertools.product((W3, W4, W5), (120.0, 240.0, 480.0), (30.0, 45.0, 65.0, 90.0)):
    tex = make.TextureParams(amp=amp, noise_sd=SD, noise_sigma=SN, density=density, widths=widths, levels=LV)
    p = prof("P8", tex)
    d = {k: v["rms_texture"] for k, v in p["dog_rms"].items()}
    c = lcost(d, tt)
    rows.append((c, amp, density, widths, d, p["noise_sd"]))
    print(f"amp {amp:3.0f} dens {density:3.0f} w{len(widths)}({widths[0][0]}): tex " + " ".join(f"{k}:{v:.2f}" for k, v in d.items()) + f"  noise_sd {p['noise_sd']:.1f} cost {c:.3f}", flush=True)
rows.sort(key=lambda r: r[0])
c, amp, density, widths, d, _ = rows[0]
tex = {"amp": amp, "noise_sd": round(SD, 2), "noise_sigma": SN, "density": density, "widths": [list(w) for w in widths], "levels": LV}
print("BEST", tex, {k: round(v, 2) for k, v in d.items()}, "cost", round(c, 3))
json.dump({"texture": tex, "phantom_texture_bands": d, "target_texture_bands": tt, "target_noise_bands": tn, "cost": c},
          open("/home/andrew/bridge_wmcov/shapetensor/results/tex_calibrated.json", "w"), indent=1)
