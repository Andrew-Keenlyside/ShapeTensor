import json, numpy as np, torch
from shapetensor.phantoms import make, stats
real = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/real_stats_S2146.json"))["bright"]
target = {k: v["rms_texture"] for k, v in real["dog_rms"].items()}
print("target (real bright):", "noise", round(real["noise_sd"], 2), {k: round(v, 2) for k, v in target.items()})
g = make.Grid(vox=0.02, core=(192, 64, 192), margin=(32, 32, 32))
rows = []
for widths in (((1.0, 0.5), (2.0, 0.35), (3.0, 0.15)), ((1.0, 0.45), (2.0, 0.3), (3.0, 0.15), (6.0, 0.10))):
    for density in (60.0, 120.0):
        for amp in (16.0, 24.0, 32.0, 44.0):
            tex = make.TextureParams(amp=amp, noise_sd=10.7, density=density, widths=widths, levels={"wm": 164.0, "gm": 126.0, "csf": 60.0})
            img, tr, meta = make.build("P8", 0, tex, g)
            core = img[32:-32, 32:-32, 32:-32]
            p = stats.profile(core.astype(np.float32))
            d = {k: v["rms_texture"] for k, v in p["dog_rms"].items()}
            cost = sum((np.log(max(d[k], 0.1)) - np.log(target[k])) ** 2 for k in target)
            rows.append((cost, amp, density, widths, p["noise_sd"], d))
            print(f"amp {amp:4.0f} dens {density:4.0f} w{len(widths)}: noise {p['noise_sd']:.2f} dog " + " ".join(f"{k}:{v:.2f}" for k, v in d.items()) + f"  cost {cost:.3f}", flush=True)
rows.sort(key=lambda r: r[0])
print("BEST", rows[0][:4], rows[0][5])
