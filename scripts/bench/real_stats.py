import json, time, sys
from shapetensor.phantoms import stats
t0 = time.time()
out = stats.real_crop_profile("/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr", level=0, threshold=142.5)
out["elapsed_s"] = time.time() - t0
json.dump(out, open("/home/andrew/bridge_wmcov/shapetensor/results/real_stats_S2146.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("shape", "bright_fraction", "elapsed_s")}))
for part in ("all", "bright", "dark"):
    p = out[part]; print(part, "noise", round(p["noise_sd"], 2), "modes", p["modes"][:4], "dog", {k: round(v["rms_texture"], 2) for k, v in p["dog_rms"].items()})
