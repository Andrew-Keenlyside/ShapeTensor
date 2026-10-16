"""Pilot: P3-15 / P3-0 / P8 / P9, seed 0, four engines, four rhos, with the calibrated texture."""
import json, sys, time
from pathlib import Path
from shapetensor.validate import suite, report

tex = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/tex_calibrated.json"))["texture"]
tex["widths"] = [tuple(w) for w in tex["widths"]]
d = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/spec_pilot.json"))
for k in ("pids", "seeds", "engines", "rhos"):
    d[k] = tuple(d[k])
spec = suite.Spec(tex=tex, **d)
t0 = time.time()
def log(m):
    print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
suite.run_all(spec, log=log)
print("REPORT", report.write(spec.tag), flush=True)
