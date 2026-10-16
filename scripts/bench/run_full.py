"""The full suite from a spec JSON, with the calibrated texture. Usage: run_full.py SPEC.json [engine_kwargs.json]"""
import json, sys, time
from shapetensor.validate import suite, report

tex = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/tex_calibrated.json"))["texture"]
tex["widths"] = [tuple(w) for w in tex["widths"]]
d = json.load(open(sys.argv[1]))
for k in ("pids", "seeds", "engines", "rhos"):
    d[k] = tuple(d[k])
ek = {**d.pop("engine_kwargs", {}), **(json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {})}
spec = suite.Spec(tex=tex, engine_kwargs=ek, **d)
t0 = time.time()
def log(m):
    print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
suite.run_all(spec, log=log)
print("REPORT", report.write(spec.tag), flush=True)
