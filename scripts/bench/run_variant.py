"""Engine variants on P3-15 / P3-0 (seed 0): the checkerboard treatment and n_pcs, scored like the pilot."""
import json, time
from shapetensor.validate import suite, report

tex = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/tex_calibrated.json"))["texture"]
tex["widths"] = [tuple(w) for w in tex["widths"]]
t0 = time.time()
def log(m):
    print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
variants = {
    "var-nonyq": {"anatomix48": {"nyquist": "none"}, "anatomix16": {"nyquist": "none"}},
    "var-filter": {"anatomix48": {"nyquist": "filter"}, "anatomix16": {"nyquist": "filter"}},
    "var-pcs8": {"anatomix48": {"n_pcs": 8}},
}
for tag, ek in variants.items():
    engines = tuple(ek.keys())
    spec = suite.Spec(pids=("P3-15", "P3-0"), seeds=(0,), engines=("intensity",) + engines, rhos=(8.0, 16.0),
                      tex=tex, engine_kwargs=ek, tag=tag, odf=False)
    for pid in spec.pids:
        suite.build_caches(spec, pid, 0, log)
        for engine in spec.engines:
            for rho in spec.rhos:
                suite.run_single(spec, pid, 0, engine, rho, log)
    print("REPORT", report.write(tag), flush=True)
print("VARIANTS DONE", flush=True)
