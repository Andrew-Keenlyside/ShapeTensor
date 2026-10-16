"""Compact comparison of single read-out scores across result tags. Usage: compare_tags.py PID tag1 tag2 ..."""
import sys
from shapetensor.validate.report import load, _get

pid, tags = sys.argv[1], sys.argv[2:]
bands = ["nointerface", "0-0.1mm", "0.1-0.25mm", "0.25-0.5mm", "0.5-1mm", "1-2mm", ">2mm"]
print(f"{'tag':12s} {'engine':11s} {'readout':14s} {'rho':>4s} " + " ".join(f"{b:>10s}" for b in bands))
for tag in tags:
    for kind in ("single", "hybrid"):
        for r in sorted(load(tag, kind), key=lambda r: (r["engine"], r["rho"])):
            if r["pid"] != pid:
                continue
            for name, sc in r["scores"].items():
                vals = [_get(sc["bands"], b, "median") for b in bands]
                print(f"{tag:12s} {r['engine']:11s} {name:14s} {r['rho']:4g} " + " ".join(f"{v:10.1f}" for v in vals))
