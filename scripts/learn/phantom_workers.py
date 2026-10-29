"""A CPU process making phantom-ladder training batches as files: steps ``rank, rank + n, ...`` of ``[start, stop)``, each
``DIR/{step:07d}.pt`` (images, targets; written under a temporary name, then renamed), at most ``--ahead`` of its own
files waiting at once (the trainer deletes each file it loads). Batches are seeded by the step, as the trainer's own.

Usage (by phantom_ladder.py --cpu-workers): python scripts/learn/phantom_workers.py DIR --rank R --n N --families ... --batch B
       --n-points P --sigma-lo S --start A --stop Z [--ahead 6]
"""
import argparse, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from shapetensor.learn import phantom2 as P2

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--rank", type=int); ap.add_argument("--n", type=int)
ap.add_argument("--families", nargs="+"); ap.add_argument("--batch", type=int); ap.add_argument("--n-points", type=int)
ap.add_argument("--sigma-lo", type=float); ap.add_argument("--start", type=int); ap.add_argument("--stop", type=int)
ap.add_argument("--ahead", type=int, default=6); ap.add_argument("--threads", type=int, default=4); a = ap.parse_args()
torch.set_num_threads(a.threads)
mine = lambda: sum(1 for f in os.listdir(a.dir) if f.endswith(".pt") and int(f[:7]) % a.n == a.rank)
parent = os.getppid()
for step in range(a.start + (a.rank - a.start) % a.n, a.stop, a.n):
    while mine() >= a.ahead:
        if not os.path.isdir(a.dir) or os.getppid() != parent:                                   # the trainer gone (killed: no atexit)
            sys.exit(0)
        time.sleep(0.2)
    imgs, ts = [], []
    for b in range(a.batch):
        fam = a.families[int(torch.randint(0, len(a.families), (1,), generator=torch.Generator().manual_seed(7919 * step + b)))]
        spec, look = P2.random_spec(fam, torch.Generator().manual_seed(step * 31 + b))
        ph = P2.make(spec, seed=step * 31 + b, look=look, device="cpu", ss=2 if fam == "fine" else 1)
        g = torch.Generator().manual_seed(104729 * step + b)
        imgs.append(ph["image"]); ts.append(P2.targets(ph, *P2.draw_queries(ph, g, a.n_points, sigma=(a.sigma_lo, 6.0))))
    tmp = os.path.join(a.dir, f".{step:07d}.tmp")
    torch.save({"step": step, "imgs": imgs, "ts": ts}, tmp); os.replace(tmp, os.path.join(a.dir, f"{step:07d}.pt"))
