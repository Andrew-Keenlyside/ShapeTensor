"""Readout variants on cached decoder outputs: the held-out phantoms' slices (as phantom_ladder.py scores them, sigma
2.5) queried once per checkpoint and cached; each variant of :func:`entity_infer.entities_from_query` then scored in
seconds -- the criteria per family and the mass budget.

Usage: python scripts/learn/readout_lab.py CKPT [--final] [--variants 'entities=lobes,kappa=votes' ...]
"""
import argparse, os, sys, json, hashlib
import numpy as np
import torch
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_train as ET, entity_infer as EI, phantom2 as P2, crossing_acceptance as CA
from shapetensor.learn.model import PopulationNet

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--final", action="store_true")
ap.add_argument("--variants", nargs="+", default=["entities=votes"]); ap.add_argument("--k", type=int, default=None)
ap.add_argument("--cache", default="/tmp/claude-1001/-home-andrew-scripts-ShapeTensors/547a4cdf-c102-4a25-a287-ec4f7517deb0/scratchpad/lab")
ap.add_argument("--families", nargs="+", default=None); ap.add_argument("--detail", nargs="*", default=None, help="print these criteria's values per phantom (e.g. C1 C6)"); ap.add_argument("--sigma", type=float, default=2.5)
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
FAMS = ("cross", "fan", "curve", "sheet", "blob", "mix", "empty", "fine")
S0 = 1_900_000 if a.final else 900_000; K_ = a.k or (4 if a.final else 2)
os.makedirs(a.cache, exist_ok=True)
key = hashlib.md5(f"{os.path.abspath(a.ckpt)}{os.path.getmtime(a.ckpt)}{S0}{K_}{a.sigma}".encode()).hexdigest()[:10]
cpath = f"{a.cache}/{key}.pt"


def slice_frame(ph):
    n = ph["n"]; spec = ph["spec"]
    if spec and "normal" in spec[0]:
        nrm = torch.nn.functional.normalize(torch.tensor(spec[0]["normal"], dtype=torch.float32), dim=0); e1, e2 = P2._frame(nrm)
        if spec[0]["kind"] == "sheet":
            R = torch.stack([e1, nrm, e2], 1); off = 0.0
        else:
            R = torch.stack([nrm, e1, e2], 1); off = spec[0].get("spacing", 0.0) / 4
    else:
        R = torch.eye(3); off = 0.0
    return R.to(dev), off


if os.path.exists(cpath):
    cache = torch.load(cpath, weights_only=False)
else:
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"]); sph = EI.sphere_dirs(200, dev)
    cache = {"dirs": sph.cpu(), "items": []}
    for fi, fam in enumerate(FAMS):
        for i in range(K_):
            spec, look = P2.random_spec(fam, torch.Generator().manual_seed(S0 + 100 * fi + i))
            ph = P2.make(spec, seed=S0 + 50_000 + 100 * fi + i, look=look, device=dev, ss=2 if fam == "fine" else 1)
            n, c = ph["n"], ph["n"] / 2.0; R, off = slice_frame(ph)
            gu, gv = torch.meshgrid(torch.arange(n, device=dev, dtype=torch.float32), torch.arange(n, device=dev, dtype=torch.float32), indexing="ij")
            pts = c + torch.stack([torch.full_like(gu, off), gu - c, gv - c], -1).view(-1, 3) @ R.T
            rotated = not torch.allclose(R, torch.eye(3, device=dev))
            disc = ((((gu - c) ** 2 + (gv - c) ** 2) <= 28 ** 2) if rotated else torch.ones_like(gu, dtype=torch.bool)).cpu().numpy()
            sig = torch.full((n * n,), a.sigma, device=dev); t = P2.targets(ph, pts, sig); K = ph["occ"].shape[0]
            tgt = {"mass": t["e_mass"][:, :K].cpu().numpy(), "k": t["e_kappa"][:, :K].cpu().numpy(), "R": t["e_R"][:, :K].cpu().numpy(),
                   "bg": t["e_mass"][:, K].cpu().numpy(), "kinds": ph["kinds"]}
            face = torch.minimum(pts, (n - 1) - pts).amin(1).view(n, n).cpu().numpy()
            with torch.no_grad():
                f = EI.features(net, ph["image"]); rs = [EI.query(net, f, pts[j:j + 1024], sig[j:j + 1024], sph, vol_shape=ph["image"].shape) for j in range(0, n * n, 1024)]
            r = {k: torch.cat([x[k] for x in rs]).cpu() for k in rs[0]}
            cache["items"].append({"fam": fam, "kinds": ph["kinds"], "n": n, "tgt": tgt, "face": face, "disc": disc, "r": r})
            print("cached", fam, i, flush=True)
    torch.save(cache, cpath)

dirs = cache["dirs"].to(dev)


def run(kw):
    acc = []
    for it in cache["items"]:
        if a.families and it["fam"] not in a.families:
            continue
        P = it["r"]["f"].shape[0]; parts = []
        for j in range(0, P, 1024):                                                              # in chunks (memory)
            parts.append(EI.entities_from_query({k: v[j:j + 1024].to(dev) for k, v in it["r"].items()}, dirs, iso_k=ET.VOTE_KAPPA_MIN, **kw))
        C = max(x["mass"].shape[1] for x in parts)
        pad = lambda x: torch.cat([x, torch.zeros((x.shape[0], C - x.shape[1]) + tuple(x.shape[2:]), device=x.device, dtype=x.dtype)], 1) if x.shape[1] < C else x
        mdl = {k: torch.cat([pad(x[k]) if k in ("mass", "k", "R") else x[k] for x in parts]).cpu().numpy() for k in parts[0]}
        orc = CA.evaluate_p2(it["n"], it["tgt"], {"mass": np.where(CA._types(it["tgt"]["k"][..., 0], it["tgt"]["k"][..., 1]) == 2, 0.0, it["tgt"]["mass"]),
                                                  "k": it["tgt"]["k"], "R": it["tgt"]["R"], "iso": (it["tgt"]["mass"] * (CA._types(it["tgt"]["k"][..., 0], it["tgt"]["k"][..., 1]) == 2)).sum(1),
                                                  "bg": it["tgt"]["bg"], "bgm": it["tgt"]["bg"]}, it["face"], it["disc"])
        if not orc["PASS"]:
            continue
        res = CA.evaluate_p2(it["n"], it["tgt"], mdl, it["face"], it["disc"])
        m = CA.match_p2(it["tgt"], mdl); valid = (it["face"].reshape(-1) >= CA.FACE_MARGIN) & it["disc"].reshape(-1)
        T = it["tgt"]["mass"][valid]; tot = max(float(T.sum()), 1e-9)
        bud = {"matched": float(m["m"][valid].sum()) / tot, "other": float(m["other"][valid].sum()) / tot, "iso": float(mdl["iso"][valid].sum()) / tot,
               "total": float((mdl["mass"].sum(1) + mdl["iso"])[valid].sum()) / tot, "target": tot}
        acc.append((it["fam"], res, bud))
    return acc


crit = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9")
for v in a.variants:
    kw = {}
    for kv in v.split(","):
        if "=" in kv:
            k_, x = kv.split("="); kw[k_] = None if x == "None" else (float(x) if x.replace(".", "").replace("-", "").isdigit() else x)
    acc = run(kw)
    tot = {c: sum(r[[k for k in r if k.startswith(c + " ")][0]][0] for _, r, _ in acc) for c in crit}
    print(f"== {v}: PASS {sum(r['PASS'] for _, r, _ in acc)}/{len(acc)} | " + " ".join(f"{c} {tot[c]}" for c in crit), flush=True)
    for fam in FAMS:
        rows = [(r, b) for f_, r, b in acc if f_ == fam]
        if not rows or fam == "empty":
            continue
        cc = " ".join(f"{c}{sum(r[[k for k in r if k.startswith(c + ' ')][0]][0] for r, _ in rows)}" for c in crit)
        bb = " ".join(f"{k} {np.median([b[k] for _, b in rows]):.2f}" for k in ("matched", "other", "iso", "total"))
        c1 = [r["C1 level"][1] for r, _ in rows]; c3 = [r["C3 count"][1]["agree"] for r, _ in rows]
        print(f"   {fam:6s} {cc} | {bb} | count agree {np.round(c3, 2)}", flush=True)
        for c in (a.detail or []):
            for r, _ in rows:
                k = [k for k in r if k.startswith(c + " ")][0]
                print(f"        {k}: " + " ".join(f"{kk} {np.round(vv, 2) if not isinstance(vv, str) else vv}" for kk, vv in r[k][1].items()), flush=True)
