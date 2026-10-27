"""What the readout's entities are at the points of held-out phantoms: per point inside a population (target mass >= half
its peak, alone there), the model's entities (mass, type, angle to the true axis) -- where the "other directions" mass and
the misses come from. Prints a summary per family: the share of points with 1, 2, 3+ entities, the mass on the matched
entity / on entities of the wrong type / at > 30 deg, and the angle distribution of the unmatched entities.

Usage: python scripts/learn/entity_diag.py CKPT [--families curve fan sheet cross] [--entities votes] [--axes votes] [--final]
"""
import argparse, sys
import numpy as np
import torch
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_train as ET, entity_infer as EI, phantom2 as P2, crossing_acceptance as CA
from shapetensor.learn.model import PopulationNet

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--families", nargs="+", default=["curve", "fan", "sheet", "cross", "blob"])
ap.add_argument("--entities", default="votes"); ap.add_argument("--axes", default="votes"); ap.add_argument("--final", action="store_true")
ap.add_argument("--n", type=int, default=2); ap.add_argument("--sigma", type=float, default=2.5); a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"])
FAMS = ("cross", "fan", "curve", "sheet", "blob", "mix", "empty", "fine"); sph = EI.sphere_dirs(200, dev)
S0 = 1_900_000 if a.final else 900_000
for fam in a.families:
    fi = FAMS.index(fam); rows = []
    for i in range(a.n):
        spec, look = P2.random_spec(fam, torch.Generator().manual_seed(S0 + 100 * fi + i))
        ph = P2.make(spec, seed=S0 + 50_000 + 100 * fi + i, look=look, device=dev, ss=2 if fam == "fine" else 1); n = ph["n"]
        g = torch.Generator(device=dev).manual_seed(i); pts = 6 + torch.rand(3000, 3, device=dev, generator=g) * (n - 13)
        sig = torch.full((pts.shape[0],), a.sigma, device=dev)
        t = P2.targets(ph, pts, sig); K = ph["occ"].shape[0]
        T = t["e_mass"][:, :K].cpu().numpy(); tk = t["e_kappa"][:, :K].cpu().numpy(); tR = t["e_R"][:, :K].cpu().numpy()
        ro = EI.readout(net, ph["image"], pts, sig, sph, floor=0.05, iso_k=ET.VOTE_KAPPA_MIN, merge_deg=15.0, entities=a.entities, axes=a.axes)
        m = ro["mass"].cpu().numpy(); k = ro["k"].cpu().numpy(); R = ro["R"].cpu().numpy(); iso = ro["iso"].cpu().numpy()
        tt = CA._types(tk[..., 0], tk[..., 1]); mt = CA._types(k[..., 0], k[..., 1])
        pk = np.array([np.percentile(T[T[:, j] > 0.02, j], 95) if (T[:, j] > 0.02).sum() > 10 else 1.0 for j in range(K)])
        for j in range(K):
            solo = (T[:, j] >= 0.5 * pk[j]) & np.all(np.delete(T, j, 1) < 0.05 * np.delete(pk, j), 1)
            ax_t = np.where((tt[:, j] == 1)[:, None], tR[:, j, :, 0], tR[:, j, :, 2])
            for p_ in np.nonzero(solo)[0]:
                live = m[p_] > 0.02 * max(T[p_, j], 1e-9)
                ax_m = np.where((mt[p_] == 1)[:, None], R[p_, :, :, 0], R[p_, :, :, 2])
                ang = np.degrees(np.arccos(np.clip(np.abs(ax_m @ ax_t[p_]), 0, 1)))
                same = (mt[p_] == tt[p_, j])
                good = live & same & (ang <= 30)
                rows.append({"n": int(live.sum()), "good": m[p_][good].sum() / T[p_, j], "wrongtype": m[p_][live & ~same].sum() / T[p_, j],
                             "far": m[p_][live & same & (ang > 30)].sum() / T[p_, j], "iso": iso[p_] / T[p_, j], "ttype": int(tt[p_, j]),
                             "angs_bad": list(ang[live & ~good]), "types_bad": list(mt[p_][live & ~good]), "kap_bad": [tuple(np.round(k[p_, c], 1)) for c in np.nonzero(live & ~good)[0]],
                             "kap_good": [tuple(np.round(k[p_, c], 1)) for c in np.nonzero(good)[0]], "tk": tuple(np.round(tk[p_, j], 1))})
    if not rows:
        print(fam, "no solo points"); continue
    nn = np.array([r["n"] for r in rows]); f = lambda key: np.median([r[key] for r in rows])
    angs = np.concatenate([r["angs_bad"] for r in rows]) if any(r["angs_bad"] for r in rows) else np.array([])
    tb = np.concatenate([r["types_bad"] for r in rows]) if angs.size else np.array([])
    print(f"{fam:6s} [{a.entities}/{a.axes}] {len(rows)} solo points; entities 1: {np.mean(nn == 1):.2f} 2: {np.mean(nn == 2):.2f} 3+: {np.mean(nn >= 3):.2f} 0: {np.mean(nn == 0):.2f} | "
          f"mass on good {f('good'):.2f} wrong type {f('wrongtype'):.2f} far {f('far'):.2f} iso {f('iso'):.2f} (medians, / target)")
    if angs.size:
        print(f"        bad entities: angle quartiles {np.percentile(angs, [25, 50, 75]).round(0)}, types peaked/girdle/iso {[(tb == x).mean().round(2) for x in (0, 1, 2)]}")
    ex = [r for r in rows if r["angs_bad"]][:3]
    for r in ex:
        print(f"        e.g. target kappa {r['tk']}: good {r['kap_good']} bad {list(zip(np.round(r['angs_bad'], 0), r['kap_bad']))}")
