"""Benchmarks of an entity-field checkpoint on fixed-seed synthetic crops (notes/plan_entity_field_model.md, section 7).

Crossing scenes with crops centred on the crossings (the generator records each crossing's angle) and single-bundle
hard negatives; queries near the crossing points and across the crop. Each query's predicted entities (decoder on a
hemisphere grid -> :func:`~shapetensor.learn.entity_infer.cluster`) are compared with the generator's entity targets:

* second-entity recall per angle bin: query windows holding two target entities (each >= ``share`` of the window's
  entity mass) whose axes differ by the bin's angle -- found when two predicted entities match the two target axes
  (best pairing, within ``match_deg``);
* pair axis error (median, matched pairs);
* false second entity: windows holding one target entity (>= 0.9 of the mass) where two or more are predicted;
* tree: Spearman correlation of predicted embedding distances with target tree distances (matched entities).

Usage: python scripts/learn/eval_entities.py CHECKPOINT [--n 160] [--size 64] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")


def axis_of(T: torch.Tensor) -> torch.Tensor:
    w, V = torch.linalg.eigh(T.double()); return V[..., -1].float()


def ang(a, b):
    return float(torch.rad2deg(torch.acos((a * b).sum(-1).abs().clamp(max=1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--n", type=int, default=160); ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--n-dirs", type=int, default=300); ap.add_argument("--share", type=float, default=0.15)
    ap.add_argument("--match-deg", type=float, default=15.0); ap.add_argument("--out", default=None); ap.add_argument("--device", default="cuda")
    ap.add_argument("--t-rel", type=float, default=None, help="sparse directions from the total ODF (None: every direction)")
    ap.add_argument("--lobes", action="store_true", help="entities from the density's lobes (entity_infer.cluster_lobes) instead of votes and embeddings")
    a = ap.parse_args()
    import shapetensor.learn.train as TR
    from shapetensor.learn import entity_infer as EI, lobe_train as LT
    from shapetensor.learn.model import PopulationNet, poincare_dist
    from shapetensor.learn.scene import draw_scene
    dev = a.device
    ck = torch.load(a.ckpt, map_location=dev); args = ck["args"]
    model = PopulationNet.from_args(args).to(dev).eval(); model.load_state_dict(ck["model"])
    TR.HEAD, TR.SH_EXACT, TR.DESC_KEY = "entities", 8, "descriptors5"
    if args.get("entity"):
        TR.ENTITY.update({k: (tuple(v) if isinstance(v, list) else v) for k, v in args["entity"].items()})
    lo_s, hi_s = TR.ENTITY["sigma"]
    dirs = EI.sphere_dirs(a.n_dirs, dev)
    bins = [(0, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 91)]
    rec = {b: {"n": 0, "found": 0, "err": []} for b in bins}
    single = {"n": 0, "false2": 0}
    rho_all = []
    gen = {**LT.difficulty_gen(1.0, {"canvas": 2 * a.size}), "crops_per_scene": 1, "cross_p": 0.85, "p_centre": 1.0,
           "deep_tree": bool(args.get("deep_tree", False) or (args.get("gen", {}).get("scene", {}) or {}).get("deep_tree", False))}
    t0 = time.time()
    for i in range(a.n):
        s, d = draw_scene((a.size,) * 3, seed=20_000_000 + i, device=dev, strength=1.0, entity_ctx=True, **gen)
        x = d["image"].float() / 255.0
        # queries: a ball around the crop centre (where the crossing was placed) and a few across the crop
        g = torch.Generator(device=dev).manual_seed(i)
        c = torch.full((3,), (a.size - 1) / 2, device=dev)
        pts = torch.cat([c + (torch.rand(48, 3, generator=g, device=dev) - 0.5) * a.size / 3,
                         6 + torch.rand(16, 3, generator=g, device=dev) * (a.size - 13)])
        sig = torch.exp(math.log(lo_s) + torch.rand(pts.shape[0], generator=g, device=dev) * math.log(min(hi_s, 4.0) / lo_s))
        t = TR.entity_queries(s, d, a.size, i, dev, points=pts, sigma=sig)
        vox = math.log2(d["voxel_um"]) if d.get("voxel_um") else None
        with torch.no_grad():
            r = EI.query(model, EI.features(model, x, vox), pts, sig, dirs, vol_shape=x.shape, t_rel=a.t_rel)
            e = EI.cluster_lobes(r, dirs) if a.lobes else EI.cluster(r, dirs)
        T6 = t["e_T6"].float(); live_t = t["e_node"] >= 0
        mt = (t["e_mass"].float() * t["e_vis"].float()) * live_t
        sh_t = mt / mt.sum(1, keepdim=True).clamp_min(1e-9)
        for q in range(pts.shape[0]):
            big = torch.nonzero(sh_t[q] >= a.share)[:, 0]
            pred = torch.nonzero(e["live"][q] & (e["share"][q] >= a.share))[:, 0]
            if big.numel() == 1 and float(sh_t[q, big[0]]) >= 0.9:
                single["n"] += 1; single["false2"] += int(pred.numel() >= 2)
            if big.numel() >= 2:
                M = lambda k: torch.stack([torch.stack([T6[q, k, 0], T6[q, k, 3], T6[q, k, 4]]), torch.stack([T6[q, k, 3], T6[q, k, 1], T6[q, k, 5]]),
                                           torch.stack([T6[q, k, 4], T6[q, k, 5], T6[q, k, 2]])])
                a0, a1 = axis_of(M(int(big[0]))), axis_of(M(int(big[1])))
                th = ang(a0, a1)
                b = next(bb for bb in bins if bb[0] <= th < bb[1])
                rec[b]["n"] += 1
                if pred.numel() >= 2:
                    pa = e["R_odf"][q, pred, :, 2]
                    best = None
                    for i0 in range(pred.numel()):
                        for i1 in range(pred.numel()):
                            if i0 == i1:
                                continue
                            e0, e1 = ang(pa[i0], a0), ang(pa[i1], a1)
                            if best is None or max(e0, e1) < max(best):
                                best = (e0, e1)
                    if best is not None and max(best) <= a.match_deg:
                        rec[b]["found"] += 1; rec[b]["err"] += list(best)
        # tree: predicted embedding distances vs target tree distances, over matched entities (one per target, by axis)
        zs, nodes = [], []
        for q in range(pts.shape[0]):
            for k in torch.nonzero(live_t[q] & (sh_t[q] >= a.share))[:, 0].tolist():
                M = T6[q, k]; Mt = torch.stack([torch.stack([M[0], M[3], M[4]]), torch.stack([M[3], M[1], M[5]]), torch.stack([M[4], M[5], M[2]])])
                ak = axis_of(Mt); pl = torch.nonzero(e["live"][q])[:, 0]
                if pl.numel() == 0:
                    continue
                j = int(pl[torch.tensor([-ang(e["R_odf"][q, p_, :, 2], ak) for p_ in pl.tolist()]).argmax()])
                zs.append(e["z"][q, j]); nodes.append(int(t["e_node"][q, k]))
        if len(nodes) >= 4:
            from shapetensor.learn.entity_train import tree_distance
            Z = torch.stack(zs); N = torch.tensor(nodes, device=dev)
            ii, jj = torch.triu_indices(len(nodes), len(nodes), 1, device=dev)
            td = tree_distance(t["e_anc"].long(), N[ii], N[jj]).cpu().numpy(); zd = poincare_dist(Z[ii], Z[jj]).cpu().numpy()
            if np.std(td) > 0 and np.std(zd) > 0:
                from scipy.stats import spearmanr
                rho_all.append(float(spearmanr(td, zd).correlation))
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{a.n} crops, {time.time() - t0:.0f} s", flush=True)
    out = {"ckpt": a.ckpt, "n_crops": a.n, "share": a.share, "match_deg": a.match_deg, "clustering": "lobes" if a.lobes else "votes",
           "second_entity": {f"{b[0]}-{b[1]}": {"windows": v["n"], "recall": (v["found"] / v["n"]) if v["n"] else None,
                                                "axis_err_med": float(np.median(v["err"])) if v["err"] else None} for b, v in rec.items()},
           "false_second": {"windows": single["n"], "rate": single["false2"] / single["n"] if single["n"] else None},
           "tree_spearman_median": float(np.median(rho_all)) if rho_all else None, "tree_crops": len(rho_all)}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
