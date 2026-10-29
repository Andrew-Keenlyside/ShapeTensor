"""For every population in a scene: its per-voxel tensors against the direction across its own material (the gradient of
its smoothed occupancy), at wall voxels: lines should lie across it at ~90 deg, sheets' normals along it. Synthetic only.

    PYTHONPATH=src python scripts/check_object_odfs.py [SEED]
"""
import sys, torch
import shapetensor.learn.train as TR
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
import shapetensor.learn.generator as G
G.ORIENT_FROM_TEXTURE = True; G.ORIENT_VIS = False; G.TARGETS_FROM_GEOMETRY = True
from shapetensor.learn import scene as SC, lobe_train as LT
from shapetensor.filters import gaussian
dev = "cuda"; torch.cuda.set_per_process_memory_fraction(0.3)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 210_000_000
gen = LT.difficulty_gen(0.6, dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA))
s, d = SC.draw_scene((64,) * 3, seed=seed, device=dev, strength=0.0, entity_ctx=True, **gen)
ctx = d["entity_ctx"]; vox, grp, obj, w, v = ctx["entries"]; gl = list(ctx["group_layer"])
cs = torch.as_tensor(ctx["canvas_shape"], device=dev); org = torch.as_tensor([int(o) for o in ctx["origin"]], device=dev)
vox = vox.long(); P = torch.stack([vox // (cs[1] * cs[2]), (vox // cs[2]) % cs[1], vox % cs[2]], -1) - org
ok = ((P >= 0) & (P < 64)).all(-1); P, grp, obj, w, v = P[ok], grp[ok].long(), obj[ok].long(), w[ok].float(), v[ok].float()
li = (P[:, 0] * 64 + P[:, 1]) * 64 + P[:, 2]
kinds = {}
for g_ in torch.unique(grp).tolist():
    L = s.layers[gl[g_]] if 0 <= g_ < len(gl) and gl[g_] < len(s.layers) else None
    kinds[g_] = (getattr(L, "name", None) or (L.role if L else "?"), L.kind if L else "?")
for g_, (nm, kd) in kinds.items():
    m = grp == g_
    W = torch.zeros(64 ** 3, device=dev).index_add_(0, li[m], w[m]).view(64, 64, 64)
    V = torch.zeros(64 ** 3, 6, device=dev).index_add_(0, li[m], v[m, :6]).view(64, 64, 64, 6)
    gz, gy, gx = torch.gradient(gaussian(W[None, None], 1.0)[0, 0]); gr = torch.stack([gz, gy, gx], -1); gn = gr.norm(dim=-1)
    sel = (W > 0.3) & (gn > 0.05)                                               # wall voxels with a clear across-direction
    if int(sel.sum()) < 20:
        continue
    t = V[sel] / W[sel][:, None]
    M = torch.stack([torch.stack([t[:, 0], t[:, 3], t[:, 4]], -1), torch.stack([t[:, 3], t[:, 1], t[:, 5]], -1), torch.stack([t[:, 4], t[:, 5], t[:, 2]], -1)], -2)
    ev, E = torch.linalg.eigh(M.double()); ev = ev.float(); E = E.float()
    across = gr[sel] / gn[sel][:, None]
    a_norm = torch.rad2deg(torch.acos((E[:, :, 0] * across).sum(-1).abs().clamp(max=1)))     # tensor normal vs across
    a_axis = torch.rad2deg(torch.acos((E[:, :, 2] * across).sum(-1).abs().clamp(max=1)))     # tensor main axis vs across
    cl = ((ev[:, 2] - ev[:, 1]) / ev[:, 2]).median(); cp = ((ev[:, 1] - ev[:, 0]) / ev[:, 2]).median()
    print(f"{nm:14s} ({kd:5s}) voxels {int(sel.sum()):6d}: line-ness {float(cl):.2f} planarity {float(cp):.2f} | "
          f"angle(normal, across) median {float(a_norm.median()):5.1f} | angle(main axis, across) median {float(a_axis.median()):5.1f}")
