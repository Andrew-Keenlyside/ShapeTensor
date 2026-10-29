"""Objects below the population (29 Oct): an unresolved bundle's strands (and its smooth partial volume) are leaves under
the bundle; a texture field's strands are grouped into compact cluster objects under the field. Merged entities are the
exact sum of their objects. CPU, synthetic only."""
import torch

from shapetensor.learn import scene as SC
from shapetensor.learn.entities import build_tree, entity_targets

N = 48


def _setup(sc, groups):
    sc.groups = groups; out = sc.render(); ent = out["entries"]
    return out, ent, build_tree(sc, ent[1], ent[2])


def _msum(t):
    live = t["node"] >= 0
    return (t["mass"][..., None, None] * t["T"] * live[..., None, None]).sum(1)


def _exact(ent, tree, pts, sig):
    kw = dict(alone_is_object=False)
    a = entity_targets(ent, tree, (N,) * 3, (0, 0, 0), 1, (N,) * 3, pts, sig, **kw)
    b = entity_targets(ent, tree, (N,) * 3, (0, 0, 0), 1, (N,) * 3, pts, sig, r_min=0.0, c_sig=0.0, emax=2048, m_min=0.0, share_keep=2.0, **kw)
    A, B = _msum(a), _msum(b)
    return a, float(((A - B).abs().amax((1, 2)) / B.abs().amax((1, 2)).clamp_min(1e-12)).max())


def _points(occ, n, thr):
    cand = torch.nonzero(occ[6:-6, 6:-6, 6:-6] > thr).float() + 6
    return cand[torch.randint(0, cand.shape[0], (n,), generator=torch.Generator().manual_seed(0))]


def test_unresolved_bundle_strands_are_leaves_under_the_bundle():
    for seed in range(1, 100):
        sc = SC.Scene((N,) * 3, seed, "cpu", voxel_um=20.0)
        g = sc.bundle(fibre={"texture": 0.5}); rec = sc.record["families"][-1]["bundle"]
        if rec.get("unresolved") and rec.get("strands", 0) > 5:
            break
    gs = g if isinstance(g, list) else [g]
    out, ent, tree = _setup(sc, gs)
    bid = int(gs[0].curves[0].seg[0]); objs = torch.unique(ent[2]).tolist()
    assert bid not in objs and all(sc.parents.get(o) == bid for o in objs)                    # strands and the tube, all children
    assert len(objs) == rec["strands"] + 1
    leaf = tree.leaves(torch.zeros(len(objs), dtype=torch.long), torch.tensor(objs))
    par = tree.parent[leaf]
    assert bool((par == par[0]).all()) and int(tree.parent[par[0]]) > 0 and int(tree.parent[tree.parent[par[0]]]) == 0   # leaf -> bundle -> population -> root
    pts = _points(out["groups"][0]["occupancy"], 120, 0.05)
    fine = entity_targets(ent, tree, (N,) * 3, (0, 0, 0), 1, (N,) * 3, pts, torch.full((120,), 1.0), alone_is_object=False)
    nodes = fine["node"][fine["node"] >= 0]
    strand = torch.isin(nodes, leaf)
    assert bool(torch.where(strand, tree.radius[nodes] >= 1.5, nodes == par[0]).all())        # the resolution rule
    coarse, err = _exact(ent, tree, pts, torch.full((120,), 8.0))
    assert err < 1e-5 and bool(((coarse["node"] >= 0).sum(1) == 1).all())


def test_texture_field_objects_are_compact_clusters_summing_to_the_field():
    for seed in range(1, 40):
        sc = SC.Scene((N,) * 3, seed, "cpu", voxel_um=5.0); gs = sc.texture()
        if gs and sc.record["families"][-1]["texture"]["mode"] == "single":
            break
    out, ent, tree = _setup(sc, gs)
    objs = torch.unique(ent[2]).tolist(); pop = sc.record["families"][-1]["texture"]["pops"][0]
    lo, hi = SC.TEXTURE_CLUSTERS
    assert min(lo, pop["strands"]) <= len(objs) <= hi and len(objs) == pop["clusters"]
    assert len({sc.parents.get(o) for o in objs}) == 1                                        # one field above them
    pts = _points(out["groups"][0]["occupancy"], 120, 0.02)
    _, err = _exact(ent, tree, pts, torch.full((120,), 6.0))
    assert err < 1e-5
