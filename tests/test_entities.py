"""The entity-field model's generator side: per-object tensors, the tree, resolution, and entity targets."""
import math

import numpy as np
import pytest
import torch

from shapetensor.learn.entities import KIND_POPULATION, KIND_ROOT, KIND_SHARED, _pulled, build_tree, entity_targets
from shapetensor.learn.scene import Scene, draw_scene

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _axis(T):
    w, V = torch.linalg.eigh(T.double()); return V[:, -1]


def _angle(a, b):
    return math.degrees(math.acos(min(1.0, abs(float(a.double() @ b.double())))))


@pytest.mark.parametrize("fam,vu", [("bundle", 2.0), ("cells", 2.0), ("foam", 5.0), ("tree", 20.0), ("myelin", 0.05)])
def test_entities_sum_to_their_populations(fam, vu):
    sc = Scene((48,) * 3, 3, DEV, vu); sc.build({k: (1.0 if k == fam else 0.0) for k in Scene.FAMILIES}, (2, 2), 0.0, 0.0)
    ren = sc.render(); vox, grp, obj, w, v = ren["entries"]; n = 48 ** 3
    for gi, r in enumerate(ren["groups"]):
        m = grp == gi
        W = torch.zeros(n, device=DEV).index_add_(0, vox[m], w[m]); TV = torch.zeros(n, 6, device=DEV).index_add_(0, vox[m], v[m, :6])
        live = W > 1e-3
        if not bool(live.any()):
            continue
        Tt = TV[live] / W[live, None]; Tt = Tt / Tt[:, :3].sum(1, keepdim=True)
        assert float((Tt - r["tensor"].reshape(6, -1).T[live]).abs().max()) < 1e-4


def _two_bundles(theta_deg, rf_um=2.0, Rb=10.0, seed=1, shape=96, vu=2.0):
    sc = Scene((shape,) * 3, seed, DEV, vu); c = torch.full((3,), (shape - 1) / 2, device=DEV)
    a1 = torch.tensor([1.0, 0.0, 0.0], device=DEV); t = math.radians(theta_deg); a2 = torch.tensor([math.cos(t), math.sin(t), 0.0], device=DEV)
    g1 = sc.bundle(None, "single", axis=a1, centre=c, radius=Rb, curv=1e-5, fibre={"rf_um": rf_um, "jitter": 0.05, "disp": 0.02})
    g2 = sc.bundle(None, "single", axis=a2, centre=c + torch.tensor([0.0, 0.0, 2.2 * Rb], device=DEV), radius=Rb, curv=1e-5,
                   fibre={"rf_um": rf_um, "jitter": 0.05, "disp": 0.02})
    sc.groups = [g1, g2]
    return sc, sc.render(), c, a1, a2


@pytest.mark.parametrize("theta", [10, 20, 45, 90])
def test_two_bundles_are_two_entities_at_any_angle(theta):
    """Two bundles stacked along z (so both pass the query's window) at an angle: two entities whose axes are the
    bundles' -- at 10 degrees too, which an 8th-order density cannot separate."""
    sc, ren, c, a1, a2 = _two_bundles(theta)
    vox, grp, obj, w, v = ren["entries"]; tree = build_tree(sc, grp, obj)
    p = (c + torch.tensor([0.0, 0.0, 11.0], device=DEV))[None]                                   # between the bundles: both in the window
    t = entity_targets(ren["entries"], tree, sc.shape, (0, 0, 0), 1, sc.shape, p, torch.tensor([6.0], device=DEV), c_sig=0.5)
    live = t["node"][0] >= 0
    assert int(live.sum()) == 2, t["node"][0]
    ax = [_axis(t["T"][0, e]) for e in range(2)]
    errs = sorted([min(_angle(ax[0], a1), _angle(ax[1], a1)), min(_angle(ax[0], a2), _angle(ax[1], a2))])
    assert errs[-1] < 2.0, errs
    leaves = set(tree.leaf_nodes.tolist())
    assert not ({int(n) for n in t["node"][0][live].tolist()} & leaves)                              # thin fibres merged: the bundles


def test_resolved_fibres_are_their_own_entities():
    sc, ren, c, a1, a2 = _two_bundles(90, rf_um=6.0, Rb=14.0)                                       # fibres 3 voxels in radius
    vox, grp, obj, w, v = ren["entries"]; tree = build_tree(sc, grp, obj)
    t = entity_targets(ren["entries"], tree, sc.shape, (0, 0, 0), 1, sc.shape, c[None], torch.tensor([1.0], device=DEV), c_sig=0.5)
    leaves = set(tree.leaf_nodes.tolist())
    nodes = t["node"][0][t["node"][0] >= 0]
    assert len(nodes) >= 1 and all(int(n) in leaves for n in nodes)                                 # single fibres
    t2 = entity_targets(ren["entries"], tree, sc.shape, (0, 0, 0), 1, sc.shape, c[None], torch.tensor([8.0], device=DEV), c_sig=0.5)
    n2 = t2["node"][0][t2["node"][0] >= 0]
    assert len(n2) >= 1 and not any(int(n) in leaves for n in n2)                                   # at a coarse scale, the bundles


def test_tree_every_leaf_reaches_the_root_through_its_population():
    sc = Scene((64,) * 3, 5, DEV, 2.0); sc.build({k: (1.0 if k in ("bundle", "cells", "tree") else 0.0) for k in Scene.FAMILIES}, (3, 3), 0.0, 0.0)
    ren = sc.render(); vox, grp, obj, w, v = ren["entries"]; tree = build_tree(sc, grp, obj)
    assert int(tree.kind[0]) == KIND_ROOT and int(tree.parent[0]) == -1
    anc = tree.ancestors(tree.leaf_nodes)
    assert bool((anc == 0).any(1).all())                                                            # all reach the root
    for row, lf in zip(anc.tolist(), tree.leaf_nodes.tolist()):
        kinds = [int(tree.kind[x]) for x in row if x >= 0]
        assert KIND_POPULATION in kinds or KIND_SHARED in kinds


def test_split_tract_halves_share_their_tract():
    fam = {k: (1.0 if k == "bundle" else 0.0) for k in Scene.FAMILIES}
    for sd in range(300):
        sc = Scene((64,) * 3, sd, DEV, 3.0).build(fam, (1, 1), 0.0, 0.0)
        if "split" in sc.record and len(sc.groups) == 2:
            break
    ren = sc.render(); vox, grp, obj, w, v = ren["entries"]; tree = build_tree(sc, grp, obj)
    pops = torch.nonzero(tree.kind == KIND_POPULATION)[:, 0]
    assert len(pops) == 2 and int(tree.parent[pops[0]]) == int(tree.parent[pops[1]]) != 0            # both halves under one shared node
    assert int(tree.kind[tree.parent[pops[0]]]) == KIND_SHARED


def test_mass_is_the_window_volume_fraction():
    fam = {k: (1.0 if k == "bundle" else 0.0) for k in Scene.FAMILIES}
    from shapetensor.filters import gaussian
    for seed in range(0, 200, 8):                                                                    # a crop with tissue in it
        s, d = draw_scene((48,) * 3, seed=seed, device=DEV, strength=0.0, canvas=96, families=fam, n_groups=(2, 2), underlay_p=0.0, voxel_um=2.0,
                          entity_ctx=True, p_warp=0.0, p_affine=0.0, p_cutout=0.0, p_sparse=0.0, p_boundary=0.0)
        occ = sum((d["occupancy"][k] for k, L in enumerate(s.layers) if hasattr(L, "desc5")), torch.zeros((48,) * 3, device=DEV))
        if float((occ[8:40, 8:40, 8:40] > 0.15).float().mean()) > 0.02:
            break
    ctx = d["entity_ctx"]
    occ = sum(d["occupancy"][k] for k, L in enumerate(s.layers) if hasattr(L, "desc5")); sm = gaussian(occ, 2.0)
    inner = torch.zeros_like(occ, dtype=torch.bool); inner[8:40, 8:40, 8:40] = True
    pts = torch.nonzero(inner & (occ > 0.15)).float()[::37][:20]
    if pts.shape[0] == 0:
        pytest.skip("no interior tissue point")
    t = entity_targets(ctx["entries"], ctx["tree"], ctx["canvas_shape"], ctx["origin"], ctx["zoom"], (48,) * 3, pts, torch.full((pts.shape[0],), 2.0, device=DEV))
    ref = sm[pts[:, 0].long(), pts[:, 1].long(), pts[:, 2].long()]
    assert float((t["mass"].sum(1) - ref).abs().median()) < 0.05


def test_pulled_points_identity_and_shear():
    Z = Y = X = 16; p = torch.tensor([[5.0, 6.0, 7.0], [8.5, 3.25, 10.0]], device=DEV)
    xp, J = _pulled(p, None, (Z, Y, X)); assert torch.allclose(xp, p) and torch.allclose(J, torch.eye(3, device=DEV).expand(2, 3, 3))
    from shapetensor.learn.generator import _coords, _warp_grid
    co = _coords((Z, Y, X), DEV); disp = torch.zeros(3, Z, Y, X, device=DEV); disp[2] = 0.1 * co[0]   # x shifted by 0.1 z
    xp, J = _pulled(p, _warp_grid(disp), (Z, Y, X))
    assert torch.allclose(xp[:, 2], p[:, 2] + 0.1 * p[:, 0], atol=1e-3)
    assert torch.allclose(J[:, 2, 0], torch.full((2,), 0.1, device=DEV), atol=1e-3)


def test_cube_augment_moves_entity_points_and_tensors():
    from shapetensor.learn.losses import augment
    g = torch.Generator().manual_seed(3)
    img = torch.zeros(1, 16, 16, 16); img[0, 2, 3, 4] = 1.0
    T6 = torch.tensor([[[1.0, 0, 0, 0, 0, 0]]])                                                         # a line along z
    tgt = {"e_points": torch.tensor([[2.0, 3.0, 4.0]]), "e_T6": T6, "e_mass": torch.ones(1, 1), "e_node": torch.zeros(1, 1, dtype=torch.long)}
    x, t = augment(img, tgt, g)
    q = torch.nonzero(x[0] > 0.5)[0].float()
    assert torch.allclose(t["e_points"][0], q)
    assert torch.isclose(t["e_T6"][0, 0, :3].sum(), torch.tensor(1.0)) and int(torch.argmax(t["e_T6"][0, 0, :3])) in (0, 1, 2)


def test_entity_queries_from_the_training_path():
    import shapetensor.learn.train as TR
    TR.HEAD = "entities"
    try:
        img, out = TR._make_pair(16, 48, DEV, 0.5, {"scene": {"canvas": 96, "crops_per_scene": 8}})
    finally:
        TR.HEAD = "slots"
    E = TR.ENTITY
    assert out["e_points"].shape == (E["n_points"], 3) and out["e_node"].shape == (E["n_points"], E["emax"])
    assert out["e_T6"].shape == (E["n_points"], E["emax"], 6) and out["e_anc"].shape == (E["u_max"], E["d_max"])
    live = out["e_node"] >= 0
    if bool(live.any()):
        assert torch.allclose(out["e_T6"].float()[live][:, :3].sum(-1), torch.ones(int(live.sum()), device=DEV), atol=1e-2)
        assert int(out["e_node"][live].max()) < int(out["e_n"])


def test_a_lone_thin_fibre_is_its_own_entity():
    """Sparse labelling: one thin fibre (radius below r_min) kept of a bundle -- alone in its window it is itself; with
    its siblings in the window it merges into the bundle."""
    sc, ren, c, a1, a2 = _two_bundles(90, rf_um=1.2, Rb=10.0)                                      # fibres 0.6 voxels in radius
    vox, grp, obj, w, v = ren["entries"]; tree = build_tree(sc, grp, obj)
    m = grp == 0; o = int(torch.mode(obj[m]).values)                                                # a fibre of the first bundle
    sel = m & (obj == o); k = int(torch.argmax(torch.where(sel, w, torch.zeros_like(w))))
    S = sc.shape; p = torch.tensor([int(vox[k]) // (S[1] * S[2]), (int(vox[k]) // S[2]) % S[1], int(vox[k]) % S[2]], device=DEV).float()[None]
    sig = torch.tensor([2.0], device=DEV)
    t_all = entity_targets(ren["entries"], tree, sc.shape, (0, 0, 0), 1, sc.shape, p, sig)
    leaf = int(tree.leaves(torch.tensor([0], device=DEV), torch.tensor([o], device=DEV))[0])
    assert leaf not in t_all["node"][0].tolist()                                                    # among its siblings: the bundle
    keep = torch.zeros(int(obj.max()) + 1, dtype=torch.bool, device=DEV); keep[o] = True
    t_one = entity_targets(ren["entries"], tree, sc.shape, (0, 0, 0), 1, sc.shape, p, sig, keep=keep)
    assert t_one["node"][0, 0].item() == leaf and int((t_one["node"][0] >= 0).sum()) == 1


def test_deep_tree_fascicles_and_branches():
    """deep_tree: a resolved bundle's fibres sit in fascicles under the bundle; a vessel tree's branches under the
    branch they leave -- trees deeper than root / population / object."""
    from shapetensor.learn.entities import build_tree
    sc = Scene((64,) * 3, 4, DEV, 3.0); sc.deep = True
    g = sc.bundle(None, "single", radius=12.0, fibre={"rf_um": 4.5})
    sc.groups = [g]
    ren = sc.render(); vox, grp, obj, w, v = ren["entries"]; tree = build_tree(sc, grp, obj)
    assert int(tree.depth.max()) >= 4                                                               # root, population, bundle, fascicle, fibre
    found = False
    for seed in range(20):
        sc2 = Scene((96,) * 3, seed, DEV, 20.0); sc2.deep = True
        gs = sc2.tree("vessel")
        if gs:
            sc2.groups = gs; ren2 = sc2.render(); t2 = build_tree(sc2, ren2["entries"][1], ren2["entries"][2])
            if int(t2.depth.max()) >= 5:
                found = True; break
    assert found


def test_entity_targets_on_a_scene_without_populations():
    """No populations drawn (empty group list): no entities, no crash."""
    sc = Scene((32,) * 3, 0, DEV, 2.0); sc.groups = []
    ren = sc.render(); vox, grp, obj, w, v = ren["entries"]; tree = build_tree(sc, grp, obj)
    p = torch.tensor([[16.0, 16.0, 16.0]], device=DEV)
    t = entity_targets(ren["entries"], tree, sc.shape, (0, 0, 0), 1, sc.shape, p, torch.tensor([2.0], device=DEV),
                       vis=torch.ones(1, 32, 32, 32, device=DEV), group_layer=[], ovis=torch.ones(1, 32, 32, 32, device=DEV))
    assert bool((t["node"] == -1).all())
