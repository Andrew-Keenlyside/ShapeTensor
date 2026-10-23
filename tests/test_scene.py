"""The object-scene generator: exact geometry, identities, topology, packing, units, compatibility with the targets."""
import math

import pytest
import torch

from shapetensor.learn.scene import Canvas, Rng, Scene, curves_like, draw_scene, render_curves
from shapetensor.learn.targets import target

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _vessel(flat=None, rad=8.0, wall=1.5, S=200):
    rng = Rng(0, DEV)
    P = torch.stack([torch.full((S,), 32.0), torch.full((S,), 32.0), torch.linspace(-4, 100, S)], -1)[None].to(DEV)
    f = torch.ones(1, S, device=DEV) if flat is None else flat
    c = curves_like(P, torch.full((1, S), rad, device=DEV), rng, obj=torch.tensor([7], device=DEV), seg=torch.tensor([0], device=DEV),
                    contrast=80.0, flat=f, wall=torch.full((1, S), wall, device=DEV))
    cv = Canvas((64, 64, 128), DEV); render_curves(cv, c)
    out = cv.finish(); ids, frac = cv.identities()
    return out, ids


def test_open_vessel_matches_the_analytic_wall_and_lumen():
    out, ids = _vessel()
    occ, D = out["occupancy"], out["desc"]
    wall_per_len = float(occ[:, :, 10:90].sum()) / 80
    assert abs(wall_per_len - math.pi * (8 ** 2 - 6.5 ** 2)) / (math.pi * (8 ** 2 - 6.5 ** 2)) < 0.05
    m = occ[:, :, 50] > 0.5
    assert abs(float(D[2][:, :, 50][m].mean()) - (6.5 / 8) ** 2) < 0.02                 # lumen fraction
    assert float(occ[32, 32, 50]) < 0.01                                                 # the lumen is empty


def test_collapse_keeps_one_population_and_closes_the_lumen():
    S = 200; s = torch.arange(S, device=DEV, dtype=torch.float32)
    flat = (1 - 0.95 * torch.exp(-0.5 * ((s - 120) / 15) ** 2))[None]
    out, ids = _vessel(flat)
    occ, D, T = out["occupancy"], out["desc"], out["tensor"]
    xs = [10, 58, 90]                                                                    # open, collapsed, open
    lum = [float(D[2][:, :, x][occ[:, :, x] > 0.5].mean()) for x in xs]
    assert lum[0] > 0.6 and lum[1] < 0.05 and lum[2] > 0.6
    assert torch.unique(ids[0][occ > 0.5]).tolist() == [7]                               # one object throughout
    assert float(T[2][occ > 0.5].min()) > 0.99                                           # the same axis throughout


def test_tree_branches_with_tapering_and_records_points():
    fam = {k: (1.0 if k == "tree" else 0.0) for k in Scene.FAMILIES}
    sc = Scene((64, 64, 64), 3, DEV, 10.0).build(fam, (1, 1), 0.0)
    tree = [g for g in sc.groups if g.name == "tree"]
    assert tree and len(tree[0].curves) >= 2                                             # several generations
    M = torch.cat(tree[0].murray)                                                        # (branchings, 3): r, r1, r2
    assert torch.allclose(M[:, 1] ** 3 + M[:, 2] ** 3, M[:, 0] ** 3, rtol=1e-4)           # Murray's law
    assert bool((M[:, 1:] < M[:, :1]).all())
    assert sc.branch and sum(p.shape[0] for p in sc.branch) >= 1


def test_bundle_fibres_are_packed_without_overlap():
    fam = {k: (1.0 if k == "bundle" else 0.0) for k in Scene.FAMILIES}
    for seed in range(30):
        sc = Scene((64, 64, 64), seed, DEV, 2.0).build(fam, (1, 1), 0.0)
        g = sc.groups[0]
        if g.curves[0].N >= 10:
            break
    c = g.curves[0]; k = c.P.shape[1] // 2
    D = torch.cdist(c.P[:, k], c.P[:, k]); D.fill_diagonal_(1e9)
    rr = c.rad[:, k]
    overlap = (D < 0.9 * (rr[:, None] + rr[None])).float().mean()
    assert float(overlap) < 0.02, float(overlap)


def test_foam_walls_have_planar_tensors():
    fam = {k: (1.0 if k == "foam" else 0.0) for k in Scene.FAMILIES}
    sc = Scene((48, 48, 48), 1, DEV, 20.0).build(fam, (1, 1), 0.0); ren = sc.render()
    g = ren["groups"][0]; occ = g["occupancy"]; T = g["tensor"]
    m = occ > 0.8
    from shapetensor.learn.generator import eig3
    ev, _ = eig3(T[:, m]); planar = (ev[1] - ev[0]) / ev[2]
    assert float(planar.median()) > 0.8 and 0.05 < float(occ.mean()) < 0.75


def test_cells_are_nested_and_do_not_overlap():
    fam = {k: (1.0 if k == "cells" else 0.0) for k in Scene.FAMILIES}
    sc = Scene((64, 64, 64), 2, DEV, 2.0).build(fam, (1, 1), 0.0)
    assert len(sc.groups) == 2 and len(sc.parents) > 0                                   # membranes and nuclei; nucleus -> cell
    C, R = sc.obstacles[0]
    D = torch.cdist(C, C); D.fill_diagonal_(1e9)
    assert float((D - (R[:, None] + R[None])).min()) > -1e-3                             # bounding spheres disjoint


def test_identities_report_two_objects_where_they_overlap():
    rng = Rng(0, DEV); S = 120
    t = torch.linspace(-4, 60, S, device=DEV)
    P1 = torch.stack([torch.full((S,), 30.0, device=DEV), torch.full((S,), 30.0, device=DEV), t], -1)
    P2 = torch.stack([torch.full((S,), 30.0, device=DEV), t, torch.full((S,), 30.0, device=DEV)], -1)
    c = curves_like(torch.stack([P1, P2]), torch.full((2, S), 3.0, device=DEV), rng, obj=torch.tensor([3, 9], device=DEV), seg=torch.zeros(2, dtype=torch.long, device=DEV), contrast=50.0)
    cv = Canvas((60, 60, 60), DEV); render_curves(cv, c); cv.finish(); ids, frac = cv.identities()
    assert sorted([int(ids[0, 30, 30, 30]), int(ids[1, 30, 30, 30])]) == [3, 9]
    assert int(ids[1, 30, 30, 50]) == -1 and int(ids[0, 30, 30, 50]) == 3


def test_draw_scene_is_deterministic_in_units_and_feeds_the_targets():
    s1, d1 = draw_scene((32, 32, 32), seed=5, device=DEV, strength=0.3, canvas=64)
    s2, d2 = draw_scene((32, 32, 32), seed=5, device=DEV, strength=0.3, canvas=64)
    assert torch.equal(d1["image"], d2["image"])
    t = target(s1, d1)
    assert torch.isfinite(t["slots"]).all() and t["mass"].shape[0] == 6
    for L, du in zip([L for L in s1.layers if hasattr(L, "desc5")], d1["desc_um"]):
        assert torch.allclose(du[0], L.desc5[0] * d1["voxel_um"])
    from shapetensor.learn.query import voxel_density
    F = voxel_density(s1, d1, 4)
    assert torch.isfinite(F).all()
