"""The object-scene generator: exact geometry, identities, topology, packing, units, compatibility with the targets."""
import math

import pytest
import torch.nn.functional as Fn
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


# ---- the generalisation round: crossings, physics, paired renders, warps, identities, families --------------------

def _crossing(interleave: bool, seed: int = 1):
    sc = Scene((96, 96, 96), seed, DEV, 2.0)
    c = torch.tensor([48.0, 48.0, 48.0], device=DEV)
    for _ in range(50):                                                                 # resolved fibres in both
        a = sc.bundle(None, "single", axis=torch.tensor([0.0, 0.0, 1.0], device=DEV), centre=c, radius=12.0, curv=1e-4)
        if a is not None and a.offsets is not None:
            break
    for _ in range(50):
        b = sc.bundle(a.centreline, "angled", curv=1e-4)
        if b is not None and not isinstance(b, list) and b.offsets is not None:
            break
    if interleave:
        sc.interleave(a, b)
    sc.groups = [a, b]; ren = sc.render()
    return ren["groups"][0]["occupancy"], ren["groups"][1]["occupancy"]


def test_interleaved_crossing_shares_no_voxels():
    oA, oB = _crossing(False); both = (oA > 0.05) & (oB > 0.05)
    oA2, oB2 = _crossing(True); both2 = (oA2 > 0.05) & (oB2 > 0.05)
    assert int(both.sum()) > 50                                                          # fibres pass through each other without it
    assert int(both2.sum()) < 0.1 * int(both.sum()), (int(both.sum()), int(both2.sum()))


def test_split_tract_is_two_populations_diverging():
    fam = {k: (1.0 if k == "bundle" else 0.0) for k in Scene.FAMILIES}
    for sd in range(400):
        sc = Scene((96, 96, 96), sd, DEV, 3.0).build(fam, (1, 1), 0.0)
        if "split" in sc.record and len(sc.groups) == 2:
            break
    g1, g2 = sc.groups
    ends = lambda g: g.curves[0].P[:, -1].mean(0) - g.curves[0].P[:, 0].mean(0)
    cos = float(Fn.normalize(ends(g1), dim=0) @ Fn.normalize(ends(g2), dim=0))
    assert cos < math.cos(math.radians(0.5 * sc.record["split"][0]))                     # the halves leave at an angle
    assert torch.unique(torch.cat([g1.curves[0].seg, g2.curves[0].seg])).numel() == 1    # one bundle


def test_tree_branch_points_separate_the_smaller_child():
    fam = {k: (1.0 if k == "tree" else 0.0) for k in Scene.FAMILIES}
    sc = Scene((96, 96, 96), 3, DEV, 10.0).build(fam, (1, 1), 0.0)
    trees = [g for g in sc.groups if g.name == "tree"]
    assert len(trees) == 2                                                               # two sides
    assert torch.unique(torch.cat([c.obj for g in trees for c in g.curves])).numel() == 1  # one object


def _two_pop(mode, t1, t2, o1=0.5, o2=0.5, seq=None, mats=None):
    from shapetensor.learn.physics import draw_material, draw_sequence, form_image
    import numpy as np
    rng = np.random.default_rng(0)
    shp = (4, 4, 4)
    occ = torch.stack([torch.full(shp, o1, device=DEV), torch.full(shp, o2, device=DEV)])
    T = torch.stack([t1.view(6, 1, 1, 1).expand(6, *shp), t2.view(6, 1, 1, 1).expand(6, *shp)]).to(DEV)
    mats = mats or [draw_material(rng, mode), draw_material(rng, mode)]
    seq = seq or draw_sequence(rng, mode)
    return form_image(mode, occ, T, torch.zeros(shp, device=DEV), mats, draw_material(rng, mode, True), seq, raw=True), mats, seq


def test_histology_overlaps_multiply_transmissions():
    import numpy as np
    from shapetensor.learn.physics import draw_material, draw_sequence, form_image
    rng = np.random.default_rng(1); shp = (2, 2, 2)
    mats = [draw_material(rng, "histology") for _ in range(2)]; seq = draw_sequence(rng, "histology"); bgm = {"c": [0.0, 0.0]}
    T = torch.zeros(2, 6, *shp, device=DEV); z = torch.zeros(shp, device=DEV)
    img = lambda a, b: float(form_image("histology", torch.stack([torch.full(shp, a, device=DEV), torch.full(shp, b, device=DEV)]), T, z, mats, bgm, seq).mean())
    lin = img(0.5, 0) + img(0, 0.5) - img(0, 0)
    assert abs(img(0.5, 0.5) - lin) > 0.5                                                 # not additive


def test_pli_crossing_fibres_cancel_and_through_plane_fibres_vanish():
    import numpy as np
    from shapetensor.learn.generator import line_tensor
    seq = {"z": np.array([1.0, 0, 0]), "rho": 0.3, "map": True, "thick": 1.5}
    mats = [{"biref": 2.0, "atten": 0.1}, {"biref": 2.0, "atten": 0.1}]
    ly = line_tensor(torch.tensor([0.0, 1.0, 0.0])); lx = line_tensor(torch.tensor([0.0, 0.0, 1.0])); lz = line_tensor(torch.tensor([1.0, 0.0, 0.0]))
    one, _, _ = _two_pop("pli", ly, ly, 0.5, 0.5, seq, mats)
    cross, _, _ = _two_pop("pli", ly, lx, 0.5, 0.5, seq, mats)
    thru, _, _ = _two_pop("pli", lz, lz, 0.5, 0.5, seq, mats)
    assert float(cross.mean()) < 0.05 * float(one.mean()) and float(thru.mean()) < 0.05 * float(one.mean())


def test_mri_signal_depends_on_fibre_angle_to_b0():
    import numpy as np
    from shapetensor.learn.generator import line_tensor
    seq = {"kind": "t2s", "TR": 50.0, "TE": 30.0, "flip": math.radians(20), "b0": np.array([1.0, 0, 0])}
    m = {"pd": 0.8, "t1": 1000.0, "t2s": 40.0, "chi1": 20.0, "chi2": 0.0, "df": 0.0}
    par, _, _ = _two_pop("mri", line_tensor(torch.tensor([1.0, 0, 0])), line_tensor(torch.tensor([1.0, 0, 0])), 0.5, 0.5, seq, [m, m])
    perp, _, _ = _two_pop("mri", line_tensor(torch.tensor([0, 1.0, 0])), line_tensor(torch.tensor([0, 1.0, 0])), 0.5, 0.5, seq, [m, m])
    assert float(par.mean()) > 1.02 * float(perp.mean())                                   # R2* is larger across B0


def test_zoom_pairs_conserve_volume_and_align_on_the_lattice():
    from shapetensor.learn.scene import draw_pair, render_scene
    (s1, d1), (s2, d2), info = draw_pair((32, 32, 32), seed=40, device=DEV, kind="zoom", strength=0.0, canvas=64, p_warp=0.0)
    assert info["scale"] == 2 and d2["voxel_um"] == pytest.approx(2 * d1["voxel_um"])
    oz, oy, ox = [8 * c for c in info["offset_cells"]]                                   # the fine crop inside the coarse one (fine voxels)
    shp1 = [L for L in s1.layers if hasattr(L, "desc5")]; shp2 = [L for L in s2.layers if hasattr(L, "desc5")]
    f = sum(L.occupancy for L in shp1); c = sum(L.occupancy for L in shp2)
    sub = c[oz // 2: oz // 2 + 16, oy // 2: oy // 2 + 16, ox // 2: ox // 2 + 16]
    assert abs(float(Fn.avg_pool3d(f[None, None], 2).mean()) - float(sub.mean())) < 0.02 + 0.05 * float(sub.mean())


def test_warp_moves_the_identities_with_the_labels():
    s, d = draw_scene((48, 48, 48), seed=9, device=DEV, strength=0.0, p_warp=1.0, canvas=64)
    assert d["warp_grid"] is not None
    occ = sum(d["occupancy"][k] for k, L in enumerate(s.layers) if hasattr(L, "desc5"))
    has = d["identity"] >= 0
    assert float((occ[has] > 0.05).float().mean()) > 0.9


def test_identity_target_uses_the_level_the_resolution_separates():
    fam = {k: (1.0 if k == "bundle" else 0.0) for k in Scene.FAMILIES}
    for sd in range(100):
        sc = Scene((64, 64, 64), sd, DEV, 4.0).build(fam, (1, 1), 0.0)
        g = sc.groups[0]
        if g.offsets is not None and float(g.curves[0].rad.max()) < 1.5:
            break
    lut = sc.identity_lut(1.5)
    fib = g.curves[0].obj
    assert torch.unique(lut[fib]).numel() == 1 and int(lut[fib[0]]) == sc.parents[int(fib[0])]


def test_plywood_and_trabecular_tensors():
    from shapetensor.learn.generator import eig3
    fam = {k: (1.0 if k == "plywood" else 0.0) for k in Scene.FAMILIES}
    sc = Scene((48, 48, 48), 2, DEV, 5.0).build(fam, (1, 1), 0.0); ren = sc.render()
    for g in ren["groups"]:
        m = g["occupancy"] > 0.3; ev, _ = eig3(g["tensor"][:, m])
        assert float(((ev[2] - ev[1]) / ev[2]).median()) > 0.9                           # lines in each lamella
    fam = {k: (1.0 if k == "trabecular" else 0.0) for k in Scene.FAMILIES}
    sc = Scene((64, 64, 64), 1, DEV, 20.0).build(fam, (1, 1), 0.0); ren = sc.render()
    by = {g["group"].name: g for g in ren["groups"]}
    ev, _ = eig3(by["plates"]["tensor"][:, by["plates"]["occupancy"] > 0.5]); assert float(((ev[1] - ev[0]) / ev[2]).median()) > 0.8
    ev, _ = eig3(by["rods"]["tensor"][:, by["rods"]["occupancy"] > 0.5]); assert float(((ev[2] - ev[1]) / ev[2]).median()) > 0.8


def test_every_family_renders():
    for name in Scene.FAMILIES:
        fam = {k: (1.0 if k == name else 0.0) for k in Scene.FAMILIES}
        sc = Scene((48, 48, 48), 4, DEV, 5.0).build(fam, (1, 1), 1.0 if name == "radial" else 0.0); ren = sc.render()
        for g in ren["groups"]:
            assert torch.isfinite(g["tensor"]).all() and float(g["occupancy"].max()) <= 1.0
