"""The compositional generator: each operator does what it says, and random sampling reaches all of them."""
import math
from collections import Counter

import pytest
import torch

from shapetensor.learn import compose as C
from shapetensor.learn.scene import Rng, Scene

DEV = "cpu"


def ang(a, b):
    return torch.rad2deg(torch.acos((a * b).sum(-1).abs().clamp(max=1)))


def test_cylinder_field_angle_and_hand():
    c = torch.tensor([20.0, 20.0, 20.0]); a = torch.tensor([0.0, 0.0, 1.0])
    f = C.Cylinder(30.0, hand=1.0, c=c, a=a)
    th = torch.linspace(0, 6.28, 50); x = c + torch.stack([8 * torch.cos(th), 8 * torch.sin(th), torch.zeros(50)], 1)
    d = f(x)
    assert torch.allclose(ang(d, a.expand_as(d)), torch.full((50,), 30.0), atol=0.5)               # 30 deg off the axis everywhere
    er = C._unit(x - c); et = torch.linalg.cross(a.expand_as(er), er, dim=-1)
    g = C.Cylinder(30.0, hand=-1.0, c=c, a=a)(x)
    assert bool(((d * et).sum(-1) > 0).all()) and bool(((g * et).sum(-1) < 0).all())              # opposite hands


def test_laminar_rotates_with_depth():
    c = torch.zeros(3); n = torch.tensor([1.0, 0, 0]); u = torch.tensor([0, 1.0, 0])
    f = C.Laminar(c, n, u, 0.0, math.radians(2.0))                                                # 2 deg per voxel along n
    x = torch.stack([torch.arange(0, 40, 5.0), torch.zeros(8), torch.zeros(8)], 1)
    d = f(x)
    th = torch.rad2deg(torch.atan2(d[:, 2], d[:, 1]))
    assert torch.allclose(th, torch.arange(0, 40, 5.0) * 2.0, atol=0.5)
    assert torch.allclose((d * n).sum(-1), torch.zeros(8), atol=1e-5)                              # in-plane (no tilt)


def test_boundary_radial_and_tangential():
    r = Rng(1, DEV); c = torch.tensor([30.0, 30.0, 30.0])
    x = c + 15 * C._unit(torch.randn(40, 3))
    rad = C.Boundary(c, 10.0, "radial", r)(x); tan = C.Boundary(c, 10.0, "tangential", r)(x)
    er = C._unit(x - c)
    assert float(ang(rad, er).max()) < 0.5 and float((tan * er).sum(-1).abs().max()) < 1e-4


def test_surface_parallel_and_normal():
    r = Rng(2, DEV); c = torch.tensor([32.0, 32.0, 32.0])
    s = C.Surface(c, torch.tensor([0, 0, 1.0]), 4.0, 30.0, "parallel", r)
    x = c + torch.randn(60, 3) * 10
    nrm = s.normal(x); par = s(x)
    assert float((par * nrm).sum(-1).abs().max()) < 1e-4
    on = x.clone(); on = on - s.sdf(on)[:, None] * torch.tensor([0, 0, 1.0])                       # onto the surface along n
    assert float(s.sdf(on).abs().max()) < 1e-3


@pytest.mark.parametrize("kind", ["smooth", "cylinder", "laminar"])
def test_paths_follow_their_field(kind):
    r = Rng(3, DEV); sc = Scene((64,) * 3, 3, DEV, 2.0)
    f = C.sample_field(sc, r, kind)
    seeds = 16 + 32 * torch.rand(30, 3, generator=torch.Generator().manual_seed(0))
    P = C.grow(f, seeds, 60.0, 0.5, (64,) * 3)
    T = C._unit(P[:, 2:] - P[:, :-2]); mid = 0.5 * (P[:, 2:] + P[:, :-2])
    moved = (P[:, 2:] - P[:, :-2]).norm(dim=-1) > 1e-3
    d = f(mid[moved])
    assert float(ang(T[moved], d).median()) < 3.0


def test_growth_stops_on_a_surface():
    f = C.Smooth(torch.tensor([1.0, 0, 0]), 0.0, 30.0, Rng(4, DEV))
    seeds = torch.tensor([[10.0, 20.0, 20.0], [12.0, 30.0, 30.0]])
    P = C.grow(f, seeds, 80.0, 0.5, (64,) * 3, both=False, stop=lambda x: x[:, 0] > 40.0)
    assert float(P[..., 0].max()) <= 40.0 + 1e-4


def test_crimp_period_and_amplitude():
    S = 801; s = torch.linspace(0, 200, S)
    P = torch.stack([s, torch.zeros(S), torch.zeros(S)], -1)[None].repeat(3, 1, 1) + torch.tensor([0.0, 20.0, 20.0])
    Q = C.crimp(P, 20.0, 1.5, 1.0, torch.zeros(3))
    lat = (Q - P).norm(dim=-1)[0]
    assert abs(float(lat.max()) - 1.5) < 0.05
    sig = (Q - P)[0].norm(dim=-1) * torch.sign(((Q - P)[0] * (Q - P)[0, S // 40]).sum(-1))
    F = torch.fft.rfft(sig - sig.mean()).abs(); f = torch.fft.rfftfreq(S, d=200 / (S - 1))
    assert abs(1 / float(f[F[1:].argmax() + 1]) - 20.0) < 1.0


def test_twisted_packing_rotates():
    r = Rng(5, DEV); S = 401
    cen = torch.stack([torch.linspace(0, 100, S), torch.full((S,), 30.0), torch.full((S,), 30.0)], -1)
    F, off = C.pack(cen, 8.0, 0.8, 2.5, math.radians(90) / 100, r, jitter=0.0)
    k = int(off.norm(dim=1).argmax())
    v0 = F[k, 0] - cen[0]; v1 = F[k, -1] - cen[-1]
    assert abs(float(ang(C._unit(v0), C._unit(v1))) - 90.0) < 3.0                                  # a quarter turn over the bundle


def test_split_merge_parts_and_rejoins():
    r = Rng(6, DEV); S = 401
    cen = torch.stack([torch.linspace(0, 120, S), torch.full((S,), 30.0), torch.full((S,), 30.0)], -1)
    F, off = C.pack(cen, 8.0, 0.8, 2.5, 0.0, r, jitter=0.0)
    G, side = C.split_merge(F, off, cen, 1, 10.0, r)
    d = (G - F).norm(dim=-1)
    assert float(d[:, S // 2].median()) > 9.0 and float(d[:, 0].max()) < 1e-4 and float(d[:, -1].max()) < 1e-4
    assert set(side.tolist()) == {-1.0, 1.0}


def test_branch_leaves_at_the_angle():
    r = Rng(7, DEV); S = 201
    P = torch.stack([torch.linspace(0, 100, S), torch.full((S,), 30.0), torch.full((S,), 30.0)], -1)[None]
    f = C.Smooth(torch.tensor([1.0, 0, 0]), 0.0, 30.0, r)
    Ch, par = C.branch(P, f, 3, 40.0, 30.0, 0.5, (128,) * 3, r)
    t0 = C._unit(Ch[:, 1] - Ch[:, 0])
    assert torch.allclose(ang(t0, torch.tensor([1.0, 0, 0]).expand_as(t0)), torch.full((3,), 40.0), atol=4.0)
    assert float((Ch[:, 0, 1:] - 30.0).abs().max()) < 1e-4                                         # starts on the parent


def test_loops_join_ends():
    a = torch.stack([torch.linspace(0, 20, 41), torch.zeros(41), torch.zeros(41)], -1)
    b = torch.stack([torch.linspace(26, 46, 41), torch.zeros(41) + 1.0, torch.zeros(41)], -1)
    L = C.loops(torch.stack([a, b]), 10.0, 21)
    assert L.shape[0] == 1 and float((L[0, 0] - a[-1]).norm()) < 1e-4 and float((L[0, -1] - b[0]).norm()) < 1e-4


def test_composed_scenes_render_with_valid_ids_and_trees():
    from shapetensor.learn.entities import build_tree
    for seed in range(4):
        sc = Scene((64,) * 3, seed, DEV, 2.0); sc.groups = C.compose_scene(sc, 3)
        for g in sc.groups:
            for cv in g.curves:
                assert all(int(o) in sc.parents for o in cv.obj.tolist())                          # every object under its population's node
        ren = sc.render(); vox, grp, obj, w, v = ren["entries"]
        assert vox.numel() > 0
        tree = build_tree(sc, grp, obj); assert int(tree.kind[0]) == 3


def test_sampling_reaches_every_operator():
    seen = Counter()
    for seed in range(100):
        sc = Scene((64,) * 3, 100 + seed, DEV, 2.0); C.compose_scene(sc, 3)
        for rec in sc.record["compose"]:
            seen[("field", rec["field"])] += 1; seen[("relation", rec["relation"])] += 1
            if rec.get("element", "line") == "line":
                seen[("arrange", rec["arrange"])] += 1
            for k in ("crimp", "wiggle", "twist", "split_merge", "branch", "loops"):
                if k in rec:
                    seen[("op", k)] += 1
    for k in C.FIELD_P:
        assert seen[("field", k)] > 0, k
    for k in C.ARRANGE_P:
        assert seen[("arrange", k)] > 0, k
    for k in C.RELATION_P:
        assert seen[("relation", k)] > 0, k
    for k in ("crimp", "wiggle", "twist", "split_merge", "branch"):
        assert seen[("op", k)] > 0, k


def test_sheets_follow_the_phase():
    """Laminar sheets: normals along the stack's normal, one sheet per spacing along it, each sheet its own object."""
    r = Rng(11, DEV); sc = Scene((64,) * 3, 11, DEV, 2.0)
    n = torch.tensor([0.0, 0.6, 0.8]); f = C.Laminar(torch.full((3,), 32.0), n, torch.tensor([1.0, 0, 0]), 0.0, 0.0)
    import random
    for attempt in range(20):                                                                    # a draw without undulation / patches
        sc = Scene((64,) * 3, 100 + attempt, DEV, 2.0)
        gs = C.sheet_population(sc, f, "none", None)
        rec = sc.record["compose"][-1]
        if gs and not rec.get("undulate") and not rec.get("patch") and not rec.get("perforated") and "n" not in rec:
            break
    occ, tens, desc, con, obj = gs[0].dense[0]
    on = occ > 0.5
    T6 = tens[:, on].T                                                                           # (n, 6): (I - n n^T) / 2
    nn_ = n / n.norm(); M = torch.eye(3) - 2 * torch.stack([torch.stack([T6[:, 0], T6[:, 3], T6[:, 4]], -1), torch.stack([T6[:, 3], T6[:, 1], T6[:, 5]], -1),
                                                       torch.stack([T6[:, 4], T6[:, 5], T6[:, 2]], -1)], -2)   # n n^T
    assert float(((M @ nn_) * nn_).sum(-1).median()) > 0.99
    # occupancy along the normal: period = the spacing
    t = torch.arange(-25.0, 25.0, 0.25); x = (32 + t[:, None] * nn_).round().long().clamp(0, 63)
    prof = occ[x[:, 0], x[:, 1], x[:, 2]]
    F = torch.fft.rfft(prof - prof.mean()).abs(); fr = torch.fft.rfftfreq(prof.numel(), d=0.25)
    per = 1 / float(fr[F[1:].argmax() + 1])
    assert abs(per - rec["spacing"]) / rec["spacing"] < 0.15, (per, rec["spacing"])
    ids = obj[on].unique()
    assert ids.numel() >= 2 and all(sc.parents[int(i)] == sc.parents[int(ids[0])] for i in ids)


def test_blobs_align_with_the_field():
    """Blobs' long axis (spindles) or short axis (discs) along a uniform field: their shape (each blob's points' second
    moment). Their material is isotropic (solids' ODF is I / 3, 29 Oct)."""
    d0 = torch.tensor([0.0, 0.0, 1.0]); checked = 0
    for seed in range(40):
        sc = Scene((64,) * 3, 300 + seed, DEV, 2.0); r = sc.r
        f = C.Smooth(d0, 0.0, 30.0, r)
        gs = C.blob_population(sc, f, "none", None, None)
        rec = sc.record["compose"][-1] if gs else None
        if not gs or rec["shape"] == "cell" or abs(math.log(rec["elong"])) < 0.5:
            continue
        Q, W, V, oid = gs[0].points[0]
        first = torch.unique(oid, return_inverse=True)[1]
        idx = torch.stack([torch.nonzero(first == j)[0, 0] for j in range(int(first.max()) + 1)])
        s6 = V[idx, :6]
        assert torch.allclose(s6, torch.tensor([1, 1, 1, 0, 0, 0.0], device=s6.device).expand_as(s6) / 3, atol=1e-4)   # material: isotropic
        M = []
        for j in range(int(first.max()) + 1):                                                   # shape: the points' spread
            q_ = Q[first == j].float(); w_ = W[first == j].float()[:, None]; q_ = q_ - (w_ * q_).sum(0) / w_.sum()
            M.append((w_ * q_).T @ q_ / w_.sum())
        ev, evec = torch.linalg.eigh(torch.stack(M))
        ax = evec[..., -1] if rec["elong"] > 1 else evec[..., 0]
        assert float(ang(ax, d0.expand_as(ax)).median()) < 10.0, rec
        checked += 1
        if checked >= 4:
            break
    assert checked >= 2


def test_sampling_reaches_every_element():
    seen = Counter()
    for seed in range(80):
        sc = Scene((64,) * 3, 700 + seed, DEV, 2.0); sc.deep = True; C.compose_scene(sc, 3)
        for rec in sc.record["compose"]:
            seen[("element", rec.get("element"))] += 1
            if rec.get("element") == "blob":
                seen[("place", rec["place"])] += 1; seen[("shape", rec["shape"])] += 1
            if rec.get("element") == "sheet":
                seen[("phase", rec["phase"])] += 1
    for k in C.ELEMENT_P:
        assert seen[("element", k)] > 0, k
    for k in C.BLOB_PLACE_P:
        assert seen[("place", k)] > 0, k
    for k in C.BLOB_SHAPE_P:
        assert seen[("shape", k)] > 0, k
    for k in ("planes", "cylinders", "folds", "shells", "lamellae"):
        assert seen[("phase", k)] > 0, (k, seen)
