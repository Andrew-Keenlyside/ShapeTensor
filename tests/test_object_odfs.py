"""Every object's per-voxel material ODF is an ellipse (29 Oct): fibres (1 - e) t t^T + e (I - t t^T) / 2 with their own
spread, solids isotropic, sheets and hollow walls planar (normal radial), a fan elongated in its plane -- checked on the
window-summed tensors of the canvas entries (what the entity targets sum). CPU, synthetic only."""
import math

import numpy as np
import pytest
import torch

from shapetensor.learn.scene import Group, Scene, curves_like, draw_scene
from shapetensor.learn import generator as GN

DEV = "cpu"
N = 48


def _render(groups, seed=0):
    sc = Scene((N,) * 3, seed, DEV, voxel_um=5.0)
    sc.groups = groups
    out = sc.render()
    return sc, out


def _window(out, x, sigma, obj=None):
    """The window-summed, trace-normalised tensor (3, 3) at ``x`` (voxels) over the entries (of ``obj`` or all)."""
    vox, grp, ob, w, v = out["entries"]
    c = torch.stack([vox // (N * N), (vox // N) % N, vox % N], 1).float()
    g = torch.exp(-0.5 * ((c - torch.tensor(x, dtype=torch.float32)) ** 2).sum(1) / sigma ** 2)
    if obj is not None:
        g = g * (ob == obj)
    t6 = (g[:, None] * v[:, :6]).sum(0) / (g * w).sum().clamp_min(1e-12)
    T = torch.stack([torch.stack([t6[0], t6[3], t6[4]]), torch.stack([t6[3], t6[1], t6[5]]), torch.stack([t6[4], t6[5], t6[2]])])
    T = T / T.trace()
    ev, V = torch.linalg.eigh(T.double())
    return ev.float(), V.float()


def _line(d, c, L, S=60):
    d = torch.tensor(d, dtype=torch.float32); d = d / d.norm()
    t = torch.linspace(-L / 2, L / 2, S)
    return (torch.tensor(c, dtype=torch.float32)[None] + t[:, None] * d[None])[None], d


def _ang(a, b):
    return math.degrees(math.acos(min(1.0, abs(float((a * b).sum())))))


@pytest.mark.parametrize("spread", [3.0, 10.0])
def test_straight_fibre_is_a_line_with_its_own_spread(spread):
    P, d = _line((1.0, 2.0, 3.0), (24, 24, 24), 60)
    rng = Scene((N,) * 3, 1, DEV).r
    cv = curves_like(P, torch.full((1, P.shape[1]), 1.2), rng, obj=torch.tensor([7]), seg=torch.tensor([0]), contrast=60.0,
                     spread=torch.tensor([spread]))
    _, out = _render([Group("fibre", "line", 1.2, 60.0, curves=[cv])])
    e = math.sin(math.radians(spread)) ** 2
    for x in ((24, 24, 24), (24 + 4 * float(d[0]), 24 + 4 * float(d[1]), 24 + 4 * float(d[2]))):
        for sig in (2.0, 4.0):
            ev, V = _window(out, x, sig, obj=7)
            assert _ang(V[:, 2], d) < 3.0
            assert abs((1 - float(ev[2])) - e) < 0.01                                          # perpendicular share = e
            assert abs(float(ev[0]) - float(ev[1])) < 0.01                                     # round about the axis


def test_default_spreads_are_drawn_per_curve_and_recorded():
    P = torch.stack([_line((0, 1.0, 0.2 * k), (24, 24, 10 + 4 * k), 40)[0][0] for k in range(6)])
    sc = Scene((N,) * 3, 3, DEV)
    cv = curves_like(P, torch.full((6, P.shape[1]), 1.0), sc.r, obj=torch.arange(1, 7), seg=torch.zeros(6, dtype=torch.long), contrast=60.0)
    lo, hi = GN.SPREAD_DEG
    assert cv.spread.shape == (6,) and bool(((cv.spread >= lo) & (cv.spread <= hi)).all()) and cv.spread.unique().numel() == 6
    sc.groups = [Group("fibres", "line", 1.0, 60.0, curves=[cv])]; sc.render()
    assert set(sc.obj_spread) == set(range(1, 7)) and sc.record["spread_deg"]["n"] == 6


@pytest.mark.parametrize("A", [np.diag([4.0, 4.0, 4.0]), np.diag([7.0, 3.5, 2.0])])
def test_solids_are_isotropic(A):
    sc = Scene((N,) * 3, 2, DEV)
    R = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(4)))[0]
    At = (R @ torch.tensor(A, dtype=torch.float32))[None]
    g = sc._solids(torch.tensor([[24.0, 24.0, 24.0]]), At, "cells", sc.ids(1, 2.0))
    _, out = _render([g])
    for sig in (2.0, 3.0):
        ev, _ = _window(out, (24, 24, 24), sig)
        assert float((ev - 1 / 3).abs().max()) < 0.05


def test_sheet_is_planar_with_its_normal():
    sc = Scene((N,) * 3, 5, DEV)
    n = torch.tensor([0.3, 0.5, 0.81]); n = n / n.norm()
    g = sc.sheet(normal=n, centre=torch.tensor([24.0, 24.0, 24.0]), half_thickness=1.5, height=(20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))   # flat
    gs = g if isinstance(g, list) else [g]
    _, out = _render(gs)
    ev, V = _window(out, (24, 24, 24), 3.0)
    assert float(ev[0]) < 0.06 and abs(float(ev[1]) - 0.5) < 0.06 and abs(float(ev[2]) - 0.5) < 0.06
    assert _ang(V[:, 0], n) < 5.0


def test_hollow_tube_wall_is_a_sheet_and_its_section_axial():
    S = 100
    P = torch.stack([torch.full((S,), 24.0), torch.full((S,), 24.0), torch.linspace(-4, 52, S)], -1)[None]
    sc = Scene((N,) * 3, 6, DEV)
    cv = curves_like(P, torch.full((1, S), 8.0), sc.r, obj=torch.tensor([9]), seg=torch.tensor([0]), contrast=80.0,
                     wall=torch.full((1, S), 1.5), spread=torch.tensor([5.0]))
    _, out = _render([Group("vessel", "line", 8.0, 80.0, curves=[cv])])
    axis = torch.tensor([0.0, 0.0, 1.0])
    for phi in (0.0, 1.1, 2.5):                                                                # small windows on the wall: planar, normal radial
        r = torch.tensor([math.cos(phi), math.sin(phi), 0.0])
        x = (24 + 7.25 * float(r[0]), 24 + 7.25 * float(r[1]), 24.0)
        ev, V = _window(out, x, 1.0, obj=9)
        assert float(ev[0]) < 0.1 and float(ev[1]) > 0.35
        assert _ang(V[:, 0], r) < 12.0
    ev, V = _window(out, (24, 24, 24), 8.0, obj=9)                                            # the whole section: axial-dominant
    assert _ang(V[:, 2], axis) < 5.0 and float(ev[2]) > 0.42 and float(ev[2]) - float(ev[1]) > 0.12


def test_fan_spreads_in_its_plane():
    sc = Scene((N,) * 3, 7, DEV)
    a = torch.tensor([0.0, 0.0, 1.0]); b = torch.tensor([0.0, 1.0, 0.0]); o = torch.tensor([24.0, 24.0, 4.0])
    th = torch.linspace(-0.6, 0.6, 9); S = 60; L = torch.linspace(0, 44, S)
    dirs = torch.cos(th)[:, None] * a + torch.sin(th)[:, None] * b
    P = o[None, None] + L[None, :, None] * dirs[:, None]
    cv = curves_like(P, torch.full((9, S), 0.8), sc.r, obj=torch.arange(1, 10), seg=torch.zeros(9, dtype=torch.long), contrast=60.0,
                     spread=torch.full((9,), 2.0))
    _, out = _render([Group("fan", "line", 0.8, 60.0, curves=[cv])])
    ev, V = _window(out, (24, 24, 30), 6.0)                                                    # all objects in the window (the population)
    assert _ang(V[:, 2], a) < 5.0
    assert float(ev[1]) > float(ev[0]) + 0.03                                                 # elongated in the fan plane
    assert _ang(V[:, 0], torch.tensor([1.0, 0.0, 0.0])) < 10.0                                # the smallest: the plane's normal


def test_universal_crops_finite_with_spreads():
    for seed in (8, 16):
        s, d = draw_scene((32,) * 3, seed=seed, device=DEV, strength=1.0, canvas=64, entity_ctx=True)
        assert torch.isfinite(d["image"].float()).all() and torch.isfinite(d["tensors"].float()).all()
        vox, grp, obj, w, v = d["entity_ctx"]["entries"]
        assert torch.isfinite(v).all() and torch.isfinite(w).all()
        tr = v[:, :3].sum(1) / w.clamp_min(1e-9)
        assert float((tr[w > 1e-3] - 1).abs().max()) < 1e-3                                    # trace 1 per entry
