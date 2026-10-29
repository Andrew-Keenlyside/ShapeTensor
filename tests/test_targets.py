"""The population model's target: populations pooled per layer, so a bend stays one population, a crossing two, faint
populations keep their own mass, the same population in two layers merges, and the rendered density is readable."""
import math
import time

import pytest
import torch

from shapetensor.learn.generator import Layer, Sample, _composite, arc_field, boundary_sample, companion_field, direction_field, principal, render_layer
from shapetensor.learn.targets import CLASSES, clean, render_odf, target
from shapetensor.lift import hemisphere

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _sample(fields, shape=(64, 64, 64), radius=1.0, density=1.0, seed=0, kind="line"):
    g = torch.Generator(device=DEV).manual_seed(seed); ones = torch.ones(shape, dtype=torch.bool, device=DEV)
    layers = []
    for f in fields:
        L = Layer(kind, f, radius, 60.0 if kind == "line" else 10.0, density, ones, 50.0)
        L.occupancy, L.tensor = render_layer(L, shape, g, DEV); layers.append(L)
    bg = torch.full(shape, 100.0, device=DEV)
    img = _composite(bg, bg.clone(), layers, DEV)
    return Sample(img, layers, bg, torch.zeros(shape, dtype=torch.long, device=DEV), "flat", 1.0, "general", {}, bg.clone())


def _cell_field(f, step=4):
    return f[:, step // 2::step, step // 2::step, step // 2::step]


def _angle(a, b):
    return torch.rad2deg(torch.acos((a * b).sum(0).abs().clamp(max=1.0)))


def _slot_axis(slots, k):
    v, ev = principal(slots[k], "max")
    return v, ev


@pytest.mark.parametrize("R", [10.0, 25.0])
def test_bend_is_one_population(R):
    g = torch.Generator(device=DEV).manual_seed(1)
    f = arc_field((64, 64, 64), R, g, DEV)
    s = _sample([f], density=2.0)
    t = target(s, clean(s))
    m = t["mass"][0] > 0.05
    assert m.sum() > 50
    assert float((t["mass"][1][m] > 0.1 * t["mass"][0][m]).float().mean()) < 0.02                 # no second population on the bend
    v, ev = _slot_axis(t["slots"], 0)
    err = _angle(v, _cell_field(f))[m]
    assert float(err.median()) < (15.0 if R == 10 else 8.0), float(err.median())


def test_bend_lowers_linearity():
    lin = []
    for R in (10.0, 40.0):
        f = arc_field((64, 64, 64), R, torch.Generator(device=DEV).manual_seed(1), DEV)
        s = _sample([f], density=2.0); t = target(s, clean(s))
        m = t["mass"][0] > 0.05; _, ev = _slot_axis(t["slots"], 0)
        lin.append(float(((ev[0] - ev[1]) / ev[0])[m].median()))
    assert lin[0] < lin[1], lin


@pytest.mark.parametrize("theta", [30.0, 45.0, 60.0, 90.0])
def test_crossing_two_slots_on_their_axes(theta):
    g = torch.Generator(device=DEV).manual_seed(2)
    d1 = direction_field((64, 64, 64), 30.0, g, DEV); d2 = companion_field(d1, theta, 30.0, g, DEV)
    s = _sample([d1, d2], density=0.8)
    t = target(s, clean(s))
    both = (t["mass"][0] > 0.02) & (t["mass"][1] > 0.02)
    sl = (slice(2, -2),) * 3
    inner = torch.zeros_like(both); inner[sl] = True
    m = both & inner
    assert m.sum() > 100 and float(m.float().sum() / inner.float().sum()) > 0.9
    a0, _ = _slot_axis(t["slots"], 0); a1, _ = _slot_axis(t["slots"], 1)
    c1, c2 = _cell_field(d1), _cell_field(d2)
    err = torch.minimum(torch.maximum(_angle(a0, c1), _angle(a1, c2)), torch.maximum(_angle(a0, c2), _angle(a1, c1)))[m]
    assert float(err.median()) < 10.0, float(err.median())


def test_same_population_merges():
    g = torch.Generator(device=DEV).manual_seed(3)
    d = direction_field((48, 48, 48), 30.0, g, DEV)
    s = _sample([d, d.clone()], shape=(48, 48, 48), density=0.6)
    t = target(s, clean(s))
    m = t["mass"][0] > 0.02
    assert float((t["mass"][1][m] > 1e-3).float().mean()) < 0.02
    tot = sum(L.occupancy for L in s.layers[:2])
    assert float(t["mass"][0][m].mean()) > 0.9 * float(t["fraction"][m].mean())


def test_faint_populations_keep_their_mass():
    g = torch.Generator(device=DEV).manual_seed(4)
    dirs = [torch.tensor(v, device=DEV, dtype=torch.float32) for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.577, 0.577, 0.577))]
    fields = [(d / d.norm()).view(3, 1, 1, 1).expand(3, 48, 48, 48).contiguous() for d in dirs]
    s = _sample(fields, shape=(48, 48, 48), density=0.05, radius=0.8)
    t = target(s, clean(s), sigma_w=6.0)
    occ_frac = torch.stack([L.occupancy.mean() for L in s.layers[:4]])
    assert float(occ_frac.max()) < 0.05                                                          # each population faint
    from shapetensor.learn.targets import _pool
    own = _pool(torch.stack([L.occupancy for L in s.layers[:4]]), 6.0, 4)                     # each population's own pooled volume
    present = (own > 1e-3).sum(0)
    ok = ((t["mass"] > 1e-3).sum(0) == present.clamp(max=6)).float().mean()
    assert float(ok) > 0.97 and float((present == 4).float().mean()) > 0.1, (float(ok), float((present == 4).float().mean()))
    want = (own * (own > 1e-3)).sum(0)                                                        # masses are the populations' own volumes, not renormalised
    assert torch.allclose(t["mass"].sum(0), want, rtol=0.02, atol=1e-4)


def test_blobs_merge_whatever_their_axis():
    from shapetensor.learn.generator import eye6
    shape = (32, 32, 32); ones = torch.ones(shape, dtype=torch.bool, device=DEV); g = torch.Generator(device=DEV).manual_seed(9)
    n = []
    for d in ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0), (0.577, 0.577, 0.577)):
        f = torch.tensor(d, device=DEV).view(3, 1, 1, 1).expand(3, *shape).contiguous()
        a = Layer("blob", f, 1.5, 3.5, 2.0, ones, 50.0); a.occupancy, a.tensor = render_layer(a, shape, g, DEV)          # blobs everywhere
        c = Layer("blob", f, 1.0, 1.0, 0.0, ones, 50.0, occupancy=torch.full(shape, 0.2, device=DEV), tensor=eye6(DEV).view(6, 1, 1, 1).expand(6, *shape) / 3)
        layers = [a, c]; bg = torch.full(shape, 100.0, device=DEV); img = _composite(bg, bg.clone(), layers, DEV)
        s = Sample(img, layers, bg, torch.zeros(shape, dtype=torch.long, device=DEV), "flat", 1.0, "general", {}, bg.clone())
        t = target(s, clean(s))
        n.append(float(((t["mass"] > 1e-3) & (t["kind"] == 2)).sum(0).float().mean()))
    assert max(n) < 1.01 and min(n) > 0.99, n                                                     # one blob population, whichever way the blobs point


def test_invisible_layer_is_isotropic():
    g = torch.Generator(device=DEV).manual_seed(5)
    d = direction_field((48, 48, 48), 30.0, g, DEV)
    s = _sample([d], shape=(48, 48, 48), density=1.0)
    lab = clean(s); vis0 = dict(lab); vis0["visibility"] = lab["visibility"].clone(); vis0["visibility"][0] = 0.0
    t1, t0 = target(s, lab), target(s, vis0)
    assert float(t0["mass"].max()) < 1e-6 and float(t1["mass"][0].max()) > 0.1
    assert float(t0["classes"][CLASSES.index("isotropic")].min()) > 0.999


def test_classes_and_fraction_consistent():
    g = torch.Generator(device=DEV).manual_seed(6)
    d1 = direction_field((48, 48, 48), 20.0, g, DEV); d2 = companion_field(d1, 70.0, 20.0, g, DEV)
    s = _sample([d1, d2], shape=(48, 48, 48), density=1.0)
    t = target(s, clean(s))
    assert torch.allclose(t["classes"].sum(0), torch.ones_like(t["fraction"]), atol=1e-4)
    assert torch.allclose(t["mass"].sum(0) + t["dropped"], t["fraction"], atol=1e-3)
    tr = t["slots"][:, :3].sum(1)
    assert torch.allclose(tr[t["mass"] > 0], torch.ones_like(tr[t["mass"] > 0]), atol=1e-3)


def test_boundary_target_has_both_states():
    b = boundary_sample((48, 48, 48), step=60.0, contrast=30.0, seed=0, device=DEV, density=1.5)
    t = target(b, clean(b))
    k = torch.where(t["mass"] > 0.01, t["kind"], torch.full_like(t["kind"], -1))
    kinds_at = k[:, 5:7]; kinds_far = k[:, 0]                                                 # cells at the step (voxels 20-27) and > 3 sigma_w away (0-3)
    assert float(((kinds_at == 3).any(0)).float().mean()) > 0.9 and float(((kinds_at == 0).any(0)).float().mean()) > 0.9
    assert float(((kinds_far == 3).any(0)).float().mean()) < 0.05 and float(((kinds_far == 0).any(0)).float().mean()) > 0.9


def test_render_odf_readable_by_peaks():
    from shapetensor.populations import peaks
    grid = hemisphere(160, DEV)
    g = torch.Generator(device=DEV).manual_seed(7)
    d1 = torch.tensor([0.0, 0.0, 1.0], device=DEV).view(3, 1, 1, 1).expand(3, 48, 48, 48).contiguous()
    d2 = companion_field(d1, 60.0, 40.0, g, DEV)
    s = _sample([d1, d2], shape=(48, 48, 48), density=0.8)
    t = target(s, clean(s))
    f = render_odf(t["slots"], t["mass"], grid)
    dirs, share = peaks(f / f.sum(0, keepdim=True).clamp_min(1e-12), grid, nmax=3, rel=0.3, min_mass=0.1, min_sep_deg=25)
    both = (t["mass"][0] > 0.05) & (t["mass"][1] > 0.05)
    n2 = ((share > 0).sum(0) >= 2)[both].float().mean()
    assert float(n2) > 0.9, float(n2)


@pytest.mark.skipif(DEV != "cuda", reason="timing on the GPU")
def test_target_is_fast():
    from shapetensor.learn.generator import draw
    s, d = draw((128, 128, 128), seed=1, device=DEV)
    target(s, d); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(3):
        target(s, d)
    torch.cuda.synchronize()
    assert (time.time() - t0) / 3 < 0.3, (time.time() - t0) / 3
