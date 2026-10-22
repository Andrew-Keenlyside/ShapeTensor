"""The synthesis for the population model. One test per defect class the review found: stationary random fields,
occupancy as a volume fraction, labels that match what was rendered, blob tensors monotone in elongation, occluders
that hide their labels, crossing groups at the drawn angle without a brightness shortcut, warps that never fold, an
interface weight independent of orientation and scale, bounded saturation, Gibbs without face ringing, a random
acquisition axis and artefact geometry, and visibility."""
import math

import pytest
import torch

from shapetensor.filters import gaussian
from shapetensor.learn.generator import (MODALITIES, STEP_FULL, Layer, _composite, blob_tensor, boundary_sample, draw, eye6, gibbs,
                                         interface, nuisance, principal, render_layer, sample, smooth_noise, visibility)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _angle(a, b):
    return torch.rad2deg(torch.acos((a * b).sum(0).abs().clamp(max=1.0)))


def _gradient_st(occ, s_int=2.0):
    """The gradient structure tensor of an occupancy: eigenvalues ascending and eigenvectors ``(..., 3, 3)``."""
    g = [gaussian(occ, 1.0, order=o) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    J = torch.stack([torch.stack([gaussian(g[i] * g[j], s_int) for j in range(3)], -1) for i in range(3)], -2)
    return torch.linalg.eigh(J)


def test_smooth_noise_stationary():
    for sig in (2.0, 20.0, 40.0):
        x = torch.stack([smooth_noise((40, 40, 40), sig, torch.Generator(device=DEV).manual_seed(s), DEV) for s in range(48)])
        corner, face, centre = x[:, 0, 0, 0].var(), x[:, 0, 20, 20].var(), x[:, 20, 20, 20].var()
        for v in (corner, face):
            assert 0.45 < float(v / centre) < 2.2, (sig, float(corner), float(face), float(centre))
        assert 0.5 < float(centre) < 1.8, (sig, float(centre))


@pytest.mark.parametrize("kind,radius,length", [("line", 0.8, 60.0), ("line", 1.5, 60.0), ("line", 3.0, 60.0), ("sheet", 0.8, 12.0),
                                                ("sheet", 2.0, 12.0), ("blob", 1.5, 3.0), ("blob", 3.0, 6.0)])
def test_isolated_shape_is_one_voxel_full(kind, radius, length):
    shape = (48, 48, 48); region = torch.zeros(shape, dtype=torch.bool, device=DEV); region[24, 24, 24] = True
    d = torch.tensor([0.3, 0.5, 0.81], device=DEV); d = d / d.norm()
    L = Layer(kind, d.view(3, 1, 1, 1).expand(3, *shape).contiguous(), radius, length, 0.0, region, 50.0)
    occ, T = render_layer(L, shape, torch.Generator(device=DEV).manual_seed(0), DEV)
    assert 0.8 <= float(occ.max()) <= 1.0, float(occ.max())
    if kind == "line":
        L.hollow = True; occ, _ = render_layer(L, shape, torch.Generator(device=DEV).manual_seed(0), DEV)
        assert 0.75 <= float(occ.max()) <= 1.0 and float(occ[24, 24, 24]) < 0.35


@pytest.mark.parametrize("kind", ["line", "sheet"])
def test_labels_match_rendered_shapes(kind):
    shape = (64, 64, 64); g = torch.Generator(device=DEV).manual_seed(3)
    from shapetensor.learn.generator import direction_field
    fld = direction_field(shape, 10.0, g, DEV)
    L = Layer(kind, fld, 1.2, 40.0 if kind == "line" else 10.0, 0.4 if kind == "line" else 0.05, torch.ones(shape, dtype=torch.bool, device=DEV), 50.0)
    occ, T = render_layer(L, shape, g, DEV)
    ev, V = _gradient_st(occ)
    lab, lev = principal(T, "max" if kind == "line" else "min")
    img_dir = (V[..., 0] if kind == "line" else V[..., 2]).movedim(-1, 0)                      # line: least gradient; sheet: most gradient
    m = occ > 0.5
    m &= (lev[0] > 0.8) if kind == "line" else ((lev[1] - lev[2]) > 0.3)                      # a single shape at the voxel
    m[:6] = m[-6:] = False; m[:, :6] = m[:, -6:] = False; m[:, :, :6] = m[:, :, -6:] = False
    err = _angle(lab, img_dir)[m]
    assert m.sum() > 200 and float(err.median()) < 10.0, (int(m.sum()), float(err.median()))


def test_blob_tensor_monotone_in_elongation():
    d = torch.tensor([[0.0, 0.0, 1.0]], device=DEV)
    aniso = []
    for L in (0.0, 1.0, 2.0, 3.0, 5.0):
        from shapetensor.learn.generator import full3
        ev = torch.linalg.eigvalsh(full3(blob_tensor(d, 1.0, L)))[0].flip(0)
        aniso.append(float(ev[0] - ev[1]))
    assert aniso[0] < 1e-6 and all(b > a for a, b in zip(aniso, aniso[1:])), aniso


def test_volume_fractions_and_tensors():
    for seed in range(3):
        for m in MODALITIES:
            s, d = draw((40, 40, 40), seed=seed, device=DEV, modality=m)
            occ = torch.stack([L.occupancy for L in s.layers])
            assert occ.min() >= 0 and occ.max() <= 1 + 1e-5
            assert torch.allclose(occ.sum(0), torch.ones_like(occ[0]), atol=1e-4)
            assert torch.allclose(d["occupancy"].sum(0), torch.ones_like(occ[0]), atol=1e-3)
            T = d["tensors"]
            assert torch.allclose(T[:, :3].sum(1), torch.ones_like(T[:, 0]), atol=1e-3)
            from shapetensor.learn.generator import full3
            assert float(torch.linalg.eigvalsh(full3(T.movedim(1, -1))).min()) > -1e-3
            assert torch.isfinite(d["image"]).all() and d["image"].min() >= 0 and d["image"].max() <= 255
            assert d["visibility"].shape == (len(s.layers),) and "axis" in d["record"]


def test_occluder_hides_what_is_under_it():
    shape = (40, 40, 40); g = torch.Generator(device=DEV).manual_seed(1); ones = torch.ones(shape, dtype=torch.bool, device=DEV)
    d = torch.tensor([0.0, 0.0, 1.0], device=DEV).view(3, 1, 1, 1).expand(3, *shape).contiguous()
    lines = Layer("line", d, 1.0, 60.0, 2.0, ones, 60.0)
    lines.occupancy, lines.tensor = render_layer(lines, shape, g, DEV)
    occl = Layer("blob", d, 6.0, 6.0, 0.0, ones, 200.0, occluding=True)
    occl.occupancy = torch.zeros(shape, device=DEV); occl.occupancy[10:30, 10:30, 10:30] = 1.0; occl.tensor = eye6(DEV).view(6, 1, 1, 1).expand(6, *shape) / 3
    layers = [lines, occl]; bg = torch.full(shape, 100.0, device=DEV)
    img = _composite(bg, bg.clone(), layers, DEV)
    inside = occl.occupancy > 0.99
    assert float((img[inside] - 200.0).abs().max()) < 1e-3 and float(lines.occupancy[inside].max()) < 1e-6
    assert torch.allclose(torch.stack([L.occupancy for L in layers]).sum(0), torch.ones(shape, device=DEV), atol=1e-5)


def test_crossing_group_angle_and_density():
    s = sample((64, 64, 64), k_layers=0, seed=2, device=DEV, style="tissue", crossing=True, cross_deg=60.0, scale=1.0)
    grp = [L for L in s.layers if L.group == 0]
    assert len(grp) >= 2
    a, b = grp[0], grp[1]
    da, _ = principal(a.tensor, "max")
    m = a.occupancy > 0.5
    err = (_angle(da, b.field)[m] - 60.0).abs()
    assert m.sum() > 100 and float(err.median()) < 8.0, float(err.median())
    tot = sum(L.occupancy for L in grp)
    assert float(tot.max()) <= 1.0 + 1e-5
    phi = s.record["crossing"]["fraction"]; region = grp[0].region
    r = float(tot[region].mean()) / phi
    assert 0.75 < r < 1.25, r                                                                    # the group fills its region to the drawn volume fraction, as one population would


def test_warp_never_folds():
    s = sample((48, 48, 48), k_layers=1, seed=4, device=DEV, crossing=False)
    for k in range(10):
        d = nuisance(s, seed=k, device=DEV, modality="mixed", warp=True, strength=2.0)
        assert d["record"]["warp"][2] > 0.2


def test_interface_independent_of_orientation():
    shape = (48, 48, 48); c = torch.stack(torch.meshgrid(*(torch.arange(n, device=DEV, dtype=torch.float32) - 23.5 for n in shape), indexing="ij"))
    for delta, want in ((7.5, 0.5), (30.0, 1.0)):
        peaks = []
        for n in (torch.tensor([1.0, 0, 0]), torch.tensor([1.0, 1.0, 1.0]) / math.sqrt(3), torch.tensor([1.0, 0.5, 0.0]) / math.sqrt(1.25)):
            n = n.to(DEV); R = 100 + delta * ((c * n.view(3, 1, 1, 1)).sum(0) > 0).float()
            normal, w = interface(R)
            core = (slice(12, 36),) * 3
            peaks.append(float(w[core].max()))
            sel = w[core] > 0.3
            assert float(_angle(normal[(slice(None),) + core], n.view(3, 1, 1, 1))[sel].median()) < 6.0
        assert all(abs(p - want) < 0.08 for p in peaks), (delta, peaks)


def test_saturation_bounded():
    bad = 0
    for k in range(30):
        _, d = draw((40, 40, 40), seed=100 + k, device=DEV)
        bad += float(((d["image"] == 0) | (d["image"] == 255)).float().mean()) > 0.10
    assert bad <= 1, bad


def test_gibbs_rings_edges_not_faces():
    x = torch.linspace(0, 100, 64, device=DEV).view(1, 1, 64).expand(16, 16, 64).contiguous()
    out = gibbs(x, [2], 0.3)
    assert float((out - x).abs()[:, :, :3].max()) < 2.0 and float((out - x).abs()[:, :, -3:].max()) < 2.0
    step = (torch.arange(64, device=DEV) >= 32).float().view(1, 1, 64).expand(16, 16, 64).contiguous() * 100
    assert float(gibbs(step, [2], 0.3).max()) > 105.0                                         # it does ring at an edge


def test_acquisition_axis_and_artefact_geometry():
    s = sample((40, 40, 40), k_layers=1, seed=5, device=DEV, crossing=False)
    axes, ring_angles, streak_angles = set(), [], []
    for k in range(40):
        rec = nuisance(s, seed=k, device=DEV, modality="hipct")["record"]
        axes.add(rec["axis"])
        if "rings" in rec:
            cy, cx = rec["rings"]["centre"]; ring_angles.append(math.atan2(cy - 19.5, cx - 19.5))
        if "streaks" in rec:
            streak_angles.append(rec["streaks"]["angle_deg"])
    assert axes == {0, 1, 2}
    assert len(ring_angles) >= 8 and any(a > 0.5 for a in ring_angles) and any(a < -0.5 for a in ring_angles) and any(abs(a) > 2.0 for a in ring_angles)
    assert len(streak_angles) >= 5 and max(streak_angles) - min(streak_angles) > 60


def test_visibility_tracks_contrast_to_noise():
    shape = (40, 40, 40); g = torch.Generator(device=DEV).manual_seed(0); ones = torch.ones(shape, dtype=torch.bool, device=DEV)
    d = torch.tensor([0.0, 0.6, 0.8], device=DEV).view(3, 1, 1, 1).expand(3, *shape).contiguous()
    occ, _ = render_layer(Layer("line", d, 1.0, 60.0, 1.5, ones, 50.0), shape, g, DEV)
    noise = torch.randn(shape, device=DEV, generator=g)
    strong = visibility(100 + 60 * occ + 10 * noise, occ[None], [1.0])
    faint = visibility(100 + 15 * occ + 300 * noise, occ[None], [1.0])
    assert float(strong[0]) > 0.95 and float(faint[0]) < 0.2, (float(strong[0]), float(faint[0]))
    R = 100 + 30.0 * (torch.arange(40, device=DEV) >= 20).float().view(40, 1, 1).expand(shape)
    assert float(visibility(R + 5 * noise, R[None], [STEP_FULL])[0]) > 0.95


def test_boundary_sample_has_both_states():
    b = boundary_sample((48, 48, 48), step=60.0, contrast=30.0, seed=0, device=DEV, density=1.5)
    line, iface, iso = b.layers
    assert iface.role == "interface" and iso.role == "isotropic"
    prof = iface.occupancy.mean((1, 2))
    assert float(prof[23]) > 0.95 * float(prof.max()) and float(prof[24]) > 0.95 * float(prof.max()) and float(prof[:16].max()) < 1e-3
    assert float(line.occupancy[20:28].mean()) > 0.05
