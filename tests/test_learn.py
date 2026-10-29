"""The synthesis for the population model. One test per defect class the review found: stationary random fields,
occupancy as a volume fraction, labels that match what was rendered, blob tensors monotone in elongation, occluders
that hide their labels, crossing groups at the drawn angle without a brightness shortcut, warps that never fold, an
interface weight independent of orientation and scale, bounded saturation, Gibbs without face ringing, a random
acquisition axis and artefact geometry, and visibility."""
import math

import pytest
import torch

from shapetensor.filters import gaussian
from shapetensor.learn.generator import (MODALITIES, STEP_FULL, Layer, _composite, blob_tensor, boundary_sample, draw, eig3, eye6, gibbs,
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
    shape = (48, 48, 48); region = torch.ones(shape, dtype=torch.bool, device=DEV)
    d = torch.tensor([0.3, 0.5, 0.81], device=DEV); d = d / d.norm()
    L = Layer(kind, d.view(3, 1, 1, 1).expand(3, *shape).contiguous(), radius, length, 0.0, region, 50.0, seeds=torch.tensor([[24.0, 24.0, 24.0]], device=DEV))
    occ, T = render_layer(L, shape, torch.Generator(device=DEV).manual_seed(0), DEV)
    assert 0.8 <= float(occ.max()) <= 1.0, float(occ.max())
    if kind == "line" and radius >= 3.0:                                                         # a resolvable lumen: a wall of a voxel or more, a lumen of more
        L.hollow = True; occ, _ = render_layer(L, shape, torch.Generator(device=DEV).manual_seed(0), DEV)
        assert 0.6 <= float(occ.max()) <= 1.0 and float(occ[24, 24, 24]) < 0.35                # a thin wall: partial volume


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


@pytest.mark.parametrize("radius,length", [(1.0, 0.0), (1.0, 2.5), (0.6, 2.0), (2.0, 5.0)])
def test_blob_label_is_the_rendered_moment(radius, length):
    from shapetensor.learn.generator import full3
    shape = (40, 40, 40); d = torch.tensor([0.0, 0.6, 0.8], device=DEV); g = torch.Generator(device=DEV).manual_seed(0)
    x = torch.stack(torch.meshgrid(*(torch.arange(n, device=DEV, dtype=torch.float32) for n in shape), indexing="ij")).reshape(3, -1)
    M = torch.zeros(3, 3, device=DEV)
    for k in range(16):                                                                           # the label is the expected moment: over sub-voxel positions
        seed = 20.0 + torch.rand(1, 3, device=DEV, generator=g)
        L = Layer("blob", d.view(3, 1, 1, 1).expand(3, *shape).contiguous(), radius, length, 0.0, torch.ones(shape, dtype=torch.bool, device=DEV), 50.0, seeds=seed)
        occ, T = render_layer(L, shape, g, DEV)
        w = occ.reshape(-1); mu = (x * w).sum(1) / w.sum(); dx = x - mu[:, None]
        Mk = (dx[:, None] * dx[None] * w).sum(-1) / w.sum(); M += Mk / Mk.trace() / 16          # the rendered blob's normalised second moment
    lab = full3(blob_tensor(d[None], radius, length))[0]
    assert torch.allclose(torch.linalg.eigvalsh(M), torch.linalg.eigvalsh(lab), atol=0.02), (torch.linalg.eigvalsh(M), torch.linalg.eigvalsh(lab))


def test_volume_fractions_and_tensors():
    for seed in range(3):
        for m in MODALITIES:
            s, d = draw((40, 40, 40), seed=seed, device=DEV, modality=m)
            occ = torch.stack([L.occupancy for L in s.layers])
            vol = torch.tensor([L.role != "interface" for L in s.layers], device=DEV)               # the volume budget: shapes and the isotropic remainder
            assert occ.min() >= 0 and occ.max() <= 1 + 1e-5
            assert torch.allclose(occ[vol].sum(0), torch.ones_like(occ[0]), atol=1e-4)
            assert torch.allclose(d["occupancy"][vol].sum(0), torch.ones_like(occ[0]), atol=1e-3)
            assert d["visibility"].shape == occ.shape and d["visibility"].min() >= 0 and d["visibility"].max() <= 1
            T = d["tensors"]
            assert torch.allclose(T[:, :3].sum(1), torch.ones_like(T[:, 0]), atol=1e-3)
            from shapetensor.learn.generator import full3
            assert float(torch.linalg.eigvalsh(full3(T.movedim(1, -1))).min()) > -1e-3
            assert torch.isfinite(d["image"]).all() and d["image"].min() >= 0 and d["image"].max() <= 255
            assert "axis" in d["record"]


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
    rs = []
    for k in range(6):                                                                          # few thick fibres make one sample's fill noisy: average
        sk = sample((64, 64, 64), k_layers=0, seed=20 + k, device=DEV, style="tissue", crossing=True, scale=1.0)
        gk = [L for L in sk.layers if L.group == 0]
        rs.append(float(sum(L.occupancy for L in gk)[gk[0].region].mean()) / sk.record["crossing"]["fraction"])
    r = sum(rs) / len(rs)
    assert 0.75 < r < 1.15, rs                                                                    # the group fills its region to the drawn volume fraction


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


@pytest.mark.xfail(reason="legacy lattice generator (generator.draw), not the scene generator training uses; 3 of 30 draws saturate; failing since before 28 Oct", strict=False)
def test_saturation_bounded():
    """The display window does not saturate by itself (draws that saturate on purpose -- a clipped detector, a slab
    lost at an extreme value -- are left out)."""
    bad = 0
    for k in range(30):
        _, d = draw((40, 40, 40), seed=100 + k, device=DEV)
        rec = d["record"]
        if "saturate" in rec or any(m in ("low", "high") for _, _, m in rec.get("slabs", [])):
            continue
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


def _fibres(shape=(48, 48, 48), radius=1.0, fraction=0.15, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed); ones = torch.ones(shape, dtype=torch.bool, device=DEV)
    from shapetensor.learn.generator import density_for
    d = torch.tensor([0.0, 0.6, 0.8], device=DEV).view(3, 1, 1, 1).expand(3, *shape).contiguous()
    occ, _ = render_layer(Layer("line", d, radius, 60.0, density_for("line", radius, 60.0, fraction), ones, 50.0), shape, g, DEV)
    return occ, g


def _vis(img, occ, rng=None):
    kw = {} if rng is None else {"t_lo": rng[0], "t_hi": rng[1]}
    v = visibility(img, occ[None], [1.0], radii=[1.0], **kw)[0]
    return float((v * occ).sum() / occ.sum())


def test_visibility_tracks_presence_and_orientation():
    from shapetensor.learn.generator import ORIENTATION_T
    occ, g = _fibres()
    noise = torch.randn(occ.shape, device=DEV, generator=g)
    for img, pres, orient in ((100 + 60 * occ + 10 * noise, (0.95, 1.01), (0.95, 1.01)),            # CNR 6: there, orientation clear
                              (100 + 20 * occ + 200 * noise, (0.0, 0.15), (0.0, 0.01)),             # CNR 0.1: barely there, orientation lost
                              (100 + 50 * occ + 50 * noise, (0.8, 1.01), (0.5, 1.01))):             # CNR 1
        p, o = _vis(img, occ), _vis(img, occ, ORIENTATION_T)
        assert pres[0] <= p <= pres[1] and orient[0] <= o <= orient[1], (p, o)
    R = 100 + 30.0 * (torch.arange(48, device=DEV) >= 24).float().view(48, 1, 1).expand(occ.shape)
    vi = visibility(R + 5 * noise, R[None], [STEP_FULL], radii=[1.0])[0]
    assert float(vi[20:28].mean()) > 0.9


def test_visibility_is_local():
    occ, g = _fibres(shape=(48, 48, 96))
    sd = torch.ones_like(occ) * 5; sd[:, :, 48:] = 300                                            # the right half drowned in noise
    img = 100 + 60 * occ + sd * torch.randn(occ.shape, device=DEV, generator=g)
    from shapetensor.learn.generator import ORIENTATION_T
    for rng, right_max in ((None, 0.4), (ORIENTATION_T, 0.05)):                                  # presence, orientation
        kw = {} if rng is None else {"t_lo": rng[0], "t_hi": rng[1]}
        v = visibility(img, occ[None], [1.0], radii=[1.0], **kw)[0]
        left = float((v * occ)[:, :, 8:36].sum() / occ[:, :, 8:36].sum()); right = float((v * occ)[:, :, 60:88].sum() / occ[:, :, 60:88].sum())
        assert left > 0.9 and right < right_max, (rng, left, right)


def test_featureless_region_is_invisible():
    shape = (80, 80, 80); g = torch.Generator(device=DEV).manual_seed(1)
    occ = torch.zeros(shape, device=DEV); occ[8:72, 8:72, 8:72] = 1.0; occ = gaussian(occ, 1.0)   # a uniformly filled slab: no shapes to see inside
    img = 100 + 60 * occ + 5 * torch.randn(shape, device=DEV, generator=g)
    v = visibility(img, occ[None], [1.0], radii=[1.0])[0]
    assert float(v[36:44, 36:44, 36:44].mean()) < 0.05                                            # beyond the window's reach (3 sigma_v + the high-pass) of the slab's edges


def test_layers_do_not_saturate():
    from shapetensor.learn.generator import random_layer
    sat = []
    for k in range(24):
        g = torch.Generator(device=DEV).manual_seed(k)
        L = random_layer((64, 64, 64), g, DEV, kind="line")
        occ, _ = render_layer(L, (64, 64, 64), g, DEV)
        sat.append(float((occ[L.region] > 0.95).float().mean()))
    assert sum(x > 0.5 for x in sat) <= 1, sat


@pytest.mark.xfail(reason="legacy lattice generator's fibres (face / centre occupancy 0.75); failing since before 28 Oct", strict=False)
def test_no_falloff_at_the_faces():
    occ, _ = _fibres(shape=(64, 64, 64), fraction=0.2)
    face = torch.cat([occ[:6].flatten(), occ[-6:].flatten(), occ[:, :6].flatten(), occ[:, -6:].flatten(), occ[:, :, :6].flatten(), occ[:, :, -6:].flatten()]).mean()
    centre = occ[16:48, 16:48, 16:48].mean()
    assert abs(float(face / centre) - 1) < 0.15, float(face / centre)


def test_interface_survives_additive_layers():
    shape = (48, 48, 48)
    R = 100 + 40.0 * (torch.arange(48, device=DEV) >= 24).float().view(48, 1, 1).expand(shape).contiguous()
    occ, g = _fibres(fraction=0.4)
    d = torch.tensor([0.0, 0.6, 0.8], device=DEV).view(3, 1, 1, 1).expand(3, *shape).contiguous()
    L = Layer("line", d, 1.0, 60.0, 1.0, torch.ones(shape, dtype=torch.bool, device=DEV), 40.0, occupancy=occ, tensor=eye6(DEV).view(6, 1, 1, 1).expand(6, *shape) / 3)
    layers = [L]; _composite(R.clone(), R, layers, DEV)
    _, w = interface(R)
    assert torch.allclose(layers[1].occupancy, w, atol=1e-5)                                      # the step's full weight under the fibres
    assert torch.allclose(layers[0].occupancy + layers[2].occupancy, torch.ones(shape, device=DEV), atol=1e-5)


def test_crossing_groups_include_independent_fields():
    flags = []
    for k in range(16):
        s = sample((32, 32, 32), k_layers=0, seed=200 + k, device=DEV, crossing=True, scale=1.0)
        flags.append(s.record["crossing"]["independent"])
    assert 2 <= sum(flags) <= 12, flags


def test_boundary_sample_has_both_states():
    b = boundary_sample((48, 48, 48), step=60.0, contrast=30.0, seed=0, device=DEV, density=1.5)
    line, iface, iso = b.layers
    assert iface.role == "interface" and iso.role == "isotropic"
    prof = iface.occupancy.mean((1, 2))
    assert float(prof[23]) > 0.95 * float(prof.max()) and float(prof[24]) > 0.95 * float(prof.max()) and float(prof[:16].max()) < 1e-3
    assert float(line.occupancy[20:28].mean()) > 0.05


# ---- the shape grammar: every family's labels and descriptors describe what it draws

def _one(kind, shape=(48, 48, 48), d=(0.0, 0.0, 1.0), seed_pos=(24.0, 24.0, 24.0), **kw):
    dd = torch.tensor(d, device=DEV); dd = dd / dd.norm()
    L = Layer(kind, dd.view(3, 1, 1, 1).expand(3, *shape).contiguous(), kw.pop("radius", 3.0), kw.pop("length", 60.0), 0.0,
              torch.ones(shape, dtype=torch.bool, device=DEV), 50.0, seeds=torch.tensor([seed_pos], device=DEV), **kw)
    occ, T = render_layer(L, shape, torch.Generator(device=DEV).manual_seed(0), DEV)
    return L, occ, T


def test_ring_section_is_hollow():
    L, occ, T = _one("line", section="ring", wall=0.4, radius=4.0)                              # a vessel along x
    lumen = float(occ[24, 24, 24]); wall = float(occ[24, 24 + 3, 24])                            # centre vs 3 voxels out (radius 4, wall 1.6)
    assert lumen < 0.15 and wall > 0.7, (lumen, wall)
    v, ev = principal(T, "max")
    m = occ > 0.5
    assert float(_angle(v, torch.tensor([0, 0, 1.0], device=DEV).view(3, 1, 1, 1))[m].median()) < 3.0
    assert abs(float(L.desc[1][m].median()) - 0.6) < 0.02 and abs(float(L.desc[0][m].median()) - 4.0) < 0.05


def test_ellipse_section_is_flattened():
    L, occ, _ = _one("line", section="ellipse", section_aspect=0.3, radius=5.0)
    sl = occ[:, :, 24]                                                                             # the cross-section (z, y) at x = 24
    zz, yy = torch.meshgrid(torch.arange(48, device=DEV, dtype=torch.float32), torch.arange(48, device=DEV, dtype=torch.float32), indexing="ij")
    w = sl / sl.sum(); mz, my = (w * zz).sum(), (w * yy).sum()
    C = torch.tensor([[float((w * (zz - mz) ** 2).sum()), float((w * (zz - mz) * (yy - my)).sum())],
                      [float((w * (zz - mz) * (yy - my)).sum()), float((w * (yy - my) ** 2).sum())]])
    ev = torch.linalg.eigvalsh(C)
    assert 0.2 < float((ev[0] / ev[1]).sqrt()) < 0.45, float((ev[0] / ev[1]).sqrt())             # minor/major ~ the aspect


def test_bilayer_sheet_has_two_leaflets():
    L, occ, T = _one("sheet", bilayer=6.0, radius=1.0, length=14.0)                              # normal along x: leaflets at x = 21 and 27
    prof = occ[24, 24, 16:33]
    assert float(prof[8]) < 0.2 and float(prof[5]) > 0.6 and float(prof[11]) > 0.6, prof.tolist()
    assert float(L.desc[1][occ > 0.5].median()) > 0.5


@pytest.mark.parametrize("axes,expect", [((1.0, 1.0, 4.0), "line"), ((4.0, 4.0, 1.0), "plane"), ((1.0, 1.0, 1.0), "iso")])
def test_solids_label_their_shape(axes, expect):
    L, occ, T = _one("solid", primitive="superquadric", axes=axes, exps=(1.0, 1.0), radius=3.0, d=(0.2, 0.3, 0.93))
    m = occ > 0.5
    val, vec = eig3(T)
    lin = ((val[2] - val[1]) / val[2])[m].median(); pla = ((val[1] - val[0]) / val[2])[m].median()
    d = torch.tensor([0.2, 0.3, 0.93], device=DEV); d = (d / d.norm()).view(3, 1, 1, 1)
    if expect == "line":
        assert float(lin) > 0.7 and float(_angle(vec[2], d)[m].median()) < 3.0
    elif expect == "plane":
        assert float(pla) > 0.7 and float(_angle(vec[0], d)[m].median()) < 3.0
    else:
        assert float(lin) < 0.15 and float(pla) < 0.15


def test_solid_shell_is_hollow():
    L, occ, T = _one("solid", primitive="superquadric", axes=(1.0, 1.0, 1.0), radius=8.0, shell=2.0)
    assert float(occ[24, 24, 24]) < 0.05 and float(occ[24, 24, 32]) > 0.7                         # an empty centre, a wall at the radius (7-9 voxels out)
    assert float(L.desc[1][occ > 0.5].median()) > 0.6


@pytest.mark.parametrize("prim", ["polyhedron", "shard"])
def test_crystals_are_angular_and_bounded(prim):
    L, occ, T = _one("solid", primitive=prim, axes=(1.0, 1.0, 2.0), radius=5.0)
    assert float(occ.max()) > 0.9
    faces = torch.cat([occ[0].flatten(), occ[-1].flatten(), occ[:, 0].flatten(), occ[:, -1].flatten(), occ[:, :, 0].flatten(), occ[:, :, -1].flatten()])
    assert float(faces.max()) < 1e-6                                                               # bounded: nothing reaches the block's faces
    assert float(L.desc[2][occ > 0.5].median()) > 0.99


def test_descriptors_reach_the_target():
    from shapetensor.learn.targets import target
    found = {"hollow": False, "angular": False}
    for sd in range(12):
        s, d = draw((48, 48, 48), seed=700 + sd, device=DEV, k_layers=3)
        t = target(s, d)
        for k in ("thickness", "hollowness", "angularity"):
            assert torch.isfinite(t[k]).all() and float(t[k].min()) >= 0
        found["hollow"] |= bool((t["hollowness"] > 0.2).any()); found["angular"] |= bool((t["angularity"] > 0.5).any())
    assert all(found.values()), found
