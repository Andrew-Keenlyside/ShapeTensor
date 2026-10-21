"""Lag-TCA on synthetic feature fields with known populations (CPU)."""
import math

import torch

from shapetensor.lagtca import invariance_direction, lag_offsets, lagtca_field, sobi_window, window_tensor


def _unit(v):
    v = torch.tensor(v, dtype=torch.float32); return v / v.norm()


def _population(d, shape, k=6, seed=0):
    """``k`` feature channels that vary only across direction ``d`` (z, y, x): sums of plane waves with wave vectors
    perpendicular to ``d`` -- invariant along the population's fibres."""
    g = torch.Generator().manual_seed(seed)
    zz, yy, xx = torch.meshgrid(*(torch.arange(n, dtype=torch.float32) for n in shape), indexing="ij")
    P = torch.stack([zz, yy, xx])
    F = torch.zeros(k, *shape)
    for c in range(k):
        for _ in range(3):
            n = torch.randn(3, generator=g); n = n - (n @ d) * d; n = n / n.norm()
            w = 2 * math.pi / (4 + 6 * torch.rand(1, generator=g).item())
            F[c] += torch.randn(1, generator=g) * torch.cos(w * (P * n.view(3, 1, 1, 1)).sum(0) + 6.28 * torch.rand(1, generator=g))
    return F


def _angle(a, b):
    return math.degrees(math.acos(min(1.0, abs(float((a * b).sum())))))


def test_invariance_direction_of_an_axial_decay():
    d = _unit([0.3, 0.5, 0.8]); off = lag_offsets(device="cpu")
    l = off[1:]; across = (l ** 2).sum(1) - (l @ d) ** 2                                       # squared distance across d
    g = torch.cat([torch.ones(1), torch.exp(-0.05 * across)])
    dd, an = invariance_direction(g, off)
    assert _angle(dd, d) < 3.0 and an > 0.5


def _mixed(seed=9):
    d1, d2 = _unit([1.0, 0.0, 0.0]), _unit([0.0, 0.0, 1.0])
    P1, P2 = _population(d1, (40, 40, 40), k=4, seed=2), _population(d2, (40, 40, 40), k=4, seed=3)
    g = torch.Generator().manual_seed(seed)
    M1, M2 = torch.randn(8, 4, generator=g), torch.randn(8, 4, generator=g)                     # both populations in every channel
    return torch.einsum("ck,kzyx->czyx", M1, P1) + torch.einsum("ck,kzyx->czyx", M2, P2), d1, d2


def _smooth_noise(k=8, n=48, sigma=2.0, seed=4):
    import torch.nn.functional as Fn
    x = torch.randn(k, 1, n, n, n, generator=torch.Generator().manual_seed(seed))
    w = torch.exp(-torch.arange(-6, 7).float() ** 2 / (2 * sigma ** 2)); w = w / w.sum()
    for ax in range(3):
        sh = [1, 1, 1]; sh[ax] = 13
        x = Fn.conv3d(x, w.view(1, 1, *sh), padding=tuple(6 if i == ax else 0 for i in range(3)))
    return x[:, 0]


def test_sobi_one_population():
    d = _unit([1.0, 0.2, 0.4]); F = _population(d, (40, 40, 40), seed=1); off = lag_offsets(device="cpu")
    T, _ = window_tensor(F, (4, 4, 4), (32, 32, 32), off, pstride=2)
    n, comps = sobi_window(T, off, keep=6)
    assert n == 1 and _angle(comps[0]["direction"], d) < 8.0, [(c["direction"], c["anisotropy"]) for c in comps]


def test_sobi_two_populations_mixed_in_the_same_channels():
    F, d1, d2 = _mixed(); off = lag_offsets(device="cpu")
    T, _ = window_tensor(F, (4, 4, 4), (32, 32, 32), off, pstride=2)
    n, comps = sobi_window(T, off, keep=8)
    dirs = [c["direction"] for c in comps]
    assert n == 2, [tuple(round(float(v), 2) for v in x) for x in dirs]
    assert min(_angle(x, d1) for x in dirs) < 10.0 and min(_angle(x, d2) for x in dirs) < 10.0


def test_sobi_isotropic_texture_has_no_population():
    off = lag_offsets(device="cpu")
    T, _ = window_tensor(_smooth_noise(), (4, 4, 4), (32, 32, 32), off, pstride=2)
    n, comps = sobi_window(T, off, keep=8)
    assert n == 0, [(c["anisotropy"], c["strength"]) for c in comps]


def test_field_interleaved_populations_stay_apart():
    """Two halves of a block, one population each: each half's positions get its own direction only."""
    d1, d2 = _unit([1.0, 0.0, 0.0]), _unit([0.0, 0.0, 1.0])
    A, B = _population(d1, (64, 40, 40), k=6, seed=2), _population(d2, (64, 40, 40), k=6, seed=3)
    F = torch.cat([A[:, :32], B[:, 32:]], 1)
    dirs, w, rep = lagtca_field(F, window=32, stride=8, keep=8)
    zc = torch.arange(2, 64, 4)
    for half, d in ((zc < 24, d1), (zc >= 40, d2)):
        D = dirs[0][:, half][:, :, 2:-2, 2:-2]; ww = w[0][half][:, 2:-2, 2:-2]
        err = torch.rad2deg(torch.acos((D * d.view(3, 1, 1, 1)).sum(0).abs().clamp(max=1)))
        assert float((ww > 0).float().mean()) > 0.9 and float(err[ww > 0].median()) < 10.0, (rep, float(err.median()))
