"""The lag-ODF pathway on small synthetic volumes of fibre tubes (CPU)."""
import math

import torch

from shapetensor.lagodf import LagODF, lag_odf, peaks, presence, response_profile
from shapetensor.separate import hemisphere


def _unit(v):
    v = torch.tensor(v, dtype=torch.float32); return v / v.norm()


def _tubes(d, n=40, count=50, width=1.2, seed=0):
    """Gaussian tubes along ``d`` through random points of an ``n``³ block, summed."""
    g = torch.Generator().manual_seed(seed)
    P = torch.stack(torch.meshgrid(*(torch.arange(n, dtype=torch.float32) for _ in range(3)), indexing="ij"), -1)
    out = torch.zeros(n, n, n)
    for _ in range(count):
        c = torch.rand(3, generator=g) * n
        r = P - c; r = r - (r @ d)[..., None] * d
        out += torch.exp(-(r ** 2).sum(-1) / (2 * width ** 2))
    return out


def _noise(n=40, sd=0.3, seed=1):
    return sd * torch.randn(n, n, n, generator=torch.Generator().manual_seed(seed))


def _angle(a, b):
    return math.degrees(math.acos(min(1.0, abs(float((a * b).sum())))))


def test_lag_odf_peaks_along_a_single_population():
    d = _unit([0.8, 0.3, 0.5]); v = _tubes(d) + _noise(); grid = hemisphere(90)
    O = lag_odf(v, grid, sigma_w=6.0, step=4, origin=18)[:, :2, :2, :2].mean((1, 2, 3))
    assert _angle(grid[O.argmax()], d) < 12.0


def test_isotropic_noise_has_low_presence():
    grid = hemisphere(90)
    O1 = lag_odf(_tubes(_unit([0.0, 0.0, 1.0])) + _noise(), grid, sigma_w=6.0, step=4, origin=10)[:, 1:-1, 1:-1, 1:-1]
    O0 = lag_odf(_noise(sd=1.0) + 0.0, grid, sigma_w=6.0, step=4, origin=10)[:, 1:-1, 1:-1, 1:-1]
    assert float(presence(O1).median()) > 3 * float(presence(O0).median())


def test_two_intermixed_populations_give_two_lobes():
    d1, d2 = _unit([1.0, 0.0, 0.0]), _unit([0.0, 0.5, 0.866])
    m = LagODF(n_dirs=90, sigma_w=6.0, step=4, origin=14, floor=0.0, full=1e-3, min_weight=0.0, edge=0.0, device="cpu", iters=200)
    O1 = m.odf(_tubes(d1, seed=3) + _noise(seed=4))
    core = torch.zeros(O1.shape[1:], dtype=torch.bool); core[1:-1, 1:-1, 1:-1] = True
    m.response = response_profile(O1, d1.view(3, 1, 1, 1).expand(3, *O1.shape[1:]), m.grid, core)
    o = m.populations(_tubes(d1, seed=5) + _tubes(d2, seed=6) + _noise(seed=7), floor=None)
    D, M = o["dirs_cells"][:, :, 1:-1, 1:-1, 1:-1], o["mass"][:, 1:-1, 1:-1, 1:-1]
    two = (M > 0).sum(0) == 2
    assert float(two.float().mean()) >= 0.6, float(two.float().mean())
    e = [min(_angle(D[0][:, i, j, k], d), _angle(D[1][:, i, j, k], d)) for d in (d1, d2)
         for i, j, k in two.nonzero().tolist()]
    assert sorted(e)[len(e) // 2] < 12.0


def test_peaks_refine_off_the_grid():
    grid = hemisphere(160); d = _unit([0.3, 0.6, 0.74])
    f = torch.exp(30.0 * ((grid @ d) ** 2 - 1))[:, None]
    D, M = peaks(f, grid, nmax=2)
    assert _angle(D[0][:, 0], d) < 2.0 and float(M[1, 0]) == 0.0


def test_ring_fraction_separates_a_sheet_from_a_crossing():
    from shapetensor.lagodf import ring_fraction
    grid = hemisphere(160); n = _unit([0.2, 0.3, 0.93])
    ring = torch.exp(-((grid @ n) ** 2) / (2 * 0.1 ** 2))[:, None]                               # mass all around a great circle
    d1, d2 = _unit([1.0, 0.0, 0.0]), _unit([0.0, 0.6, 0.8])
    cross = (torch.exp(30.0 * ((grid @ d1) ** 2 - 1)) + torch.exp(30.0 * ((grid @ d2) ** 2 - 1)))[:, None]
    for f, lo, hi in ((ring, 0.4, 1.0), (cross, 0.0, 0.2)):
        D, M = peaks(f, grid, nmax=2)
        r = float(ring_fraction(f, grid, D, M)[0])
        assert lo <= r <= hi, r


def test_label_populations_keeps_two_crossing_bundles_and_follows_a_bend():
    from shapetensor.lagodf import label_populations
    Z, Y, X = 12, 3, 12
    dirs = torch.zeros(2, 3, Z, Y, X); mass = torch.zeros(2, Z, Y, X)
    # bundle A along z in columns x = 2..3, bending to x for z >= 8 (rows z = 8..9, x = 2..11)
    for z in range(8):
        for x in (2, 3):
            dirs[0, :, z, :, x] = torch.tensor([1.0, 0, 0])[:, None]; mass[0, z, :, x] = 1
    for z in (8, 9):
        for x in range(2, 12):
            t = min(1.0, (x - 2) / 3)                                                               # turns over three cells
            d = torch.tensor([math.cos(t * math.pi / 2), 0, math.sin(t * math.pi / 2)])
            dirs[0, :, z, :, x] = d[:, None]; mass[0, z, :, x] = 1
    # bundle B along x in rows z = 4..5, crossing A at x = 2..3 (there as the second fixel)
    for z in (4, 5):
        for x in range(X):
            k = 1 if x in (2, 3) else 0
            dirs[k, :, z, :, x] = torch.tensor([0, 0, 1.0])[:, None]; mass[k, z, :, x] = 1
    lab, table = label_populations(dirs, mass, max_angle_deg=35.0, min_fixels=4)
    a = {int(v) for v in lab[0, :4, :, 2:4].flatten()} | {int(v) for v in lab[0, 8:10, :, 8:].flatten()}
    b = {int(v) for v in lab[0, 4:6, :, 6:].flatten()} | {int(v) for v in lab[1, 4:6, :, 2:4].flatten()}
    assert len(a) == 1 and len(b) == 1 and a != b and 0 not in a | b, (a, b, table)


def test_isotropic_surrogate_removes_presence():
    from shapetensor.lagodf import iso_surrogate
    grid = hemisphere(90); v = _tubes(_unit([0.0, 0.6, 0.8]), count=80) + _noise()
    p = presence(lag_odf(v, grid, sigma_w=6.0, step=4, origin=10)[:, 1:-1, 1:-1, 1:-1])
    ps = presence(lag_odf(iso_surrogate(v), grid, sigma_w=6.0, step=4, origin=10)[:, 1:-1, 1:-1, 1:-1])
    assert float(ps.median()) < 0.5 * float(p.median()), (float(ps.median()), float(p.median()))
