"""The chain's stages on small synthetic fields (CPU): the lift peaks along a population and is flat on noise, the
surrogate has no presence, two populations give two lobes, a boundary's sheet does not become a population, the field
at every voxel follows the lattice, and a chunked run equals one pass."""
import math

import numpy as np
import pytest
import torch

from shapetensor.field import Orientations, orientations
from shapetensor.lift import hemisphere, iso_surrogate, lift, presence, shell_lags
from shapetensor.populations import label_populations, momentum, peaks

DEV = "cpu"


def lines(shape, d, n=400, seed=0, width=1.0, amp=40.0):
    """Dark lines along unit direction ``d`` (z, y, x) in a bright block, Gaussian-blurred, plus noise."""
    g = torch.Generator().manual_seed(seed)
    Z, Y, X = shape
    img = torch.full(shape, 160.0)
    coords = torch.stack(torch.meshgrid(*(torch.arange(n_, dtype=torch.float32) for n_ in shape), indexing="ij"), -1)
    d = torch.as_tensor(d, dtype=torch.float32); d = d / d.norm()
    for _ in range(n):
        p0 = torch.rand(3, generator=g) * torch.tensor(shape, dtype=torch.float32)
        rel = coords - p0
        perp = rel - (rel @ d)[..., None] * d
        img -= amp * torch.exp(-0.5 * (perp.norm(dim=-1) / width) ** 2)
    img += 8 * torch.randn(shape, generator=g)
    return img


def angle(a, b):
    a = torch.as_tensor(a, dtype=torch.float32); b = torch.as_tensor(b, dtype=torch.float32)
    return math.degrees(math.acos(min(1.0, abs(float((a / a.norm()) @ (b / b.norm()))))))


def test_shell_lags_are_half_the_shell():
    L = shell_lags(2.5, 6.5)
    n = np.linalg.norm(L, axis=1)
    assert (n >= 2.5).all() and (n <= 6.5).all()
    assert not any((-l).tolist() in L.tolist() for l in L[:50])                                 # one of each pair


def test_lift_peaks_along_a_population_and_not_on_noise():
    grid = hemisphere(96, DEV)
    v = lines((48, 48, 48), (0.0, 0.0, 1.0), n=300, seed=1)
    O = lift(v, grid, r_min=1.5, r_max=4.0, sigma_w=6.0, sigma_m=2.0, step=4, origin=2)
    c = O[:, 6, 6, 6]
    assert angle(grid[int(c.argmax())], (0.0, 0.0, 1.0)) < 15.0
    p = presence(O)[3:-3, 3:-3, 3:-3].median()
    noise = 8 * torch.randn((48, 48, 48), generator=torch.Generator().manual_seed(2)) + 160
    pn = presence(lift(noise, grid, r_min=1.5, r_max=4.0, sigma_w=6.0, sigma_m=2.0))[3:-3, 3:-3, 3:-3].median()
    assert p > 3 * pn


def test_isotropic_surrogate_has_no_presence():
    grid = hemisphere(96, DEV)
    v = lines((48, 48, 48), (1.0, 0.0, 0.0), n=300, seed=3)
    s = iso_surrogate(v)
    assert abs(float(s.std() - (v - v.mean()).std())) < 0.1 * float(v.std())
    p, ps = presence(lift(v, grid, sigma_w=6.0, sigma_m=2.0, r_max=5.5)), presence(lift(s, grid, sigma_w=6.0, sigma_m=2.0, r_max=5.5))
    assert float(p[3:-3, 3:-3, 3:-3].median()) > 3 * float(ps[3:-3, 3:-3, 3:-3].median())


def test_two_populations_two_lobes_and_identities():
    grid = hemisphere(96, DEV)
    v = lines((48, 48, 48), (0.0, 0.0, 1.0), n=200, seed=4) + lines((48, 48, 48), (1.0, 0.0, 0.0), n=200, seed=5) - 160
    O = lift(v, grid, r_min=2.5, r_max=5.5, sigma_w=6.0, sigma_m=2.0)
    d, m = peaks(O - O.min(0, keepdim=True).values, grid, nmax=2, rel=0.3, min_sep_deg=30.0)
    c = (slice(4, 8),) * 3
    assert float((m[1][c] > 0).float().mean()) > 0.8
    e = []
    for dd in (d[0][:, 6, 6, 6], d[1][:, 6, 6, 6]):
        e.append(min(angle(dd, (0, 0, 1)), angle(dd, (1, 0, 0))))
    assert max(e) < 20.0
    labels, table = label_populations(d[:, :, 2:-2, 2:-2, 2:-2], m[:, 2:-2, 2:-2, 2:-2], min_fixels=4)
    assert len(table) >= 2 and table[0]["fixels"] > 50 and table[1]["fixels"] > 50


def test_momentum_bridges_but_does_not_extend():
    grid = torch.cat([hemisphere(48, DEV), torch.tensor([[0.0, 0.0, 1.0]])]); nd = grid.shape[0]
    k = nd - 1                                                                                     # the x axis itself
    O = torch.zeros(nd, 5, 5, 24)
    O[k, 2, 2, 4:10] = 1.0; O[k, 2, 2, 14:20] = 1.0                                               # a gap of 4 cells, both sides hold
    Om = momentum(O, grid, sigma_along=2.0, reach=5)
    assert float(Om[k, 2, 2, 11]) > 0.8                                                            # bridged
    assert float(Om[k, 2, 2, 22]) < 0.05 and float(Om[k, 2, 2, 1]) < 0.05                          # free ends not extended


def test_field_follows_the_lattice_and_chunks_match_one_pass(tmp_path):
    torch.manual_seed(0)
    v = lines((80, 80, 80), (0.0, 0.0, 1.0), n=500, seed=6, width=1.5)
    model = Orientations(scale=10.0, n_dirs=96, device=DEV, iters=100)
    out = model.run(v)
    core = (slice(28, 52),) * 3
    e = torch.rad2deg(torch.acos(out["vec"][(slice(None),) + core][2].abs().clamp(max=1)))
    assert float(e.median()) < 12.0
    assert float((out["N"][core] >= 1).float().mean()) > 0.7
    # chunked: the same lattice, the same floor and response -> the same field on the core
    import zarr
    from shapetensor.volume import run_volume
    zarr.open_array(str(tmp_path / "v.zarr"), mode="w", shape=v.shape, dtype="f4", chunks=(40, 40, 40))[:] = v.numpy()
    run_volume(str(tmp_path / "v.zarr"), str(tmp_path / "o.zarr"), chunk=(40, 40, 40), floor=out["floor"], response=out["response"], device=DEV,
               scale=10.0, n_dirs=96, iters=100, labels=False, log=lambda *a: None)
    g = zarr.open_group(str(tmp_path / "o.zarr"), mode="r")
    vec = torch.as_tensor(np.asarray(g["vec"]).astype(np.float32)).flip(0)
    sel = out["weight"] > 0.5
    ang = torch.rad2deg(torch.acos((vec * out["vec"]).sum(0).abs().clamp(max=1)))
    assert float(ang[sel].median()) < 1.0


def test_orientations_anisotropic_spacing_returns_the_input_shape():
    v = lines((32, 48, 48), (0.0, 0.0, 1.0), n=150, seed=7)
    out = orientations(v, spacing=(1.5, 1.0, 1.0), device=DEV, scale=6.0, n_dirs=64, iters=60)
    assert out["vec"].shape == (3, 32, 48, 48) and out["vecs"].shape[0] == 2


def test_two_dimensional_array():
    g = torch.Generator().manual_seed(8)
    Y = X = 96
    img = torch.full((Y, X), 160.0)
    yy, xx = torch.meshgrid(torch.arange(Y, dtype=torch.float32), torch.arange(X, dtype=torch.float32), indexing="ij")
    for _ in range(60):                                                                             # dark lines along x
        y0 = float(torch.rand(1, generator=g)) * Y
        img -= 40 * torch.exp(-0.5 * ((yy - y0) / 1.2) ** 2)
    img += 8 * torch.randn((Y, X), generator=g)
    out = orientations(img, device=DEV, scale=8.0, n_dirs=96, iters=60)
    assert out["vec"].shape == (3, Y, X) and out["vecs"].shape == (2, 3, Y, X)
    core = (slice(30, 66), slice(30, 66))
    e = torch.rad2deg(torch.acos(out["vec"][2][core].abs().clamp(max=1)))                        # x is the last component
    assert float(e.median()) < 10.0 and float(out["vec"][0].abs().max()) < 1e-6
