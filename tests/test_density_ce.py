"""The normalised density head (``dens_ce``) and the 28 Oct target / augmentation fixes (CPU, seconds)."""
import math

import numpy as np
import pytest
import torch

from shapetensor.learn.model import PopulationNet, fib_hemisphere, CE_GRID


def _net():
    torch.manual_seed(0)
    net = PopulationNet.from_args({"head": "entities", "odirect": 256, "od_sigmas": [2, 4, 6, 8], "st_path": True, "dens_ce": True,
                                   "od_norm": True}).eval()
    dec = net.ent
    with torch.no_grad():                                                                       # non-trivial heads
        for m in (dec.ce_sh[2], dec.ce_floor[2], dec.ce_od):
            m.weight.normal_(0, 0.1); m.bias.normal_(0, 0.1)
    return dec


def _inputs(dec, n=7):
    g = torch.Generator().manual_seed(1)
    p = torch.randn(n, dec.ce_sh[0].in_features - dec.od_lv - 7, generator=g)
    ov = torch.rand(n, dec.od_lv, dec.odirect, generator=g) + 0.1
    a = torch.randn(n, 3, generator=g); a = a / a.norm(dim=1, keepdim=True)
    M = torch.eye(3) - 0.8 * a[:, :, None] * a[:, None, :]                                        # a line's tensor
    j6 = torch.stack([M[:, 0, 0], M[:, 1, 1], M[:, 2, 2], M[:, 0, 1], M[:, 0, 2], M[:, 1, 2]], 1) * 1e-3
    return p, ov, j6, torch.full((n,), 3.0)


def test_density_normalised_positive_antipodal():
    dec = _net(); p, ov, j6, s = _inputs(dec)
    dense = fib_hemisphere(4000)
    with torch.no_grad():
        out = dec.ce_density(p, ov, j6, s, dirs=torch.cat([dense, -dense]))
    G = out["logq_g"].shape[1]
    assert G == CE_GRID
    on_grid = torch.exp(out["logq_g"]).sum(1) * 4 * math.pi / G                                 # exact on the grid
    assert torch.allclose(on_grid, torch.ones_like(on_grid), atol=1e-4)
    q = torch.exp(out["logq"]); half = q.shape[1] // 2
    dense_int = q[:, :half].sum(1) * 4 * math.pi / half                                        # off the grid: a quadrature check
    assert torch.allclose(dense_int, torch.ones_like(dense_int), atol=0.02), dense_int
    assert bool((q > 0).all())
    assert torch.allclose(q[:, :half], q[:, half:], rtol=1e-4)                                   # q(u) = q(-u)


def test_zero_heads_are_the_structure_tensor_reader():
    torch.manual_seed(0)
    dec = PopulationNet.from_args({"head": "entities", "odirect": 256, "od_sigmas": [2, 4, 6, 8], "st_path": True, "dens_ce": True,
                                   "od_norm": True}).eval().ent
    p, ov, j6, s = _inputs(dec, 3)
    with torch.no_grad():
        out = dec.ce_density(p, ov, j6, s)
    a = torch.linalg.eigh(torch.stack([torch.tensor([[r[0], r[3], r[4]], [r[3], r[1], r[5]], [r[4], r[5], r[2]]]) for r in j6]))[1][:, :, 0]
    peak = dec.ce_grid[out["logq_g"].argmax(1)]
    ang = torch.rad2deg(torch.acos((peak * a).sum(1).abs().clamp(max=1)))
    assert bool((ang < 10).all()), ang                                                          # the peak on the tensor's line axis
    assert torch.allclose(out["w0"], torch.full_like(out["w0"], 0.5))


def _rods(axis, N=48, spacing=4.0, radius=1.2):
    c = (N - 1) / 2
    z, y, x = torch.meshgrid(*(torch.arange(N, dtype=torch.float32),) * 3, indexing="ij")
    a = torch.tensor(axis, dtype=torch.float32); a = a / a.norm()
    P = torch.stack([z, y, x], -1) - c; perp = P - (P @ a)[..., None] * a
    b = torch.linalg.svd(torch.eye(3) - a[:, None] * a[None])[0][:, :2]                          # a basis of the plane across the rods
    uv = perp @ b
    return (torch.sqrt((((uv + spacing / 2) % spacing - spacing / 2) ** 2).sum(-1)) < radius).float()


@pytest.mark.parametrize("deg", [60, 90])
def test_crossing_core_visible(deg):
    from shapetensor.learn.entities import orientation_visibility
    th = math.radians(deg); a1 = (0.0, 0.0, 1.0); a2 = (0.0, math.sin(th), math.cos(th))
    img = torch.maximum(_rods(a1), _rods(a2))
    pts = torch.tensor([[23.5, 23.5, 23.5]]); sig = torch.tensor([4.0])
    R = torch.zeros(1, 2, 3, 3)
    for k, ax in enumerate((a1, a2)):                                                           # frames with the line axis in column 2
        ax = torch.tensor(ax); o1 = torch.tensor([1.0, 0, 0]); o2 = torch.linalg.cross(ax, o1)
        R[0, k] = torch.stack([o1, o2 / o2.norm(), ax], 1)
    kappa = torch.tensor([[[17.0, 17.0], [17.0, 17.0]]]); mass = torch.tensor([[0.3, 0.3]])
    v = orientation_visibility(img, pts, sig, R, kappa, mass)
    assert float(v[0]) > 0.9, float(v[0])


def test_missing_lines_no_alias():
    from shapetensor.learn import augment as A
    x = torch.rand(16, 16, 16)
    for seed in range(300):                                                                     # the neighbour copy at the edge overlapped
        A.missing_lines(x, np.random.default_rng(seed), {})
