"""Bingham orientation densities (the entity ODF): normaliser and moments against Monte Carlo and asymptotics, moment
matching round trips, the isotropic / girdle / peaked limits, unit mean, the SH rendering against direct quadrature and
the repo's Watson harmonics, mixtures, the 6-D rotation head and the frame loss."""
import math

import pytest
import torch

from shapetensor.learn import bingham as BG
from shapetensor.learn.sh import SHBasis, fibonacci_sphere, real_sh

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def _rand_rot(n, device=DEV):
    q = torch.nn.functional.normalize(torch.randn(n, 4, device=device, dtype=torch.float64), dim=1); w, x, y, z = q.unbind(1)
    return torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1).float()


def _grid(n=400000):
    return fibonacci_sphere(n, DEV).double()


@pytest.mark.parametrize("k1,k2", [(0.0, 0.0), (0.7, 0.2), (3.0, 3.0), (12.0, 1.5), (25.0, 0.0), (40.0, 40.0), (80.0, 20.0)])
def test_normaliser_and_moments_match_quadrature(k1, k2):
    g = _grid(); R = torch.eye(3, device=DEV, dtype=torch.float64)                                  # a = z, b = y, m = x
    e = torch.exp(-k1 * g[:, 0] ** 2 - k2 * g[:, 1] ** 2)
    kap = torch.tensor([k1, k2], device=DEV, dtype=torch.float64)
    assert abs(float(BG.log_normaliser(kap)) - math.log(float(e.mean()))) < 2e-4
    M = BG.moments(R, kap)
    ref = (g.T * e) @ g / e.sum()
    assert torch.allclose(M, ref, atol=2e-4)


@pytest.mark.parametrize("k1,k2", [(150.0, 150.0), (300.0, 120.0), (500.0, 500.0)])
def test_sharp_normaliser_matches_laplace(k1, k2):
    # two antipodal Gaussian caps: c ~ 1 / (2 sqrt(k1 k2)) (1 + O(1/kappa))
    c = math.exp(float(BG.log_normaliser(torch.tensor([k1, k2], dtype=torch.float64))))
    lap = 1 / (2 * math.sqrt(k1 * k2))
    assert abs(c / lap - 1) < 3 * (1 / k1 + 1 / k2)


@pytest.mark.parametrize("k1,k2", [(2.0, 0.5), (30.0, 10.0), (300.0, 300.0), (200.0, 0.0)])
def test_moments_match_samples(k1, k2):
    R = _rand_rot(1)[0]; kap = torch.tensor([k1, k2], device=DEV)
    g = torch.Generator(device=DEV).manual_seed(1)
    x = BG.sample(R, kap, 200000, generator=g).double()
    emp = x.T @ x / x.shape[0]
    assert torch.allclose(BG.moments(R.double(), kap.double()), emp, atol=4e-3)


def test_fit_round_trip_over_a_grid():
    ks = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0]
    pairs = torch.tensor([(a, b) for a in ks for b in ks if b <= a], device=DEV, dtype=torch.float64)
    R = _rand_rot(pairs.shape[0]).double()
    T = BG.moments(R, pairs)
    Rf, kf = BG.fit(T)
    rel = ((kf - pairs).abs() / pairs).max()
    assert float(rel) < 0.02, float(rel)
    assert torch.allclose(torch.linalg.det(Rf), torch.ones(Rf.shape[0], device=DEV, dtype=Rf.dtype), atol=1e-6)
    peaked = pairs[:, 1] > 2.0                                                                      # the mode axis is defined
    cos_m = (Rf[:, :, 2] * R[:, :, 2]).sum(-1).abs()
    assert float(cos_m[peaked].min()) > 0.9999
    aniso = (pairs[:, 0] - pairs[:, 1]) > 2.0                                                       # the a axis is defined
    cos_a = (Rf[:, :, 0] * R[:, :, 0]).sum(-1).abs()
    assert float(cos_a[aniso].min()) > 0.999


def test_limits():
    eye = torch.eye(3, device=DEV)
    _, k = BG.fit(eye[None] / 3)                                                                    # isotropic
    assert float(k.max()) < 1e-2
    n = torch.tensor([0.0, 0.0, 1.0], device=DEV)                                                   # a sheet with normal x: in-plane directions
    _, k = BG.fit((eye - torch.outer(n, n))[None] / 2 * 0.999 + eye[None] / 3 * 0.001)
    assert float(k[0, 0]) > 100 and float(k[0, 1]) < 0.5                                            # a girdle about a = n
    m = torch.nn.functional.normalize(torch.tensor([1.0, 2.0, 3.0], device=DEV), dim=0)
    R, k = BG.fit((torch.outer(m, m) * 0.995 + eye / 3 * 0.005)[None])                              # nearly a line
    assert float(k[0, 1]) > 50 and abs(float(k[0, 0] / k[0, 1]) - 1) < 0.05
    assert float((R[0, :, 2] * m).sum().abs()) > 0.9999


@pytest.mark.parametrize("k1,k2", [(0.0, 0.0), (6.0, 2.0), (40.0, 5.0), (100.0, 100.0)])
def test_pdf_has_unit_mean(k1, k2):
    g = _grid().float(); R = _rand_rot(1)[0]; kap = torch.tensor([k1, k2], device=DEV)
    assert abs(float(BG.pdf(g, R, kap).double().mean()) - 1) < 2e-3


@pytest.mark.parametrize("k1,k2", [(0.0, 0.0), (1.0, 0.3), (8.0, 8.0), (20.0, 2.0), (60.0, 0.0), (250.0, 120.0)])
def test_to_sh_matches_quadrature(k1, k2):
    lmax = 8; R = _rand_rot(3); kap = torch.tensor([[k1, k2]] * 3, device=DEV)
    c = BG.to_sh(R, kap, lmax)
    g = _grid(800000).float(); Y = real_sh(g, lmax).double()
    for i in range(3):
        p = BG.pdf(g, R[i], kap[i]).double()
        ref = (Y.T @ p) * (4 * math.pi / g.shape[0])
        assert float((c[i].double() - ref).norm() / ref.norm()) < 3e-3, (i, float((c[i].double() - ref).norm() / ref.norm()))
    assert abs(float(c[0, 0]) - math.sqrt(4 * math.pi)) < 1e-3


def test_to_sh_reduces_to_the_repo_watson():
    from shapetensor.learn.lobes import lobes_to_sh
    lmax = 8; R = _rand_rot(4)
    for k in (0.5, 4.0, 15.0):
        c = BG.to_sh(R, torch.tensor([[k, k]] * 4, device=DEV), lmax)                             # a line: Watson kappa about m
        w = lobes_to_sh(R[:, :, 2][None, :, :, None], torch.ones(1, 4, 1, device=DEV), torch.full((1, 4, 1), k, device=DEV), lmax)
        # lobes_to_sh sums its J lobes: compare component by component
        for j in range(4):
            wj = lobes_to_sh(R[j:j + 1, :, 2][None, :, :, None], torch.ones(1, 1, 1, device=DEV), torch.full((1, 1, 1), k, device=DEV), lmax)[0, :, 0]
            assert torch.allclose(c[j], wj, atol=2e-3 * float(wj.norm())), (k, j)
        cg = BG.to_sh(R, torch.tensor([[k, 0.0]] * 4, device=DEV), lmax)                          # a girdle: Watson -kappa about a
        for j in range(4):
            wj = lobes_to_sh(R[j:j + 1, :, 0][None, :, :, None], torch.ones(1, 1, 1, device=DEV), torch.full((1, 1, 1), -k, device=DEV), lmax)[0, :, 0]
            assert torch.allclose(cg[j], wj, atol=2e-3 * float(wj.norm())), (k, j)


def test_sh_round_trip_for_smooth_densities():
    lmax = 16; R = _rand_rot(2); kap = torch.tensor([[3.0, 1.0], [6.0, 0.0]], device=DEV)
    c = BG.to_sh(R, kap, lmax); g = fibonacci_sphere(2000, DEV); Y = real_sh(g, lmax)
    for i in range(2):
        f = Y @ c[i]; p = BG.pdf(g, R[i], kap[i])
        assert float((f - p).abs().max() / p.max()) < 1e-2                                          # smooth: order 16 resolves it


def test_mixture_sums_components():
    g = fibonacci_sphere(500, DEV); R = _rand_rot(6).view(2, 3, 3, 3); kap = torch.rand(2, 3, 2, device=DEV) * 20
    kap = torch.stack([kap.max(-1).values, kap.min(-1).values], -1); mass = torch.rand(2, 3, device=DEV)
    mask = torch.tensor([[True, True, False], [True, False, True]], device=DEV)
    f = BG.mixture_pdf(g, R, kap, mass, mask)
    ref = torch.stack([sum(float(mass[b, e]) * BG.pdf(g, R[b, e], kap[b, e]) for e in range(3) if mask[b, e]) for b in range(2)])
    assert torch.allclose(f, ref, rtol=1e-5, atol=1e-6)


def test_rot6d_is_a_proper_rotation():
    R = BG.rot6d_to_matrix(torch.randn(1000, 6, device=DEV))
    I = torch.eye(3, device=DEV).expand(1000, 3, 3)
    assert torch.allclose(R.transpose(1, 2) @ R, I, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(1000, device=DEV), atol=1e-5)


def test_frame_loss_ignores_signs_and_undefined_axes():
    R = _rand_rot(5)
    flip = R * torch.tensor([-1.0, 1.0, -1.0], device=DEV)                                         # columns' signs flipped
    kap = torch.tensor([[50.0, 10.0]] * 5, device=DEV)
    assert float(BG.frame_loss(flip, R, kap).max()) < 1e-6
    swap = torch.stack([R[:, :, 1], R[:, :, 0], -R[:, :, 2]], -1)                                  # a <-> b: meaningless for a peaked line
    assert float(BG.frame_loss(swap, R, torch.tensor([[40.0, 40.0]] * 5, device=DEV)).max()) < 1e-6
    assert float(BG.frame_loss(swap, R, kap).min()) > 0.45                                         # but not for a fan
    other = _rand_rot(5)
    assert float(BG.frame_loss(other, R, kap).mean()) > 0.2
