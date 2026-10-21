"""Flexible-N populations: the distribution targets and the population-specific momentum (CPU)."""
import math

import torch

from shapetensor.classical import lobes
from shapetensor.populations import kernel_odf, momentum
from shapetensor.separate import hemisphere


def test_kernel_odf_sums_to_the_mass():
    grid = hemisphere(60)
    d = torch.nn.functional.normalize(torch.randn(5, 2, 3), dim=-1)
    w = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 0.0], [0.3, 0.2], [1.0, 0.0]])
    O = kernel_odf(d, w, grid, 12.0)
    assert torch.allclose(O.sum(1), w.sum(1), atol=1e-5)


def _crossing_field(n=32, gap=(12, 20), free_end=False):
    """Two straight populations along x and along z on a lattice, both missing where they cross (the overlap)."""
    grid = hemisphere(60)
    ex, ez = torch.tensor([0.0, 0.0, 1.0]), torch.tensor([1.0, 0.0, 0.0])
    ix, iz = (grid @ ex).abs().argmax(), (grid @ ez).abs().argmax()
    O = torch.zeros(60, n, n, n)
    g0, g1 = gap
    O[ix, g0:g1, :, :] = 1.0; O[iz, :, :, g0:g1] = 1.0
    O[:, g0:g1, :, g0:g1] = 0.0
    if free_end:
        O[iz, g1:, :, g0:g1] = 0.0                                                                   # the z population ends at the overlap
    return O, grid, ix, iz


def test_momentum_bridges_a_crossing_into_two_lobes():
    O, grid, ix, iz = _crossing_field()
    Om = momentum(O, grid, sigma_along=3.0, reach=8, angular_deg=0.0)
    c = (slice(None), 16, 16, 16)
    assert float(Om[ix][16, 16, 16]) > 0.3 and float(Om[iz][16, 16, 16]) > 0.3
    S, mass = lobes(Om[c].reshape(60, 1, 1, 1), grid, nmax=3, rel=0.25, min_sep_deg=25.0)
    assert int((mass.flatten() > 0.1).sum()) == 2


def test_momentum_does_not_extend_a_free_end():
    O, grid, ix, iz = _crossing_field(free_end=True)
    Om = momentum(O, grid, sigma_along=3.0, reach=8, angular_deg=0.0)
    assert float(Om[iz][16, 16, 16]) < 1e-6 and float(Om[iz][24, 16, 16]) < 1e-6                 # nothing carried past the end
    assert float(Om[ix][16, 16, 16]) > 0.3                                                         # the through-going population is bridged


def test_momentum_smooths_a_spike_along_the_population():
    O, grid, ix, iz = _crossing_field()
    O[ix, 14, 5, 5] = 5.0                                                                           # a spike inside the x population
    Om = momentum(O, grid, sigma_along=3.0, reach=8, angular_deg=0.0)
    assert float(Om[ix][14, 5, 5]) < 2.5
