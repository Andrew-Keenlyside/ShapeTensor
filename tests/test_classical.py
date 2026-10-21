"""The classical pathway's pieces (CPU): NNLS by FISTA, and a two-direction mixture deconvolved into two lobes."""
import itertools
import math

import torch

from shapetensor.classical import lobes, nnls_fista
from shapetensor.separate import hemisphere
from shapetensor.tensors import direction_and_gap


def _quartic(u):
    """Features linear in the fourth-order moment of a direction: every monomial u_i u_j u_k u_l (15 distinct)."""
    idx = sorted({tuple(sorted(c)) for c in itertools.product(range(3), repeat=4)})
    return torch.stack([u[:, a] * u[:, b] * u[:, c] * u[:, d] for a, b, c, d in idx], 1)


def _angle(a, b):
    return math.degrees(math.acos(min(1.0, abs(float((a * b).sum())))))


def test_nnls_fista_matches_a_known_solution():
    g = torch.Generator().manual_seed(0)
    A = torch.rand(12, 6, generator=g)
    X = torch.rand(50, 6, generator=g) * (torch.rand(50, 6, generator=g) > 0.5)
    Xh = nnls_fista(A, X @ A.T, iters=3000)
    assert float((Xh - X).abs().max()) < 1e-2


def test_mixture_of_two_directions_gives_two_lobes():
    grid = hemisphere(160)
    D = _quartic(grid)                                                                             # (n_dirs, 15): each direction's signature
    d1 = torch.tensor([[1.0, 0.0, 0.0]]); d2 = torch.nn.functional.normalize(torch.tensor([[math.cos(math.radians(70)), 0.0, math.sin(math.radians(70))]]), dim=1)
    y = 0.5 * _quartic(d1) + 0.5 * _quartic(d2)                                                    # a crossing: the two signatures added
    a = nnls_fista(D.T, y, iters=4000)                                                             # (1, n_dirs)
    S, mass = lobes(a.T[:, :, None, None, None].reshape(160, 1, 1, 1), grid, nmax=3, rel=0.2, min_sep_deg=25.0)
    found = []
    for r in range(3):
        if float(mass[r].flatten()[0]) > 0:
            d, _ = direction_and_gap(S[r]); found.append(d[:, 0, 0, 0])
    assert len(found) == 2, len(found)
    errs = sorted(min(_angle(f, t[0]) for f in found) for t in (d1, d2))
    assert errs[-1] < 10.0, errs
