"""The populations' density in spherical harmonics, and the local-contrast input."""
import math

import pytest
import torch

from shapetensor.learn.generator import eye6, line_tensor, sheet_tensor
from shapetensor.learn.model import PopulationNet
from shapetensor.learn.sh import SHBasis, bands, density_sh, n_coef, peaks

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _cell(tensors, masses):
    S = torch.stack(tensors).view(1, len(tensors), 6, 1).to(DEV); m = torch.tensor(masses, device=DEV).view(1, -1, 1)
    return S, m


def test_basis_is_orthonormal():
    b = SHBasis(8, 3000, DEV)
    G = b.Y.T @ b.Y * (4 * math.pi / 3000)
    assert float((G - torch.eye(n_coef(8), device=DEV)).abs().max()) < 2e-3


def test_density_is_additive_and_isotropic_only_in_order_zero():
    S, m = _cell([line_tensor(torch.tensor([1.0, 0, 0])), eye6("cpu", ()) / 3], [0.2, 0.7])
    both = density_sh(S, m, 8)
    line = density_sh(S[:, :1], m[:, :1], 8); iso = density_sh(S[:, 1:], m[:, 1:], 8)
    assert torch.allclose(both, line + iso, atol=1e-4)
    for b in bands(8)[1:]:
        assert float(iso[:, b].abs().max()) < 1e-3 * float(iso[:, 0].abs().max())


@pytest.mark.parametrize("deg", [90.0, 60.0])
def test_crossing_gives_two_peaks(deg):
    a = math.radians(deg)
    S, m = _cell([line_tensor(torch.tensor([1.0, 0, 0])), line_tensor(torch.tensor([math.cos(a), math.sin(a), 0.0]))], [0.1, 0.07])
    D, V = peaks(density_sh(S, m, 8)[0], 8, rel=0.3)
    assert int((V[:, 0] > 0).sum()) == 2
    sep = math.degrees(math.acos(abs(float((D[0, :, 0] * D[1, :, 0]).sum()))))
    assert abs(sep - deg) < 8.0, sep


def test_sheet_has_no_line_peak_pair():
    """A sheet is a girdle: its density has no two isolated peaks 90 degrees apart standing out like a crossing's."""
    S, m = _cell([sheet_tensor(torch.tensor([1.0, 0, 0]))], [0.1])
    c = density_sh(S, m, 8)[0]
    D, V = peaks(c, 8, rel=0.3)
    on_girdle = (D[:, 0, 0].abs() < 0.2) | (V[:, 0] == 0)                   # any peak lies in the plane normal to z
    assert bool(on_girdle.all())


def test_local_contrast_input_is_amplitude_invariant():
    torch.manual_seed(0)
    net = PopulationNet(lcn=2.0).to(DEV)
    x = torch.rand(1, 1, 32, 32, 32, device=DEV)
    a, b = net.local_contrast(0.4 * x + 0.3), net.local_contrast(0.1 * x + 0.6)
    assert float((a - b).abs().max()) < 0.05


def test_from_args_round_trip():
    net = PopulationNet.from_args({"widths": [16, 32, 64], "sh_lmax": 8, "lcn": 2.0})
    assert net.sh_lmax == 8 and net.lcn == 2.0 and net.sh.out_channels == n_coef(8) and net.stem.in_channels == 2
