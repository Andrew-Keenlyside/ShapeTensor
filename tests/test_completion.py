"""Population completion across crossings (CPU): a set is carried across a gap only where it holds on both sides
along its own direction."""
import torch

from shapetensor.completion import _dyad_products, complete


class _Readout:
    def dyad_products(self, F, prepped=True):
        return torch.zeros((6,) + tuple(F.shape[1:]))


class _Cal:
    readout = _Readout()


def _two_bands(n=40, gap=(16, 24)):
    L = torch.zeros(2, n, n, n)
    g0, g1 = gap
    L[0, g0:g1, :, :] = 3.0; L[1, :, :, g0:g1] = 3.0                                   # a band along x, a band along z
    L[0, g0:g1, :, g0:g1] = 0.0; L[1, g0:g1, :, g0:g1] = 0.0                           # both lost where they cross
    d = torch.zeros(2, 3, n, n, n); d[0, 2] = 1.0; d[1, 0] = 1.0
    return L, d


def test_crossing_is_bridged_and_free_end_is_not():
    L, d = _two_bands()
    Lc, prods, rep = complete(_Cal(), torch.zeros(4, *L.shape[1:]), L, d, mode="image", half=10)
    assert float(Lc[0, 20, 20, 20]) > 1.0 and float(Lc[1, 20, 20, 20]) > 1.0            # both carried into the overlap
    assert rep["bridged_voxels"][0] > 0 and len(prods) == 2
    L2 = L.clone(); L2[1, 24:, :, 16:24] = 0.0                                          # the z band ends at the overlap
    Lc2, _, _ = complete(_Cal(), torch.zeros(4, *L.shape[1:]), L2, d, mode="image", half=10)
    assert float(Lc2[1, 20, 20, 20]) == 0.0                                             # a free end is not extended
    assert float(Lc2[0, 20, 20, 20]) > 1.0


def test_continuation_products_point_along_the_set():
    L, d = _two_bands()
    _, prods, _ = complete(_Cal(), torch.zeros(4, *L.shape[1:]), L, d, mode="image_cont", half=10)
    S = prods[1][:, 20, 20, 20]                                                          # bridged voxel of the z band
    ref = _dyad_products(torch.tensor([1.0, 0.0, 0.0]).view(3, 1))[:, 0]
    assert torch.allclose(S / S[:3].sum(), ref / ref[:3].sum(), atol=1e-5)
