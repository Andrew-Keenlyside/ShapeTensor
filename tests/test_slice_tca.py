"""sliceTCA denoising (experimental): the decomposition recovers a slice-rank structure under noise, and the
feature denoiser keeps the field's shape and reduces noise around a smooth field."""
import pytest

torch = pytest.importorskip("torch")

from shapetensor.experimental.slice_tca import denoise_features, slice_tca


def test_slice_tca_recovers_low_slice_rank():
    g = torch.Generator().manual_seed(0)
    P, C, L = 200, 16, 27
    T0 = (torch.randn(P, 1, 1, generator=g) * torch.randn(1, C, L, generator=g)            # one position-slicing component
          + torch.randn(1, C, 1, generator=g) * torch.randn(P, 1, L, generator=g))         # one channel-slicing component
    noise = 0.3 * torch.randn(P, C, L, generator=g)
    That, info = slice_tca(T0 + noise, ranks=(1, 1, 0), iters=10)
    err_noisy = float(((T0 + noise - T0) ** 2).mean())
    err_fit = float(((That - T0) ** 2).mean())
    assert err_fit < 0.25 * err_noisy
    assert 0.0 < info["explained"] <= 1.0


def test_denoise_features_shape_and_noise():
    g = torch.Generator().manual_seed(1)
    C, Z, Y, X = 4, 24, 16, 24
    z = torch.linspace(0, 6.28, Z)[:, None, None]
    clean = torch.stack([torch.sin(z + k).expand(Z, Y, X) for k in range(C)])              # smooth, shared structure
    noisy = clean + 0.4 * torch.randn(C, Z, Y, X, generator=g)
    out, rep = denoise_features(noisy, ranks=(1, 1, 1), spacing=1, patch=3, window=16, pos_stride=1, iters=6)
    assert out.shape == noisy.shape and rep["uncovered"] == 0
    assert float(((out - clean) ** 2).mean()) < 0.5 * float(((noisy - clean) ** 2).mean())
