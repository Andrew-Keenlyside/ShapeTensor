"""The anatomix engine on the GPU (needs the weights): the feature field's shape and finiteness, and that tiles with
discarded margins give the same orientation read-out as one pass."""
import numpy as np
import pytest
import torch

DEV = "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_enc2_features_and_tiling():
    pytest.importorskip("anatomix")
    from shapetensor.features import make_engine
    from shapetensor.phantoms import Grid, build
    from shapetensor.tensors import layer_tensor
    g = Grid(core=(96, 32, 96), margin=(32, 32, 32))
    img = torch.as_tensor(build("bundle", 0, g, angle=40.0, device=DEV)[0], device=DEV).float()
    one = make_engine("anatomix_enc2", nyquist="none")(img)
    assert one.shape == (32,) + img.shape and torch.isfinite(one).all()
    tiled = make_engine("anatomix_enc2", nyquist="none", tile=64, margin=48)(img)
    # the read-out that matters: the orientation of each channel's structure tensor, inside the core
    inner = (slice(None), slice(40, -40), slice(8, -8), slice(40, -40))
    angs = []
    for c in range(0, 32, 8):
        _, d1, _ = layer_tensor(one[c].float(), 1.0, 4.0); _, d2, _ = layer_tensor(tiled[c].float(), 1.0, 4.0)
        angs.append(float(torch.rad2deg(torch.arccos((d1[inner] * d2[inner]).sum(0).abs().clamp(max=1))).median()))
    assert max(angs) < 2.0
