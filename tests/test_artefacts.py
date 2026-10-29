"""The artefact generators (CPU)."""
import numpy as np
import torch

from shapetensor import artefacts, st


def flat(shape=(48, 48, 48), level=128):
    return np.full(shape, level, np.uint8)


def test_zero_amplitude_is_identity():
    img = (np.random.default_rng(0).random((24, 24, 24)) * 255).astype(np.uint8)
    for kind in ("streaks", "rings"):
        out, info = artefacts.add(kind, img, 0.0, device="cpu")
        assert out.dtype == np.uint8 and out.shape == img.shape
        assert np.array_equal(out, img) and info["rms"] == 0.0


def test_streaks_lie_in_z_slices_along_their_direction():
    for angle in (0.0, 90.0, 30.0):
        out, info = artefacts.streaks(flat(), 60.0, per_slice=2.0, angle_deg=angle, spread_deg=0.0, device="cpu", seed=1)
        assert info["n"] > 0 and info["rms"] > 0
        v = torch.as_tensor(out).float()
        d = st.structure_tensor(v, sigma=1.0, rho=4.0, truncate=3.0, excise=False, order="zyx", device="cpu")
        S = artefacts.streak_direction(angle)
        dev = (v - 128).abs() > 20                                                       # on the streaks
        cos = (d * torch.as_tensor(S).view(3, 1, 1, 1)).sum(0).abs()
        assert float(cos[dev].median()) > 0.9, (angle, float(cos[dev].median()))


def test_rings_are_constant_along_z():
    out, info = artefacts.rings(flat(), 50.0, device="cpu", seed=2)
    assert info["n"] == 6 and info["rms"] > 0
    assert (out == out[:1]).all()                                                        # the same in every slice
    assert (out != 128).any()


def test_seeded():
    a, _ = artefacts.streaks(flat(), 40.0, device="cpu", seed=3)
    b, _ = artefacts.streaks(flat(), 40.0, device="cpu", seed=3)
    c, _ = artefacts.streaks(flat(), 40.0, device="cpu", seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
