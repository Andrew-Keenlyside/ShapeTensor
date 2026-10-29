"""The opt-in HiP-CT acquisition profile (``draw_scene(profile="hipct")``): only HiP-CT nuisances, an isotropic PSF,
X-ray image formation, nearly clean at strength 0; the batch-time routes' HiP-CT subset. CPU, synthetic only."""
import numpy as np
import pytest
import torch

from shapetensor.learn import augment as AU
from shapetensor.learn import generator as GN
from shapetensor.learn.lobe_train import difficulty_gen
from shapetensor.learn.scene import FILL_QUOTA, draw_scene

GEN = difficulty_gen(1.0, {"canvas": 64, "crops_per_scene": 8, "cross_p": 0.5, "p_centre": 0.5, "fill_quota": FILL_QUOTA})
KEPT_ROUTES = {"xray", "texture", "merge", "grain"}


@pytest.mark.parametrize("i", range(8))
def test_hipct_profile_draws_only_hipct_nuisances(i):
    strength = (0.0, 0.3, 0.6, 1.0)[i % 4]
    _, d = draw_scene((32,) * 3, seed=5_100_000 + 8 * i, device="cpu", strength=strength, profile="hipct", **GEN)
    r = d["record"]
    assert r["modality"] == "hipct" and r["physics"] == "xray"
    assert not [k for k in GN.HIPCT_REMOVED if k in r], r
    assert set(r["routes"]) <= KEPT_ROUTES, r["routes"]
    psf = r["psf"]
    assert len(set(psf)) == 1 and GN.HIPCT_PSF[0] - 1e-6 <= psf[0] <= GN.HIPCT_PSF[1] + 0.01
    assert not [k for k in GN.EXTREME_P if k in r], r                                              # the universal path's wider pool: never here
    assert not [f for f in r.get("families", []) if "texture" in f], r                             # nor the oriented-texture family
    dfm = r.get("deform")                                                                          # the deformation of before 29 Oct
    if dfm:
        assert set(dfm.get("warp_kinds") or []) <= {"smooth"} and not any(dfm.get("shift") or [])
        if dfm.get("affine"):
            A = dfm["affine"]; assert max(abs(A[k]) for k in (1, 2, 3, 5, 6, 7)) <= 0.15 + 1e-6
    if strength == 0.0:                                                                            # nearly clean: no artefact, no noise
        assert not [k for k in ("corr_noise", "rings", "seams", "fringes", "local_tomography") if k in r], r
        assert psf[0] == pytest.approx(GN.HIPCT_PSF[0], abs=0.01)


def test_unknown_profile_is_refused():
    with pytest.raises(ValueError):
        draw_scene((32,) * 3, seed=8, device="cpu", strength=0.3, profile="mri", **GEN)


def test_hipct_routes_use_only_the_hipct_subset():
    x = torch.rand(24, 24, 24)
    old = AU.PROFILE
    try:
        AU.PROFILE = "hipct"
        for k in range(40):
            rec = {}
            y = AU.contrast_route(x, np.random.default_rng(k), 1.0, rec=rec)
            assert set(rec) <= {"lut", "gamma", "field", "noise"}, rec
            assert rec.get("lut", ("global", True)) == ("global", True)                           # monotone, global
            assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
    finally:
        AU.PROFILE = old
