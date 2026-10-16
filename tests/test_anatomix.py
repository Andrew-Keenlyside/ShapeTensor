"""GPU checks of the anatomix engine: tiles with discarded margins equal a one-pass block; the [1 2 1] Nyquist
filter and the 8-phase average agree on the structure that matters (the gradient direction). Needs the weights."""
import time

import numpy as np
import torch

from shapetensor.features.anatomix import AnatomixEngine

DEV = "cuda"


def _texture(n=256, seed=0):
    """A 256^3 block of the calibrated homogeneous fibre phantom (P8): the texture the suite runs on."""
    from shapetensor.phantoms import make
    g = make.Grid(vox=0.02, core=(n // 2, n // 2, n // 2), margin=(n // 4, n // 4, n // 4))
    tex = make.TextureParams(amp=100, noise_sd=95.7, noise_sigma=1.1, density=120,
                             widths=((1.0, 0.45), (2.0, 0.3), (3.0, 0.15), (6.0, 0.1)), levels={"wm": 164.0, "gm": 126.0, "csf": 60.0})
    img, _, _ = make.build("P8", seed, tex, g)
    return torch.as_tensor(img, device=DEV).float()


def test_tiled_orientations_match_one_pass():
    """Tiles with a 64-voxel margin (and bf16) against one fp32 pass: the features differ at the 1e-2 level in
    under 1% of values (a tile edge is a half-space perturbation the UNet answers slowly), but the read-out --
    the orientation -- moves by well under a degree (measured 2026-10-16: median 0.06 deg, p90 0.1, max 1.7)."""
    from shapetensor.pca import PCA
    from shapetensor.readout import common, single
    from shapetensor.tensor import gradients, k_sites
    v = _texture(256)
    kw = dict(layer="l58", nyquist="phase", phases=2, margin=64, n_pcs=None, l2=True, out_dtype=torch.float32)
    t0 = time.time(); f1 = AnatomixEngine(tile=256, amp=False, **kw)(v); t1 = time.time() - t0
    t0 = time.time(); f2 = AnatomixEngine(tile=128, amp=True, **kw)(v); t2 = time.time() - t0
    scale = float(f1.abs().amax())
    d = (f1 - f2).abs()[:, 16:-16, 16:-16, 16:-16] / scale
    print(f"one pass fp32 {t1:.1f}s, 128-tiles bf16 {t2:.1f}s; share of values > 1e-2 of peak: {float((d > 1e-2).float().mean()):.2e}", flush=True)
    pca = PCA.fit(f1, 16)
    def dirs(F):
        G = gradients(pca(F), 1.0, 3.0, out_dtype=torch.float32)
        K, _ = k_sites(G, 8.0, 3.0, 2, [128], ((64, 192), (64, 192)))
        Kt = common.unpack(K.reshape(-1, K.shape[-1]), 48)
        return single.eigen(Kt)["dir"], single.tca(Kt)["dir"]
    e1, c1 = dirs(f1); e2, c2 = dirs(f2)
    ae, ac = common.angle_deg(e1, e2), common.angle_deg(c1, c2)
    print(f"orientation change: eigen median {float(ae.median()):.2f} p90 {float(ae.quantile(0.9)):.2f}; "
          f"tca median {float(ac.median()):.2f} p90 {float(ac.quantile(0.9)):.2f} deg", flush=True)
    assert float((d > 1e-2).float().mean()) < 0.02
    assert float(ae.quantile(0.9)) < 0.5 and float(ac.quantile(0.9)) < 0.5


def test_phase_average_vs_nyquist_filter():
    v = _texture(192, seed=1)
    a = AnatomixEngine(layer="l58", nyquist="phase", phases=2, tile=256, margin=64, n_pcs=None, out_dtype=torch.float32)(v)
    b = AnatomixEngine(layer="l58", nyquist="filter", tile=256, margin=64, n_pcs=None, out_dtype=torch.float32)(v)
    c = AnatomixEngine(layer="l58", nyquist="none", tile=256, margin=64, n_pcs=None, out_dtype=torch.float32)(v)
    # Nyquist power along each axis, summed over channels, in the core
    def nyq(f):
        core = f[:, 24:-24, 24:-24, 24:-24]
        out = []
        for ax in (1, 2, 3):
            alt = core - core.roll(1, dims=ax)
            out.append(float((alt ** 2).mean() / (core ** 2).mean()))
        return out
    print("even/odd power ratio (none / phase / filter):", [f"{x:.4f}" for x in nyq(c)], [f"{x:.4f}" for x in nyq(a)], [f"{x:.4f}" for x in nyq(b)])
    cos = (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-6)
    print(f"cosine(phase avg, [1 2 1] filter) median {float(cos.median()):.4f}")
    # the read-out is what matters: TCA directions from the two de-checkerboarded fields
    from shapetensor.pca import PCA
    from shapetensor.readout import common, single
    from shapetensor.tensor import gradients, k_sites
    pca = PCA.fit(a, 16)
    def dirs(F):
        G = gradients(pca(F), 1.0, 3.0, out_dtype=torch.float32)
        K, _ = k_sites(G, 8.0, 3.0, 2, [96], ((48, 144), (48, 144)))
        return single.tca(common.unpack(K.reshape(-1, K.shape[-1]), 48))["dir"]
    ang = common.angle_deg(dirs(a), dirs(b)); ang0 = common.angle_deg(dirs(a), dirs(c))
    print(f"TCA orientation: phase-avg vs [1 2 1] median {float(ang.median()):.2f} p90 {float(ang.quantile(0.9)):.2f}; "
          f"phase-avg vs none median {float(ang0.median()):.2f} p90 {float(ang0.quantile(0.9)):.2f} deg", flush=True)
    assert float(ang.quantile(0.9)) < 3.0


if __name__ == "__main__":
    test_tiled_orientations_match_one_pass(); print("ok tiled")
    test_phase_average_vs_nyquist_filter(); print("ok nyquist")
