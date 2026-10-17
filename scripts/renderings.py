"""Ways of drawing the shape tensor's single field, on the mid plane of the HiP-CT block: orientation alone (DEC),
DEC with the activation as brightness (linear, compressed, or cut at the null), and DEC with the activation as
opacity or saturation over the grey image, with the excised structure tensor's DEC for reference.

    python scripts/renderings.py              -> docs/img/renderings.png (from results/renderings_block.npz)
    python scripts/renderings.py --recompute  -> runs the block on the GPU first and refreshes the cache

The cache holds the mid-plane slices only (a few hundred kB), so the renderings can be redrawn without the GPU or
the data. The block is calibrated on the 256³ around it (as ``run_volume`` does) with the engine on smaller tiles
(tile 128, margin 32), so that it fits beside another job on the GPU; the features differ from the default tiling
by about 1% rms.
"""
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
CACHE = Path(__file__).resolve().parents[1] / "results" / "renderings_block.npz"


def compute():
    import torch
    import torch.nn.functional as Fn
    from figures import BRIGHT, RHO, SIGMA, _hipct_blocks
    from shapetensor import st
    from shapetensor.pipeline import apply, calibrate
    vc, v = _hipct_blocks(); o = 64; vc = vc[o:o + 256, o:o + 256, o:o + 256].contiguous(); tc, t = vc >= BRIGHT, v >= BRIGHT
    cal = calibrate(vc, tc, sigma=SIGMA, rho=RHO, engine_kwargs={"tile": 128, "margin": 32}); del vc, tc
    out = apply(cal, v, t); d_st = st.structure_tensor(v, sigma=SIGMA, rho=RHO, mask=t, order="zyx")
    S = st.excised_structure_tensor_3d(v, SIGMA, RHO, mask=t); val, _ = st.eig_special_3d(S); gap_st = ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1); del S, val
    m = 24; sl = (slice(m, 192 - m), 96, slice(m, 192 - m)); dsl = (slice(None),) + sl; zs, _, xs = sl
    W = out["weight"]; total = Fn.avg_pool3d(W[None, None], cal.smooth, stride=1, padding=cal.smooth // 2)[0, 0]
    np.savez_compressed(CACHE, image=v[sl].cpu().numpy(), vec=out["vec"][dsl].cpu().numpy(), weight=W[sl].cpu().numpy(), smoothed_weight=total[sl].cpu().numpy(),
                        weight_p99=float(W[zs, :, xs].float().quantile(0.99)), tau=out["tau"], n_populations=int(out["modes"].shape[0]),
                        st=d_st[dsl].cpu().numpy(), st_gap=gap_st[sl].cpu().numpy(), tissue=t[sl].cpu().numpy())
    print(f"cached {CACHE.name}: {int(out['modes'].shape[0])} populations, tau {out['tau']:.3f}")


def render():
    from shapetensor.viz import dec_activation_overlay, dec_rgb
    c = np.load(CACHE); img, d, w, d_st = c["image"], c["vec"], c["weight"], c["st"]
    wn = np.clip(w / c["weight_p99"], 0, 1); above = c["smoothed_weight"] > c["tau"]
    panels = [
        ("orientation only: DEC at every voxel", dec_rgb(d)),
        ("brightness = activation (linear, 99th percentile = 1)", dec_rgb(d, wn)),
        ("brightness = √activation (faint tissue kept visible)", dec_rgb(d, np.sqrt(wn))),
        (f"cut at the null: smoothed activation > τ ({100 * above.mean():.0f}% of the slice)", dec_rgb(d, above.astype(np.float32))),
        ("the image", None),
        ("opacity = activation over the image", dec_activation_overlay(d, w, img, "opacity")),
        ("saturation = activation over the image (grey = unsupported)", dec_activation_overlay(d, w, img, "saturation")),
        ("excised structure tensor: DEC at every voxel", dec_rgb(d_st)),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(17.6, 9.4), dpi=100, constrained_layout=True)
    for ax, (title, rgb) in zip(axs.ravel(), panels):
        if rgb is None:
            ax.imshow(img, cmap="gray", origin="lower")
        else:
            ax.imshow(rgb, origin="lower", interpolation="nearest")
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"The same single field drawn six ways (S-21-46, 20 µm, {int(c['n_populations'])} populations; DEC: R = |x|, G = |y|, B = |z|). "
                 "Top: on black. Bottom: over the image, and the structure tensor for reference.", x=0.01, ha="left", fontsize=11.5)
    fig.savefig(OUT / "renderings.png"); plt.close(fig); print("wrote renderings.png")


if __name__ == "__main__":
    if "--recompute" in sys.argv or not CACHE.exists():
        compute()
    render()
