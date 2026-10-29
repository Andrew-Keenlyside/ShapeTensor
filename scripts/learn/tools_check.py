"""TorchIO and MONAI against our own routes, on our data: what each does to a synthetic training crop and to a real
HiP-CT crop, and what it costs.

    python scripts/learn/tools_check.py [--out DIR] [--device cpu]

Per operation (MRI motion, k-space spikes, ghosting, per-axis resolution / anisotropy, bias field, a random intensity
curve / histogram shift, smooth-field contrast, Gibbs ringing, labels to image (SynthSeg-style)): the library
version and ours, middle slices side by side (``tools_vs_ours.png``) and the time per 96³ volume (``tools_timing.json``).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HIP = "/hdd/andrew/BRIDGE/20um_cortex_S2146/sub-S2146/micr/sub-S2146_sample-cortex_acq-20um_XPCT.ome.zarr"


def timed(fn, *a):
    t = time.perf_counter(); y = fn(*a); return y, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/scene_generator")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(); dev = a.device; out = Path(a.out)
    import torchio as tio
    import monai.transforms as mt
    from shapetensor.learn import augment as AU
    from shapetensor.learn.generator import gibbs
    from shapetensor.learn.scene import draw_scene
    # inputs: a synthetic training crop (additive, clean) and a real HiP-CT crop
    s, d = draw_scene((96,) * 3, seed=8 * 4331, device=dev, strength=0.1, canvas=128, physics="additive", p_warp=0.0, p_affine=0.0,
                      p_boundary=0.0, p_blend=0.0, p_aniso=0.0, p_scramble=0.0, p_texture=0.0, p_merge=0.0, modality="mixed")
    syn = d["image"].float() / 255
    import zarr
    arr = zarr.open_group(HIP, mode="r")["0"]; o = [336, 336, 336]
    r = np.asarray(arr[o[0]:o[0] + 96, o[1]:o[1] + 96, o[2]:o[2] + 96]).astype(np.float32)
    lo, hi = np.quantile(r, [0.005, 0.995]); real = torch.tensor(np.clip((r - lo) / (hi - lo), 0, 1), device=dev)
    occ = torch.stack([L.occupancy for L in s.layers if hasattr(L, "desc5")])
    rng = lambda k: np.random.default_rng(k)
    tio_img = lambda x: tio.ScalarImage(tensor=x[None].cpu())
    def tio_apply(t, x):
        t_ = t(tio_img(x)); return t_.data[0].float().to(dev)
    def monai_apply(t, x, seed=0):
        t.set_random_state(seed=seed); y = t(x[None]); y = y.as_tensor() if hasattr(y, "as_tensor") else y
        return torch.as_tensor(y, device=dev)[0].float()
    norm = lambda y: ((y - y.min()) / (y.max() - y.min()).clamp_min(1e-6))
    ops = [
        ("MRI motion", lambda x: tio_apply(tio.RandomMotion(degrees=5, translation=4, num_transforms=3), x), lambda x: AU.kspace_motion(x, rng(1), {})),
        ("k-space spikes", lambda x: tio_apply(tio.RandomSpike(num_spikes=2, intensity=(0.5, 1.0)), x), lambda x: AU.kspace_spikes(x, rng(2), {})),
        ("ghosting", lambda x: tio_apply(tio.RandomGhosting(num_ghosts=(3, 6), intensity=(0.5, 1.0)), x),
         lambda x: x + 0.3 * torch.roll(x, 24, 1)),
        ("anisotropy / resolution", lambda x: tio_apply(tio.RandomAnisotropy(downsampling=(2, 5)), x), lambda x: AU.per_axis_resolution(x, rng(3), {})),
        ("bias field", lambda x: tio_apply(tio.RandomBiasField(coefficients=0.6), x),
         lambda x: x * torch.exp(0.4 * AU.smooth_field(x.shape, 40.0, x.device))),
        ("intensity curve", lambda x: monai_apply(mt.RandHistogramShift(num_control_points=(4, 10), prob=1.0), x), lambda x: AU.apply_lut(x, *AU.random_lut(rng(4), False))),
        ("smooth-field contrast", lambda x: monai_apply(mt.RandSmoothFieldAdjustContrast(spatial_size=tuple(x.shape), rand_size=(4, 4, 4), gamma=(0.4, 2.5), prob=1.0), x),
         lambda x: (lambda L1, L2, w: w * AU.apply_lut(x, *L1) + (1 - w) * AU.apply_lut(x, *L2))(AU.random_lut(rng(5), True), AU.random_lut(rng(6), True),
                                                                                              torch.sigmoid(2 * AU.smooth_field(x.shape, 25.0, x.device)))),
        ("Gibbs ringing", lambda x: monai_apply(mt.RandGibbsNoise(prob=1.0, alpha=(0.6, 0.8)), x), lambda x: gibbs(x, [0, 1, 2], 0.3)),
        ("compression (ours only)", None, lambda x: AU.compress(x, rng(7), {})),
    ]
    rows, timing = [], {}
    for name, lib, ours in ops:
        row = []
        for tag, x in (("synthetic", syn), ("real HiP-CT", real)):
            yl, tl = timed(lib, x) if lib is not None else (None, float("nan"))
            yo, to = timed(ours, x)
            row.append((tag, x, yl, yo)); timing.setdefault(name, {})[tag] = {"library_s": tl, "ours_s": to}
        rows.append((name, row)); print(name, json.dumps(timing[name]), flush=True)
    # labels -> image (SynthSeg): TorchIO's RandomLabelsToImage on our population maps vs our texture + shared-intensity routes
    lab = torch.cat([1 - occ.sum(0, keepdim=True).clamp(0, 1), occ]).cpu()
    li = tio.RandomLabelsToImage(label_key="label", used_labels=None, std=[(0.02, 0.15)] * lab.shape[0])
    subj = tio.Subject(label=tio.LabelMap(tensor=lab))
    t0 = time.perf_counter(); synth = li(subj)["image_from_labels"].data[0].float(); tl = time.perf_counter() - t0
    _, d2 = draw_scene((96,) * 3, seed=8 * 4331, device=dev, strength=0.1, canvas=128, physics="additive", p_warp=0.0, p_affine=0.0, p_boundary=0.0,
                       p_blend=0.0, p_aniso=0.0, p_scramble=0.0, p_texture=1.0, p_merge=1.0, modality="mixed")
    timing["labels to image"] = {"synthetic": {"library_s": tl}}
    n = len(rows) + 1
    fig, ax = plt.subplots(n, 6, figsize=(18, 3 * n))
    for i, (name, row) in enumerate(rows):
        for j, (tag, x, yl, yo) in enumerate(row):
            for k, (im, t) in enumerate(((x, f"{tag} input"), (yl, f"{name}: library"), (yo, f"{name}: ours"))):
                a_ = ax[i, 3 * j + k]; a_.axis("off")
                if im is not None:
                    a_.imshow(norm(im)[48].cpu().numpy(), cmap="gray"); a_.set_title(t, fontsize=8)
    ax[-1, 0].imshow(syn[48].cpu().numpy(), cmap="gray"); ax[-1, 0].set_title("synthetic input", fontsize=8)
    ax[-1, 1].imshow(norm(synth)[48].numpy(), cmap="gray"); ax[-1, 1].set_title("labels to image: TorchIO (SynthSeg-style)", fontsize=8)
    ax[-1, 2].imshow(d2["image"][48].float().cpu().numpy(), cmap="gray"); ax[-1, 2].set_title("ours: texture + shared intensities", fontsize=8)
    for a_ in ax[-1]:
        a_.axis("off")
    plt.tight_layout(); fig.savefig(out / "tools_vs_ours.png", dpi=50); plt.close(fig)
    json.dump(timing, open(out / "tools_timing.json", "w"), indent=1); print("wrote", out / "tools_vs_ours.png")


if __name__ == "__main__":
    main()
