"""sliceTCA denoising of the shape features before stage 2 (prototype; not in the docs).

    python scripts/prototype_slicetca.py [OUT_DIR]     (default /home/andrew/bridge_wmcov/shapetensor/prototypes/slicetca)

The encoder-2 features of every volume are denoised with ``shapetensor.experimental.slice_tca.denoise_features`` at a
small grid of rank (components of each kind kept) and scale (the lag step of the neighbourhood: 1 -> a 5-voxel patch,
3 -> 13 voxels), and the pipeline's own stage 2 (``apply``) runs on them unchanged. The readers are calibrated on the
denoised calibration bundles (the validation's twelve straight bundles), so they read the features they are given.
Three phantoms at the realistic ("calibrated") texture: the 90-degree bend, the 90-degree band crossing and the
whole-block 90-degree crossing. Scores as prototype_fibre_sets.py; presence as the share of windows whose
orientation-density prominence exceeds the largest on fibre-free texture of the same size (denoised alike).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

sys.path.insert(0, str(Path(__file__).parent))
import prototype_fibre_sets as PF                                                        # noqa: E402  (phantom scores, texture, nulls)
from shapetensor.experimental import fibre_sets as FS                                    # noqa: E402
from shapetensor.experimental.slice_tca import denoise_features                          # noqa: E402
from shapetensor.features import make_engine                                             # noqa: E402
from shapetensor.phantoms import BEND_GRID, Grid, build                                  # noqa: E402
from shapetensor.pipeline import apply, calibrate                                        # noqa: E402
from shapetensor.validate import ANGLES                                                  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/prototypes/slicetca")
DEV = "cuda"
SETTING = "calibrated"
SIGMA, RHO, TRUNC = 1.0, 10.0, 3.0
CONFIGS = [("raw", None, None), ("rank 2, scale 1", (2, 2, 2), 1), ("rank 8, scale 1", (8, 8, 8), 1),
           ("rank 2, scale 3", (2, 2, 2), 3), ("rank 8, scale 3", (8, 8, 8), 3)]
CASES = [("bend", "bend", {}), ("band crossing 90", "band_crossing", {"cross_deg": 90.0}),
         ("whole-block crossing 90", "crossing", {"cross_deg": 90.0})]
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#eda100", "#e87ba4", "#008300"]


def den(F, cfg):
    _, ranks, spacing = cfg
    if ranks is None:
        return F.float(), {}
    return denoise_features(F, ranks=ranks, spacing=spacing)


def strongest(A, grid):
    """The strongest population per (z, x) of the y-mean activation, 0 where none reaches the null."""
    ys = grid.inner[1]
    Ay = A[:, :, ys, :].mean(2)
    if Ay.shape[0] == 0:
        return np.zeros(Ay.shape[1:], int)
    tot = Ay.reshape(Ay.shape[0], -1).sum(1); order = torch.argsort(tot, descending=True)
    rank = torch.empty_like(order); rank[order] = torch.arange(len(order), device=order.device)
    lab = rank[Ay.argmax(0)] + 1
    lab = torch.where(Ay.max(0).values >= 1.0, lab, torch.zeros_like(lab))
    return lab[grid.inner[0], grid.inner[2]].cpu().numpy()


def window_truth(tr, starts, w):
    """Share of each window's (z, x) footprint inside a fascicle."""
    npop = tr["npop"]
    out = np.zeros((len(starts[0]), len(starts[2])))
    for i, z0 in enumerate(starts[0]):
        for j, x0 in enumerate(starts[2]):
            out[i, j] = float((npop[z0:z0 + w[0], x0:x0 + w[2]] > 0).mean())
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(*a):
        print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)

    tex, texture = PF.tex_for(SETTING)
    eng = make_engine("anatomix_enc2", nyquist="none")
    cg = Grid()
    vols = [torch.as_tensor(build("bundle", 100 + k, cg, tex, texture=texture, angle=float(a), device=DEV)[0], device=DEV).float()
            for k, a in enumerate(ANGLES)]
    Fb = [eng(v).cpu() for v in vols]
    nulls_v = {"grid": PF.null_volume(tex, texture, grid=Grid()), "bend": PF.null_volume(tex, texture, grid=BEND_GRID)}
    Fn_ = {k: eng(v).cpu() for k, v in nulls_v.items()}
    phantoms = {}
    for label, name, kw in CASES:
        img, tr, meta = build(name, 0, tex=tex, texture=texture, device=DEV, **kw)
        grid = BEND_GRID if name in ("bend", "band_crossing") else Grid()
        v = torch.as_tensor(img, device=DEV).float()
        tissue = (v > PF.otsu(img)) if name in ("bend", "band_crossing") else torch.ones_like(v, dtype=torch.bool)
        phantoms[label] = {"name": name, "img": img, "tr": tr, "grid": grid, "v": v, "tissue": tissue, "F": eng(v).cpu()}
    log("raw features ready")
    # the channel shown in the images: the most variable raw channel of the band crossing
    Fband = phantoms["band crossing 90"]["F"].float()
    show_c = int(Fband.reshape(Fband.shape[0], -1).var(1).argmax())

    results, images = {}, {}
    for cfg in CONFIGS:
        cname = cfg[0]; tc = time.time()
        Fd = [den(F.to(DEV), cfg)[0] for F in Fb]
        torch.manual_seed(0)
        cal = calibrate(vols, sigma=SIGMA, rho=RHO, truncate=TRUNC, margin=cg.margin[0], features=Fd, device=DEV)
        del Fd; torch.cuda.empty_cache()
        prom_null = {}
        for k, g in (("grid", Grid()), ("bend", BEND_GRID)):
            Fx, _ = den(Fn_[k].to(DEV), cfg)
            D, _, _ = FS.window_densities(cal, cal.dictionary.prep(Fx), g.inner, 32, 8)
            prom_null[k] = float(FS.prominence(D).max()); del D, Fx
        results[cname] = {"calibration": {"tau": round(cal.tau, 3), "readout_overlap": round(cal.readout_overlap, 3)}, "presence_null": prom_null}
        for label, ph in phantoms.items():
            v, tissue, tr, grid = ph["v"], ph["tissue"], ph["tr"], ph["grid"]
            F, rep = den(ph["F"].to(DEV), cfg)
            o = apply(cal, v, tissue, features=F, device=DEV)
            o["activation"] = o["activation"] / max(cal.tau, 1e-12)
            if ph["name"] == "bend":
                sc = PF.score_bend(o, tr, grid)
                sc = sc[0] if isinstance(sc, tuple) else sc
            elif ph["name"] == "band_crossing":
                sc, _ = PF.score_band(o, tr, grid)
            else:
                sc = PF.score_crossing(o, v, tr, grid, SETTING)
            # presence: windows with fibres above the texture null, and fibre-free windows above it
            D, starts, w = FS.window_densities(cal, cal.dictionary.prep(F), grid.inner, 32, 8)
            prom = FS.prominence(D).max(1).values.cpu().numpy()                             # (nz, nx): the y windows' largest
            wt = window_truth(tr, starts, w)
            pn = prom_null["bend" if grid is BEND_GRID else "grid"]
            fib = wt > 0.25; free = wt == 0
            sc["presence: fibre windows above the texture null"] = round(float((prom[fib] > pn).mean()), 3) if fib.any() else None
            sc["presence: fibre-free windows above it"] = round(float((prom[free] > pn).mean()), 3) if free.any() else None
            sc["denoise"] = rep
            results[cname][label] = sc
            ys = grid.inner[1].start + (grid.inner[1].stop - grid.inner[1].start) // 2
            images.setdefault(label, {})[cname] = {"strongest": strongest(o["activation"], grid),
                                                     "feature": F[show_c, :, ys, :][grid.inner[0], grid.inner[2]].float().cpu().numpy()}
            images[label]["_image"] = v[:, grid.inner[1], :].mean(1)[grid.inner[0], grid.inner[2]].cpu().numpy()
            del F, o, D; torch.cuda.empty_cache()
        log(f"{cname} done in {time.time() - tc:.0f}s: " + json.dumps({k: v_ for k, v_ in results[cname].items()}, default=float)[:1500])
    (OUT / "slicetca_scores.json").write_text(json.dumps(results, indent=1, default=float) + "\n")

    # images: the shown feature channel and the strongest population, per phantom and setting
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([(0.12, 0.12, 0.12)] + CAT + ["#9a9890"])
    for kind in ("feature", "strongest"):
        fig, axs = plt.subplots(len(CASES), len(CONFIGS) + 1, figsize=(3.1 * (len(CONFIGS) + 1), 3.3 * len(CASES)), dpi=120)
        for r, (label, _, _) in enumerate(CASES):
            axs[r, 0].imshow(images[label]["_image"], cmap="gray", origin="lower"); axs[r, 0].set_ylabel(label, fontsize=10)
            if r == 0:
                axs[r, 0].set_title("phantom (mean over y)", fontsize=10)
            lo, hi = np.percentile(images[label]["raw"]["feature"], [1, 99])
            for c, cfg in enumerate(CONFIGS, 1):
                im = images[label][cfg[0]]
                if kind == "feature":
                    axs[r, c].imshow(im["feature"], cmap="magma", vmin=lo, vmax=hi, origin="lower")
                else:
                    axs[r, c].imshow(im["strongest"], cmap=cmap, vmin=-0.5, vmax=len(CAT) + 1.5, origin="lower", interpolation="nearest")
                if r == 0:
                    axs[r, c].set_title(cfg[0], fontsize=10)
        for ax in axs.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        what = f"encoder-2 channel {show_c} (mid-y plane)" if kind == "feature" else "strongest population of the current stage 2 (black: none above the null)"
        fig.suptitle(f"sliceTCA denoising, realistic texture: {what}", x=0.01, ha="left", fontsize=11)
        fig.tight_layout(); fig.savefig(OUT / f"slicetca_{kind}.png"); plt.close(fig)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
