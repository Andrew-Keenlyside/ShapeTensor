"""Why the activation image is spiky: a spectral diagnosis on a single-population phantom.

The activation a_r(v) is a linear (non-negative) coefficient of the voxel's feature vector. The features are
single rectified filter responses, so they oscillate with the *phase* of the fibre texture (on a fibre / between
fibres), the way one half of a quadrature pair does; nothing in the unmixing squares or exponentiates. The
structure tensor is smooth for the opposite reason: it squares the gradient (an energy) and integrates it over
rho, i.e. over several fibre periods. Here the 1-D power spectrum across the fibres and the relative fluctuation
are measured for the image, the gradient energy, the tensor trace at rho 1/3/5/10, the feature norm, the
activation (dictionary at smooth 1) under box averages 1/3/5/7, the activation from compressed features (sqrt,
log1p, unit-norm) and the compressed activation (sqrt, log1p) -- so the fibre period shows up, or not, in each.

    python scripts/bench/activation_sensitivity.py   -> figures/tca_field/activation_sensitivity.png, results/activation_sensitivity.json
"""
import json, sys, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fn

from shapetensor import DATA_ROOT, st
from shapetensor.features import make_engine
from shapetensor.features.cache import cached_features
from shapetensor.filters import gaussian
from shapetensor.phantoms import make
from shapetensor.readout.layers import SignatureDictionary, unmix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fig_population_layers import ANG, SIGMA, TR, build, teacher, unit  # noqa: E402

FIGS = Path(__file__).resolve().parents[2] / "figures" / "tca_field"
SEQ = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def spectrum(q, axis, inner):
    """Mean 1-D power spectrum of ``q`` along ``axis`` over all lines of the inner region (line means removed,
    Hann window), normalised to unit total power; returns periods (voxels) and power."""
    x = q[inner].float()
    x = x.movedim(axis, -1); x = x - x.mean(-1, keepdim=True)
    n = x.shape[-1]; w = torch.hann_window(n, device=x.device)
    P = (torch.fft.rfft(x * w, dim=-1).abs() ** 2).reshape(-1, n // 2 + 1).mean(0)
    P = P[1:] / P[1:].sum()
    freq = torch.fft.rfftfreq(n)[1:]
    return (1.0 / freq).numpy(), P.cpu().numpy(), freq.numpy()


def fluct(q, inner, axis):
    """Relative fluctuation inside the core: std / mean, and the mean 2-voxel difference along ``axis`` / mean."""
    x = q[inner].float(); mu = x.mean().clamp_min(1e-9)
    dx = (x.narrow(axis, 2, x.shape[axis] - 2) - x.narrow(axis, 0, x.shape[axis] - 2)).abs().mean()
    return float(x.std() / mu), float(dx / mu)


def main():
    t0 = time.time()
    tex = json.load(open(f"{DATA_ROOT}/results/tex_calibrated.json"))["texture"]; tex["widths"] = [tuple(w) for w in tex["widths"]]
    tex = make.TextureParams(**{**tex, "dispersion_deg": 8.0, "pitch_deg": 4.0})
    g = make.Grid(vox=0.02, core=(96, 32, 96), margin=(32, 32, 32)); Z, Y, X = g.shape; y_mid = Y // 2
    inner = (slice(36, Z - 36), slice(12, Y - 12), slice(36, X - 36))
    eng = make_engine("anatomix_enc2", nyquist="none")
    from shapetensor.features.anatomix import load_net; net = load_net("cuda")
    acts = sorted({type(m).__name__ for m in net.modules() if "ReLU" in type(m).__name__}); norms = sorted({type(m).__name__ for m in net.modules() if "Norm" in type(m).__name__})
    blocks, bg = [], []
    for k, ang in enumerate(ANG):
        v, fib, _ = build([(ang, 1.0)], 100 + k, tex, g)
        F = cached_features(eng, v); d, gap = teacher(v)
        w = torch.zeros_like(gap); w[inner] = (gap[inner] > torch.quantile(gap[inner], 0.7)).float() * fib[inner].float()
        wb = torch.zeros_like(gap); wb[inner] = (gap[inner] < torch.quantile(gap[inner], 0.3)).float() * (~fib[inner]).float()
        blocks.append((F, d, w)); bg.append((F, wb))
    dics = {t: SignatureDictionary(smooth=1, transform=t).fit(blocks, background=bg) for t in ("none", "sqrt", "log1p", "unit")}
    del blocks; torch.cuda.empty_cache()
    print(f"[{time.time() - t0:4.0f}s] dictionaries fitted (smooth 1); network nonlinearity {acts}, normalisation {norms}", flush=True)
    # the single population: fibres along unit(0); the across-fibre axis in the x-z plane is the one its direction avoids
    v, fib, _ = build([(0.0, 1.0)], 500, tex, g); F = cached_features(eng, v); t = unit(0.0)
    across = 0 if abs(float(t[0])) < abs(float(t[2])) else 2; along = 2 - across
    mode = t[None]
    Q = {"image": v}
    gx = [gaussian(v, SIGMA, order=o, truncate=TR) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    Q["gradient energy |∇I|² (σ 1, no ρ)"] = sum(c ** 2 for c in gx)
    for rho in (1.0, 3.0, 5.0, 10.0):
        Q[f"tensor trace, ρ {rho:g}"] = gaussian(Q["gradient energy |∇I|² (σ 1, no ρ)"], rho, truncate=TR)
    Xn = dics["none"].standardise(F.float().reshape(F.shape[0], -1).T); Q["feature norm ‖x‖ (standardised)"] = Xn.norm(dim=1).reshape(v.shape); del Xn
    A0, _ = unmix(F, dics["none"], mode); a0 = A0[0]
    Q["activation (smooth 1)"] = a0
    for s in (3, 5, 7):
        Q[f"activation, box {s}"] = Fn.avg_pool3d(a0[None, None], s, stride=1, padding=s // 2)[0, 0]
    for tname, label in (("sqrt", "activation from √features"), ("log1p", "activation from log1p features"), ("unit", "activation from unit-norm features")):
        Q[label] = unmix(F, dics[tname], mode)[0][0]
    Q["√activation"] = a0.clamp_min(0).sqrt(); Q["log1p(10·activation)"] = torch.log1p(10 * a0.clamp_min(0))
    res = {"network": {"nonlinearity": acts, "normalisation": norms}, "fibre_density_per_mm2": tex.density, "mean_fibre_spacing_vox": round(float(1 / np.sqrt(tex.density * (g.vox ** 2))), 2),
           "across_axis": ["z", "y", "x"][across], "quantities": {}}
    spectra = {}
    for name, q in Q.items():
        per, P, fr = spectrum(q, across, inner); per_al, P_al, _ = spectrum(q, along, inner)
        tex_band = (per >= 3) & (per <= 8)
        cv, rg = fluct(q, inner, across); cv_al, rg_al = fluct(q, inner, along)
        res["quantities"][name] = {"peak_period_across_vox": round(float(per[np.argmax(P)]), 1), "power_share_period_3_to_8_across": round(float(P[tex_band].sum()), 3),
                                   "power_share_period_3_to_8_along": round(float(P_al[tex_band].sum()), 3), "cv_core": round(cv, 3), "rel_2vox_diff_across": round(rg, 3), "rel_2vox_diff_along": round(rg_al, 3)}
        spectra[name] = (per, P * fr / (P * fr).sum(), per_al, P_al)              # variance-preserving: power per octave
        print(f"  {name:40s} peak period {per[np.argmax(P)]:5.1f} vox; texture-band power across {P[tex_band].sum():.2f} / along {P_al[tex_band].sum():.2f}; CV {cv:.2f}; rel 2-vox diff across {rg:.2f}", flush=True)
    json.dump(res, open(f"{DATA_ROOT}/results/activation_sensitivity.json", "w"), indent=1)
    # ---- figure: strips, spectra, fluctuation
    strips = ["image", "gradient energy |∇I|² (σ 1, no ρ)", "tensor trace, ρ 3", "tensor trace, ρ 10", "activation (smooth 1)", "activation, box 3", "activation, box 5", "activation, box 7",
              "activation from unit-norm features", "√activation"]
    fig = plt.figure(figsize=(30, 16), dpi=90); gs = fig.add_gridspec(3, 5, height_ratios=[1.0, 1.1, 1.0])
    row = (slice(36, Z - 36), y_mid, slice(36, X - 36))
    for i, name in enumerate(strips):
        ax = fig.add_subplot(gs[i // 5, i % 5]); im = Q[name][row].cpu().numpy()
        ax.imshow(im, cmap="gray" if name == "image" else "Blues", origin="lower", vmin=np.percentile(im, 1), vmax=np.percentile(im, 99), interpolation="nearest")
        r = res["quantities"][name]; ax.set_title(f"{name}\nCV {r['cv_core']:.2f}, texture-band power {r['power_share_period_3_to_8_across']:.2f}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[2, 0:3])
    groups = [("image", "#444444", "-"), ("gradient energy |∇I|² (σ 1, no ρ)", "#e34948", "-"), ("tensor trace, ρ 3", "#e34948", "--"), ("tensor trace, ρ 10", "#e34948", ":"),
              ("activation (smooth 1)", SEQ[5], "-"), ("activation, box 3", SEQ[3], "-"), ("activation, box 5", SEQ[2], "-"), ("activation, box 7", SEQ[0], "-"),
              ("activation from unit-norm features", "#2a9d8f", "-"), ("activation from √features", "#2a9d8f", "--"), ("√activation", "#eda100", "-")]
    for name, c, ls in groups:
        per, P, _, _ = spectra[name]; ax.plot(per, P, color=c, ls=ls, lw=1.8, label=name)
    ax.set_xscale("log"); ax.set_xlim(2, 60); ax.set_ylim(0, None); ax.axvspan(3, 8, color="#eda100", alpha=0.12, lw=0)
    ax.axvline(res["mean_fibre_spacing_vox"], color="#eda100", lw=1.2, ls="--"); ax.text(res["mean_fibre_spacing_vox"] * 1.03, ax.get_ylim()[1] * 0.95, f"mean fibre spacing {res['mean_fibre_spacing_vox']:.1f} vox", fontsize=9, color="#9a6a00", va="top")
    ax.set_xlabel("period across the fibres (voxels)"); ax.set_ylabel("power per octave (share)"); ax.legend(fontsize=8, ncol=2); ax.set_title("power spectrum across the fibres (core region, line means removed; power × frequency, so equal areas per octave)", fontsize=11)
    ax.grid(alpha=0.25, which="both")
    ax = fig.add_subplot(gs[2, 3:5])
    names = [n for n, _, _ in groups]; vals = [res["quantities"][n]["cv_core"] for n in names]; vals2 = [res["quantities"][n]["power_share_period_3_to_8_across"] for n in names]
    yy = np.arange(len(names)); ax.barh(yy - 0.2, vals, 0.4, color=SEQ[3], label="relative fluctuation (std / mean)"); ax.barh(yy + 0.2, vals2, 0.4, color="#eda100", label="power at periods 3–8 vox (share)")
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis(); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.25, axis="x"); ax.set_title("fluctuation in the core", fontsize=11)
    fig.suptitle(f"Why the activation is spiky: a single-population phantom (fibres {tex.density:g}/mm², mean spacing {res['mean_fibre_spacing_vox']:.1f} voxels; network {', '.join(acts)} after {', '.join(norms)}). "
                 "The activation is a linear read of phase-sensitive rectified filters, so it carries the fibre period like the raw gradient energy does; the tensor loses it by integrating over ρ.",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIGS / "activation_sensitivity.png"); plt.close(fig); print("wrote", FIGS / "activation_sensitivity.png")


if __name__ == "__main__":
    main()
