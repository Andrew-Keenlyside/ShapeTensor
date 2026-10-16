"""Graphical abstract of the shapetensor pipeline and its validation verdict (figures/tca_field/graphical_abstract.{png,svg}).

    python scripts/graphical_abstract.py     (shapetensors env; CPU only)
"""
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "tca_field"
PH = Path("/home/andrew/bridge_wmcov/shapetensor/phantoms")
CROP = "/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr"

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
LIGHT = {BLUE: "#e3eefb", ORANGE: "#fdebe3", AQUA: "#e0f5ed", YELLOW: "#fff3d6", MAGENTA: "#fbe8ef", GREEN: "#e3f1e3", VIOLET: "#ebe8fa", RED: "#fbe5e5"}


def box(ax, x, y, w, h, title, lines=(), colour=BLUE, title_size=10.5, size=8.6, fill=None, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02", fc=fill or LIGHT[colour],
                                ec=colour, lw=lw, mutation_aspect=1))
    ax.text(x + w / 2, y + h - 0.03, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=INK)
    for i, ln in enumerate(lines):
        ax.text(x + 0.018, y + h - 0.075 - 0.04 * i, ln, ha="left", va="top", fontsize=size, color=INK)


def arrow(ax, x0, y0, x1, y1, colour=MUTED, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14, lw=lw, color=colour,
                                 shrinkA=2, shrinkB=2, connectionstyle="arc3,rad=0"))


def thumb(ax, img, x, y, w, h, label, cmap="gray", vmin=None, vmax=None):
    ia = ax.inset_axes([x, y, w, h])
    ia.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    ia.set_xticks([]); ia.set_yticks([])
    for s in ia.spines.values():
        s.set_edgecolor(GRID)
    ax.text(x + w / 2, y - 0.012, label, ha="center", va="top", fontsize=7.6, color=MUTED)


def phantom_slice(pid, seed=0, crop=True):
    p = PH / f"{pid}_s{seed}.npz"
    if not p.exists():
        import json
        from shapetensor.phantoms import make
        tex = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/tex_calibrated.json"))["texture"]
        tex["widths"] = [tuple(w) for w in tex["widths"]]
        core = (224, 32, 224) if pid.startswith(("P1", "P2", "P5")) else (256, 32, 192)
        g = make.Grid(vox=0.02, core=core, margin=(32, 32, 32))
        img, _, _ = make.build(pid, seed, make.TextureParams(**tex), g, device="cpu")
        return img[32:-32, img.shape[1] // 2, 32:-32]
    with np.load(p) as z:
        img = z["image"]
        import json
        g = json.loads(str(z["meta"]))["grid"]
    mz, my, mx = g["margin"]; nz, ny, nx = g["core"]
    return img[mz:mz + nz, img.shape[1] // 2, mx:mx + nx] if crop else img[:, img.shape[1] // 2, :]


def real_slice():
    try:
        import zarr
        z0 = zarr.open(CROP, mode="r")["0"]
        return np.asarray(z0[384, 128:640, 128:640])
    except Exception:
        return None


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
    fig = plt.figure(figsize=(17, 10.2), dpi=130)
    fig.patch.set_facecolor(SURF)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    fig.text(0.02, 0.965, "Orientation fields for BRIDGE from multichannel shape features: tensor component analysis under test at 20 µm",
             fontsize=15, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.02, 0.928, "shapetensor: image block → feature engine → channel-resolved tensor K and gradient-direction ODF → read-outs → field; "
             "validated on calibrated phantoms with twins along a step-to-texture axis.", fontsize=10, ha="left", va="top", color=MUTED)

    # ---------------- row 1: the pipeline ----------------
    y0, h = 0.585, 0.30
    # 1 input
    box(ax, 0.02, y0, 0.15, h, "1  Image block", ["HiP-CT, 20 µm, uint8", "+ context margins (3ρ + engine)"], colour=BLUE)
    rs = real_slice()
    if rs is not None:
        thumb(ax, rs, 0.045, y0 + 0.035, 0.10, 0.115, "S-21-46 crop, internal capsule")
    # 2 engines
    box(ax, 0.20, y0, 0.17, h, "2  Feature engine (pluggable)", [], colour=ORANGE)
    for i, (name, note, col) in enumerate((("intensity", "1 ch · the control", BLUE), ("jet", "27 ch · derivative jet, no network", AQUA),
                                           ("anatomix16", "16 ch · UNet output", YELLOW), ("anatomix48", "48→16 ch · UNet layer 58", MAGENTA))):
        yy = y0 + h - 0.075 - i * 0.052
        ax.add_patch(FancyBboxPatch((0.215, yy - 0.040), 0.14, 0.042, boxstyle="round,pad=0.004", fc="white", ec=col, lw=1.2))
        ax.text(0.222, yy - 0.013, name, fontsize=8.6, fontweight="bold", va="center", color=INK)
        ax.text(0.352, yy - 0.030, note, fontsize=6.8, va="center", ha="right", color=MUTED)
    # 3 tensor
    box(ax, 0.40, y0, 0.19, h, "3  Tensor pass (GPU)", ["∇σ=1 of every channel → g (3C)", "K = G_ρ ∗ (g gᵀ)  per site", "   = the structure tensor when C = 1",
                                                        "ODF: |g|²-weighted directions in SH₈", "excision weights (fix A) optional", "sites: block-reduce b = ρ/4"], colour=AQUA)
    # 4 read-outs
    box(ax, 0.62, y0, 0.19, h, "4  Read-outs", ["eigen  →  e₃ of Σ_c K  (ST)", "TCA  →  CP(K): null dir of pieces' Gram", "CSD  →  ring-deconvolved ODF peaks",
                                               "detectors: gm, cl, gap, TCA pieces", "hybrid / gated excision"], colour=YELLOW)
    # 5 field
    box(ax, 0.84, y0, 0.14, h, "5  Field → BRIDGE", ["dir (excised ST), dir_st", "gm, cl, n: boundary", "gap: quality", "fixels: ≤3 dirs + fractions",
                                                     "zarr store, coarse grid", "→ interleaved vertices"], colour=VIOLET)
    for xa, xb in ((0.17, 0.20), (0.37, 0.40), (0.59, 0.62), (0.81, 0.84)):
        arrow(ax, xa, y0 + h / 2, xb, y0 + h / 2)

    # ---------------- row 2: validation ----------------
    y1, h1 = 0.07, 0.45
    box(ax, 0.02, y1, 0.40, h1, "Validation: nine phantom families with twins, calibrated to the real crop", [], colour=BLUE, fill="white")
    ax.text(0.04, y1 + h1 - 0.075, "noise (correlated), texture bands, intensity levels and orientation gap measured label-free on the 20 µm crop;",
            fontsize=8, color=INK)
    ax.text(0.04, y1 + h1 - 0.10, "difficulty axis = step size × fibre-texture amplitude t (real tissue: between t0.25 and t0.5); 2 seeds, ρ 8 / 16 / 32 voxels",
            fontsize=8, color=INK)
    for i, (pid, lab) in enumerate((("P3-40-t0.25", "P3 interface (step 40, t0.25)"), ("P1", "P1 gyrus: GW step + fan"), ("P4-lines-c60-f1", "P4 60° crossing"),
                                   ("P6", "P6 vessels"))):
        s = phantom_slice(pid)
        if s is not None:
            thumb(ax, s, 0.04 + i * 0.095, y1 + 0.17, 0.085, 0.14, lab)
    ax.text(0.04, y1 + 0.13, "scored by distance to the interface (0–0.1 … > 2 mm), boundary-parallel share, N confusion, matched errors,", fontsize=8, color=INK)
    ax.text(0.04, y1 + 0.105, "fraction error, detector precision–recall; nulls per texture level calibrate every threshold.", fontsize=8, color=INK)
    ax.text(0.04, y1 + 0.055, "Controls: intensity ST at matched ρ · fix A excision · no-network jet · de-checkerboarding and tiling checked at the", fontsize=8, color=MUTED)
    ax.text(0.04, y1 + 0.03, "read-out level (≤ 0.1°).", fontsize=8, color=MUTED)

    # verdict tiles
    box(ax, 0.45, y1, 0.53, h1, "What the phantoms showed (20 µm, realistic statistics)", [], colour=ORANGE, fill="white")
    tiles = [
        (RED, "Structure tensor at an interface", "next to a 40-level step: 41 / 13 / 7° (t0.5), 58 / 53 / 42° (t0.25) at ρ 8 / 16 / 32;\nclear tissue 3–6°. A step-to-texture-ratio effect; the shell thickens with ρ."),
        (RED, "Shape-space TCA does not rescue it", "anatomix16: blind to weak texture; anatomix48: 26–30° floor, still carries the step;\njet: step-insensitive but orientation-poor. TCA pieces do not name the step (PR-AUC ≤ 0.4)."),
        (GREEN, "Gradient excision (fix A) does", "7.9 / 2.7 / 1.8° where the plain tensor is 60 / 50 / 37°; no cost in clear tissue;\n0–1% of voxels excised. Detector: coarse gradient magnitude (PR-AUC 0.85–0.96)."),
        (GREEN, "N populations: CSD of the gradient ODF", "rank-free. 90°: all engines; 60°: anatomix48 100% both within 15° (4°), 45°: 61%;\nintensity ODF resolves 90° only. BIC, over-complete CP, TCA refinement: fail."),
    ]
    for i, (col, t, body) in enumerate(tiles):
        yy = y1 + h1 - 0.085 - i * 0.092
        ax.add_patch(FancyBboxPatch((0.465, yy - 0.075), 0.50, 0.082, boxstyle="round,pad=0.004", fc=LIGHT[col], ec=col, lw=1.2))
        ax.text(0.475, yy - 0.006, t, fontsize=9, fontweight="bold", va="top", color=INK)
        ax.text(0.475, yy - 0.032, body, fontsize=7.6, va="top", color=INK, linespacing=1.35)

    fig.text(0.02, 0.03, "Prototype · Andrew Keenlyside · shapetensor 0.1 · docs/TCA_FIELD_REVIEW.md §11, TCA_FIELD_RESULTS.md · phantom-only truth; real data shown for consistency only",
             fontsize=7.8, color=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "graphical_abstract.png", facecolor=SURF)
    fig.savefig(OUT / "graphical_abstract.svg", facecolor=SURF)
    print("wrote", OUT / "graphical_abstract.png")


if __name__ == "__main__":
    main()
