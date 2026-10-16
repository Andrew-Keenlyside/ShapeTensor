"""Tables and figures from the suite's results: ``REPORT.md`` plus PNGs under ``DATA_ROOT/figures/<tag>/``.

Every number is the mean over seeds of a per-seed statistic (median or share), with the min-max over seeds in
brackets when there is more than one seed. Arms are drawn in a fixed colour order (the same arm has the same colour
in every figure); error maps use one sequential hue, light = small error.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import DATA_ROOT
from .score import BANDS_MM, band_name
from .suite import RESULTS_DIR

FIG_DIR = Path(DATA_ROOT) / "figures"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BAND_NAMES = [band_name(lo, hi) for lo, hi in BANDS_MM]
#: arm = (kind, engine, readout) -> label; the order fixes the colour slot
ARMS = [
    ("single", "intensity", "eigen", "intensity ST (eigen)"),
    ("single", "jet", "tca", "jet TCA R3"),
    ("single", "anatomix16", "tca", "anatomix16 TCA R3"),
    ("single", "anatomix48", "tca", "anatomix48 TCA R3"),
    ("single", "anatomix48", "eigen", "anatomix48 eigen"),
    ("hybrid", "anatomix48", "blend_cl", "hybrid blend (cl)"),
    ("hybrid", "anatomix48", "excise_all", "excision everywhere (fix A)"),
    ("hybrid", "anatomix48", "excise_cl", "excision gated by cl"),
]   # eight arms: one categorical colour slot each, never cycled (the tables list every read-out)


def load(tag: str, kind: str) -> list[dict]:
    d = RESULTS_DIR / tag / kind
    if not d.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


def _fmt(vals) -> str:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "–"
    if len(v) == 1:
        return f"{v[0]:.1f}"
    return f"{np.mean(v):.1f} [{min(v):.1f}–{max(v):.1f}]"


def _fmt_pct(vals) -> str:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "–"
    if len(v) == 1:
        return f"{100 * v[0]:.0f}%"
    return f"{100 * np.mean(v):.0f}% [{100 * min(v):.0f}–{100 * max(v):.0f}]"


def _get(d: dict, *path, default=float("nan")):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def single_table(tag: str, pid: str, stat: str = "median") -> str:
    rows = defaultdict(list)                                           # (engine, readout, rho) -> per-seed score dicts
    for r in load(tag, "single"):
        if r["pid"] != pid:
            continue
        for name, sc in r["scores"].items():
            rows[(r["engine"], name, r["rho"])].append(sc["bands"])
    for r in load(tag, "hybrid"):
        if r["pid"] != pid:
            continue
        for name, sc in r["scores"].items():
            rows[(r["engine"], name, r["rho"])].append(sc["bands"])
    if not rows:
        return ""
    cols = ["nointerface"] + BAND_NAMES + ["grey_matter"]
    out = [f"### {pid}: angle to the true fibre ({stat}, deg) by distance to the interface\n",
           "| engine | read-out | ρ | " + " | ".join(cols) + " |", "|---|---|---|" + "---|" * len(cols)]
    for (eng, name, rho), lst in sorted(rows.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        cells = [_fmt([_get(b, c, stat) for b in lst]) for c in cols]
        out.append(f"| {eng} | {name} | {rho:g} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def parallel_table(tag: str, pid: str) -> str:
    rows = defaultdict(list)
    truth = []
    for kind in ("single", "hybrid"):
        for r in load(tag, kind):
            if r["pid"] != pid:
                continue
            for name, sc in r["scores"].items():
                rows[(r["engine"], name, r["rho"])].append(sc["parallel"])
            if kind == "single" and r["engine"] == "intensity":
                truth.append(r["parallel_truth"])
    if not rows:
        return ""
    out = [f"### {pid}: share of directions within 30° of the interface's tangent plane (boundary-parallel)\n",
           "| engine | read-out | ρ | " + " | ".join(BAND_NAMES) + " |", "|---|---|---|" + "---|" * len(BAND_NAMES)]
    if truth:
        out.append("| *truth* | | | " + " | ".join(_fmt_pct([_get(t, b, "share") for t in truth]) for b in BAND_NAMES) + " |")
    for (eng, name, rho), lst in sorted(rows.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        out.append(f"| {eng} | {name} | {rho:g} | " + " | ".join(_fmt_pct([_get(p, b, "share") for p in lst]) for b in BAND_NAMES) + " |")
    return "\n".join(out) + "\n"


def detector_table(tag: str, pid: str) -> str:
    rows = defaultdict(list)
    for r in load(tag, "single"):
        if r["pid"] != pid:
            continue
        for name, d in r["detectors"].items():
            rows[(r["engine"], name, r["rho"])].append(d)
    if not rows:
        return ""
    out = [f"### {pid}: boundary detectors, precision-recall AUC for 'within 0.5 mm of the interface'\n",
           "| engine | detector | ρ | PR-AUC | p50 (WM) | p99 (WM) |", "|---|---|---|---|---|---|"]
    for (eng, name, rho), lst in sorted(rows.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        out.append(f"| {eng} | {name} | {rho:g} | {_fmt([_get(d, 'pr_auc') for d in lst])} | "
                   f"{_fmt([_get(d, 'quantiles_wm', '50') for d in lst])} | {_fmt([_get(d, 'quantiles_wm', '99') for d in lst])} |")
    return "\n".join(out) + "\n"


def multi_table(tag: str, pid: str) -> str:
    rows = defaultdict(list)
    for r in load(tag, "multi"):
        if r["pid"] != pid:
            continue
        for name, m in r["methods"].items():
            rows[(r["engine"], name, r["rho"])].append(m)
    if not rows:
        return ""
    out = [f"### {pid}: N populations\n",
           "| engine | method | ρ | N correct | found 2 | both < 15° | err 1 / 2 (median) | frac abs err | false 2nd (single) | false any (null) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for (eng, name, rho), lst in sorted(rows.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        nc = [np.mean(list(m["n_correct_share"].values())) if m.get("n_correct_share") else float("nan") for m in lst]
        out.append(f"| {eng} | {name} | {rho:g} | {_fmt_pct(nc)} | {_fmt_pct([_get(m, 'two', 'found2_share') for m in lst])} | "
                   f"{_fmt_pct([_get(m, 'two', 'both_within_15_share') for m in lst])} | "
                   f"{_fmt([_get(m, 'two', 'err1', 'median') for m in lst])} / {_fmt([_get(m, 'two', 'err2', 'median') for m in lst])} | "
                   f"{_fmt([_get(m, 'two', 'frac_abs_err', 'median') for m in lst])} | "
                   f"{_fmt_pct([_get(m, 'single', 'false_second_share') for m in lst])} | "
                   f"{_fmt_pct([_get(m, 'null', 'false_any_share') for m in lst])} |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------------

def _style(plt):
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white"})


def _arm_series(tag: str, pid: str, rho: float, stat: str = "median") -> dict:
    """arm label -> {band: [per-seed values]}"""
    by = {}
    for kind in ("single", "hybrid"):
        for r in load(tag, kind):
            if r["pid"] != pid or r["rho"] != rho:
                continue
            for name, sc in r["scores"].items():
                for k, eng, rd, label in ARMS:
                    if k == kind and eng == r["engine"] and rd == name:
                        by.setdefault(label, defaultdict(list))
                        for b in ["nointerface"] + BAND_NAMES:
                            v = _get(sc["bands"], b, stat)
                            if np.isfinite(v):
                                by[label][b].append(v)
    return by


def fig_error_vs_band(tag: str, pid: str, rho: float, out: Path, stat: str = "median"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    by = _arm_series(tag, pid, rho, stat)
    if not by:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=130)
    xs = np.arange(len(BAND_NAMES))
    for i, (k, eng, rd, label) in enumerate(ARMS):
        if label not in by:
            continue
        ys = [np.mean(by[label][b]) if by[label][b] else np.nan for b in BAND_NAMES]
        lo = [min(by[label][b]) if by[label][b] else np.nan for b in BAND_NAMES]
        hi = [max(by[label][b]) if by[label][b] else np.nan for b in BAND_NAMES]
        ax.plot(xs, ys, color=CAT[i % len(CAT)], lw=2, marker="o", ms=5, label=label)
        ax.fill_between(xs, lo, hi, color=CAT[i % len(CAT)], alpha=0.12, lw=0)
    ax.set_xticks(xs); ax.set_xticklabels(BAND_NAMES)
    ax.set_xlabel("distance to the interface"); ax.set_ylabel(f"angle to the true fibre ({stat}, deg)")
    ax.set_ylim(0, 90); ax.set_title(f"{pid}, ρ {rho:g} vox: error by distance band (band = min–max over seeds)", loc="left", fontsize=10.5)
    ax.legend(fontsize=8.5, ncol=2)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_rho_sweep(tag: str, pid: str, out: Path, bands=("0-0.1mm", "0.25-0.5mm", ">2mm", "nointerface")):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    rhos = sorted({r["rho"] for r in load(tag, "single") if r["pid"] == pid})
    if not rhos:
        return None
    present = [b for b in bands if any(_arm_series(tag, pid, r).get(l, {}).get(b) for r in rhos for l in [a[3] for a in ARMS])]
    if not present:
        return None
    fig, axs = plt.subplots(1, len(present), figsize=(4.2 * len(present), 4.2), dpi=130, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, b in zip(axs, present):
        for i, (k, eng, rd, label) in enumerate(ARMS):
            ys = []
            for r in rhos:
                v = _arm_series(tag, pid, r).get(label, {}).get(b, [])
                ys.append(np.mean(v) if v else np.nan)
            if np.isfinite(ys).any():
                ax.plot(rhos, ys, color=CAT[i % len(CAT)], lw=2, marker="o", ms=5, label=label)
        ax.set_xscale("log", base=2); ax.set_xticks(rhos); ax.set_xticklabels([f"{r:g}" for r in rhos])
        ax.set_xlabel("ρ (voxels)"); ax.set_title(b, fontsize=10); ax.set_ylim(0, 90)
    axs[0].set_ylabel("median angle to the true fibre (deg)")
    axs[-1].legend(fontsize=8)
    fig.suptitle(f"{pid}: the integration scale", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_maps(tag: str, pid: str, seed: int, rho: float, out: Path, arms=None):
    """Per-site error maps for the arms present, with the phantom's mid-y image."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from ..phantoms import make
    from .suite import result_path, Spec
    _style(plt)
    spec = Spec(pids=(pid,), tag=tag)
    arms = arms or ARMS
    panels = []
    for k, eng, rd, label in arms:
        p = result_path(spec, k, pid, seed, eng, rho).with_suffix(".npz")
        if not p.exists():
            continue
        with np.load(p) as z:
            if f"err_{rd}" not in z.files:
                continue
            panels.append((label, z[f"err_{rd}"].reshape(len(z["z"]), len(z["x"])), z["z"], z["x"]))
    if not panels:
        return None
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ); cmap.set_bad(GRID)
    n = len(panels) + 1
    fig, axs = plt.subplots(1, n, figsize=(3.4 * n, 4.6), dpi=110)
    try:
        img, tr, meta, grid = make.load(pid, seed)
        z0, z1 = grid.core_zx[0]; x0, x1 = grid.core_zx[1]
        ext = (0, (x1 - x0) * grid.vox, 0, (z1 - z0) * grid.vox)
        axs[0].imshow(img[z0:z1, grid.y_mid, x0:x1], cmap="gray", vmin=0, vmax=255, origin="lower", extent=ext)
        axs[0].set_title("image (mid y)", fontsize=9.5)
    except FileNotFoundError:                                        # the phantom file was regenerated since
        zs, xs = panels[0][2], panels[0][3]
        ext = (0, (xs[-1] - xs[0]) * 0.02, 0, (zs[-1] - zs[0]) * 0.02)
        axs[0].set_title("(phantom image not kept)", fontsize=9.5)
    axs[0].grid(False)
    for ax, (label, err, zs, xs) in zip(axs[1:], panels):
        im = ax.imshow(np.ma.masked_invalid(err), cmap=cmap, vmin=0, vmax=90, origin="lower", extent=ext, interpolation="nearest")
        med = np.nanmedian(err)
        ax.set_title(f"{label}\nmedian {med:.1f}°", fontsize=9); ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=axs[1:], fraction=0.025, pad=0.01); cb.set_label("angle to the true fibre (deg)")
    fig.suptitle(f"{pid} seed {seed}, ρ {rho:g} vox: per-site error (light = close to the truth)", x=0.01, ha="left", fontsize=10.5)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def fig_difficulty(tag: str, family: str, rho: float, out: Path, band: str = "0-0.1mm", stat: str = "median"):
    """For an interface family (``P3``, ``P1``, ``P7`` ...): the ``band`` error against the texture level
    (the ``-t<f>`` suffix; 1 = calibrated), one panel per step, one line per arm."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..phantoms.make import split_pid
    _style(plt)
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))     # step -> arm -> tfrac -> values
    for kind in ("single", "hybrid"):
        for r in load(tag, kind):
            base, t = split_pid(r["pid"])
            if not base.startswith(family) or r["rho"] != rho:
                continue
            step = base.split("-", 1)[1] if "-" in base else base
            for name, sc in r["scores"].items():
                for k, eng, rd, label in ARMS:
                    if k == kind and eng == r["engine"] and rd == name:
                        v = _get(sc["bands"], band, stat)
                        if np.isfinite(v):
                            rows[step][label][t].append(v)
    steps = sorted(rows, key=lambda s: float(s) if s.replace(".", "").isdigit() else 0)
    if not steps:
        return None
    fig, axs = plt.subplots(1, len(steps), figsize=(4.4 * len(steps), 4.3), dpi=130, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, step in zip(axs, steps):
        for i, (k, eng, rd, label) in enumerate(ARMS):
            if label not in rows[step]:
                continue
            ts = sorted(rows[step][label])
            ys = [np.mean(rows[step][label][t]) for t in ts]
            ax.plot(ts, ys, color=CAT[i % len(CAT)], lw=2, marker="o", ms=5, label=label)
        ax.set_xscale("log"); ax.set_xticks([0.1, 0.25, 1.0]); ax.set_xticklabels(["0.1", "0.25", "1"])
        ax.set_xlabel("fibre texture amplitude (x calibrated)"); ax.set_title(f"{family} step {step}", fontsize=10)
        ax.set_ylim(0, 90)
    axs[0].set_ylabel(f"angle to the true fibre, {band} ({stat}, deg)")
    axs[-1].legend(fontsize=8)
    fig.suptitle(f"{family}: the step-to-texture axis at ρ {rho:g} vox", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def write(tag: str = "main", out: str | None = None) -> Path:
    pids = sorted({r["pid"] for kind in ("single", "multi", "hybrid") for r in load(tag, kind)})
    fdir = FIG_DIR / tag
    fdir.mkdir(parents=True, exist_ok=True)
    md = [f"# Validation report: tag `{tag}`\n", "Numbers: mean over seeds of the per-seed statistic, [min–max] over seeds.\n"]
    figs = []
    for pid in pids:
        md.append(f"\n## {pid}\n")
        for fn in (single_table, parallel_table, detector_table, multi_table):
            t = fn(tag, pid)
            if t:
                md.append(t)
        rhos = sorted({r["rho"] for r in load(tag, "single") if r["pid"] == pid})
        seeds = sorted({r["seed"] for r in load(tag, "single") if r["pid"] == pid})
        if len(rhos) > 1:
            f = fig_rho_sweep(tag, pid, fdir / f"{pid}_rho_sweep.png")
            if f:
                figs.append(f); md.append(f"![]({f})\n")
        for rho in rhos:
            f = fig_error_vs_band(tag, pid, rho, fdir / f"{pid}_rho{rho:g}_bands.png")
            if f:
                figs.append(f); md.append(f"![]({f})\n")
        if seeds and rhos:
            rho_mid = rhos[min(len(rhos) - 1, 2)]
            f = fig_maps(tag, pid, seeds[0], rho_mid, fdir / f"{pid}_s{seeds[0]}_rho{rho_mid:g}_maps.png")
            if f:
                figs.append(f); md.append(f"![]({f})\n")
    from ..phantoms.make import split_pid
    families = sorted({split_pid(p)[0].split("-")[0] for p in pids if split_pid(p)[1] != 1.0})
    if families:
        md.append("\n## The step-to-texture axis\n")
        for fam in families:
            for rho in sorted({r["rho"] for r in load(tag, "single") if split_pid(r["pid"])[0].startswith(fam)}):
                for band in ("0-0.1mm", "0.25-0.5mm", ">2mm"):
                    f = fig_difficulty(tag, fam, rho, fdir / f"{fam}_rho{rho:g}_{band.replace('>', 'gt')}_difficulty.png", band=band)
                    if f:
                        figs.append(f); md.append(f"![]({f})\n")
    from .scorecard import write as write_scorecard
    sc = write_scorecard(tag)
    md.append(f"\n## Pre-registered claims\n\nSee `{sc.name}` beside this report: every claim, every arm, PASS / FAIL.\n")
    path = Path(out) if out else RESULTS_DIR / tag / "REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md))
    return path
