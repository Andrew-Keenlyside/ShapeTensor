"""How a ladder run's correctness develops over training: from its mid-run scorings (OUT_s{step}_acceptance.json and
_budget.json, written by phantom_ladder.py --score-every) and the final one (OUT_acceptance.json): per criterion, the
number of held-out phantoms passing against the step (small multiples, one panel per criterion); and per family, where
the populations' mass goes (matched / in other directions / read as isotropic where there are none) against the step.

Usage: python scripts/learn/ladder_progress.py OUT_STEM [--name "overnight pilot"]
"""
import argparse, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("stem"); ap.add_argument("--name", default=None); a = ap.parse_args()
stem = a.stem[:-4] if a.stem.endswith(".png") else a.stem
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]                                                       # reference palette slots 1-3
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CRIT = ["C1 level", "C2 flat", "C3 count", "C4 leakage", "C5 edges", "C6 direction", "C7 background", "C8 ODF shape", "C9 match rate"]
LABEL = {"C1 level": "C1 mass level", "C2 flat": "C2 flat across overlaps", "C3 count": "C3 count", "C4 leakage": "C4 leakage",
         "C5 edges": "C5 edges", "C6 direction": "C6 direction", "C7 background": "C7 background", "C8 ODF shape": "C8 ODF shape",
         "C9 match rate": "C9 found"}

runs = []
for f in glob.glob(stem + "_s*_acceptance.json"):
    m = re.search(r"_s(\d+)_acceptance\.json$", f)
    if m:
        runs.append((int(m.group(1)), f, f.replace("_acceptance.json", "_budget.json")))
if os.path.exists(stem + "_acceptance.json"):
    ck = stem + ".pt"
    import torch
    final = torch.load(ck, map_location="cpu", weights_only=False).get("step") if os.path.exists(ck) else None
    runs.append((final or (max(r[0] for r in runs) + 1000 if runs else 0), stem + "_acceptance.json", stem + "_budget.json"))
runs.sort()
if not runs:
    raise SystemExit("no scorings found for " + stem)
steps = [r[0] for r in runs]; acc = [json.load(open(r[1])) for r in runs]
bud = [json.load(open(r[2])) if os.path.exists(r[2]) else [] for r in runs]
n_ph = len(acc[-1])

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig = plt.figure(figsize=(15, 9.5), layout="constrained")
sub = fig.subfigures(1, 2, width_ratios=[1.0, 1.05])
ax1 = sub[0].subplots(3, 3, sharex=True, sharey=True)
for c, ax in zip(CRIT, ax1.ravel()):
    y = [sum(1 for p in A if p.get(c, {}).get("pass")) for A in acc]
    ax.plot(steps, y, color=SERIES[0], lw=2, marker="o", ms=4)
    ax.set_title(LABEL[c], color=INK, fontsize=9, loc="left"); ax.set_ylim(-0.5, n_ph + 0.5); ax.grid(color=GRID, lw=0.6)
    ax.annotate(f"{y[-1]}/{n_ph}", (steps[-1], y[-1]), textcoords="offset points", xytext=(4, 4), color=INK, fontsize=8)
for ax in ax1[-1]:
    ax.set_xlabel("training step")
for ax in ax1[:, 0]:
    ax.set_ylabel("phantoms passing")
allp = [sum(1 for p in A if p.get("PASS")) for A in acc]
sub[0].suptitle(f"Criteria on the {n_ph} held-out phantoms over training (all criteria at the last scoring: {allp[-1]}/{n_ph})", fontsize=10, color=INK)

fams = [f for f in ("cross", "fan", "curve", "sheet", "blob", "mix", "fine") if any(lab.startswith(f + " ") for B in bud for lab, _ in B)]
ax2 = sub[1].subplots(int(np.ceil(len(fams) / 2)), 2, sharex=True, squeeze=False)
keys = [("matched", "on the right population"), ("other", "in other directions"), ("iso spurious", "isotropic where there is none")]
for fam, ax in zip(fams, ax2.ravel()):
    for (k, lab), col in zip(keys, SERIES):
        y = [np.median([v[k] for l_, v in B if l_.startswith(fam + " ")]) if any(l_.startswith(fam + " ") for l_, _ in B) else np.nan for B in bud]
        ax.plot(steps, y, color=col, lw=2, marker="o", ms=4, label=lab)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls="--"); ax.set_title(fam, color=INK, fontsize=9, loc="left"); ax.grid(color=GRID, lw=0.6)
    ax.set_ylim(0, max(1.5, ax.get_ylim()[1]))
for ax in ax2.ravel()[len(fams):]:
    ax.axis("off")
for ax in ax2[-1]:
    ax.set_xlabel("training step")
for ax in ax2[:, 0]:
    ax.set_ylabel("share of the target mass (median)")
h, l_ = ax2.ravel()[0].get_legend_handles_labels()
sub[1].legend(h, l_, loc="outside lower center", ncol=3, frameon=False)
sub[1].suptitle("Where the populations' mass goes, per family (dashed: the target)", fontsize=10, color=INK)
fig.suptitle(a.name or os.path.basename(stem), fontsize=11, color=INK)
plt.savefig(stem + "_progress.png", dpi=80, facecolor="#fcfcfb"); print("saved", stem + "_progress.png")
