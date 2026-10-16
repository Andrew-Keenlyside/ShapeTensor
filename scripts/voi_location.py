"""Where the subcortical VOI of shape_real.py sits: the S2146 100 um block (675, 577, 775) on the whole brain (OME-Zarr
level 3, 0.8 mm) with SynthSeg outlines, and the 10 mm VOI itself at 100 um.

    python voi_location.py OUT.png        (bridge-gpu-zv3 env)
"""
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent))
import anat_lib as L

OMEZ = "/home/andrew/bridge_wmcov/hundred_um/new/sub-S2146/micr/sub-S2146_sample-wholebrain_acq-100um_XPCT.ome.zarr"
A100 = nib.load("/hdd/BRIDGE_tutorial/multires_tutorial/wholebrain_100um.nii.gz").affine
C = np.array([675, 577, 775])                                             # x, y, z at 100 um
an = L.Anatomy(L.SS_S2146)
l3 = np.asarray(zarr.open(OMEZ, mode="r")["3"]).transpose(2, 1, 0).astype(np.float32)   # x, y, z at 0.8 mm
c3 = C // 8
# SynthSeg resampled onto level 3
X, Y, Z = np.meshgrid(*[np.arange(n) * 8 + 4 for n in l3.shape], indexing="ij")
w = nib.affines.apply_affine(A100, np.stack([X, Y, Z], -1).reshape(-1, 3))
q = np.clip(np.rint(an.vox(w)).astype(int), 0, np.array(an.S.shape) - 1)
S = an.S[tuple(q.T)].reshape(l3.shape)
lab = an.sample(nib.affines.apply_affine(A100, C[None].astype(float)))
names = {2: "L cerebral WM", 41: "R cerebral WM", 10: "L thalamus", 49: "R thalamus", 11: "L caudate", 50: "R caudate",
         12: "L putamen", 51: "R putamen", 13: "L pallidum", 52: "R pallidum", 4: "L lat. ventricle", 43: "R lat. ventricle"}
v100 = np.asarray(zarr.open(OMEZ, mode="r")["0"][C[2] - 50:C[2] + 50, C[1] - 50:C[1] + 50, C[0] - 50:C[0] + 50]).transpose(2, 1, 0)
# SynthSeg labels in the 10 mm VOI
Xv, Yv, Zv = np.meshgrid(*[np.arange(-50, 50, 5) + c for c in C], indexing="ij")
qv = np.clip(np.rint(an.vox(nib.affines.apply_affine(A100, np.stack([Xv, Yv, Zv], -1).reshape(-1, 3)))).astype(int), 0, np.array(an.S.shape) - 1)
u, cnt = np.unique(an.S[tuple(qv.T)], return_counts=True)
comp = ", ".join(f"{names.get(int(a), int(a))} {b / cnt.sum():.0%}" for a, b in sorted(zip(u, cnt), key=lambda t: -t[1]) if b / cnt.sum() >= 0.02)
fig, axs = plt.subplots(2, 3, figsize=(20, 13), dpi=100)
planes = (("axial (z)", lambda a, i: a[:, :, i].T, 2, (0, 1)), ("coronal (y)", lambda a, i: a[:, i, :].T, 1, (0, 2)),
          ("sagittal (x)", lambda a, i: a[i, :, :].T, 0, (1, 2)))
for k, (pn, take, ax_i, (h, v_)) in enumerate(planes):
    ax = axs[0, k]; ax.imshow(take(l3, c3[ax_i]), cmap="gray", origin="lower")
    seg = take(S, c3[ax_i])
    for lbs, col in (((10, 49), "#377eb8"), ((13, 52), "#ff7f00"), ((12, 51), "#4daf4a"), ((11, 50), "#984ea3"), ((4, 43), "c")):
        ax.contour(np.isin(seg, lbs).astype(float), [0.5], colors=col, linewidths=0.8)
    for half, col, lw in ((50, "r", 1.6), (96, "y", 1.0)):
        ax.add_patch(Rectangle(((C[h] - half) / 8, (C[v_] - half) / 8), 2 * half / 8, 2 * half / 8, fill=False, ec=col, lw=lw))
    ax.set_title(f"S2146 whole brain, {pn} through the VOI\nred = 10 mm VOI scored, yellow = 19.2 mm read for anatomix", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax = axs[1, k]; ax.imshow(take(v100, 50), cmap="gray", origin="lower"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"the VOI at 100 um, {pn} (10 x 10 mm)", fontsize=10)
fig.suptitle(f"Subcortical VOI used so far: S2146 100 um voxel {tuple(int(x) for x in C)} (occipital periventricular WM); SynthSeg at its centre: {names.get(int(lab['label'][0]), int(lab['label'][0]))}, "
             f"{lab['sdf'][0]:.1f} mm inside WM.\nVOI content (SynthSeg 1 mm): {comp}.  Thalamus blue, pallidum orange, putamen green, "
             "caudate purple, ventricles cyan", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(sys.argv[1]); print("wrote", sys.argv[1], "|", comp)
