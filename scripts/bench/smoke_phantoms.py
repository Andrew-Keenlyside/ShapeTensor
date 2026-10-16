import time, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shapetensor.phantoms import make
g = make.Grid(vox=0.02, core=(256, 32, 192), margin=(48, 48, 48))
gg = make.Grid(vox=0.02, core=(224, 32, 224), margin=(48, 48, 48))
tex = make.TextureParams()
ids = ["P3-15", "P1", "P2", "P4-lines-c60-f1", "P4-rods-c60-f1", "P5", "P6", "P7", "P8", "P9"]
fig, axs = plt.subplots(2, len(ids), figsize=(3.2 * len(ids), 7), dpi=80)
for i, pid in enumerate(ids):
    grid = gg if pid.startswith(("P1", "P2", "P5")) else g
    t0 = time.time(); img, tr, meta = make.build(pid, 0, tex, grid); dt = time.time() - t0
    jy = grid.y_mid
    axs[0, i].imshow(img[:, jy, :], cmap="gray", vmin=0, vmax=255, origin="lower"); axs[0, i].set_title(f"{pid}\n{dt:.1f}s {img.shape}", fontsize=9)
    d = tr["dir1"]; ang = np.degrees(np.arctan2(d[..., 0], d[..., 2]))
    axs[1, i].imshow(np.where(np.isnan(ang), np.nan, ang % 180), cmap="hsv", vmin=0, vmax=180, origin="lower")
    axs[1, i].contour(np.nan_to_num(tr["dist"], nan=99), levels=[0.1, 0.5], colors="k", linewidths=0.5)
    axs[1, i].set_title(f"npop {np.unique(tr['npop']).tolist()} region {np.unique(tr['region']).tolist()}", fontsize=8)
    print(pid, img.shape, f"{dt:.1f}s", "mean", img.mean().round(1), "sd", img.std().round(1), flush=True)
for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
fig.suptitle("Smoke build at 20 um (small grids): image (top), truth dir1 angle + dist contours 0.1/0.5 mm (bottom)")
fig.tight_layout(); fig.savefig("/home/andrew/bridge_wmcov/shapetensor/figures/smoke_phantoms.png"); print("fig written")
