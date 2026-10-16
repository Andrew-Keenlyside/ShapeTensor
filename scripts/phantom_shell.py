"""A gyrus phantom for the boundary shell: two white-matter blades on a WM base, a 2.5 mm grey-matter layer, CSF in
the sulcus and above, extruded along y (the gyral axis). True fibres are straight lines fanning up from below
(focus at x = 10, z = -15 mm) that CROSS the grey/white boundary and continue faintly into grey matter -- so any
streamline running along the boundary is an artefact by construction.

  A "contrast": WM 190, GM 150, CSF 60 (GM around the 144.5 mask threshold, as at 100 um)
  B "flat":     every class 170 -- no intensity step; only the fibre texture (WM amplitude 12, GM 4)

    python phantom_shell.py make              -> /home/andrew/bridge_wmcov/phantom_shell/{A,B}.nii.gz, truth.npz
    PYTHONPATH=<bridge_snap> python phantom_shell.py build A|B
    PYTHONPATH=<bridge_snap>:. python phantom_shell.py analyse OUT.png
"""
import json, sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

ROOT = Path("/home/andrew/bridge_wmcov/phantom_shell")
VOX = 0.1
NX, NY, NZ = 200, 120, 200                                  # x, y, z voxels: 20 x 12 x 20 mm
FOCUS = np.array([10.0, -15.0])                             # fan focus in (x, z), mm
BLADES = ((5.0, 1.4), (15.0, 1.4))                          # (centre x, half-width) mm
CROWN_Z, BASE_Z, GM_MM = 14.0, 7.0, 2.5
INT = {"A": (190.0, 150.0, 60.0), "B": (170.0, 170.0, 170.0)}
THRESH = {"A": 144.5, "B": 100.0}


def tissue_2d():
    x = (np.arange(NX) + 0.5) * VOX; z = (np.arange(NZ) + 0.5) * VOX
    X, Z = np.meshgrid(x, z, indexing="ij")
    wm = Z < BASE_Z
    for cx, hw in BLADES:
        stalk = (np.abs(X - cx) <= hw) & (Z <= CROWN_Z - hw)
        cap = (X - cx) ** 2 + (Z - (CROWN_Z - hw)) ** 2 <= hw ** 2
        wm |= stalk | cap
    d_out = ndi.distance_transform_edt(~wm) * VOX
    gm = ~wm & (d_out <= GM_MM)
    sdf = ndi.distance_transform_edt(wm) * VOX - d_out          # + inside WM, mm
    return wm, gm, sdf, X, Z


def make():
    ROOT.mkdir(parents=True, exist_ok=True)
    wm, gm, sdf, X, Z = tissue_2d()
    rng = np.random.default_rng(7)
    # fibre lines: straight, from a jittered focus, at a fixed y each; deposit points every 0.05 mm, then splat
    dens = np.zeros((NX, NY, NZ), np.float32)
    n_lines = 3200
    th = np.radians(rng.uniform(-38, 38, n_lines)); fx = FOCUS[0] + rng.uniform(-2.5, 2.5, n_lines)
    ys = rng.uniform(0, NY * VOX, n_lines)
    s = np.arange(0, 45, 0.05)
    for t, f0, y in zip(th, fx, ys):
        px = f0 + s * np.sin(t); pz = FOCUS[1] + s * np.cos(t)
        ok = (px >= 0) & (px < NX * VOX) & (pz >= 0) & (pz < NZ * VOX)
        ix = (px[ok] / VOX).astype(int); iz = (pz[ok] / VOX).astype(int); iy = np.full(ix.shape, int(y / VOX))
        np.add.at(dens, (ix, iy, iz), 1.0)
    dens = ndi.gaussian_filter(dens, 0.7); dens /= np.percentile(dens[dens > 0], 99)
    for name, (iw, ig, ic) in INT.items():
        base2d = np.where(wm, iw, np.where(gm, ig, ic)).astype(np.float32)
        base2d = ndi.gaussian_filter(base2d, 1.0)                                         # partial volume
        amp2d = np.where(wm, 12.0, np.where(gm, 4.0, 0.0)).astype(np.float32)
        vol = base2d[:, None, :] - amp2d[:, None, :] * np.clip(dens, 0, 1.5)
        vol += rng.normal(0, 8.0, vol.shape).astype(np.float32)
        img = np.clip(np.rint(vol), 0, 255).astype(np.uint8)
        nib.save(nib.Nifti1Image(img, np.diag([VOX, VOX, VOX, 1.0])), ROOT / f"{name}.nii.gz")
    np.savez_compressed(ROOT / "truth.npz", wm=wm, gm=gm, sdf=sdf)
    print("wrote", ROOT)


def build(name):
    from bridge import AcqKey, Study
    root = ROOT / f"study_{name}"
    try:
        study = Study.open(root)
    except FileNotFoundError:
        root.mkdir(parents=True, exist_ok=True)
        study = Study.create(root, name=f"boundary-shell phantom {name}", authors=["Andrew Keenlyside"])
    acq = study.add_acquisition(AcqKey(subject="phantom", sample=f"shell{name}", acq="100um"), source=str(ROOT / f"{name}.nii.gz"))
    acq.update_config(resolution_mm=(VOX, VOX, VOX), GPU_limit=0.8, preflight="warn", run_pyramid=False, run_exports=False,
                      vertices={"truncate": 2.0, "max_nodes_per_chunk": 40_000, "min_fa": 0.0},
                      masking={"threshold": THRESH[name]})
    for stage, r in acq.run().items():
        print(name, stage, r.status, round(float(r.wall_seconds), 1), flush=True)


def truth_dir(px, pz):
    d = np.stack([px - FOCUS[0], pz - FOCUS[1]], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def analyse(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import anat_lib as L
    t = np.load(ROOT / "truth.npz"); wm, gm, sdf = t["wm"], t["gm"], t["sdf"]
    gx, gz = np.gradient(ndi.gaussian_filter(sdf, 5.0)); nrm = np.stack([gx, gz], -1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-9)
    ext = (0, NX * VOX, 0, NZ * VOX); jy = NY // 2; res = {}
    fig, axs = plt.subplots(2, 4, figsize=(24, 12.5), dpi=120)
    for row, name in enumerate(("A", "B")):
        vol = np.asarray(nib.load(ROOT / f"{name}.nii.gz").dataobj).astype(np.float32)
        # BRIDGE's tensor on the middle slab
        slab = vol[:, jy - 12:jy + 13, :]
        g = [ndi.gaussian_filter(slab, 1.0, order=o, truncate=2.0) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        T = np.empty(slab.shape + (3, 3), np.float32)
        for i in range(3):
            for j in range(i, 3):
                T[..., i, j] = T[..., j, i] = ndi.gaussian_filter(g[i] * g[j], 3.0, truncate=2.0)
        w, v = np.linalg.eigh(T[:, 12]); e3 = v[..., :, 0]
        img = slab[:, 12]; mask = img >= THRESH[name]
        X, Z = np.meshgrid((np.arange(NX) + 0.5) * VOX, (np.arange(NZ) + 0.5) * VOX, indexing="ij")
        tru = truth_dir(X, Z)
        e2 = e3[..., [0, 2]]; inpl = np.linalg.norm(e2, axis=-1)
        ang = np.degrees(np.arccos(np.clip(np.abs((e2 * tru).sum(-1)), 0, 1)))       # in-slice angle to truth
        tang = np.abs((e2 / np.maximum(inpl[..., None], 1e-9) * nrm).sum(-1)) < 0.5   # in-slice e3 parallel to boundary
        bins = [-1.0, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0]; prof = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            m = mask & (sdf >= b0) & (sdf < b1) & (inpl > 0.5)
            prof.append({"depth_mm": [b0, b1], "e3_parallel_to_boundary": float(tang[m].mean()) if m.any() else None,
                         "angle_to_truth_median": float(np.median(ang[m])) if m.any() else None})
        res[name] = {"voxel_profile": prof}
        # panel 1: image + a few true fibre lines
        ax = axs[row, 0]; ax.imshow(img.T, cmap="gray", origin="lower", extent=ext, vmin=40, vmax=220)
        for a in np.radians(np.linspace(-36, 36, 13)):
            ax.plot([FOCUS[0], FOCUS[0] + 45 * np.sin(a)], [FOCUS[1], FOCUS[1] + 45 * np.cos(a)], color="y", lw=0.6, alpha=0.6)
        ax.contour(X, Z, wm.astype(float), [0.5], colors="c", linewidths=0.8)
        ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:]); ax.set_title(f"{name}: image, true fibres (yellow), WM boundary (cyan)", fontsize=10)
        # panel 2: e3 glyphs coloured by angle to truth
        ax = axs[row, 1]; ax.imshow(img.T, cmap="gray", origin="lower", extent=ext, vmin=40, vmax=220)
        st = 4; xi, zi = np.meshgrid(np.arange(2, NX, st), np.arange(2, NZ, st), indexing="ij"); sel = mask[xi, zi]
        xi, zi = xi[sel], zi[sel]; d = e2[xi, zi]; Lg = 0.35 * inpl[xi, zi]; d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        cx, cz = X[xi, zi], Z[xi, zi]
        segs = np.stack([np.stack([cx - d[:, 0] * Lg / 2, cz - d[:, 1] * Lg / 2], 1), np.stack([cx + d[:, 0] * Lg / 2, cz + d[:, 1] * Lg / 2], 1)], 1)
        cmap = plt.get_cmap("RdYlGn_r"); ax.add_collection(LineCollection(segs, colors=cmap(ang[xi, zi] / 90), linewidths=1.0))
        ax.contour(X, Z, wm.astype(float), [0.5], colors="c", linewidths=0.6)
        ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:]); ax.set_title("tensor e3 (sigma 1, rho 3 vox), colour = angle to truth\ngreen 0 deg ... red 90 deg; short = through the slice", fontsize=10)
        # panel 3: BRIDGE streamlines (replayed walks)
        try:
            rep = L.extract(ROOT / f"study_{name}", f"phantom_shell_{name}", np.diag([VOX, VOX, VOX, 1.0]))
            world = L.world_of(rep); flat, offs = L.pointer_walk(rep)
            P = world[flat]; sid = np.repeat(np.arange(len(offs) - 1), np.diff(offs)); same = sid[1:] == sid[:-1]
            a, b = P[:-1][same], P[1:][same]; ky = (np.abs(a[:, 1] - jy * VOX) <= 1.0) & (np.abs(b[:, 1] - jy * VOX) <= 1.0)
            dd = np.abs(b - a); dd /= np.maximum(np.linalg.norm(dd, axis=1, keepdims=True), 1e-9)
            ax = axs[row, 2]; ax.imshow(img.T, cmap="gray", origin="lower", extent=ext, vmin=40, vmax=220)
            ax.add_collection(LineCollection(np.stack([a[ky][:, [0, 2]], b[ky][:, [0, 2]]], 1), colors=np.clip(dd[ky], 0, 1), linewidths=0.5, alpha=0.85))
            ax.contour(X, Z, wm.astype(float), [0.5], colors="c", linewidths=0.6)
            ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:]); ax.set_title(f"BRIDGE streamlines (default walk), slab +-1 mm\n{len(offs) - 1:,} streamlines; colour x red / y green / z blue", fontsize=10)
            # streamline measures against the true boundary
            R, o2 = L.resample_nb(P, offs, 0.5); tt, s2 = L.tangents(R, o2, 2)
            qi = np.clip((R[:, [0, 2]] / VOX).astype(int), 0, [NX - 1, NZ - 1]); dep = sdf[qi[:, 0], qi[:, 1]]; nn = nrm[qi[:, 0], qi[:, 1]]
            t2 = tt[:, [0, 2]]; tn = np.linalg.norm(t2, axis=1)
            par = (np.abs((t2 * nn).sum(1)) < 0.5 * np.maximum(tn, 1e-9)) & (tn > 0.5)
            near = np.abs(dep) <= 1.0
            rl, st_ = L.runs(near & par, s2); rmm = rl * 0.5
            tr = truth_dir(R[:, 0], R[:, 2]); tang_err = np.degrees(np.arccos(np.clip(np.abs((t2 / np.maximum(tn[:, None], 1e-9) * tr).sum(1)), 0, 1)))
            cross = np.zeros(len(o2) - 1, bool); cross[s2[dep < -1.0]] = True
            res[name]["streamlines"] = {"n": int(len(offs) - 1), "arc_share_within_1mm_of_boundary": float(near.mean()),
                                        "parallel_share_within_1mm": float(par[near].mean()),
                                        "arc_share_in_parallel_runs_ge5mm": float(rmm[rmm >= 5].sum() / (len(R) * 0.5)),
                                        "angle_to_truth_median_deep_wm": float(np.median(tang_err[dep > 2])),
                                        "angle_to_truth_median_within_1mm": float(np.median(tang_err[near])),
                                        "streamlines_entering_gm_1mm": float(cross.mean())}
        except Exception as ex:                                                   # build not there yet
            res[name]["streamlines"] = repr(ex)
        # panel 4: profiles
        ax = axs[row, 3]
        mids = [np.mean(p["depth_mm"]) for p in prof]
        ax.plot(mids, [p["e3_parallel_to_boundary"] for p in prof], "-o", label="tensor e3 parallel to boundary")
        ax.plot(mids, [p["angle_to_truth_median"] / 90 if p["angle_to_truth_median"] is not None else None for p in prof], "-s",
                label="tensor e3 angle to truth / 90 deg")
        # truth's own parallel share for reference
        tpar = np.abs((tru * nrm).sum(-1)) < 0.5
        ax.plot(mids, [float(tpar[(sdf >= p["depth_mm"][0]) & (sdf < p["depth_mm"][1])].mean()) for p in prof], "--", color="k", label="true fibres parallel to boundary")
        ax.axvline(0, color="c"); ax.set_xlabel("depth below the WM boundary (mm; < 0 = in grey matter)"); ax.set_ylim(0, 1)
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("by depth (in-slice, inside the mask)", fontsize=10)
        for a_ in axs[row, :3]:
            a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("Boundary-shell phantom: identical straight fanning fibres that cross the grey/white boundary.  "
                 "A = realistic WM/GM/CSF contrast;  B = no intensity step (texture only)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out)
    (ROOT / "phantom_shell_results.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make":
        make()
    elif cmd == "build":
        build(sys.argv[2])
    else:
        analyse(sys.argv[2])
