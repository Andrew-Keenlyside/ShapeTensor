"""Fix A through the real pipeline on the surface phantoms: build each phantom with BRIDGE's defaults, base and with
gradient excision (the study snapshot at /home/andrew/bridge_wmcov/edge_aware/bridge_snap), replay the pointer walks
and score the streamlines against the phantom's truth.

    PYTHONPATH=<snap>:. python edge_aware_builds.py build NAME VARIANT     (VARIANT base | gm3; sets the env itself)
    PYTHONPATH=<snap>:. python edge_aware_builds.py score NAME VARIANT ...
NAME: A | B (phantom_shell)  C | C0 (U-fibres)  D0 | D8 | D15 | D30 (tract interface)
"""
import json, os, subprocess, sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import phantom_shell as PS
import surface_phantoms as SP

OUT = Path("/home/andrew/bridge_wmcov/edge_aware")
SNAP = OUT / "bridge_snap"
VOX = PS.VOX
AFF = np.diag([VOX, VOX, VOX, 1.0])


def source(name):
    return (PS.ROOT if name in ("A", "B") else SP.ROOT) / f"{name}.nii.gz"


def thresh(name):
    return PS.THRESH[name] if name in PS.THRESH else SP.THRESH[name]


def global_median(name):
    """The in-mask median coarse-gradient magnitude over the whole phantom (one number, as the pipeline would use)."""
    vol = np.asarray(nib.load(source(name)).dataobj).astype(np.float32)
    gc = np.sqrt(sum(ndi.gaussian_filter(vol, 3.0, order=o, mode="nearest", truncate=2.0) ** 2
                     for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))))
    return float(np.median(gc[vol >= thresh(name)]))


def build(name, variant):
    if variant != "base" and "BRIDGE_EDGE_F" not in os.environ:
        env = dict(os.environ, BRIDGE_EDGE_F=variant[2:], BRIDGE_EDGE_GMED=repr(global_median(name)))
        print(name, variant, "global median", env["BRIDGE_EDGE_GMED"], flush=True)
        sys.exit(subprocess.call([sys.executable, __file__, "build", name, variant], env=env))
    if variant == "base":
        os.environ.pop("BRIDGE_EDGE_F", None); os.environ.pop("BRIDGE_EDGE_GMED", None)
    from bridge import AcqKey, Study
    import bridge
    assert str(SNAP) in bridge.__file__, bridge.__file__
    root = OUT / "builds" / f"{name}_{variant}"
    try:
        study = Study.open(root)
    except FileNotFoundError:
        root.mkdir(parents=True, exist_ok=True)
        study = Study.create(root, name=f"edge-aware phantom {name} {variant}", authors=["Andrew Keenlyside"])
    acq = study.add_acquisition(AcqKey(subject="phantom", sample=f"{name.lower()}{variant}", acq="100um"), source=str(source(name)))
    acq.update_config(resolution_mm=(VOX, VOX, VOX), GPU_limit=0.8, preflight="warn", run_pyramid=False, run_exports=False,
                      vertices={"truncate": 2.0, "max_nodes_per_chunk": 40_000, "min_fa": 0.0},
                      masking={"threshold": thresh(name)})
    for stage, r in acq.run().items():
        print(name, variant, stage, r.status, round(float(r.wall_seconds), 1), flush=True)


def truth(name):
    """(truth_fn(points) -> (n, 2) x-z unit or nan, depth_fn(points) -> signed mm from the surface, normal_fn) for a phantom."""
    if name in ("A", "B", "C", "C0"):
        t = np.load(PS.ROOT / "truth.npz") if name in ("A", "B") else np.load(SP.ROOT / "C_truth.npz")
        sdf = t["sdf"]
        gx, gz = np.gradient(ndi.gaussian_filter(sdf, 5.0)); nrm = np.stack([gx, gz], -1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-9)
        tru = None if name in ("A", "B") else t["tru"]

        def idx(P):
            return tuple(np.clip((P[:, [0, 2]] / VOX).astype(int), 0, [SP.NX - 1, SP.NZ - 1]).T)
        tf = (lambda P: PS.truth_dir(P[:, 0], P[:, 2])) if tru is None else (lambda P: tru[idx(P)])
        return tf, (lambda P: sdf[idx(P)]), (lambda P: nrm[idx(P)])
    td = np.load(SP.ROOT / "D_truth.npz"); tru = td["tru"]

    def idx(P):
        return tuple(np.clip((P[:, [0, 2]] / VOX).astype(int), 0, [SP.NX - 1, SP.NZ - 1]).T)
    return (lambda P: tru[idx(P)]), (lambda P: P[:, 2] - SP.D_PLANE_Z), (lambda P: np.tile([0.0, 1.0], (len(P), 1)))


def score(name, variant):
    import anat_lib as L
    root = OUT / "builds" / f"{name}_{variant}"
    rep = L.extract(root, f"ea_{name}_{variant}", AFF)
    world = L.world_of(rep); flat, offs = L.pointer_walk(rep)
    P = world[flat]
    R, o2 = L.resample_nb(P, offs, 0.5); tt, s2 = L.tangents(R, o2, 2)
    tf, df, nf = truth(name)
    tr = tf(R); dep = df(R); nn = nf(R)
    t2 = tt[:, [0, 2]]; tn = np.linalg.norm(t2, axis=1)
    ok = np.isfinite(tr).all(1)
    ang = np.full(len(R), np.nan)
    ang[ok] = np.degrees(np.arccos(np.clip(np.abs((tt[ok][:, [0, 2]] * tr[ok]).sum(1)), 0, 1)))   # 3D: truth has no y
    par = (np.abs((t2 * nn).sum(1)) < 0.5 * np.maximum(tn, 1e-9)) | (tn <= 0.5)       # along the surface, incl. along y
    near = np.abs(dep) <= 1.0
    rl, _ = L.runs(near & par, s2); rmm = rl * 0.5
    lens = np.diff(o2) * 0.5
    r = {"streamlines": int(len(offs) - 1), "median_length_mm": float(np.median(lens)),
         "arc_within_1mm": float(near.mean()), "arc_in_parallel_runs_ge5mm": float(rmm[rmm >= 5].sum() / (len(R) * 0.5)),
         "angle_median_all": float(np.nanmedian(ang)), "angle_median_within_1mm": float(np.nanmedian(ang[near])),
         "angle_median_beyond_2mm": float(np.nanmedian(ang[np.abs(dep) > 2]))}
    if name in ("A", "B"):
        cross = np.zeros(len(o2) - 1, bool); cross[s2[dep < -1.0]] = True
        r["streamlines_entering_gm_1mm"] = float(cross.mean())
    if name.startswith("D"):
        lo = np.zeros(len(o2) - 1, bool); hi = lo.copy(); lo[s2[dep < -1.0]] = True; hi[s2[dep > 1.0]] = True
        long = lens >= 5
        r["streamlines_crossing_interface"] = float((lo & hi)[long].mean())
        for x0, x1, a in SP.D_BANDS:
            sel = (R[:, 0] >= x0 + 1) & (R[:, 0] < x1 - 1)
            r[f"angle_{a:.0f}deg_within_1mm"] = float(np.nanmedian(ang[sel & near]))
    if name in ("C", "C0"):
        band = (dep >= 0.15) & (dep <= 1.2)
        r["angle_median_ufibre_band"] = float(np.nanmedian(ang[band]))
    return r


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2], sys.argv[3])
    else:
        name = sys.argv[2]; res = {}
        p = OUT / "builds_scores.json"; old = json.loads(p.read_text()) if p.exists() else {}
        for v in sys.argv[3:]:
            res[f"{name}:{v}"] = score(name, v); print(name, v, json.dumps({k: round(x, 3) for k, x in res[f'{name}:{v}'].items()}), flush=True)
        old.update(res); p.write_text(json.dumps(old, indent=1))
