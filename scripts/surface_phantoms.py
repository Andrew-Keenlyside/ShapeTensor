"""Surface-versus-fibre phantoms: does the orientation field follow fibres, or the surfaces they cross?

All 0.1 mm voxels, 20 x 12 x 20 mm, extruded along y; fibres are lines in the x-z plane at random y, splatted
into a density that darkens the tissue (dehydrated-WM convention), plus Gaussian noise 8.

  A, B   phantom_shell.py's gyrus (grey/white step / no step), fan of fibres crossing the boundary
  C, C0  the same gyrus with U-FIBRES: fibres run PARALLEL to the boundary in the superficial 0.15-1.2 mm band and
         a fan only below 1.6 mm. A fix must keep these (truth there IS boundary-parallel). C0 has no step.
  D<d>   a tract INTERFACE inside white matter: one WM slab, the half z >= 10 mm brighter by d (8, 15, 30; D0 none).
         Three x-bands of straight fibres crossing the interface at 90, 60 and 30 degrees.

    python surface_phantoms.py make
    python surface_phantoms.py make_pad     D0 / D15 / D30 grown by D_PAD voxels per x and z face (make_D_padded)
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import phantom_shell as PS

ROOT = Path("/home/andrew/bridge_wmcov/edge_aware/phantoms")
VOX, NX, NY, NZ = PS.VOX, PS.NX, PS.NY, PS.NZ
NOISE, AMP = 8.0, 12.0
D_STEPS = (0, 8, 15, 30)
D_BANDS = ((0.0, 20 / 3, 90.0), (20 / 3, 40 / 3, 60.0), (40 / 3, 20.0, 30.0))    # x0, x1 (mm), angle to the plane
D_PLANE_Z = 10.0
THRESH = {"C": 144.5, "C0": 100.0, **{f"D{d}": 100.0 for d in D_STEPS}}


def splat(lines, rng):
    """Density from polylines [(px, pz)] in mm, each at a random y, normalised to its own 99th percentile."""
    dens = np.zeros((NX, NY, NZ), np.float32)
    for px, pz in lines:
        ok = (px >= 0) & (px < NX * VOX) & (pz >= 0) & (pz < NZ * VOX)
        ix = (px[ok] / VOX).astype(int); iz = (pz[ok] / VOX).astype(int)
        iy = np.full(ix.shape, int(rng.uniform(0, NY * VOX) / VOX))
        np.add.at(dens, (ix, iy, iz), 1.0)
    dens = ndi.gaussian_filter(dens, 0.7)
    return dens / np.percentile(dens[dens > 0], 99)


def resample_line(px, pz, step=0.05):
    d = np.r_[0, np.cumsum(np.hypot(np.diff(px), np.diff(pz)))]
    s = np.arange(0, d[-1], step)
    return np.interp(s, d, px), np.interp(s, d, pz)


def make_C(rng):
    from skimage import measure
    wm, gm, sdf, X, Z = PS.tissue_2d()
    lines = []
    # deep fan, clipped to sdf > 1.6 mm
    s = np.arange(0, 45, 0.05)
    for _ in range(2000):
        t = np.radians(rng.uniform(-38, 38)); f0 = PS.FOCUS[0] + rng.uniform(-2.5, 2.5)
        px = f0 + s * np.sin(t); pz = PS.FOCUS[1] + s * np.cos(t)
        ins = (px >= 0) & (px < NX * VOX) & (pz >= 0) & (pz < NZ * VOX)
        ix = np.clip((px / VOX).astype(int), 0, NX - 1); iz = np.clip((pz / VOX).astype(int), 0, NZ - 1)
        keep = ins & (sdf[ix, iz] > 1.6)
        lines.append((px[keep], pz[keep]))
    fan = splat(lines, rng); lines = []
    # U-fibres: pieces of the iso-depth contours, 6-12 mm long
    for _ in range(2400):
        d = rng.uniform(0.15, 1.2)
        cs = [c for c in measure.find_contours(sdf, d) if len(c) > 100]
        c = cs[rng.integers(len(cs))] * VOX + VOX / 2
        px, pz = resample_line(c[:, 0], c[:, 1])
        L = int(rng.uniform(6, 12) / 0.05); i0 = rng.integers(0, max(1, len(px) - L))
        lines.append((px[i0:i0 + L], pz[i0:i0 + L]))
    dens = np.maximum(fan, splat(lines, rng))
    amp2d = np.where(wm, AMP, 0.0).astype(np.float32)
    for name, (iw, ig, ic) in (("C", PS.INT["A"]), ("C0", PS.INT["B"])):
        base2d = ndi.gaussian_filter(np.where(wm, iw, np.where(gm, ig, ic)).astype(np.float32), 1.0)
        save(name, base2d[:, None, :] - amp2d[:, None, :] * np.clip(dens, 0, 1.5), rng)
    # truth: contour tangent in the band, fan direction in deep WM, else nan
    gx, gz = np.gradient(ndi.gaussian_filter(sdf, 5.0)); nrm = np.stack([gx, gz], -1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-9)
    tru = np.full(sdf.shape + (2,), np.nan, np.float32)
    band = (sdf >= 0.15) & (sdf <= 1.2); deep = sdf > 1.6
    tru[band] = np.stack([-nrm[..., 1], nrm[..., 0]], -1)[band]
    tru[deep] = PS.truth_dir(X, Z)[deep]
    np.savez_compressed(ROOT / "C_truth.npz", tru=tru, sdf=sdf, nrm=nrm, wm=wm)


def make_D(rng):
    x = (np.arange(NX) + 0.5) * VOX; z = (np.arange(NZ) + 0.5) * VOX
    X, Z = np.meshgrid(x, z, indexing="ij")
    lines = []; tru = np.zeros((NX, NZ, 2), np.float32)
    for x0, x1, ang in D_BANDS:
        a = np.radians(ang); dx, dz = np.cos(a), np.sin(a)                 # angle to the z = const plane
        sel = (X >= x0) & (X < x1); tru[sel] = (dx, dz)
        n_lines = int(1100 * 20.0 / min((x1 - x0) / dx, NZ * VOX / max(dz, 1e-9)))   # equal density per band
        for _ in range(n_lines):
            # a line through a random point of the band, clipped to the band
            cx, cz = rng.uniform(x0, x1), rng.uniform(0, NZ * VOX)
            s = np.arange(-30, 30, 0.05); px = cx + s * dx; pz = cz + s * dz
            keep = (px >= x0) & (px < x1) & (pz >= 0) & (pz < NZ * VOX)
            lines.append((px[keep], pz[keep]))
    dens = splat(lines, rng)
    for d in D_STEPS:
        base2d = ndi.gaussian_filter(np.where(Z >= D_PLANE_Z, 180.0 + d, 180.0).astype(np.float32), 1.0)
        save(f"D{d}", base2d[:, None, :] - AMP * np.clip(dens, 0, 1.5), rng)
    np.savez_compressed(ROOT / "D_truth.npz", tru=tru, depth=np.abs(Z - D_PLANE_Z), X=X)


D_PAD = 96                                                              # voxels added on each x and z face
PAD_ROOT = ROOT.parent / f"phantoms_pad{D_PAD}"


def make_D_padded(rng, pad=D_PAD, root=PAD_ROOT, steps=(0, 15, 30)):
    """The D phantoms on a grid grown by pad voxels on each x and z face, so that orientations can be computed away
    from the block faces and cropped back to the original 20 x 20 mm region, [pad, pad + NX) x [pad, pad + NZ).
    y is not grown: the fibres lie in x-z planes and the phantom is uniform in y, so a face mirror there is more of the
    same. Same bands (the 90 and 30 deg bands extend outwards), interface, step, noise and line density (line length
    per mm^2, as make_D); coordinates in mm are the original ones (the padded grid starts at -pad * VOX).
    A new random realisation: compare methods on this set with each other, not with the unpadded numbers."""
    o = pad * VOX; nx, nz = NX + 2 * pad, NZ + 2 * pad; z0, z1 = -o, NZ * VOX + o
    x = (np.arange(nx) + 0.5) * VOX - o; z = (np.arange(nz) + 0.5) * VOX - o
    X, Z = np.meshgrid(x, z, indexing="ij")
    bands = ((-o, D_BANDS[0][1], 90.0), D_BANDS[1], (D_BANDS[2][0], NX * VOX + o, 30.0))
    dens = np.zeros((nx, NY, nz), np.float32); tru = np.zeros((nx, nz, 2), np.float32)
    for x0, x1, ang in bands:
        a = np.radians(ang); dx, dz = np.cos(a), np.sin(a)
        sel = (X >= x0) & (X < x1); tru[sel] = (dx, dz)
        L = min((x1 - x0) / max(dx, 1e-9), (z1 - z0) / max(dz, 1e-9))         # longest chord through the band
        n_lines = int(1100 * 20.0 * ((x1 - x0) * (z1 - z0) / (20 / 3 * 20.0)) / L)   # make_D's density, this area
        s = np.arange(-L, L, 0.05)
        for _ in range(n_lines):
            cx, cz = rng.uniform(x0, x1), rng.uniform(z0, z1)
            px = cx + s * dx; pz = cz + s * dz
            k = (px >= x0) & (px < x1) & (pz >= z0) & (pz < z1)
            ix = ((px[k] + o) / VOX).astype(int); iz = ((pz[k] + o) / VOX).astype(int)
            iy = np.full(ix.shape, int(rng.uniform(0, NY * VOX) / VOX))
            np.add.at(dens, (np.clip(ix, 0, nx - 1), iy, np.clip(iz, 0, nz - 1)), 1.0)
    dens = ndi.gaussian_filter(dens, 0.7); dens /= np.percentile(dens[dens > 0], 99)
    root.mkdir(parents=True, exist_ok=True)
    for d in steps:
        base2d = ndi.gaussian_filter(np.where(Z >= D_PLANE_Z, 180.0 + d, 180.0).astype(np.float32), 1.0)
        save(f"D{d}", base2d[:, None, :] - AMP * np.clip(dens, 0, 1.5), rng, root)
    np.savez_compressed(root / "D_truth.npz", tru=tru, depth=np.abs(Z - D_PLANE_Z), X=X, pad=pad)


def save(name, vol, rng, root=ROOT):
    vol = vol + rng.normal(0, NOISE, vol.shape).astype(np.float32)
    img = np.clip(np.rint(vol), 0, 255).astype(np.uint8)
    nib.save(nib.Nifti1Image(img, np.diag([VOX, VOX, VOX, 1.0])), root / f"{name}.nii.gz")
    print("wrote", root / name, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "make":
        ROOT.mkdir(parents=True, exist_ok=True)
        make_C(np.random.default_rng(11)); make_D(np.random.default_rng(13))
    elif sys.argv[1] == "make_pad":
        make_D_padded(np.random.default_rng(17))
