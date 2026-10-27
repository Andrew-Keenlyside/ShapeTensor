"""DiSCo for the test harness only (``disco_test.py``, ``disco_fibres.py``): the strands' reader, the crop choice and a
greyscale renderer. Never imported by training, the generator or model selection (notes/plan_full_generator.md, D2).

DiSCo (Rafael-Patino et al. 2020; data doi:10.17632/fgf86jdfg6, CC BY 4.0): strands (centrelines, diameters, ROI pairs)
in a 1 mm sphere, coordinates scaled x40 (1 unit = 25 um, the 40^3 maps' voxel). The centrelines are polylines of ~15 um
segments turning ~21 deg (median) at each vertex -- tortuous at every scale (chords of 15-120 um all turn ~20 deg; arc /
chord 1.34). ``spline=True`` renders each strand along a uniform Catmull-Rom spline through its points instead (the same
points, no kinks: a tangent defined everywhere, as an axon's), the truth's tangents from the spline's derivative.
"""
import math
from pathlib import Path
import numpy as np
import torch

UNIT = 25.0                                                                                        # um per DiSCo coordinate unit


def read_tck(path):
    """MRtrix .tck (Float32LE): the strands as (N_i, 3) arrays, in the file's coordinates."""
    raw = Path(path).read_bytes(); hdr = raw[:raw.index(b"END\n")].decode("latin1")
    off = int([l for l in hdr.splitlines() if l.startswith("file:")][0].split()[-1])
    assert "Float32LE" in hdr
    v = np.frombuffer(raw[off:], dtype="<f4"); v = v[: len(v) // 3 * 3].reshape(-1, 3)
    brk = np.where(~np.isfinite(v).all(1))[0]; out = []; s = 0
    for b in brk:
        if b > s + 1:
            out.append(v[s:b].astype(np.float64))
        s = b + 1
    return out


def select_crops(folder, crops=(3, 2, 1), seed=0):
    """Crop centres (DiSCo voxels, (x, y, z)) and labels: ``crops`` of each of 4+, 2-3 and 1 bundles in DiSCo's 25-um
    bundle-count map, >= 125 um inside the sphere and >= 150 um apart, the rarest placed first; listed 4+ first."""
    import nibabel as nib
    from scipy.ndimage import binary_erosion
    D = Path(folder); stem = D.name
    bc = np.asarray(nib.load(D / f"{stem}_Strand_Bundle_Count.nii.gz").dataobj).astype(int)       # (x, y, z) index = coordinate
    inner = binary_erosion(np.asarray(nib.load(D / f"{stem}_mask.nii.gz").dataobj) > 0, iterations=5)
    rng = np.random.default_rng(seed); centres = []
    for (lo_, hi_), k_ in list(zip(((4, 99), (2, 3), (1, 1)), crops))[::-1]:
        cand = np.argwhere(inner & (bc >= lo_) & (bc <= hi_)); rng.shuffle(cand); got = 0
        for c_ in cand:
            if got == k_:
                break
            if all(np.abs(c_ - c2).max() >= 6 for c2, _ in centres):
                centres.append((c_, f"{int(bc[tuple(c_)])} bundles")); got += 1
    return centres[::-1]


class Disco:
    """The strands of one DiSCo phantom on ``device``, ready to render crops."""

    def __init__(self, folder, device="cuda", spline=False):
        D = Path(folder); stem = D.name; self.dev = device; self.spline = spline
        strands = read_tck(D / f"{stem}_Strands_Trajectories.tck")
        radius = np.loadtxt(D / f"{stem}_Strands_Diameters.txt") * 1e3 / 2                       # um
        pairs = np.sort(np.loadtxt(D / f"{stem}_Strands_ROIs_Pairs.txt", dtype=int), 1)
        _, conn = np.unique(pairs, axis=0, return_inverse=True); conn = conn.ravel()
        assert len(strands) == len(radius) == len(conn), (len(strands), len(radius), len(conn))
        self.NC = int(conn.max()) + 1; self.NS = len(strands); self.radius = radius
        ctl = []                                                                                   # per segment: P[i-1], P[i], P[i+1], P[i+2]
        for s in strands:                                                                          # (z, y, x) um; the ends extrapolated
            P = s[:, ::-1] * UNIT; P = np.concatenate([2 * P[:1] - P[1:2], P, 2 * P[-1:] - P[-2:-1]])
            ctl.append(np.stack([P[:-3], P[1:-2], P[2:-1], P[3:]], 1))
        self.ctl = torch.tensor(np.concatenate(ctl), device=device)                               # (M, 4, 3) float64
        self.seg_r = torch.tensor(np.concatenate([np.full(len(s) - 1, radius[i]) for i, s in enumerate(strands)]), device=device, dtype=torch.float32)
        self.seg_s = torch.tensor(np.concatenate([np.full(len(s) - 1, i) for i, s in enumerate(strands)]), device=device)
        self.conn = torch.tensor(conn, device=device)
        print(f"{stem}: {self.NS} strands, {len(self.seg_r)} segments, {self.NC} connections, radius {radius.min():.2f}-{radius.max():.2f} um"
              f"{', Catmull-Rom splines' if spline else ''}", flush=True)

    def _curve(self, c, t):
        """Points and tangents on segments ``c (S, 4, 3)`` at ``t (S,)`` in [0, 1)."""
        A, B, C, D = c.unbind(1); t = t[:, None]
        if not self.spline:
            return B + t * (C - B), C - B
        a1, a2, a3 = C - A, 2 * A - 5 * B + 4 * C - D, -A + 3 * B - 3 * C + D
        return B + 0.5 * (a1 * t + a2 * t ** 2 + a3 * t ** 3), 0.5 * (a1 + 2 * a2 * t + 3 * a3 * t ** 2)

    def render(self, centre_um, v, n, ss):
        """The crop of ``n^3`` ``v``-um voxels centred at ``centre_um`` ((z, y, x) um), from a grid ``ss`` times finer:
        ``occ (NC, n, n, n)`` per connection, its direction dyad ``dyad (NC, 6, n, n, n)`` (zz yy xx zy zx yx), and
        ``purity (n, n, n)``: the share of the voxel's occupied fine cells held by its most common strand."""
        dev = self.dev; NC = self.NC
        f = v / ss; nf = n * ss; o = torch.tensor(centre_um, device=dev, dtype=torch.float64) - n * v / 2
        rmax = float(self.seg_r.max()); pad = rmax + 1 + (4.0 if self.spline else 0.0)            # (a spline strays ~2 um from its chord)
        lo, hi = o - pad, o + n * v + pad
        p0, p1 = self.ctl[:, 1], self.ctl[:, 2]
        keep = ((torch.maximum(p0, p1) >= lo) & (torch.minimum(p0, p1) <= hi)).all(1)
        c4, rr, sid = self.ctl[keep], self.seg_r[keep], self.seg_s[keep]
        L = (c4[:, 2] - c4[:, 1]).norm(dim=1) * (1.15 if self.spline else 1.0)
        step = torch.minimum(rr.double() / 2, torch.tensor(f, device=dev, dtype=torch.float64))
        k = (L / step).ceil().long().clamp_min(1); idx = torch.repeat_interleave(torch.arange(len(L), device=dev), k)
        t = (torch.arange(int(k.sum()), device=dev) - torch.repeat_interleave(k.cumsum(0) - k, k)).double() / k[idx].double()
        s, d = self._curve(c4[idx], t); q = ((s - o) / f - 0.5).float()                            # fine-cell coordinates
        tan = torch.nn.functional.normalize(d.float(), dim=1); r_f = (rr[idx] / f).float(); s_s = sid[idx]
        R = int(math.ceil(float(r_f.max()))); g = torch.arange(-R, R + 1, device=dev)
        off = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        lab = torch.zeros(nf ** 3, dtype=torch.int16, device=dev); tg = torch.zeros(nf ** 3, 3, dtype=torch.float16, device=dev)
        ch = max(1, int(1.5e7 // off.shape[0]))
        for i in range(0, q.shape[0], ch):
            qq = q[i:i + ch]; cell = qq.round().long()[:, None] + off[None]                          # (S, O, 3)
            ok = ((cell - qq[:, None]).float().norm(dim=-1) <= r_f[i:i + ch, None]) & ((cell >= 0) & (cell < nf)).all(-1)
            lin = (cell[..., 0] * nf + cell[..., 1]) * nf + cell[..., 2]
            sel = ok.nonzero(as_tuple=True)
            lab[lin[sel]] = (s_s[i:i + ch][sel[0]] + 1).to(torch.int16); tg[lin[sel]] = tan[i:i + ch][sel[0]].half()
        zf = (lab > 0).nonzero()[:, 0]; cz, cy, cx = zf // (nf * nf) // ss, zf // nf % nf // ss, zf % nf // ss   # occupied fine cells only
        coarse = (cz * n + cy) * n + cx; st_ = lab[zf].long() - 1; lc = self.conn[st_]
        occ = torch.zeros(NC, n ** 3, device=dev); occ.index_put_((lc, coarse), torch.ones(len(zf), device=dev), accumulate=True)
        u = tg[zf].float(); dy = torch.stack([u[:, 0] ** 2, u[:, 1] ** 2, u[:, 2] ** 2, u[:, 0] * u[:, 1], u[:, 0] * u[:, 2], u[:, 1] * u[:, 2]], 1)
        dyad = torch.zeros(NC, 6, n ** 3, device=dev)
        for j in range(6):
            dyad[:, j].index_put_((lc, coarse), dy[:, j], accumulate=True)
        key, cnt = torch.unique(coarse * self.NS + st_, return_counts=True)                       # the dominant strand's share per voxel
        top = torch.zeros(n ** 3, device=dev).scatter_reduce_(0, key // self.NS, cnt.float(), "amax")
        tot = torch.zeros(n ** 3, device=dev).index_add_(0, coarse, torch.ones(len(zf), device=dev))
        return {"occ": (occ / ss ** 3).view(NC, n, n, n), "dyad": (dyad / ss ** 3).view(NC, 6, n, n, n),
                "purity": (top / tot.clamp_min(1)).view(n, n, n)}


def image(tot, look, seed, blur_fn):
    """The greyscale image of the total occupancy ``tot (n, n, n)``: ``look`` background + contrast x occupancy,
    ``blur_fn(x, sigma)`` blur, noise (seeded), 8-bit levels."""
    img = look["bg"] + look["contrast"] * tot
    img = blur_fn(img, look["blur"]) if look["blur"] > 0 else img
    img = (img + look["noise"] * torch.randn(img.shape, generator=torch.Generator(device=img.device).manual_seed(seed), device=img.device)).clamp(0, 1)
    return (img * 255).round() / 255
