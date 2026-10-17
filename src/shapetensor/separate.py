"""Stage 2 — separating fibre populations in the shape space, without labels.

The shape features of an oriented voxel encode its orientation (results in ``notes/``: encoder block 2 of anatomix
carries a near-linear image of the orientation dyad d dᵀ). Two readers of that code are fitted on the data itself,
with the excised structure tensor at its confident sites as the teacher:

* :class:`OrientationReadout` — features -> a distribution over hemisphere directions (a small MLP with a softmax
  head). Summed over a window it is the window's orientation density, whose modes are the populations present.
* :class:`SignatureDictionary` — the mean standardised feature pattern of every orientation, plus a background
  pattern from the least-oriented voxels. :func:`unmix` decomposes every voxel non-negatively onto the signatures
  of the populations found, which gives one **activation volume per population**: a full-volume layer of that
  population's shape, not a mask of the image.

:func:`activation_null` calibrates the activation a wrong population reaches on single-population voxels, and
:func:`population_masks` thresholds the (averaged) activations at it.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fn


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------

def hemisphere(n: int = 160, device="cpu") -> torch.Tensor:
    """Roughly uniform unit vectors on a hemisphere (z >= 0), a Fibonacci set, ``(n, 3)`` in (z, y, x)."""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = i / n
    phi = i * math.pi * (3 - math.sqrt(5))
    r = torch.sqrt(1 - z * z)
    return torch.stack([z, r * torch.sin(phi), r * torch.cos(phi)], 1).float().to(device)


def sample_idx(w: torch.Tensor, n: int, g: torch.Generator) -> torch.Tensor:
    """``n`` indices drawn without replacement with probability ∝ ``w`` (flat weights, any device), over the nonzero
    entries only. On the GPU by Gumbel-top-k (``argmax`` of ``log w + Gumbel`` taken ``n`` times is sampling without
    replacement ∝ ``w``; ``torch.multinomial`` is capped at 2^24 categories and a block has more), seeded from ``g``."""
    nz = (w > 0).nonzero(as_tuple=True)[0]
    k = min(n, len(nz))
    if k == 0:
        return nz
    if w.is_cuda:
        gd = torch.Generator(device=w.device).manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g)))
        u = torch.rand(len(nz), device=w.device, generator=gd).clamp(1e-12, 1 - 1e-12)
        keys = torch.log(w[nz].float()) - torch.log(-torch.log(u))
        return nz[keys.topk(k).indices]
    return nz[torch.multinomial(w[nz] / w[nz].sum(), k, replacement=False, generator=g)]


_IDENTITY = lambda x: x                                                                   # the prep of features already averaged


def _draw(blocks, per: int, seed: int):
    """Standardisation-ready samples from ``blocks`` = iterable of ``(F (C, Z, Y, X), d (3, Z, Y, X), w (Z, Y, X))``,
    after the ``prep`` average: features ``(n, C)`` and directions ``(n, 3)``."""
    g = torch.Generator().manual_seed(seed)
    Xs, Ds = [], []
    for F, d, w, prep in blocks:
        f = prep(F)
        idx = sample_idx(w.reshape(-1), per, g)
        Xs.append(f.reshape(f.shape[0], -1)[:, idx].T)
        Ds.append(None if d is None else d.reshape(3, -1)[:, idx].T)
    return torch.cat(Xs), (None if Ds[0] is None else torch.cat(Ds))


class _Prep:
    """The feature average before any read-out: a ``smooth``³ box (the fibre texture's period is ~5 voxels at 20 µm;
    the average integrates the phase of the rectified filter responses, see notes/results §25)."""

    def __init__(self, smooth: int):
        self.smooth = int(smooth)

    def __call__(self, F: torch.Tensor) -> torch.Tensor:
        f = F.float()
        if self.smooth > 1:
            f = Fn.avg_pool3d(f[None], self.smooth, stride=1, padding=self.smooth // 2)[0]
        return f


# ---------------------------------------------------------------------------------------------------------------
# the per-voxel orientation distribution
# ---------------------------------------------------------------------------------------------------------------

class OrientationReadout:
    """Features -> a distribution over ``n_dirs`` hemisphere directions (softmax), trained on the self-calibrated
    teacher as a soft target (a kernel of ``target_deg`` around the teacher direction). A voxel holding two
    populations keeps two bumps rather than their average, and a window's population density is the sum of its
    voxels' distributions."""

    def __init__(self, smooth: int = 3, n_dirs: int = 160, target_deg: float = 12.0, hidden: int = 64, steps: int = 2000,
                 lr: float = 2e-3, weight_decay: float = 1e-4, seed: int = 0):
        self.prep = _Prep(smooth)
        self.n_dirs, self.target_deg, self.hidden, self.steps = int(n_dirs), float(target_deg), int(hidden), int(steps)
        self.lr, self.wd, self.seed = float(lr), float(weight_decay), int(seed)
        self.mean = self.std = self.net = self.grid = None

    def fit(self, blocks, n_sample: int = 240_000, prepped: bool = False):
        """``blocks``: ``(F, d, w)`` triples of features (already averaged if ``prepped``), teacher directions and
        sampling weights."""
        X, D = _draw([(F, d, w, _IDENTITY if prepped else self.prep) for F, d, w in blocks], max(1, n_sample // max(1, len(blocks))), self.seed)
        self.grid = hemisphere(self.n_dirs, X.device)
        k = 1.0 / (2 * math.sin(math.radians(self.target_deg)) ** 2)
        Y = torch.exp(k * ((D @ self.grid.T) ** 2 - 1)); Y = Y / Y.sum(1, keepdim=True)
        self.mean, self.std = X.mean(0), X.std(0).clamp_min(1e-6)
        Xn = (X - self.mean) / self.std
        torch.manual_seed(self.seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], self.hidden), torch.nn.GELU(), torch.nn.Linear(self.hidden, self.hidden),
                                       torch.nn.GELU(), torch.nn.Linear(self.hidden, self.n_dirs)).to(X.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        n = len(Xn); bs = min(8192, n)
        for _ in range(self.steps):
            i = torch.randint(0, n, (bs,), device=X.device)
            loss = -(Y[i] * torch.log_softmax(self.net(Xn[i]), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            P = torch.cat([torch.softmax(self.net(Xn[a:a + 65536]), 1) for a in range(0, n, 65536)])
            self.fit_overlap = float((P * Y).sum(1).mean() / (Y * Y).sum(1).mean())            # 1 = perfect
        return self

    def _chunks(self, F: torch.Tensor, z_chunk: int | None, prepped: bool = False):
        """The per-voxel distributions in slabs of ``z_chunk`` planes (by default as many as keep a slab's
        distributions under ~2.5 GB); ``prepped``: ``F`` is already the averaged features."""
        f = F if prepped else self.prep(F); C = f.shape[0]
        z_chunk = z_chunk or _slab(f.shape[1:], 4_000_000)
        with torch.no_grad():
            for z in range(0, f.shape[1], z_chunk):
                fz = f[:, z:z + z_chunk]
                yield z, fz.shape[1:], torch.softmax(self.net((fz.reshape(C, -1).T - self.mean) / self.std), 1)

    def predict_dist(self, F: torch.Tensor, z_chunk: int | None = None, prepped: bool = False) -> torch.Tensor:
        """Per-voxel distributions ``(n_dirs, Z, Y, X)``."""
        out = torch.empty((self.n_dirs,) + tuple(F.shape[1:]), device=F.device)
        for z, shp, P in self._chunks(F, z_chunk, prepped):
            out[:, z:z + shp[0]] = P.T.reshape(self.n_dirs, *shp)
        return out

    def density(self, F: torch.Tensor, region: torch.Tensor | None = None, z_chunk: int | None = None, prepped: bool = False) -> torch.Tensor:
        """The window's orientation density ``(n_dirs,)``: the distributions summed over ``region`` (all voxels when
        None), without holding them all."""
        dens = torch.zeros(self.n_dirs, device=F.device)
        for z, shp, P in self._chunks(F, z_chunk, prepped):
            if region is None:
                dens += P.sum(0)
            else:
                dens += (P * region[z:z + shp[0]].reshape(-1, 1).float()).sum(0)
        return dens

    def window_densities(self, F: torch.Tensor, region: torch.Tensor, window: int, z_chunk: int | None = None, prepped: bool = False) -> torch.Tensor:
        """The orientation density of every ``window``³ sub-window of the block (``(n_windows, n_dirs)``), in one pass
        over the features. The block's own density is their sum."""
        Z, Y, X = region.shape
        nz, ny, nx = (Z + window - 1) // window, (Y + window - 1) // window, (X + window - 1) // window
        ar = lambda n: torch.arange(n, device=F.device) // window
        wid = (ar(Z)[:, None, None] * ny + ar(Y)[None, :, None]) * nx + ar(X)[None, None, :]
        out = torch.zeros(nz * ny * nx, self.n_dirs, device=F.device)
        for z, shp, P in self._chunks(F, z_chunk, prepped):
            out.index_add_(0, wid[z:z + shp[0]].reshape(-1), P * region[z:z + shp[0]].reshape(-1, 1).float())
        return out

    def predict(self, F: torch.Tensor, z_chunk: int | None = None, prepped: bool = False):
        """The distribution's peak direction per voxel ``(3, Z, Y, X)`` and its confidence (the peak's share)."""
        P = self.predict_dist(F, z_chunk, prepped)
        j = P.argmax(0)
        return self.grid[j].permute(3, 0, 1, 2).contiguous(), P.max(0).values.clamp(0, 1)


def modes_from_density(dens: torch.Tensor, u: torch.Tensor, rel: float = 0.3, min_sep_deg: float = 25.0, nmax: int = 3,
                       mass_floor: float = 0.1):
    """Peaks of a density ``dens (m,)`` over grid directions ``u (m, 3)``: greedy, above ``rel`` × the top, at least
    ``min_sep_deg`` apart, each owning at least ``mass_floor`` of the density (the grid points nearer to it than to
    any kept mode). Returns ``modes (nmax, 3)`` (zero rows where absent), their relative heights and N."""
    modes = torch.zeros(nmax, 3, device=dens.device); heights = torch.zeros(nmax, device=dens.device)
    top = float(dens.max()); taken = torch.zeros(len(u), dtype=torch.bool, device=dens.device)
    cos_sep = math.cos(math.radians(min_sep_deg)); n = 0
    total = float(dens.clamp_min(0).sum())
    for r in range(nmax):
        d = dens.masked_fill(taken, -1.0)
        j = int(d.argmax())
        if d[j] < rel * top or d[j] <= 0:
            break
        if mass_floor > 0 and r > 0:
            near = ((u @ u[j]) ** 2 >= (u @ modes[:r].T).pow(2).max(1).values)
            if float(dens.clamp_min(0)[near].sum()) / max(total, 1e-12) < mass_floor:
                taken[j] = True; continue
        modes[r] = u[j]; heights[r] = d[j] / top; n += 1
        taken |= (u @ u[j]).abs() >= cos_sep
    return modes, heights, n


# ---------------------------------------------------------------------------------------------------------------
# the signature dictionary and the unmixing into population layers
# ---------------------------------------------------------------------------------------------------------------

class SignatureDictionary:
    """The mean standardised feature pattern of every orientation (kernel-smoothed over a hemisphere grid), and a
    background pattern from voxels with no orientation."""

    def __init__(self, smooth: int = 3, n_dirs: int = 160, kernel_deg: float = 12.0, seed: int = 0):
        self.prep = _Prep(smooth)
        self.n_dirs, self.kernel_deg, self.seed = int(n_dirs), float(kernel_deg), int(seed)
        self.mean = self.std = self.grid = self.D = self.background = None

    def standardise(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std

    def _kernel(self, c2: torch.Tensor) -> torch.Tensor:
        return torch.exp((c2 - 1) / (2 * math.sin(math.radians(self.kernel_deg)) ** 2))

    def fit(self, blocks, n_sample: int = 240_000, background=None, prepped: bool = False):
        """``blocks``: ``(F, d, w)`` triples of features (already averaged if ``prepped``), teacher directions and
        sampling weights; ``background``: ``(F, w)`` pairs of features and weights of voxels with no orientation."""
        prep = _IDENTITY if prepped else self.prep
        X, D = _draw([(F, d, w, prep) for F, d, w in blocks], max(1, n_sample // max(1, len(blocks))), self.seed)
        self.mean, self.std = X.mean(0), X.std(0).clamp_min(1e-6)
        Xn = self.standardise(X)
        self.grid = hemisphere(self.n_dirs, X.device)
        W = self._kernel((D @ self.grid.T) ** 2)                                           # (n, n_dirs)
        self.D = (W.T @ Xn) / W.sum(0)[:, None].clamp_min(1e-6)                            # (n_dirs, C)
        if background:
            B, _ = _draw([(F, None, w, prep) for F, w in background], max(1, n_sample // max(1, len(background))), self.seed)
            self.background = self.standardise(B).mean(0)
        return self

    def signature(self, direction: torch.Tensor) -> torch.Tensor:
        """The signature of an arbitrary direction (kernel-interpolated over the grid)."""
        w = self._kernel((self.grid @ direction) ** 2)
        return (w[:, None] * self.D).sum(0) / w.sum().clamp_min(1e-6)

    def matrix(self, modes: torch.Tensor) -> torch.Tensor:
        """The unmixing matrix ``(C, N [+ 1])``: the modes' signatures and the background."""
        S = torch.stack([self.signature(m) for m in modes], 1)
        return S if self.background is None else torch.cat([S, self.background[:, None]], 1)


def nnls_batched(A: torch.Tensor, Y: torch.Tensor, iters: int = 40, tol: float = 1e-4) -> torch.Tensor:
    """Non-negative least squares ``min ||A x - y||²`` for many ``y``: ``A (C, N)``, ``Y (M, C)`` -> ``X (M, N)`` ≥ 0.
    Cyclic coordinate descent from the clamped normal-equation solution (each coordinate minimised exactly with
    the others fixed), at most ``iters`` sweeps, stopped when a sweep moves no coordinate by more than ``tol`` of the
    mean activation. For the pipeline's N ≤ 5 it reaches the exact solution in ~25 sweeps, where 60 projected
    gradient steps were still 10% off on correlated signatures, at a fifth of the time."""
    AtA = A.T @ A; AtY = Y @ A; N = A.shape[1]
    X0 = (AtY @ torch.linalg.pinv(AtA)).clamp_min(0)
    if N == 1:
        return (AtY / AtA[0, 0].clamp_min(1e-12)).clamp_min(0)
    cols = [X0[:, k].contiguous() for k in range(N)]
    aty = [AtY[:, k].contiguous() for k in range(N)]
    G = AtA.detach().cpu().double().numpy(); diag = [max(float(G[k, k]), 1e-12) for k in range(N)]
    scale = float(X0.mean().clamp_min(1e-12)) * tol
    for sweep in range(iters):
        moved = 0.0
        for k in range(N):
            r = aty[k].clone()
            for j in range(N):
                if j != k:
                    r.add_(cols[j], alpha=-float(G[j, k]))
            new = (r / diag[k]).clamp_min_(0)
            if (sweep + 1) % 5 == 0:
                moved = max(moved, float((new - cols[k]).abs().max()))
            cols[k] = new
        if (sweep + 1) % 5 == 0 and moved < scale:
            break
    return torch.stack(cols, 1)


def _slab(shape, voxels: int) -> int:
    """How many z planes of a ``(Z, Y, X)`` block hold about ``voxels`` voxels (at least one plane)."""
    return max(1, int(voxels // max(1, int(shape[1]) * int(shape[2]))))


def unmix(F: torch.Tensor, dictionary: SignatureDictionary, modes: torch.Tensor, z_chunk: int | None = None, iters: int = 60, prepped: bool = False):
    """Per-voxel non-negative unmixing of the features onto the signatures of ``modes (N, 3)``: the activation
    layers ``A (N, Z, Y, X)`` ≥ 0 and the residual energy share ``R (Z, Y, X)`` in [0, 1]. Slabs of ``z_chunk`` planes
    (by default about 8M voxels each); ``prepped``: ``F`` is already the averaged features."""
    f = F if prepped else dictionary.prep(F)
    C, Z, Y, X = f.shape
    z_chunk = z_chunk or _slab(f.shape[1:], 8_000_000)
    S = dictionary.matrix(modes); N = modes.shape[0]
    A_out = torch.empty((N, Z, Y, X), device=F.device); R_out = torch.empty((Z, Y, X), device=F.device)
    for z in range(0, Z, z_chunk):
        fz = f[:, z:z + z_chunk]
        Xn = dictionary.standardise(fz.reshape(C, -1).T)
        a = nnls_batched(S, Xn, iters)
        res = ((Xn - a @ S.T) ** 2).sum(1) / (Xn ** 2).sum(1).clamp_min(1e-9)
        A_out[:, z:z + z_chunk] = a[:, :N].T.reshape(N, *fz.shape[1:]); R_out[z:z + z_chunk] = res.clamp(0, 1).reshape(fz.shape[1:])
    return A_out, R_out


def direction_gains(dictionary: SignatureDictionary, readout: OrientationReadout, blocks, per: int = 60_000, seed: int = 0, floor: float = 0.2, prepped: bool = False):
    """The response of the readers per direction, measured on the calibration samples against the teacher, so
    that every direction counts alike: ``density_gain (n_dirs,)`` = the readout's density of the samples over the
    teacher's own kernel-smoothed histogram, and ``activation_gain (n_dirs,)`` = the mean activation a sample reaches
    on its own direction's signature. Both are kernel-smoothed over the grid, normalised to a mean of 1 and
    floored at ``floor``."""
    g = torch.Generator().manual_seed(seed)
    u = dictionary.grid; Xs, Ds, Ps = [], [], []
    for F, d, w in blocks:
        f = F if prepped else dictionary.prep(F)
        idx = sample_idx(w.reshape(-1), per, g)
        Xs.append(dictionary.standardise(f.reshape(f.shape[0], -1)[:, idx].T)); Ds.append(d.reshape(3, -1)[:, idx].T)
        fr = f if (prepped or readout.prep.smooth == dictionary.prep.smooth) else readout.prep(F)
        with torch.no_grad():
            Ps.append(torch.softmax(readout.net((fr.reshape(fr.shape[0], -1)[:, idx].T - readout.mean) / readout.std), 1))
    Xn, D, P = torch.cat(Xs), torch.cat(Ds), torch.cat(Ps)
    K = dictionary._kernel((D @ u.T) ** 2)                                                 # (n, n_dirs): each sample's kernel on the grid
    hist = K.sum(0); dens = P.sum(0)
    density_gain = (dens / hist.clamp_min(1e-6))
    # the activation on the own signature: samples grouped by nearest grid direction
    j = ((D @ u.T) ** 2).argmax(1); act = torch.zeros(len(u), device=u.device); cnt = torch.zeros(len(u), device=u.device)
    for jj in j.unique():
        sel = j == jj
        S = torch.stack([dictionary.D[jj], dictionary.background], 1) if dictionary.background is not None else dictionary.D[jj][:, None]
        a = nnls_batched(S, Xn[sel])[:, 0]
        act[jj] = a.mean(); cnt[jj] = sel.sum()
    Kg = dictionary._kernel((u @ u.T) ** 2)                                                # smoothing over the grid
    def smooth(x, wgt):
        y = (Kg * (x * wgt)[None]).sum(1) / (Kg * wgt[None]).sum(1).clamp_min(1e-6)
        return (y / y.mean().clamp_min(1e-9)).clamp_min(floor)
    return smooth(density_gain, hist), smooth(act, cnt)


def activation_null(dictionary: SignatureDictionary, blocks, offsets_deg=(45.0, 60.0, 75.0), q: float = 0.8, per: int = 60_000,
                    smooth: int = 3, seed: int = 0, activation_gain: torch.Tensor | None = None, prepped: bool = False) -> float:
    """The activation a *wrong* population's signature reaches on single-population voxels: each calibration block's
    confident voxels are unmixed onto their own direction's signature, signatures rotated by ``offsets_deg`` in the
    fibre plane, and the background; the ``q`` quantile of the rotated activations (after a ``smooth``³ average) is
    the threshold above which an activation counts as a population's presence."""
    g = torch.Generator().manual_seed(seed)
    vals = []
    for F, d, w in blocks:
        f = F if prepped else dictionary.prep(F)
        idx = sample_idx(w.reshape(-1), per, g)
        Xn = dictionary.standardise(f.reshape(f.shape[0], -1)[:, idx].T)
        dd = d.reshape(3, -1)[:, idx].T
        _, V = torch.linalg.eigh(dd.T @ dd); d0 = V[:, -1]                                 # the axial mean direction
        ref = torch.tensor([0.0, 1.0, 0.0], device=d0.device) if abs(float(d0[1])) < 0.9 else torch.tensor([1.0, 0.0, 0.0], device=d0.device)
        e2 = torch.linalg.cross(d0, ref); e2 = e2 / e2.norm().clamp_min(1e-9)
        dirs = torch.stack([d0] + [math.cos(math.radians(o)) * d0 + math.sin(math.radians(o)) * e2 for o in offsets_deg])
        a = nnls_batched(dictionary.matrix(dirs), Xn)
        if activation_gain is not None:
            a = a.clone(); a[:, :len(dirs)] = a[:, :len(dirs)] / gain_at(dictionary, activation_gain, dirs)[None]
        vals.append(a[:, 1:1 + len(offsets_deg)].reshape(-1))
    v = torch.cat(vals)
    if smooth > 1:                                                                        # the null of the averaged activation
        n = smooth ** 3; m = (len(v) // n) * n
        v = v[torch.randperm(len(v), device=v.device)[:m]].reshape(-1, n).mean(1)
    return float(torch.quantile(v, q))


def gain_at(dictionary: SignatureDictionary, gain: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """A per-direction gain ``(n_dirs,)`` interpolated at ``directions (N, 3)`` (the dictionary's kernel)."""
    w = dictionary._kernel((directions @ dictionary.grid.T) ** 2)                            # (N, n_dirs)
    return (w * gain[None]).sum(1) / w.sum(1).clamp_min(1e-6)


def population_masks(A: torch.Tensor, tau: float, smooth: int = 3):
    """Where each population is present: ``masks (N, Z, Y, X)`` = the ``smooth``³-averaged activation above the
    null ``tau``; ``union`` = any population; ``strongest`` int8 = 1 + the index of the largest activation inside
    the union (0 outside)."""
    As = Fn.avg_pool3d(A[None], smooth, stride=1, padding=smooth // 2)[0] if smooth > 1 else A
    masks = As > tau
    union = masks.any(0)
    strongest = (As.argmax(0) + 1).to(torch.int8) * union.to(torch.int8)
    return masks, union, strongest
