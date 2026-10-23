"""Image formation from materials: the image as a function of what each population is made of, not a sum of grey levels.

Every population (layer) of a scene carries a **material** -- the properties each imaging physics reads -- and so does
the background (modulated by its texture and regions). Per voxel the volume fractions ``o_L`` (summing to at most
one; the remainder is background) mix the materials, and the image follows from the physics:

* ``additive``: grey level = sum of fractions x contrast (the old compositor; a special case);
* ``xray``: attenuation ``mu`` and phase decrement ``delta`` mix linearly; the reconstructed volume is a blend of the
  absorption image, the phase map after single-distance phase retrieval (a low-pass of ``delta``: Paganin) and a
  propagation edge term (``-gamma laplacian(delta)``, the bright/dark fringe pair at every interface);
* ``darkfield``: X-ray small-angle scattering, orientation-dependent -- a fibre scatters across its axis, so its signal
  goes as ``1 - (g.t)^2`` for the grating's sensitive direction ``g`` (``1 - g^T T g`` for any tensor);
* ``histology``: Beer--Lambert in colour -- each population holds stain concentrations (two stains, each with an RGB
  absorption vector), the section's optical density per channel is their sum, the transmission ``exp(-OD)``, and
  the grey image a random channel mix (a camera, a channel, a stain's luminance). Overlaps multiply transmissions:
  not additive;
* ``pli``: polarised light -- each population's retardance goes as its in-plane fraction (``1 - z^T T z``, z the light
  axis) with azimuth ``phi``; retardances add as vectors in the ``2 phi`` plane, so fibres crossing at 90 degrees
  cancel; the image is the retardation map ``|sin(R / 2)|`` or one polariser angle's transmittance;
* ``mri``: each population a compartment with proton density, T1, T2* and a susceptibility anisotropy (R2* gains
  ``c1 sin^2 theta + c2 sin^4 theta`` and the frequency shifts by ``(3 cos^2 theta - 1)`` against B0); a spoiled
  gradient echo with random TR / TE / flip; the voxel's signal the magnitude of the compartments' complex sum.

The orientation terms use the population's tensor (``u^T T u``: ``cos^2`` of a line's angle to ``u``, half the
in-plane share for a sheet), so they hold for any shape. The result is a float image in grey levels (about 0-255)
before the acquisition (:func:`~shapetensor.learn.generator.nuisance`: PSF, noise, artefacts, the display window).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..filters import gaussian

MODES = ("additive", "xray", "darkfield", "histology", "pli", "mri")
#: the physics each modality draws from (with ``additive`` mixed in everywhere)
BY_MODALITY = {"hipct": ("xray", "xray", "xray", "darkfield"), "ct": ("xray", "xray", "darkfield"),
               "histology": ("histology", "histology", "pli"), "mri": ("mri",), "mixed": MODES[1:],
               "em": ("histology", "xray"),                              # heavy-metal stain absorbs / backscatters: a dark-stain absorption image
               "fluorescence": ("darkfield", "additive"),                # emission on a dark field (the populations bright)
               "cryoet": ("xray",)}                                      # weak phase objects: the phase map and its propagation fringes


def _quad(T: torch.Tensor, g) -> torch.Tensor:
    """``g^T T g`` for tensors ``T (6, ...)`` (zz, yy, xx, zy, zx, yx) and a unit direction ``g (3,)``."""
    gz, gy, gx = (float(v) for v in g)
    return T[0] * gz * gz + T[1] * gy * gy + T[2] * gx * gx + 2 * (T[3] * gz * gy + T[4] * gz * gx + T[5] * gy * gx)


def draw_material(rng: np.random.Generator, mode: str, background: bool = False) -> dict:
    """One population's (or the background's) material for ``mode``, drawn on the CPU."""
    u = lambda a, b: float(rng.uniform(a, b))
    lu = lambda a, b: float(math.exp(rng.uniform(math.log(a), math.log(b))))
    if mode == "xray":
        return {"mu": lu(0.2, 3.0), "delta": lu(0.2, 3.0)}
    if mode == "darkfield":
        return {"scatter": lu(0.05, 1.0) if background else lu(0.3, 4.0), "aniso": 0.0 if background else u(0.3, 1.0), "mu": lu(0.2, 3.0)}
    if mode == "histology":
        return {"c": [lu(0.02, 0.6), lu(0.02, 0.6)] if background else [lu(0.02, 2.5), lu(0.02, 2.5)]}
    if mode == "pli":
        return {"biref": 0.0 if background else lu(0.3, 3.0), "atten": lu(0.05, 0.6) if background else lu(0.1, 2.0)}
    if mode == "mri":
        return {"pd": u(0.4, 1.0), "t1": lu(300, 3500), "t2s": lu(8, 120), "chi1": 0.0 if background else u(0, 25), "chi2": 0.0 if background else u(-10, 10),
                "df": 0.0 if background else u(-12, 12)}
    return {"contrast": 0.0}


def draw_sequence(rng: np.random.Generator, mode: str) -> dict:
    """The acquisition parameters of a physics (directions, sequence timings, stains, propagation)."""
    u = lambda a, b: float(rng.uniform(a, b))
    lu = lambda a, b: float(math.exp(rng.uniform(math.log(a), math.log(b))))
    unit = lambda: (lambda v: v / np.linalg.norm(v))(rng.normal(size=3))
    if mode == "xray":
        return {"w_abs": u(0, 1), "w_phase": u(0, 1), "paganin": lu(0.5, 4.0), "edge": u(0, 1.5), "edge_sigma": lu(0.5, 1.5)}
    if mode == "darkfield":
        return {"g": unit(), "mix_abs": u(0, 0.4)}
    if mode == "histology":
        A = np.abs(rng.normal(size=(2, 3))) + 0.05; A /= np.linalg.norm(A, axis=1, keepdims=True)    # stain RGB absorption vectors
        w = rng.dirichlet(np.ones(3)) if rng.random() < 0.6 else np.eye(3)[rng.integers(0, 3)]       # the camera / a channel
        return {"A": A, "w": w, "thick": lu(0.5, 2.0)}
    if mode == "pli":
        return {"z": unit(), "rho": u(0, math.pi), "map": bool(rng.random() < 0.5), "thick": lu(0.5, 2.5)}
    if mode == "mri":
        kind = ["t1", "t2", "pd", "t2s"][rng.integers(0, 4)]
        TR = {"t1": lu(300, 900), "t2": lu(2500, 6000), "pd": lu(2500, 6000), "t2s": lu(30, 80)}[kind]
        TE = {"t1": lu(2, 15), "t2": lu(60, 120), "pd": lu(2, 15), "t2s": lu(15, 40)}[kind]
        flip = {"t1": u(60, 90), "t2": u(70, 90), "pd": u(70, 90), "t2s": u(10, 30)}[kind]
        return {"kind": kind, "TR": TR, "TE": TE, "flip": math.radians(flip), "b0": unit()}
    return {}


def form_image(mode: str, occ: torch.Tensor, tens: torch.Tensor, bg: torch.Tensor, mats: list, bgmat: dict, seq: dict,
               contrasts=None, raw: bool = False, tex: torch.Tensor | None = None) -> torch.Tensor:
    """The image of populations with fractions ``occ (L, Z, Y, X)`` (summing to at most 1), tensors ``tens (L, 6, ...)``
    and materials ``mats`` over a background whose texture ``bg (Z, Y, X)`` (about zero mean, unit spread) modulates
    its material ``bgmat``; ``seq`` from :func:`draw_sequence`. Grey levels, about 0-255 (``raw``: the physical map).
    ``tex (L, ...)``: a multiplicative texture of each population's material (its density varies inside it)."""
    _to_grey_ = (lambda x: x) if raw else _to_grey
    rest = (1 - occ.sum(0)).clamp(0, 1)
    if tex is not None:                                                                            # texture inside each population: its
        occ = occ * tex                                                                            # material's density varies (SynthSeg's per-label
                                                                                                   # noise, anatomix's per-label textures)
    tex = (1 + 0.25 * bg).clamp_min(0.05)                                                       # the background's material varies with its texture
    if mode == "additive":
        img = 128 + 30 * bg
        for k in range(occ.shape[0]):
            img = img + contrasts[k] * occ[k]
        return img
    if mode == "xray":
        mu = rest * tex * bgmat["mu"] + sum(occ[k] * m["mu"] for k, m in enumerate(mats))
        de = rest * tex * bgmat["delta"] + sum(occ[k] * m["delta"] for k, m in enumerate(mats))
        ph = gaussian(de, seq["paganin"], truncate=3.0)                                           # phase retrieval: a low-pass of delta
        sm = gaussian(de, seq["edge_sigma"], truncate=3.0)
        lap = sum(torch.gradient(torch.gradient(sm, dim=a)[0], dim=a)[0] for a in range(3))
        img = seq["w_abs"] * mu + seq["w_phase"] * ph - seq["edge"] * lap
        return _to_grey_(img)
    if mode == "darkfield":
        g = seq["g"]
        sc = rest * tex * bgmat["scatter"] + sum(occ[k] * m["scatter"] * ((1 - m["aniso"]) + m["aniso"] * (1 - _quad(tens[k], g))) for k, m in enumerate(mats))
        mu = rest * tex * bgmat["mu"] + sum(occ[k] * m["mu"] for k, m in enumerate(mats))
        return _to_grey_(sc + seq["mix_abs"] * mu)
    if mode == "histology":
        A, w, th = seq["A"], seq["w"], seq["thick"]
        c = [rest * tex * bgmat["c"][s] + sum(occ[k] * m["c"][s] for k, m in enumerate(mats)) for s in range(2)]
        I = sum(float(w[ch]) * torch.exp(-th * (c[0] * float(A[0, ch]) + c[1] * float(A[1, ch]))) for ch in range(3))
        return 255 * I                                                                            # bright background, dark stain
    if mode == "pli":
        z = seq["z"]; th = seq["thick"]
        # the in-plane frame: two unit vectors across the light axis
        a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(z, a); e1 /= np.linalg.norm(e1); e2 = np.cross(z, e1)
        Rc = torch.zeros_like(rest); Rs = torch.zeros_like(rest); att = rest * tex * bgmat["atten"]
        for k, m in enumerate(mats):
            T = tens[k]
            t11, t22 = _quad(T, e1), _quad(T, e2)
            t12 = (T[0] * e1[0] * e2[0] + T[1] * e1[1] * e2[1] + T[2] * e1[2] * e2[2]
                   + T[3] * (e1[0] * e2[1] + e1[1] * e2[0]) + T[4] * (e1[0] * e2[2] + e1[2] * e2[0]) + T[5] * (e1[1] * e2[2] + e1[2] * e2[1]))
            inpl = (t11 + t22).clamp_min(1e-6)                                                    # the in-plane share (1 - z^T T z)
            r = occ[k] * m["biref"] * th * inpl
            Rc = Rc + r * (t11 - t22) / inpl; Rs = Rs + r * 2 * t12 / inpl                         # retardance vectors in the 2 phi plane
            att = att + occ[k] * m["atten"]
        R = torch.sqrt(Rc ** 2 + Rs ** 2); Phi = 0.5 * torch.atan2(Rs, Rc)
        trans = torch.exp(-th * att)
        if seq["map"]:
            return 255 * torch.abs(torch.sin(R / 2)) * (0.5 + 0.5 * trans)
        return 255 * trans * (0.15 + 0.85 * torch.sin(2 * (Phi - seq["rho"])) ** 2 * torch.sin(R / 2) ** 2)
    if mode == "mri":
        b0 = seq["b0"]; TR, TE, fl = seq["TR"], seq["TE"], seq["flip"]
        def sig(m, cos2):
            E1 = math.exp(-TR / m["t1"])
            s = m["pd"] * math.sin(fl) * (1 - E1) / (1 - math.cos(fl) * E1)
            if cos2 is None:
                return torch.tensor(s * math.exp(-TE / m["t2s"]), device=occ.device), None
            sin2 = (1 - cos2).clamp(0, 1)
            r2 = 1.0 / m["t2s"] + 1e-3 * (m["chi1"] * sin2 + m["chi2"] * sin2 ** 2)              # orientation-dependent R2* (1/ms)
            phase = 2 * math.pi * 1e-3 * TE * m["df"] * (3 * cos2 - 1) / 2                         # orientation-dependent frequency shift
            return s * torch.exp(-TE * r2), phase
        sb, _ = sig(bgmat, None)
        re = rest * tex * sb; im = torch.zeros_like(re)
        for k, m in enumerate(mats):
            s, ph = sig(m, _quad(tens[k], b0))
            re = re + occ[k] * s * torch.cos(ph); im = im + occ[k] * s * torch.sin(ph)
        return _to_grey_(torch.sqrt(re ** 2 + im ** 2))
    raise ValueError(mode)


def _to_grey(img: torch.Tensor) -> torch.Tensor:
    """A physical map to grey levels: its 1-99 % range to 30-220 (the acquisition's window comes later)."""
    flat = img.flatten()[:: max(1, img.numel() // 200000)]
    lo, hi = float(torch.quantile(flat, 0.01)), float(torch.quantile(flat, 0.99))
    return 30 + 190 * (img - lo) / max(hi - lo, 1e-6)
