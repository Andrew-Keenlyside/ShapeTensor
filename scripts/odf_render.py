"""ODF glyphs as spherical-harmonic surfaces, drawn with matplotlib (no OpenGL needed).

Densities sampled on a hemisphere of directions are fitted with real, even spherical harmonics (DIPY's
``descoteaux07`` basis, order 8). Each fit is evaluated on a 724-vertex sphere mesh and drawn as a shaded triangle
surface: radius = the min-max normalised ODF, colour = the direction (DEC: R = x, G = y, B = z), Lambert shading,
painter's order along the viewing axis. A density whose peak is within 15 % of its mean is drawn as a small grey
sphere (no orientation to show).
"""
import numpy as np
from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.reconst.shm import sf_to_sh, sh_to_sf_matrix
from matplotlib.collections import PolyCollection

MESH = get_sphere(name="repulsion724")
ORDER = 8
_B = sh_to_sf_matrix(MESH, sh_order_max=ORDER, basis_type="descoteaux07", return_inv=False)
_LIGHT = np.array([0.35, -0.45, 0.82]); _LIGHT = _LIGHT / np.linalg.norm(_LIGHT)


def fit_sh(q: np.ndarray, dirs_zyx: np.ndarray, order: int = ORDER) -> np.ndarray:
    """SH coefficients ``(N, n_coef)`` of densities ``q (N, G)`` sampled at hemisphere directions ``dirs_zyx (G, 3)``."""
    xyz = np.asarray(dirs_zyx)[:, ::-1]
    sph = Sphere(xyz=np.concatenate([xyz, -xyz]))
    return sf_to_sh(np.concatenate([q, q], 1), sph, sh_order_max=order, basis_type="descoteaux07")


def draw(ax, centres, coefs, size: float, view: str = "z", flat_ratio: float = 1.15):
    """Draw glyphs at display ``centres (N, 2)`` (column, row) for SH ``coefs (N, n_coef)``; ``size`` the glyph radius
    in display units. ``view``: the axis looked along -- "z" (columns = x, rows = y) or "y" (columns = x, rows = z)."""
    v = MESH.vertices; faces = MESH.faces
    cols, rows, depth = {"z": (0, 1, 2), "y": (0, 2, 1)}[view]
    polys, colours, depths = [], [], []
    for (cx, cy), c in zip(centres, coefs):
        r = c @ _B
        flat = r.max() / max(r.mean(), 1e-12) < flat_ratio
        rn = np.full_like(r, 0.35) if flat else 0.15 + 0.85 * (r - r.min()) / max(r.max() - r.min(), 1e-12)
        p = v * rn[:, None] * size
        tri = p[faces]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        nrm *= np.sign((nrm * tri.mean(1)).sum(1))[:, None]                                 # outward
        view_dir = np.zeros(3); view_dir[depth] = -1.0
        front = (nrm @ view_dir) < 0.25                                                     # cull most of the back
        shade = 0.45 + 0.55 * np.clip(nrm @ _LIGHT, 0, 1)
        dirc = np.abs(v[faces].mean(1)); dirc /= np.linalg.norm(dirc, axis=1, keepdims=True) + 1e-12
        col = np.full((len(faces), 3), 0.62) if flat else dirc                               # x, y, z -> R, G, B
        col = np.clip(col * shade[:, None], 0, 1)
        for t, cc, dd in zip(tri[front], col[front], tri[front][:, :, depth].mean(1)):
            polys.append(np.stack([cx + t[:, cols], cy + t[:, rows]], 1)); colours.append(cc); depths.append(dd)
    o = np.argsort(depths)[::-1]                                                            # far first (viewer at -depth)
    ax.add_collection(PolyCollection([polys[i] for i in o], facecolors=[colours[i] for i in o], edgecolors="none", antialiased=False))
