"""The stage-2 prototypes (shapetensor.experimental.fibre_sets) on synthetic evidence, no network: an arc turning
through 90° must be one population and a crossing two, for both formulations; and the bend phantom's truth."""
import math

import numpy as np
import pytest
import torch

from shapetensor import separate
from shapetensor.experimental import fibre_sets as FS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VOX, STRIDE = 0.02, 2                                  # the strided grid is 40 µm


def _evidence(shape, dirs_of, kappa=20.0, floor=0.002):
    """Synthetic lifted evidence on a (Z, Y, X) grid: Watson bumps about each voxel's directions (``dirs_of(z, x)`` ->
    list of (z, y, x) unit vectors) over a uniform floor, normalised per voxel."""
    u = separate.hemisphere(160, DEV)
    Z, Y, X = shape
    U = torch.full((160, Z, Y, X), floor, device=DEV)
    tissue = torch.zeros(shape, dtype=torch.bool, device=DEV)
    for z in range(Z):
        for x in range(X):
            for d in dirs_of(z, x):
                w = torch.exp(kappa * ((u @ torch.tensor(d, dtype=torch.float32, device=DEV)) ** 2 - 1))
                U[:, z, :, x] += w[:, None] / w.sum()
                tissue[z, :, x] = True
    return U / U.sum(0, keepdim=True), u, tissue


def _arc(z, x, c=(4, 40), r0=14, r1=20):
    """A band turning through 90°: up z at x < c[1] below c[0], round the centre, along x above it (grid units)."""
    zc, xc = c
    if z < zc and xc - r1 <= x < xc - r0:
        return [(1.0, 0.0, 0.0)]
    rad = math.hypot(z - zc, xc - x)
    if z >= zc and x <= xc and r0 <= rad < r1:
        ph = math.atan2(z - zc, xc - x)
        return [(math.cos(ph), 0.0, math.sin(ph))]
    if x > xc and zc + r0 <= z < zc + r1:
        return [(0.0, 0.0, 1.0)]
    return []


def _crossing(z, x, half=30.0, w=4, n=40):
    """Two straight bands through the middle at ±``half`` from z; both directions where they overlap."""
    out = []
    for s in (1, -1):
        a = math.radians(half) * s
        d = (math.cos(a), 0.0, math.sin(a))
        # distance from the band's centre line through (n/2, n/2)
        dist = abs((x - n / 2) * d[0] - (z - n / 2) * d[2])
        if dist < w:
            out.append(d)
    return out


def _null(U, u):
    return float(FS.peak_mass(torch.full_like(U[:, :1, :1, :1], 1.0 / U.shape[0]), u).max()) * 2.0


def test_connected_sets_keep_an_arc_whole_and_part_a_crossing():
    U, u, tissue = _evidence((40, 6, 44), _arc)
    A, rep = FS.connected_sets(U, u, tissue, _null(U, u), vox_mm=VOX, stride=STRIDE, min_volume_mm3=0.0005)
    assert rep["sets_kept"] == 1, rep
    U, u, tissue = _evidence((40, 6, 40), _crossing)
    A, rep = FS.connected_sets(U, u, tissue, _null(U, u), vox_mm=VOX, stride=STRIDE, min_volume_mm3=0.0005)
    assert rep["sets_kept"] == 2, rep


def test_smooth_fields_keep_an_arc_whole_and_part_a_crossing():
    U, u, tissue = _evidence((40, 6, 44), _arc)
    A, rep = FS.smooth_fields(U, u, tissue, 0.3, vox_mm=VOX, stride=STRIDE, min_volume_mm3=0.0005, sigma_f=1.5)
    assert rep["fields"] == 1 and rep["sets_kept"] == 1, rep
    U, u, tissue = _evidence((40, 6, 40), lambda z, x: _crossing(z, x, half=45.0))
    A, rep = FS.smooth_fields(U, u, tissue, 0.3, vox_mm=VOX, stride=STRIDE, min_volume_mm3=0.0005, sigma_f=1.5)
    assert rep["fields"] == 2, rep


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the phantom splats on the GPU")
def test_bend_phantom_truth():
    from shapetensor.phantoms import BEND_RADII, build
    img, tr, meta = build("bend", 0, texture=0.5)
    fid = tr["fid"]
    assert set(np.unique(fid)) == set(range(len(BEND_RADII) + 1))
    t = tr["dir1"][fid > 0]
    assert np.allclose(np.linalg.norm(t, axis=1), 1, atol=1e-5) and np.all(t[:, 1] == 0)
    assert tr["bend"].any() and not build("bend", 0, texture=0.5, straight=True)[1]["bend"].any()
    inside = np.broadcast_to(fid[:, None, :] > 0, img.shape)
    assert img[inside].mean() > img[~inside].mean() + 10                                  # the fascicles are brighter


def test_merge_by_agreement_joins_a_hand_over_and_keeps_a_crossing():
    """Two layers meeting along a fibre with the same local direction are one set; two co-located layers with
    directions 60° apart stay two; a layer that never meets another stays alone."""
    sh = (8, 8, 40)
    A = torch.zeros((4,) + sh); D = torch.zeros((4, 3) + sh); G = torch.full((4,) + sh, 0.8)
    A[0, :, :, :24] = 2.0; A[1, :, :, 16:] = 2.0                                          # a hand-over over x 16-24
    D[0, 2] = 1.0; D[1, 2] = 1.0                                                         # both along x
    A[2] = 2.0; A[3] = 2.0                                                               # a crossing everywhere
    D[2, 0] = math.cos(math.radians(30)); D[2, 2] = math.sin(math.radians(30))
    D[3, 0] = math.cos(math.radians(30)); D[3, 2] = -math.sin(math.radians(30))
    mask = torch.ones(sh, dtype=torch.bool)
    groups, _ = FS.merge_by_agreement(A[:2], D[:2], G[:2], mask)
    assert sorted(map(sorted, groups)) == [[0, 1]]
    groups, _ = FS.merge_by_agreement(A[2:], D[2:], G[2:], mask)
    assert sorted(map(sorted, groups)) == [[0], [1]]
    B = torch.zeros((2,) + sh); B[0, :, :, :10] = 2.0; B[1, :, :, 30:] = 2.0               # never meet
    groups, _ = FS.merge_by_agreement(B, D[:2], G[:2], mask)
    assert sorted(map(sorted, groups)) == [[0], [1]]
