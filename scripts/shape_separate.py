"""PROTOTYPE: separate the boundary and fibre components of anatomix shape space, then take orientation from the fibre
components only. Tract-interface phantoms: D0 has no step, D8/D15/D30 identical fibres with the upper half brighter.

The model (learned on one paired phantom, applied to all):
  boundary components   directions u of the 16-channel feature space along which the step moves the features:
                        the generalised eigenvectors of C_near u = lambda C_far u, where C_near / C_far are the
                        covariances of the paired difference f_Dstep - f_D0 within 3 mm of the interface and beyond
                        5 mm (the far part is only the different noise draws, so lambda >> 1 means "the step did it")
  fibre components      the rest: P_k = I - Q Q^T, Q an orthonormal basis of the top k boundary directions
  rmE<k>                the same, but the boundary directions are the plain principal components of the change
                        near the interface (most change energy, not most change relative to noise)
  keep<k>               keep only D0's own top-k principal components (the fibre space seen with no boundary)

Orientation: the shape tensor of the projected features, G_rho * sum_c grad (P f)_c grad (P f)_c^T, e3.

    python shape_separate.py score            (chunkreg env)
    python shape_separate.py fig
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy import linalg

sys.path.insert(0, str(Path(__file__).parent))
import shape_tensor as ST
import surface_phantoms as SP

B = 12
OUT = ST.OUT
NEAR, FAR = 3.0, 5.0


def depth():
    z = (np.arange(SP.NZ) + 0.5) * SP.VOX
    return np.abs(z - SP.D_PLANE_Z)


def core(F):
    return F[:, B:-B, B:-B, B:-B]


def boundary_model(train):
    """Generalised eigen-decomposition of the paired difference for D<train> vs D0: (lambda desc, U columns)."""
    dF = core(ST.features(train)) - core(ST.features("D0")); C = dF.shape[0]
    dz = depth()[B:-B]
    near = np.broadcast_to(dz[None, None, :] < NEAR, dF.shape[1:]); far = np.broadcast_to(dz[None, None, :] > FAR, dF.shape[1:])
    Xn = dF[:, near]; Xf = dF[:, far]
    Cn = Xn @ Xn.T / Xn.shape[1]; Cf = Xf @ Xf.T / Xf.shape[1] + 1e-8 * np.eye(C)
    lam, U = linalg.eigh(Cn, Cf)
    o = np.argsort(lam)[::-1]
    we, E = np.linalg.eigh(Cn); oe = np.argsort(we)[::-1]                  # energy: plain PCA of the change
    return lam[o], U[:, o], we[oe] / we.sum(), E[:, oe]


def fibre_space():
    """D0's principal components (desc variance)."""
    X = core(ST.features("D0")).reshape(16, -1)
    w, V = np.linalg.eigh(np.cov(X)); o = np.argsort(w)[::-1]
    return w[o], V[:, o]


def projector(kind, k, model):
    if kind in ("rm", "rmE"):                                              # remove k boundary directions
        Q, _ = np.linalg.qr(model["U" if kind == "rm" else "E"][:, :k])
        return np.eye(16, dtype=np.float32) - (Q @ Q.T).astype(np.float32)
    V = model["V"][:, :k]                                                  # keep D0's top-k components
    return (V @ V.T).astype(np.float32)


def stats(P, model, train):
    """Share of the step's feature change (near the interface) the projection removes, and share of D0's own
    feature variance it keeps."""
    dF = core(ST.features(train)) - core(ST.features("D0")); dz = depth()[B:-B]
    near = np.broadcast_to(dz[None, None, :] < NEAR, dF.shape[1:])
    Xn = dF[:, near]; X0 = core(ST.features("D0")).reshape(16, -1); X0 = X0 - X0.mean(1, keepdims=True)
    return {"step_change_removed": float(1 - (P @ Xn).var(1).sum() / Xn.var(1).sum()),
            "d0_variance_kept": float((P @ X0).var(1).sum() / X0.var(1).sum())}


def orient(name, P, rho):
    F = ST.features(name)
    PF = np.tensordot(P, F, axes=(1, 0))
    m = int(np.ceil(2 * rho)) + 4
    T0 = ST.t0_shape(PF[:, :, ST.JY - m:ST.JY + m + 1, :])
    from scipy import ndimage as ndi
    S = [ndi.gaussian_filter(t, rho, mode="nearest", truncate=2.0)[:, m, :] for t in T0]
    M = np.empty((SP.NX, SP.NZ, 3, 3), np.float32)
    import tensor_variants as TV
    for k, (i, j) in enumerate(TV.PAIRS):
        M[..., i, j] = M[..., j, i] = S[k]
    return np.linalg.eigh(M)[1][..., :, 0]


def score():
    res = {"models": {}, "scores": {}}
    models = {}
    for train in ("D15", "D30"):
        lam, U, we, E = boundary_model(train)
        models[train] = {"U": U, "E": E}
        res["models"][train] = {"generalised_eigenvalues": [round(float(x), 2) for x in lam],
                                "change_energy_share": [round(float(x), 4) for x in we]}
        print(train, "change energy share", np.round(we, 3), flush=True)
        print(train, "boundary eigenvalues", np.round(lam, 1), flush=True)
    w, V = fibre_space()
    res["models"]["D0_pca_explained"] = [round(float(x), 4) for x in w / w.sum()]
    variants = [("none", None, 0, None)]
    for train in ("D15", "D30"):
        variants += [(f"rm{k}_{train}", "rm", k, train) for k in (1, 2, 3, 4, 6)]
        variants += [(f"rmE{k}_{train}", "rmE", k, train) for k in (1, 2, 4, 8)]
    variants += [(f"keep{k}_D0", "keep", k, None) for k in (3, 5, 8)]
    for vname, kind, k, train in variants:
        if kind is None:
            P = np.eye(16, dtype=np.float32)
        else:
            P = projector(kind, k, {"U": models[train]["U"] if train else None, "E": models[train]["E"] if train else None, "V": V})
        st = stats(P, None, "D15") | {"step_change_removed_D30": stats(P, None, "D30")["step_change_removed"]}
        res["scores"][vname] = {"stats": st}
        for rho in (3.0, 6.0):
            for name in ST.NAMES:
                r, _ = ST.score_e3(orient(name, P, rho))
                res["scores"][vname].setdefault(f"r{rho:g}", {})[name] = r
            line = "  ".join(f"{n} {res['scores'][vname][f'r{rho:g}'][n]['within_0.5mm']:5.1f}/{res['scores'][vname][f'r{rho:g}'][n]['deep_3-8mm']:4.1f}"
                             for n in ST.NAMES)
            print(f"{vname:9s} r{rho:g}  removed D15 {st['step_change_removed']:.2f} D30 {st['step_change_removed_D30']:.2f}  "
                  f"D0 kept {st['d0_variance_kept']:.2f} | <0.5mm/deep: {line}", flush=True)
        (OUT / "shape_separate.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score()
    elif sys.argv[1] == "fig":
        import shape_separate_fig
        shape_separate_fig.main()
