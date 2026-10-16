"""Figure for shape_separate.py: the boundary and fibre components of anatomix shape space on D0 / D15 / D30, and what
the orientation of the fibre components does.

    python shape_separate.py fig        (chunkreg env)
"""
import json

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import edge_aware_phantom_fig as EPF
import shape_separate as SS
import shape_tensor as ST

ROWS = ("D0", "D15", "D30")
K = 4                                                                      # boundary directions removed in the maps


def main():
    out = ST.FIGS; out.mkdir(parents=True, exist_ok=True)
    _, _, we, E = SS.boundary_model("D15")
    Q, _ = np.linalg.qr(E[:, :K]); P = np.eye(16, dtype=np.float32) - (Q @ Q.T).astype(np.float32)
    F = {n: ST.features(n) for n in ROWS}
    X0 = SS.core(F["D0"]).reshape(16, -1); mu0 = X0.mean(1, keepdims=True)
    d0_share = ((E.T @ (X0 - mu0)) ** 2).mean(1); d0_share /= d0_share.sum()
    # shared RGB basis of the fibre components (projected features), fitted on all three phantoms
    rng = np.random.default_rng(0)
    Xs = np.concatenate([P @ SS.core(F[n]).reshape(16, -1)[:, rng.choice(SS.core(F[n])[0].size, 60_000, replace=False)] for n in ROWS], 1)
    mu = Xs.mean(1, keepdims=True); w, V = np.linalg.eigh(np.cov(Xs - mu)); V = V[:, np.argsort(w)[::-1][:3]].T
    q = V @ (Xs - mu); lo, hi = np.percentile(q, 1, axis=1), np.percentile(q, 99, axis=1)
    tru, draw, _ = EPF.truth_and_bounds("D15")
    sc = json.loads((ST.OUT / "shape_separate.json").read_text())["scores"]
    fig = plt.figure(figsize=(23, 22), dpi=100); gs = fig.add_gridspec(4, 4)
    for r, n in enumerate(ROWS):
        f = F[n][:, :, ST.JY, :]
        ax = fig.add_subplot(gs[r, 0]); EPF.base_image(ax, ST.volume(n)[:, ST.JY, :], draw); ax.set_ylabel(f"{n}: step {n[1:]}", fontsize=13)
        ax.set_title("intensity" if r == 0 else "", fontsize=11)
        ax = fig.add_subplot(gs[r, 1]); b = np.linalg.norm(Q.T @ (f.reshape(16, -1) - mu0), axis=0).reshape(f.shape[1:])
        ax.imshow(b.T, cmap="magma", origin="lower", extent=EPF.EXT, vmin=0, vmax=0.6); draw(ax); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"boundary component: |f| along the top {K} step directions\n(learned from D15 - D0)" if r == 0 else "", fontsize=11)
        ax = fig.add_subplot(gs[r, 2]); pf = V @ (P @ f.reshape(16, -1) - mu)
        rgb = np.clip((pf - lo[:, None]) / (hi - lo)[:, None], 0, 1).T.reshape(f.shape[1:] + (3,))
        ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", extent=EPF.EXT); draw(ax); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"fibre component: the rest of shape space (RGB, shared basis)" if r == 0 else "", fontsize=11)
        ax = fig.add_subplot(gs[r, 3]); EPF.base_image(ax, ST.volume(n)[:, ST.JY, :], draw)
        e = SS.orient(n, P, 6.0); EPF.glyphs(ax, e, np.ones(e.shape[:2], bool), tru)
        s_, _ = ST.score_e3(e)
        EPF.label(ax, f"within 0.5 mm: {s_['within_0.5mm']:.0f} deg\n3-8 mm away: {s_['deep_3-8mm']:.0f} deg")
        ax.set_title("shape tensor of the fibre component, rho 0.6 mm\n(colour = angle to the true fibres)" if r == 0 else "", fontsize=11)
    # the overlap: each step direction's share of the step change and of D0's own (fibre) variance
    ax = fig.add_subplot(gs[3, 0]); k = np.arange(1, 17)
    ax.bar(k - 0.2, we, 0.4, label="share of the step's change (near the interface)")
    ax.bar(k + 0.2, d0_share, 0.4, label="share of D0's own variance (fibres, no step)")
    ax.set_xlabel("step direction (principal components of f_D15 - f_D0)"); ax.set_ylabel("share")
    ax.set_title("the step moves the features along the same axes\nthat carry the fibre texture", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[3, 1:3]); ks = (1, 2, 4, 8)
    for n, c in (("D0", "C2"), ("D8", "C0"), ("D15", "C1"), ("D30", "C3")):
        ax.plot([0] + list(ks), [sc["none"]["r6"][n]["within_0.5mm"]] + [sc[f"rmE{k_}_D15"]["r6"][n]["within_0.5mm"] for k_ in ks],
                "-o", color=c, label=f"{n}, within 0.5 mm of the interface")
    rem = [0] + [sc[f"rmE{k_}_D15"]["stats"]["step_change_removed"] for k_ in ks]
    kept = [1] + [sc[f"rmE{k_}_D15"]["stats"]["d0_variance_kept"] for k_ in ks]
    for i, (x, a_, b_) in enumerate(zip([0] + list(ks), rem, kept)):
        ax.annotate(f"removes {a_:.0%} of the step change\nkeeps {b_:.0%} of D0 variance", (x, 84 if i % 2 else 76),
                    fontsize=7.5, ha="center")
    ax.axhline(59.1, color="grey", ls=":", label="BRIDGE now, D15 (intensity, rho 0.3)")
    ax.axhline(8.8, color="k", ls="--", label="gradient excision (fix A), D15")
    ax.set_xlabel("boundary directions removed (learned on D15 - D0)"); ax.set_ylabel("median angle to the true fibres (deg)")
    ax.set_ylim(0, 90); ax.set_xticks([0] + list(ks)); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="center right")
    ax.set_title("orientation of the fibre component, rho 0.6 mm", fontsize=11)
    ax = fig.add_subplot(gs[3, 3]); dep = SS.depth()
    labs = np.digitize(dep, [0.5, 1.0, 2.0]); cols = ("#e41a1c", "#ff7f00", "#377eb8", "#4daf4a")
    for n, mk in (("D0", "o"), ("D15", "^")):
        f = F[n][:, 12:-12:3, ST.JY - 20:ST.JY + 21:4, 12:-12:2]
        c2 = E[:, :2].T @ (f.reshape(16, -1) - mu0); lz = np.broadcast_to(labs[None, None, 12:-12:2], f.shape[1:]).ravel()
        for kk in range(4):
            s = lz == kk
            ax.scatter(c2[0, s], c2[1, s], s=2, alpha=0.35, lw=0, c=cols[kk], marker=mk,
                       label=f"{n} {('<0.5', '0.5-1', '1-2', '>2')[kk]} mm" if kk in (0, 3) else None)
    ax.set_xlabel("step direction 1"); ax.set_ylabel("step direction 2"); ax.grid(alpha=0.3); ax.legend(fontsize=8, markerscale=4)
    ax.set_title("D0 (o) and D15 (^) on the two main step directions", fontsize=11)
    fig.suptitle("Separating anatomix shape space into boundary and fibre components (tract-interface phantoms; the step "
                 "directions are learned from the paired difference D15 - D0)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.975]); fig.savefig(out / "shape_space_separation.png"); plt.close(fig)
    print("wrote", out / "shape_space_separation.png")
