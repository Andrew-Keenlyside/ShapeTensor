"""Two co-registered stains of the same tissue, read together.

The phantom generator gives the same geometry (and truth) under different texture realisations and texture levels,
so a pair of phantoms stands in for two stains of one block: a *fibre-rich* stain (full fibre texture, a mild
boundary step) and a *cell-like* stain (weak fibre texture, a strong boundary step), plus a pair of independent
realisations of the same stain (a pure SNR control). Each stain is read alone and the two together, with the
channels stacked: the intensity tensors add; the anatomix48 features of both stains are concatenated (8 PCs each)
and read as one channel-resolved tensor (TCA, rank 3) and as one summed gradient-direction ODF (CSD) -- "the shape
features of both stains represented together". Scored against the truth in the usual bands.

    python scripts/bench/two_stain_test.py          -> results/two_stain_test.json
"""
import json, time
import numpy as np, torch
from shapetensor import DATA_ROOT, st
from shapetensor.features import make_engine
from shapetensor.pca import PCA
from shapetensor.phantoms import make
from shapetensor.readout import common, multi, single
from shapetensor.tensor import gradients, k_sites, odf_sites
from shapetensor.odf import fit_ring_width, ring_profile, zonal_kernel
from shapetensor.validate import score

PAIRS = {  # name: ((pid A, seed A), (pid B, seed B)) -- same geometry and truth, different texture realisation / level
    "step: fibre-rich (P3-15, t1) + cell-like (P3-40, t0.25)": (("P3-15", 0), ("P3-40-t0.25", 1)),
    "step: two realisations of P3-40-t0.5": (("P3-40-t0.5", 0), ("P3-40-t0.5", 1)),
    "crossing 60: t1 + t0.5": (("P4-lines-c60-f1", 0), ("P4-lines-c60-f1-t0.5", 1)),
    "single line (null for the crossing): t1 + t0.5": (("P4-lines-single", 0), ("P4-lines-single-t0.5", 1)),
}
RHOS = (10.0, 16.0); SIGMA, TR, STRIDE, LMAX, NPCS = 1.0, 4.0, 4, 8, 8
EK = {"nyquist": "none"}
BANDS = ("0-0.1mm", "0.25-0.5mm", ">2mm")


def load(pid, seed):
    img, tr, meta, grid = make.load(pid, seed)
    return torch.as_tensor(img, device="cuda").float(), tr, grid


def feats(v, engine):
    if engine == "intensity":
        return v[None].half()
    F = make_engine(engine, **EK)(v)
    return PCA.fit(F, NPCS, seed=0)(F).half()


def read_single(G, rho, grid, zs, xs, t):
    """Directions from the channel-resolved tensor of ``G (3C, ...)``: eigen of the summed 3x3 (the stacked structure
    tensor) and TCA rank 3 on the full 3C x 3C; medians by band."""
    b = max(1, int(rho // 4)); s = max(1, STRIDE // b)
    K, sites = k_sites(G, rho, TR, b, [grid.y_mid], grid.core_zx)
    Kp = K[0, ::s, ::s].reshape(-1, K.shape[-1])
    C = common.nc3_from_packed(Kp.shape[-1]) // 3
    Kf = common.unpack(Kp, 3 * C)                                            # (N, 3C, 3C)
    S3 = sum(Kf[:, 3 * c:3 * c + 3, 3 * c:3 * c + 3] for c in range(C))      # the summed 3 x 3 tensor
    out = {}
    d = single.eigen(S3)["dir"].cpu().numpy(); out["eigen"] = bands(d, t)
    if C > 1:
        d = single.tca(Kf, rank=3)["dir"].cpu().numpy(); out["tca"] = bands(d, t)
    return out


def bands(d, t):
    err = score.angle_err(d, t["dir1"]); bs = score.band_summary(err, t["dist"], t["region"])
    return {k: round(float(bs[k]["median"]), 2) for k in BANDS if k in bs and "median" in bs[k]}


def read_multi(G, rho, grid, t, kern, rel=0.5):
    b = max(1, int(rho // 4)); s = max(1, STRIDE // b)
    O, sites = odf_sites(G, rho, TR, b, [grid.y_mid], grid.core_zx, lmax=LMAX)
    O = O[0, ::s * 2, ::s * 2]; ts = score.sample_truth(t, sites.z[::s * 2], sites.x[::s * 2])
    r = multi.n_by_peaks(O.reshape(-1, O.shape[-1]), LMAX, mode="csd", kernel=kern, rel=rel)
    fib, frac, cnt = [x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x) for x in (r["fibres"], r["fracs"], r["count"])]
    sc = score.multi_scores(fib, frac, cnt, ts["dir1"], ts["dir2"], ts["npop"], ts["frac1"])
    two, sg = sc.get("two") or {}, sc.get("single") or {}
    return {"found2": two.get("found2_share"), "both15": two.get("both_within_15_share"), "false2": sg.get("false_second_share")}


def kernel_for(Gs_single, rho, grid, tr_single):
    """The ring response fitted on the single-line phantom pair, for the stain set at hand."""
    b = max(1, int(rho // 4)); s = max(1, STRIDE // b)
    O, sites = odf_sites(Gs_single, rho, TR, b, [grid.y_mid], grid.core_zx, lmax=LMAX)
    O = O[0, ::s * 2, ::s * 2]; ts = score.sample_truth(tr_single, sites.z[::s * 2], sites.x[::s * 2])
    width = fit_ring_width(O.reshape(-1, O.shape[-1]), torch.as_tensor(ts["dir1"], device="cuda"), LMAX)
    return zonal_kernel(LMAX, ring_profile(width)), float(np.degrees(width))


def main():
    t0 = time.time(); out = {}
    for name, ((pa, sa), (pb, sb)) in PAIRS.items():
        vA, trA, grid = load(pa, sa); vB, trB, _ = load(pb, sb)
        (z0, z1), (x0, x1) = grid.core_zx
        zs, xs = np.arange(z0, z1, STRIDE), np.arange(x0, x1, STRIDE)
        t = score.sample_truth(trA, zs, xs)
        same = np.nanmax(np.abs(trA["dir1"] - trB["dir1"])) < 1e-6
        res = {"same_truth": bool(same)}
        is_multi = pa.startswith("P4")
        for engine in ("intensity", "anatomix48"):
            FA, FB = feats(vA, engine), feats(vB, engine)
            GA, GB = gradients(FA, SIGMA, TR), gradients(FB, SIGMA, TR); del FA, FB
            GAB = torch.cat([GA, GB], 0)
            for rho in RHOS:
                for tag, G in (("A", GA), ("B", GB), ("A+B", GAB)):
                    key = f"{engine}|rho{rho:g}|{tag}"
                    if not is_multi:
                        res[key] = read_single(G, rho, grid, zs, xs, t)
                    else:
                        # the ring kernel from the single-line pair read the same way (A, B or A+B)
                        sA, _, _ = load("P4-lines-single", 0); sB, trS, _ = load("P4-lines-single-t0.5", 1)
                        GS = {"A": gradients(feats(sA, engine), SIGMA, TR), "B": gradients(feats(sB, engine), SIGMA, TR)}
                        GS["A+B"] = torch.cat([GS["A"], GS["B"]], 0); del sA, sB
                        kern, width = kernel_for(GS[tag], rho, grid, trS); del GS
                        res[key] = {**read_multi(G, rho, grid, trA, kern), "ring_width_deg": round(width, 1)}
                    print(f"[{time.time() - t0:5.0f}s] {name} | {key}: {res[key]}", flush=True)
                    torch.cuda.empty_cache()
            del GA, GB, GAB; torch.cuda.empty_cache()
        out[name] = res
        del vA, vB; torch.cuda.empty_cache()
        json.dump(out, open(f"{DATA_ROOT}/results/two_stain_test.json", "w"), indent=1)
    print("TWO STAIN TEST DONE", flush=True)


if __name__ == "__main__":
    main()
