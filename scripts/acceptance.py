"""The pilot acceptance criteria (notes/core_model_design_2026-10-29.md), checked from evaluation results.

    python scripts/acceptance.py FINAL.json HALF.json TRAIN.log
Prints each criterion with its numbers and PASS / FAIL, and exits 0 only if all pass.
"""
import json
import re
import sys

fin, half, log = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), open(sys.argv[3]).read()
f, h = fin["synthetic_s0.3"], half["synthetic_s0.3"]
res = []


def check(name, ok, text):
    res.append(ok); print(f"{name:3s} {'PASS' if ok else 'FAIL'}  {text}")


check("A", f["A_axis_model_deg"] <= f["A_axis_st_deg"],
      f"single-line axis error {f['A_axis_model_deg']:.1f} deg vs structure tensor {f['A_axis_st_deg']:.1f} deg (n {f['A_n']})")
check("B", f["B_cross_found"] >= 0.25, f"crossings both found {f['B_cross_found'] * 100:.0f} % (n {f['B_n']}; need >= 25 %)")
check("C", f["C_exist_f1"] >= f["C_exist_f1_trivial"] + 0.05 and f["C_share_err"] <= 0.8 * f["C_share_err_trivial"],
      f"existence F1 {f['C_exist_f1']:.3f} vs every-slot {f['C_exist_f1_trivial']:.3f}; share error {f['C_share_err']:.3f} vs trivial {f['C_share_err_trivial']:.3f}")
tex = fin["phantoms"].get("texture_tex0.5", {}).get("dominant_lineness_below_0.4", float("nan"))
check("D", f["D_shape_acc"] >= 0.6 and tex >= 0.6, f"shape class {f['D_shape_acc'] * 100:.0f} % (need >= 60); texture phantom isotropic {tex * 100:.0f} % (need >= 60)")
vals = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"val (\d+)[^\n]*?val_pop ([0-9.]+)", log)]
vals = [v for v in vals if v[0] > 0]
if len(vals) >= 2:
    first, last = vals[0][1], vals[-1][1]
    better = [f["A_axis_model_deg"] < h["A_axis_model_deg"], f["B_cross_found"] > h["B_cross_found"],
              f["C_exist_f1"] > h["C_exist_f1"], f["D_shape_acc"] > h["D_shape_acc"]]
    worse = [f["A_axis_model_deg"] > h["A_axis_model_deg"] + 2, f["B_cross_found"] < h["B_cross_found"] - 0.02,
             f["C_exist_f1"] < h["C_exist_f1"] - 0.02, f["D_shape_acc"] < h["D_shape_acc"] - 0.02]
    check("E", last <= 0.8 * first and sum(better) >= 2 and not any(worse),
          f"val_pop {first:.3f} (step {vals[0][0]}) -> {last:.3f} (step {vals[-1][0]}); A-D better than halfway: {sum(better)}/4, worse: {sum(worse)}")
else:
    check("E", False, "not enough validations")
r = f["F_exist_rate_pred"] / max(f["F_exist_rate_true"], 1e-9)
check("F", 0.5 <= r <= 2.0 and f["F_lineness_sd"] > 0.1, f"existence rate ratio {r:.2f} (0.5-2); line-ness sd {f['F_lineness_sd']:.3f} (> 0.1)")
print("ALL PASS" if all(res) else f"{len(res) - sum(res)} criteria failed")
sys.exit(0 if all(res) else 1)
