"""
step09_basis_diagnostics.py

Prints all basis vectors and projection coefficients, then
compares two partial reconstructions of insulin side by side:
    - V1 + V2 only (mass + variance)
    - V1 + V3 only (mass + defect)

Also tests unit-step sensitivity, meaning spectrum comparison when
we move exactly 1 unit along V2 or V3 from the mass-only baseline. 

Outputs:
    figures/step09_submodels.png             V1+V2 vs V1+V3 comparison
    figures/step09_unit_steps.png            unit-step sensitivity along V2, V3
    results/step09_diagnostics.csv           all k-coefficients and Wasserstein distances
    results/step09_k_threshold_comparison.csv
        k_mass, k_var, k_def under both the tight-threshold configuration
        (used only in step08's own standalone) and the mixed-threshold configuration used 
        everywhere else from this point onward (steps 09, 11, 12, 14).
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from IsoSpecPy import Iso, IsoTotalProb

from step00_shared_utils import (
    elements, ins_formula, save_fig, results_dir,
    parse_formula_to_dict, formula_dict_to_string, vector_from_formula_dict,
    spectrum_variance, aggregate_close_peaks,
)
from step08_basis_vectors import (
    compute_mass_vector, compute_variance_vector, compute_defect_vector,
    orthogonalize_vectors, get_residue_h,
)

#Mixed-threshold configuration 
exp_spec = IsoTotalProb(0.999, formula=ins_formula)
exp_spec.normalize()
exp_masses = list(exp_spec.masses)
exp_probs = list(exp_spec.probs)
exp_var = spectrum_variance(exp_masses, exp_probs)

ins_dict = parse_formula_to_dict(ins_formula)
ins_vec = vector_from_formula_dict(ins_dict)
ins_mass = float(Iso(ins_formula).getTheoreticalAverageMass())
H_mass = float(Iso("H1").getTheoreticalAverageMass())

mass_vec = compute_mass_vector(elements)
V_var = compute_variance_vector(elements, ins_dict, exp_var)
V_def = compute_defect_vector(elements)
v1, v2, v3 = orthogonalize_vectors(mass_vec, V_var, V_def)

k1 = np.dot(ins_vec, v1)
k2 = np.dot(ins_vec, v2)
k3 = np.dot(ins_vec, v3)

print(f"Formula: {ins_formula}")
print(f"k_mass={k1:.6f}  k_var={k2:.6f}  k_def={k3:.6f}")

#Tight-threshold configuration 
exp_spec_tight = IsoTotalProb(0.999999999, formula=ins_formula)
exp_spec_tight.normalize()
exp_var_tight = spectrum_variance(list(exp_spec_tight.masses), list(exp_spec_tight.probs))

V_var_tight = compute_variance_vector(elements, ins_dict, exp_var_tight)
v1_tight, v2_tight, v3_tight = orthogonalize_vectors(mass_vec, V_var_tight, V_def)
k1_tight = np.dot(ins_vec, v1_tight)
k2_tight = np.dot(ins_vec, v2_tight)
k3_tight = np.dot(ins_vec, v3_tight)

out_k = results_dir / "step09_k_threshold_comparison.csv"
with open(out_k, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["formula", "baseline_threshold", "k_mass", "k_var", "k_def"])
    w.writerow([ins_formula, "0.999999999 (tight; used only in step08)",
                round(k1_tight, 6), round(k2_tight, 6), round(k3_tight, 6)])
    w.writerow([ins_formula, "0.999 (mixed; used in steps 09/11/12/14)",
                round(k1, 6), round(k2, 6), round(k3, 6)])
print(f"  \n  Saved results: {out_k}")


def make(F):
    """
    Round vector F to a formula, fixing mass with the H correction
    """
    d = get_residue_h(F, elements, H_mass, ins_mass)
    return formula_dict_to_string(d)


def spec_and_w(formula):
    """
    Generate the formula's spectrum and its W distance to insulin
    """
    s = IsoTotalProb(0.999, formula=formula)
    s.normalize()
    return s, s.wassersteinDistance(exp_spec)

# Two sub-models - mass+variance (V1+V2) and mass+defect (V1+V3)
f_v12 = make(k1*v1 + k2*v2)
f_v13 = make(k1*v1 + k3*v3)

sp_v12, w12 = spec_and_w(f_v12)
sp_v13, w13 = spec_and_w(f_v13)
sp_orig = IsoTotalProb(0.999, formula=ins_formula)
sp_orig.normalize()

mo, po = aggregate_close_peaks(list(sp_orig.masses), list(sp_orig.probs))
m12, p12 = aggregate_close_peaks(list(sp_v12.masses),  list(sp_v12.probs))
m13, p13 = aggregate_close_peaks(list(sp_v13.masses),  list(sp_v13.probs))

# Plot submodels comparison (V1+V2 vs V1+V3)
fig, ax = plt.subplots(figsize=(11, 5))
ax.vlines(mo,  0, po,  color="pink",    linewidth=2,
          label=f"Reference: {ins_formula}")
ax.vlines(m12, 0, p12, color="steelblue", linewidth=1.5, alpha=0.85,
          label=f"V1+V2 reconstruction: {f_v12}  (W = {w12:.4f} Da)")
ax.vlines(m13, 0, p13, color="purple", linewidth=1.5, alpha=0.85,
          label=f"V1+V3 reconstruction: {f_v13}  (W = {w13:.4f} Da)")
ax.set_xlabel("m/z (Da)", fontsize=12)
ax.set_ylabel("Intensity", fontsize=12)
ax.set_title("Mass + variance (V1+V2) vs. mass + defect (V1+V3) sub-models", fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(5650, 6100)
ax.set_ylim(0, 0.25)
ax.grid(True, alpha=0.2)
plt.tight_layout()
save_fig("step09_submodels.png")
plt.close()

# Unit-step sensitivity analysis along V2 and V3 vectors
f_uv = make(k1*v1 + 1.0*v2)
f_ud = make(k1*v1 + 1.0*v3)
sp_uv, _ = spec_and_w(f_uv)
sp_ud, _ = spec_and_w(f_ud)
muv, puv  = aggregate_close_peaks(list(sp_uv.masses), list(sp_uv.probs))
mud, pud  = aggregate_close_peaks(list(sp_ud.masses), list(sp_ud.probs))

fig, ax = plt.subplots(figsize=(11, 5))
ax.vlines(mo,  0, po,  color="pink",    linewidth=2,
          label=f"Reference: {ins_formula}")
ax.vlines(muv, 0, puv, color="steelblue", linewidth=1.5, alpha=0.85,
          label=f"Mass + V2×1 reconstruction: {f_uv}")
ax.vlines(mud, 0, pud, color="purple", linewidth=1.5, alpha=0.85,
          label=f"Mass + V3×1 reconstruction: {f_ud}")
ax.set_xlabel("m/z (Da)", fontsize=12)
ax.set_ylabel("Intensity", fontsize=12)
ax.set_title("Unit-step sensitivity along V2 (variance) and V3 (mass defect)", fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(5650, 6100)
ax.set_ylim(0, 0.25)
ax.grid(True, alpha=0.2)
plt.tight_layout()
save_fig("step09_unit_steps.png")
plt.close()

out = results_dir / "step09_diagnostics.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["formula", "model", "reconstructed", "wasserstein_Da"])
    for model, recon, wass in [("V1+V2", f_v12, w12), ("V1+V3", f_v13, w13)]:
        w.writerow([ins_formula, model, recon, round(wass, 6)])
print(f"  Saved results: {out}")