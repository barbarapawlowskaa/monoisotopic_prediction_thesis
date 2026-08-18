"""
step07_variance_only.py

Projects the insulin formula onto the Averagine direction (as V1)/Mass vector (asV1) 
and the variance gradient (as V2) and compare with the original spectrum.

Outputs:
    figures/step07_variance_averagine.png     Averagine direction + V2
    figures/step07_variance_mass_vector.png   Mass vector + V2 (pipeline-consistent)
    results/step07_variance_only.csv          Averagine+V2 result
    results/step07_variance_mass_vector.csv   Mass_vec+V2 result
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from IsoSpecPy import IsoTotalProb, Iso

from step00_shared_utils import (
    elements, ins_formula, save_fig, results_dir,
    parse_formula_to_dict, formula_dict_to_string, vector_from_formula_dict,
    spectrum_variance, aggregate_close_peaks,
)
from step01_averagine     import calculate_mass
from step08_basis_vectors import compute_variance_vector, compute_mass_vector, get_residue_h

# Experimental spectrum generation
exp_spec = IsoTotalProb(0.999999999, formula=ins_formula)
exp_spec.normalize()
exp_masses = list(exp_spec.masses)
exp_probs = list(exp_spec.probs)
exp_var = spectrum_variance(exp_masses, exp_probs)

ins_dict = parse_formula_to_dict(ins_formula)
ins_vec = vector_from_formula_dict(ins_dict)
mass_ins = float(Iso(ins_formula).getTheoreticalAverageMass())

# Define V1 as the Averagine direction at insulin's mass scale
_, base_formula, _ = calculate_mass(mass_ins)
A_vec = vector_from_formula_dict(base_formula)
A_unit = A_vec / np.linalg.norm(A_vec)

# Define V2 as variance gradient, made orthogonal via Gram-Schmidt
V_var = compute_variance_vector(elements, ins_dict, exp_var)
V_var_ort = V_var - (np.dot(V_var, A_vec) / np.dot(A_vec, A_vec)) * A_vec
V_var_norm = np.linalg.norm(V_var_ort)
V_var_unit = V_var_ort / V_var_norm if V_var_norm > 0 else V_var_ort

# Project insulin vector onto 2D subspace
k_mass = np.dot(ins_vec, A_unit)
k_var  = np.dot(ins_vec, V_var_unit)
F_vec  = k_mass * A_unit + k_var * V_var_unit

H_mass = float(Iso("H1").getTheoreticalAverageMass())
F_dict = get_residue_h(F_vec, elements, H_mass, mass_ins)
final_formula = formula_dict_to_string(F_dict)

final_spec = IsoTotalProb(0.999999999, formula=final_formula)
final_spec.normalize()
final_var  = spectrum_variance(list(final_spec.masses), list(final_spec.probs))
wasser     = final_spec.wassersteinDistance(exp_spec)

em, ep = aggregate_close_peaks(exp_masses, exp_probs)
fm, fp = aggregate_close_peaks(list(final_spec.masses), list(final_spec.probs))

# Render first 2D experiment plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.vlines(em, 0, ep, color="pink",    linewidth=2,
          label=f"Reference: {ins_formula}")
ax.vlines(fm, 0, fp, color="steelblue", linewidth=1.5,
          label=f"Averagine + V2 reconstruction: {final_formula}")
ax.set_xlabel("m/z (Da)", fontsize=12)
ax.set_ylabel("Intensity", fontsize=12)
ax.set_title("Averagine direction (V1) + variance gradient (V2)\n"
             f"Formula: {final_formula}  |  W = {wasser:.4f} Da  |  "
             f"Var (exp) = {exp_var:.2f}  |  Var (model) = {final_var:.2f}", fontsize=10)
ax.legend(fontsize=11)
ax.set_xlim(5727, 5745)
ax.set_ylim(0, 0.25)
ax.grid(True, alpha=0.2)
plt.tight_layout()
save_fig("step07_variance_averagine.png")
plt.close()

out = results_dir / "step07_variance_only.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["reference_formula", "model_formula", "wasserstein_Da",
                "variance_exp_Da2", "variance_model_Da2", "k_mass", "k_var"])
    w.writerow([ins_formula, final_formula, round(wasser, 6),
                round(exp_var, 4), round(final_var, 4),
                round(k_mass, 4), round(k_var, 4)])

#Second experiment - Using atomic mass vector as V1 (pipeline consistent)

exp_spec2 = IsoTotalProb(0.999999999, formula=ins_formula)
exp_spec2.normalize()
exp_masses2 = list(exp_spec2.masses)
exp_probs2 = list(exp_spec2.probs)
exp_var2 = spectrum_variance(exp_masses2, exp_probs2)

ins_dict2 = parse_formula_to_dict(ins_formula)
ins_vec2 = vector_from_formula_dict(ins_dict2)
mass_ins2 = float(Iso(ins_formula).getTheoreticalAverageMass())
H_mass2 = float(Iso("H1").getTheoreticalAverageMass())

mass_vec2 = compute_mass_vector(elements)
mass_norm2 = np.linalg.norm(mass_vec2)
mass_unit2 = mass_vec2 / mass_norm2

V_var2 = compute_variance_vector(elements, ins_dict2, exp_var2)
proj2 = np.dot(V_var2, mass_vec2) / np.dot(mass_vec2, mass_vec2)
V_var_ort2 = V_var2 - proj2 * mass_vec2
V_var_norm2 = np.linalg.norm(V_var_ort2)
V_var_unit2 = V_var_ort2 / V_var_norm2 if V_var_norm2 > 0 else V_var_ort2

k_mass2 = np.dot(ins_vec2, mass_unit2)
k_var2 = np.dot(ins_vec2, V_var_unit2)
F_vec2 = k_mass2 * mass_unit2 + k_var2 * V_var_unit2

F_vec_rounded2 = np.rint(F_vec2).astype(int)
F_vec_rounded2 = np.maximum(F_vec_rounded2, 0)
F_dict2 = {el: int(F_vec_rounded2[i]) for i, el in enumerate(elements)}

iso_new2 = Iso(formula=formula_dict_to_string(F_dict2))
mass_new2 = float(iso_new2.getTheoreticalAverageMass())
delta_H2 = int(round((mass_ins2 - mass_new2) / H_mass2))
F_dict2["H"] = max(F_dict2.get("H", 0) + delta_H2, 0)

final_formula2 = formula_dict_to_string(F_dict2)

final_spec2 = IsoTotalProb(0.999999999, formula=final_formula2)
final_spec2.normalize()
final_masses2 = list(final_spec2.masses)
final_probs2 = list(final_spec2.probs)
final_var2 = spectrum_variance(final_masses2, final_probs2)
wasser2 = final_spec2.wassersteinDistance(exp_spec2)

exp_m2, exp_p2 = aggregate_close_peaks(exp_masses2, exp_probs2)
final_m2, final_p2 = aggregate_close_peaks(final_masses2, final_probs2)

fig, ax = plt.subplots(figsize=(10, 5))
ax.vlines(exp_m2,   0, exp_p2,   color="pink",    linewidth=2,
          label=f"Reference: {ins_formula}")
ax.vlines(final_m2, 0, final_p2, color="steelblue", linewidth=1.5,
          label=f"V1 + V2 reconstruction: {final_formula2}")
ax.set_xlabel("m/z (Da)", fontsize=12)
ax.set_ylabel("Intensity", fontsize=12)
ax.set_title("Mass vector (V1) + variance gradient (V2)\n"
             f"Formula: {final_formula2}  |  W = {wasser2:.4f} Da  |  "
             f"Var (exp) = {exp_var2:.2f}  |  Var (model) = {final_var2:.2f}", fontsize=10)
ax.legend(fontsize=11)
ax.set_xlim(5727, 5745)
ax.set_ylim(0, 0.25)
ax.grid(True, alpha=0.2)
plt.tight_layout()
save_fig("step07_variance_mass_vector.png")
plt.close()

out2 = results_dir / "step07_variance_mass_vector.csv"
with open(out2, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["reference_formula", "model_formula", "wasserstein_Da",
                "variance_exp_Da2", "variance_model_Da2", "k_mass", "k_var"])
    w.writerow([ins_formula, final_formula2, round(wasser2, 6),
                round(exp_var2, 4), round(final_var2, 4),
                round(k_mass2, 4), round(k_var2, 4)])
print(f"  Saved results: {out2}")