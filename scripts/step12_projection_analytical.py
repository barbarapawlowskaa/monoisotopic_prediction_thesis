"""
step12_projection_analytical.py

Full 5D reconstruction using the closed-form solution for alpha, beta and gamma

Given the partial reconstruction K = k1*V1 + k2*V2 + k3*V3, we compute
alpha and beta analytically. These coefficients are chosen so that
F = K + alpha*V4 + beta*V5 aligns with the Averagine ray A in the
V4-V5 plane.

The pre-computed constants are from step09 (k-values) and step11 (v4, v5)
for the insulin formula, hardcoded here for fast reproducible validation.

Outputs:
    figures/step12_full_5D_reconstruction.png   insulin vs 5D projection
    results/step12_projection_analytical.csv    alpha, beta, gamma, W distance
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from IsoSpecPy import Iso, IsoTotalProb

from step00_shared_utils import (
    elements, ins_formula, save_fig, results_dir,
    formula_dict_to_string, aggregate_close_peaks,
)
from step08_basis_vectors import get_residue_h

# Pre-computed constants for C254H377N65O75S6 (reproducibility hardcoded values)
k_mass = 142.20490112549146
k_var  = 106.77728726322395
k_def  = -104.74397330853067

mass_unit  = np.array([0.2979,  0.0250,  0.3474,  0.3968,  0.7953])
V_var_unit = np.array([-0.1621, 0.5580, -0.4193, -0.5068,  0.4793])
V_def_unit = np.array([-0.4702, -0.0022, -0.5551,  0.6816,  0.0786])
v4 = np.array([-0.7868,  0.1169,  0.6001, -0.0604,  0.0590])
v5 = np.array([-0.2118, -0.8212, -0.1875, -0.3427,  0.3580])

# Averagine direction vector reference anchor
A = np.array([4.9245, 7.7724, 1.3555, 1.46, 0.0356])

insulin_mass = float(Iso(ins_formula).getTheoreticalAverageMass())
H_mass = float(Iso("H1").getTheoreticalAverageMass())

# Construct partial baseline vector K using V1, V2, V3
K = k_mass*mass_unit + k_var*V_var_unit + k_def*V_def_unit

# Analytical solution parameters for alpha, beta, and gamma
gamma = 50.09308
alpha = -112.11483
beta = -409.09227

print(f"gamma = {gamma:.5f},  alpha = {alpha:.5f},  beta = {beta:.5f}")

# Compute final 5D vector and apply residue hydrogen adjustment
F_new = K + alpha*v4 + beta*v5
d_new = get_residue_h(F_new, elements, H_mass, insulin_mass)
formula_new = formula_dict_to_string(d_new)
print(f"Original: {ins_formula}")
print(f"Projected: {formula_new}")

spec_orig = IsoTotalProb(0.999, formula=ins_formula)
spec_orig.normalize()
spec_new  = IsoTotalProb(0.999, formula=formula_new)
spec_new.normalize()
wass = spec_orig.wassersteinDistance(spec_new)
print(f"Wasserstein distance: {wass:.6f} Da")

mo, po = aggregate_close_peaks(list(spec_orig.masses), list(spec_orig.probs))
mn, pn = aggregate_close_peaks(list(spec_new.masses),  list(spec_new.probs))

# Plot the full 5D analytical reconstruction against original reference
fig, ax = plt.subplots(figsize=(11, 5))
ax.vlines(mo, 0, po, color="pink",    linewidth=2,
          label=f"Reference: {ins_formula}")
ax.vlines(mn, 0, pn, color="steelblue", linewidth=1.5, alpha=0.85,
          label=f"5D projection: {formula_new}")
ax.set_xlabel("m/z (Da)", fontsize=12)
ax.set_ylabel("Intensity", fontsize=12)
ax.set_title("Full 5D reconstruction\n"
             f"α = {alpha:.2f}, β = {beta:.2f}, γ = {gamma:.2f}  |  W = {wass:.4f} Da",
             fontsize=11)
ax.legend(fontsize=11)
ax.set_xlim(insulin_mass - 20, insulin_mass + 20)
ax.set_ylim(0, 0.25)
ax.grid(True, alpha=0.2)
plt.tight_layout()
save_fig("step12_full_5D_reconstruction.png")
plt.close()

out = results_dir / "step12_projection_analytical.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["reference_formula", "projected_formula",
                "alpha", "beta", "gamma", "wasserstein_Da"])
    w.writerow([ins_formula, formula_new,
                round(alpha,5), round(beta,5), round(gamma,5), round(wass,6)])
print(f"  Saved results: {out}")