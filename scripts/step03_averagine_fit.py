"""
step03_averagine_fit.py

Fits the Averagine formula to the insulin isotopic envelope by minimising
the Wasserstein distance over a +-5 hydrogen adjustment range.

Establishes how well can the Averagine model alone reproduce the spectrum 
without any variance or mass defect correction. The residual Wasserstein distance motivates 
the additional basis vectors introduced in steps 07-11.

Outputs:
    figures/step03_averagine_fit.png       insulin vs best Averagine fit (aggregated)
    figures/step03_averagine_fit_raw.png   same comparison without peak aggregation
    results/step03_averagine_fit.csv       best formula and Wasserstein distance
"""

import csv
import matplotlib.pyplot as plt
from IsoSpecPy import IsoTotalProb, Iso

from step00_shared_utils import (ins_formula, save_fig, results_dir, 
                                 aggregate_close_peaks, formula_dict_to_string)
from step01_averagine import calculate_mass


# Experimental spectrum generation
exp_spec = IsoTotalProb(0.999, formula=ins_formula)
exp_spec.normalize()
exp_masses = list(exp_spec.masses)
exp_probs  = list(exp_spec.probs)

mass_ins = float(Iso(formula=ins_formula).getTheoreticalAverageMass())
_, base_formula, _ = calculate_mass(mass_ins)

best_distance = float("inf")
best_formula  = None

# Iterate through a local search grid of hydrogen atoms (+-5 range)
# to optimize the fit against the experimental envelope 
for k in range(-5, 6):
    test = base_formula.copy()
    test["H"] += k
    
    # Skip invalid chemical compositions with negative hydrogen counts
    if test["H"] < 0:
        continue
        
    test_str = formula_dict_to_string(test)
    test_spec = IsoTotalProb(0.999, formula=test_str)
    test_spec.normalize()
    
    dist = test_spec.wassersteinDistance(exp_spec)
    
    if dist < best_distance:
        best_distance = dist
        best_formula  = test.copy()

best_str = formula_dict_to_string(best_formula)
best_spec = IsoTotalProb(0.999, formula=best_str)
best_spec.normalize()

print(f"Best Averagine formula: {best_str}")
print(f"Wasserstein distance: {best_distance:.5f} Da")

em, ep = aggregate_close_peaks(exp_masses, exp_probs)
bm, bp = aggregate_close_peaks(list(best_spec.masses), list(best_spec.probs))

# Plotting the comparative overlay between real insulin and the optimized Averagine fit
fig, ax = plt.subplots(figsize=(10, 5))
ax.vlines(em, 0, ep, color="pink",    linewidth=2,
          label=f"Reference: {ins_formula}")
ax.vlines(bm, 0, bp, color="steelblue", linewidth=1.5,
          label=f"Best Averagine fit: {best_str}")
          
ax.set_xlabel("m/z (Da)", fontsize=12)
ax.set_ylabel("Intensity", fontsize=12)
ax.set_title("Averagine fit with peak aggregation\n"
             f"Formula: {best_str}  |  W = {best_distance:.4f} Da", fontsize=11)
ax.legend(fontsize=11)
ax.set_xlim(5727, 5745)
ax.set_ylim(0, 0.25)
ax.grid(True, alpha=0.2)
plt.tight_layout()

save_fig("step03_averagine_fit.png")
plt.close()

# Same comparison without peak aggregation, for reference against the aggregated spectrum
fig, ax = plt.subplots(figsize=(10, 5))
ax.vlines(exp_masses, 0, exp_probs, color="pink",    linewidth=1,
          label=f"Reference: {ins_formula}")
ax.vlines(list(best_spec.masses), 0, list(best_spec.probs), color="steelblue", linewidth=0.8,
          label=f"Best Averagine fit: {best_str}")

ax.set_xlabel("m/z (Da)", fontsize=12)
ax.set_ylabel("Intensity", fontsize=12)
ax.set_title("Averagine fit without peak aggregation\n"
             f"Formula: {best_str}  |  W = {best_distance:.4f} Da", fontsize=11)
ax.legend(fontsize=11)
ax.set_xlim(5727, 5745)
ax.set_ylim(0, 0.25)
ax.grid(True, alpha=0.2)
plt.tight_layout()

save_fig("step03_averagine_fit_raw.png")
plt.close()

out = results_dir / "step03_averagine_fit.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["reference_formula", "best_averagine_formula", "wasserstein_Da"])
    w.writerow([ins_formula, best_str, round(best_distance, 6)])
print(f"  Saved results: {out}")