"""
step06_baryon_montecarlo.py

Monte Carlo search for the formula direction, orthogonal to the
Averagine ray, that maximises baryon mass defect while keeping average
mass and variance close to insulin. Informs the physical motivation
for V3.

Score function:
    score = delta(baryon_mass) - lambda_avg * |delta_avg_mass| - lambda_var * |delta_variance|

Outputs:
    figures/step06_baryon_montecarlo.png   insulin vs max-defect formula
    results/step06_baryon_montecarlo.csv   best formula, score, direction
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from IsoSpecPy import Iso, IsoTotalProb

from step00_shared_utils import (
    elements, ins_formula, save_fig, results_dir,
    parse_formula_to_dict, formula_dict_to_string,
    vector_from_formula_dict, spectrum_variance, aggregate_close_peaks,
)
from step01_averagine    import calculate_mass
from step05_baryon_analysis import count_baryons_mono, average_baryon_mass

# Hyperparameters for optimization constraints and sample size
lambda_avg = 1e3
lambda_var = 1e3
n_samples = 3000
np.random.seed(42)

# Parse reference insulin parameters
ins_dict = parse_formula_to_dict(ins_formula)
ins_vec = vector_from_formula_dict(ins_dict)
n_el = len(elements)

exp_iso = Iso(formula=ins_formula)
exp_mono = float(exp_iso.getMonoisotopicPeakMass())
exp_avg = float(exp_iso.getTheoreticalAverageMass())

spec_ref = IsoTotalProb(0.999, formula=ins_formula)
spec_ref.normalize()
exp_var = spectrum_variance(list(spec_ref.masses), list(spec_ref.probs))

ins_p, ins_n = count_baryons_mono(ins_dict)
exp_bmass = average_baryon_mass(exp_mono, ins_p, ins_n)

# Establish Averagine reference ray direction vector
_, avg_formula, _ = calculate_mass(exp_avg)
avg_vec  = vector_from_formula_dict(avg_formula)
avg_unit = avg_vec / np.linalg.norm(avg_vec)


def get_properties(fd):
    """
    Return (mono_mass, avg_mass, variance, avg_baryon_mass) for a formula dict.
    """
    formula_str = formula_dict_to_string(fd)
    iso = Iso(formula=formula_str)
    mono = float(iso.getMonoisotopicPeakMass())
    avg = float(iso.getTheoreticalAverageMass())
    sp= IsoTotalProb(0.999, formula=formula_str)
    sp.normalize()
    var = spectrum_variance(list(sp.masses), list(sp.probs))
    p, n = count_baryons_mono(fd)
    return mono, avg, var, average_baryon_mass(mono, p, n)


# Initialize optimization state variables
best_formula = ins_dict.copy()
best_score   = -np.inf
best_dir     = np.zeros(n_el)

# Monte Carlo loop (sample random perturbation vectors in orthogonal subspace)
for _ in range(n_samples):
    rand = np.random.normal(size=n_el)
    rand -= np.dot(rand, avg_unit) * avg_unit  
    nrm = np.linalg.norm(rand)
    if nrm < 1e-12:
        continue
    rand /= nrm
    limits = []
    for i in range(n_el):
        if rand[i] < 0:
            limits.append(-ins_vec[i] / rand[i])
    max_step = min(limits) if limits else 1.0
    step     = 0.5 * max_step

    # Build the trial formula
    trial_vec  = np.maximum(ins_vec + step * rand, 0)
    trial_dict = {}
    for i, el in enumerate(elements):
        trial_dict[el] = int(round(trial_vec[i]))

    unchanged = True
    for el in elements:
        if trial_dict[el] != ins_dict[el]:
            unchanged = False
            break
    if unchanged:
        continue

    # Score the trial
    _, t_avg, t_var, t_bm = get_properties(trial_dict)
    score = ((t_bm - exp_bmass)
             - lambda_avg * abs(t_avg - exp_avg)
             - lambda_var * abs(t_var - exp_var))

    if score > best_score:
        best_score   = score
        best_formula = trial_dict.copy()
        best_dir     = rand.copy()

best_str      = formula_dict_to_string(best_formula)
best_dir_norm = best_dir / np.linalg.norm(best_dir) if np.linalg.norm(best_dir) > 0 else best_dir

print(f"Best formula: {best_str}")
print(f"Score: {best_score:.5e}")
print(f"Best direction: {best_dir_norm}")

spec_best = IsoTotalProb(0.999, formula=best_str)
spec_best.normalize()
wasser = spec_best.wassersteinDistance(spec_ref)
print(f"Wasserstein distance: {wasser:.5f} Da")

em, ep = aggregate_close_peaks(list(spec_ref.masses),  list(spec_ref.probs))
bm, bp = aggregate_close_peaks(list(spec_best.masses), list(spec_best.probs))

# Plot results comparing original insulin against the max-defect formula
fig, ax = plt.subplots(figsize=(10, 5))
ax.vlines(em, 0, ep, color="pink",    linewidth=2,
          label=f"Reference: {ins_formula}")
ax.vlines(bm, 0, bp, color="steelblue", linewidth=1.5,
          label=f"Max baryon defect: {best_str}")
ax.set_xlabel("m/z (Da)", fontsize=12)
ax.set_ylabel("Intensity", fontsize=12)
ax.set_title(f"Monte Carlo search (N={n_samples})\n"
             f"Formula: {best_str}  |  W = {wasser:.4f} Da  |  Score = {best_score:.3e}",
             fontsize=10)
ax.legend(fontsize=11)
ax.set_xlim(5727, 5745)
ax.set_ylim(0, 0.25)
ax.grid(True, alpha=0.2)
plt.tight_layout()
save_fig("step06_baryon_montecarlo.png")
plt.close()

out = results_dir / "step06_baryon_montecarlo.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["reference_formula", "best_formula", "score", "wasserstein_Da",
                "n_samples", "lambda_avg", "lambda_var"]
               + [f"best_dir_{el}" for el in elements])
    w.writerow([ins_formula, best_str, round(best_score, 6), round(wasser, 6),
                n_samples, lambda_avg, lambda_var]
               + [round(x, 6) for x in best_dir_norm])
print(f"  Saved results: {out}")