"""
step02_isospec_intro.py

Computes the theoretical isotopic spectrum of bovine insulin and compares it 
to a mock experimental spectrum using the Wasserstein distance.

The Wasserstein distance (earth-mover distance) measures the minimum
transport cost to reshape one probability distribution into another.

Run this first to verify that IsoSpecPy is correctly installed.

Outputs:
    figures/step02_isospec_intro.png   theoretical vs mock experimental spectrum
    results/step02_wasserstein.csv     Wasserstein distance summary
"""

import csv
import matplotlib.pyplot as plt
from IsoSpecPy import IsoTotalProb, IsoDistribution

from step00_shared_utils import save_fig, results_dir

formula = "C254H377N65O75S6"

# Theoretical spectrum generation
theo = IsoTotalProb(0.999, formula=formula)
theo_masses = list(theo.masses)
theo_probs = list(theo.probs)

# Normalise theoretical probabilities so they sum to 1
theo_total = sum(theo_probs)
theo_probs_norm = [p / theo_total for p in theo_probs]

# Mock experimental spectrum (hand-crafted for illustration only, not real data)
exp_mz = [5729.5, 5731.0, 5732.0, 5733.0, 5734.8,
                 5736.0, 5736.2, 5736.9, 5737.4, 5737.6, 5738.5]
exp_intensity = [15, 50, 90, 100, 55, 65, 5, 25, 5, 25, 15]
total = sum(exp_intensity)
exp_probs = [i / total for i in exp_intensity]
exp_spectrum = IsoDistribution(masses=exp_mz, probs=exp_probs)

theo_dist = IsoDistribution(masses=theo_masses, probs=theo_probs_norm)
distance = theo_dist.wassersteinDistance(exp_spectrum)
print(f"Wasserstein distance: {distance:.5f} Da")

fig, ax = plt.subplots(figsize=(10, 5))
ax.vlines(theo_masses, 0, theo_probs_norm,
          color="steelblue", linewidth=1.5, label=f"Theoretical (IsoSpec): {formula}")
ax.vlines(exp_mz, 0, exp_probs,
          color="pink", linewidth=2, linestyle="--", label="Mock experimental spectrum")
ax.set_xlabel("m/z (Da)", fontsize=12)
ax.set_ylabel("Intensity", fontsize=12)
ax.set_title("Theoretical vs. mock experimental spectrum\n"
             f"W = {distance:.4f} Da", fontsize=12)
ax.legend(fontsize=11)
ax.set_xlim(5727, 5745)
ax.set_ylim(0, 0.25)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig("step02_isospec_intro.png")
plt.close()

out = results_dir / "step02_wasserstein.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["formula", "comparison", "wasserstein_Da"])
    w.writerow([formula, "mock_experimental", round(distance, 6)])
print(f"  Saved results: {out}")