"""
step02_isospec_intro.py

Computes the theoretical isotopic spectrum of bovine insulin and compares it 
to a mock experimental spectrum using the Wasserstein distance.

The Wasserstein distance ,easures the minimum transport cost to reshape one probability 
distribution into another.

Run this first to verify that IsoSpecPy is correctly installed.

Outputs:
    figures/step02_isospec_intro.png   theoretical vs mock experimental spectrum
    figures/step02_spectrum.png        theoretical envelope alone, monoisotopic peak marked
    results/step02_wasserstein.csv     Wasserstein distance summary
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from IsoSpecPy import IsoTotalProb, IsoDistribution, Iso

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


# Zommed in isualization for the thesis
# Theoretical isotopic envelope of bovine insulin with monoisotopic peak marked 

mono_mass = float(Iso(formula=formula).getMonoisotopicPeakMass())
mono_idx = min(range(len(theo_masses)), key=lambda i: abs(theo_masses[i] - mono_mass))

# Find the 2nd tallest peak
best_i, best_p = 0, 0
for i in range(len(theo_masses)):
    if 5730.58 < theo_masses[i] < 5730.62 and theo_probs_norm[i] > best_p:
        best_p = theo_probs_norm[i]
        best_i = i
peak_x, peak_y = theo_masses[best_i], theo_probs_norm[best_i]

fig, ax = plt.subplots(figsize=(10, 5))
ax.vlines(theo_masses, 0, theo_probs_norm, color="steelblue", linewidth=1.5,
          label=f"Bovine insulin {formula}")
ax.plot(theo_masses[mono_idx], theo_probs_norm[mono_idx], marker="*",
        color="palevioletred", markersize=16, linestyle="none", zorder=5,
        label="Monoisotopic peak")
ax.set_xlabel("m/z (Da)", fontsize=12)
ax.set_ylabel("Intensity", fontsize=12)
ax.set_title("Isotopic envelope of bovine insulin", fontsize=12)
ax.legend(fontsize=11, loc = "upper left")
ax.set_xlim(5727, 5741)
ax.set_ylim(0, 0.15)
ax.grid(True, which="major", axis="y", alpha=0.3)
ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
ax.grid(True, which="minor", axis="x", alpha=0.3)

left, bottom, width, height = 0.66, 0.38, 0.32, 0.42
axins = ax.inset_axes([left, bottom, width, height])
axins.vlines(theo_masses, 0, theo_probs_norm, color="steelblue", linewidth=1.2)
axins.set(xlim=(5730.585, 5730.62), ylim=(0, 0.09))
axins.set_xticks([5730.59, 5730.60, 5730.61])
axins.set_xticklabels([".59", ".60", ".61"])
axins.set_xlabel("m/z - 5730 (Da)", fontsize=8)
axins.tick_params(labelsize=6)
axins.set_title("Second cluster: 5 subpeaks", fontsize=8, pad=2)
for spine in axins.spines.values():
    spine.set_edgecolor("gray")

line_args = dict(arrowstyle="-", color="gray", linewidth=1.2, linestyle=(0, (1, 2)), alpha=0.7)
ax.annotate("", xy=(peak_x, peak_y), xycoords="data",
            xytext=(left, bottom + height), textcoords="axes fraction",
            arrowprops=line_args)
ax.annotate("", xy=(peak_x, 0), xycoords="data",
            xytext=(left, bottom), textcoords="axes fraction",
            arrowprops=line_args)

plt.tight_layout()
save_fig("step02_spectrum.png")
plt.close()