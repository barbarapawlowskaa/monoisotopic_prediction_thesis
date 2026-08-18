"""
step04_mass_defect_intro.py

Physical background. Computes the per element mass defect of insulin
and determines the gradient direction of maximum defect increase.
This is the same definition used throughout the pipeline.

Outputs:
    results/step04_mass_defect_per_element.csv   per-element defects
    results/step04_mass_defect_insulin.csv       total defect and gradient
"""

import csv
import re
import numpy as np
from IsoSpecPy import Iso

from step00_shared_utils import results_dir, parse_formula_to_dict as parse_formula, ins_formula


def per_element_defects(elements):
    """
    Compute the fractional mass defect for each element.

    For each element, function iterates over all naturally occurring isotopes
    and computes the probability weighted sum of fractional masses:
        defect = sum( (m - floor(m)) * abundance )

    Parameters:
        elements: list of element symbols

    Returns:
        dict {element: defect_Da}
    """
    defects = {}
    for el in elements:
        iso = Iso(el)
        masses = iso.isotopeMasses[0]
        probs = iso.isotopeProbabilities[0]
        mass_defect = 0.0
        for m, p in zip(masses, probs):
            fractional = m - int(m)
            mass_defect += fractional * p
        defects[el] = mass_defect
    return defects


ins_dict = parse_formula(ins_formula)
elements = list(ins_dict.keys())
defects  = per_element_defects(elements)

total_defect = sum(ins_dict[el] * defects[el] for el in elements)

print("Mass defect per element:")
for el in elements:
    print(f"  {el}: {defects[el]:.6f} Da")
print(f"\nTotal mass defect for {ins_formula}: {total_defect:.6f} Da")

defect_vec = np.array([defects[el] for el in elements])
gradient = defect_vec / np.linalg.norm(defect_vec)
print(f"\nGradient (normalised): {gradient}")

out_el = results_dir / "step04_mass_defect_per_element.csv"
with open(out_el, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["element", "mass_defect_Da"])
    for el in elements:
        w.writerow([el, round(defects[el], 8)])
print(f"  \n  Saved results: {out_el}")

out_ins = results_dir / "step04_mass_defect_insulin.csv"
with open(out_ins, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["formula", "total_defect_Da"] +
               [f"gradient_{el}" for el in elements])
    w.writerow([ins_formula, round(total_defect, 6)] +
               [round(g, 6) for g in gradient])
print(f"  Saved results: {out_ins}")