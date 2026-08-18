"""
step01_averagine.py

Averagine model estimates the elemental composition of a protein
from its average molecular mass alone.

The "Averagine" concept (Senko et al. 1995) defines a hypothetical
average amino-acid residue whose elemental ratios are derived from
large proteome databases. 

Two coefficient sets are relevant for thesis:
    - Senko et al. 1995:     unit_mass = 111.1254 Da (original)
    - Radziński et al. 2022: unit_mass = 110.4728 Da (used here, consistent
      with the Envemind benchmark used in step15)

Outputs:
    results/step01_averagine_scaling.csv   scaling table for a range of test masses

"""

import math
import csv
from step00_shared_utils import results_dir

# Averagine coefficients (Radziński et al. 2022)
averagine_unit_mass = 110.4728
averagine_comp = {"C": 4.9245, "H": 7.7724, "N": 1.3555, "O": 1.46, "S": 0.0356}

atomic_mass = {"C": 12.0107, "H": 1.00794, "N": 14.0067, "O": 15.9994, "S": 32.065}


def calculate_mass(target_mass):
    """
    Estimate the molecular formula for a protein of target_mass Da.

    Scales the Averagine composition to the target mass, floors each element 
    count, then closes the remaining mass gap with extra hydrogen atoms.

    Parameters:
        target_mass: desired average molecular mass (Da)

    Returns:
        formula_str: e.g. "C254H375N65O73"
        formula:     {element: int count}
        actual_mass: actual average mass of the returned formula
    """
    f = target_mass / averagine_unit_mass

    formula = {}
    for e in averagine_comp:
        formula[e] = math.floor(f * averagine_comp[e])

    current_mass = 0.0
    for e in formula:
        current_mass += formula[e] * atomic_mass[e]

    extra_H = max(0, round((target_mass - current_mass) / atomic_mass["H"]))
    formula["H"] += extra_H
    actual_mass = current_mass + extra_H * atomic_mass["H"]

    parts = []
    for e in formula:
        if formula[e] > 0:
            parts.append(f"{e}{formula[e]}")
    formula_str = "".join(parts)

    return formula_str, formula, actual_mass


if __name__ == "__main__":
    test_masses = [1000, 2500, 5733, 10000, 20000, 57000, 90000]

    rows = []
    print(f"{'Target mass (Da)':>12}  {'Formula':<30}  {'Actual mass (Da)':>12}  {'Deviation (Da)':>10}")
    for mass in test_masses:
        s, _, m = calculate_mass(mass)
        delta = m - mass
        print(f"{mass:>12}  {s:<30}  {m:>12.2f}  {delta:>+10.4f}")
        rows.append({"target_mass_Da": mass, "formula": s,
                     "actual_mass_Da": round(m, 4), "delta_Da": round(delta, 4)})

    out = results_dir / "step01_averagine_scaling.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved results: {out}")