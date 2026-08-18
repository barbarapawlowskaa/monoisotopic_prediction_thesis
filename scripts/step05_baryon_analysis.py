"""
step05_baryon_analysis.py

Compares the baryon structure and nuclear mass defect of the insulin formula vs 
its Averagine approximation.

A baryon is a proton or neutron. The total baryon count of a molecule is:
    baryons = sum_el( (Z_el + N_el) * count_el )

where Z_el = proton number and N_el = neutron number of the most abundant
stable isotope. The average baryon mass (monoisotopic_mass / baryons)
measures how much rest mass each nucleon carries on average.

Outputs:
    results/step05_baryon_analysis.csv    per-formula baryon comparison

Exported functions:
    count_baryons_mono(formula_dict) -> (protons, neutrons)
    average_baryon_mass(mono_mass, protons, neutrons) : float
"""

import csv
import re
from IsoSpecPy import Iso

from step00_shared_utils import results_dir
from step01_averagine import calculate_mass

# Proton + neutron counts for the most abundant stable isotope of each element
monoisotopic_baryons = {
    "C": (6,  6),    # 12C
    "H": (1,  0),    # 1H
    "N": (7,  7),    # 14N
    "O": (8,  8),    # 16O
    "S": (16, 16),   # 32S
}


def parse_baryon(formula):
    result = {}
    for el, n in re.findall(r'([A-Z][a-z]*)(\d*)', formula):
        if el:
            result[el] = int(n) if n else 1
    return result


def count_baryons_mono(formula_dict):
    """
    Count protons and neutrons in the monoisotopic molecule.

    Parameters:
        formula_dict: {element: count}

    Returns:
        (protons, neutrons)
    """
    p = n = 0
    for el, count in formula_dict.items():
        if el in monoisotopic_baryons:
            zp, zn = monoisotopic_baryons[el]
            p += zp * count
            n += zn * count
    return p, n


def average_baryon_mass(mono_mass, protons, neutrons):
    """
    Average mass per baryon = monoisotopic_mass / (protons + neutrons).
    Lower value means more nuclear binding energy per nucleon.
    """
    total = protons + neutrons
    return mono_mass / total if total > 0 else 0.0


def energy_difference_J(bm1, bm2, total_baryons):
    """Energy equivalent (J) of a baryon-mass difference via E = delta_m * c^2."""
    C2 = (299_792_458) ** 2
    da_to_kg = 1.66053906660e-27
    return abs(bm1 - bm2) * da_to_kg * total_baryons * C2


if __name__ == "__main__":
    ins_formula = "C254H377N65O75S6"
    ins_dict = parse_baryon(ins_formula)

    exp_iso = Iso(formula=ins_formula)
    exp_mono = float(exp_iso.getMonoisotopicPeakMass())
    mass_ins = float(exp_iso.getTheoreticalAverageMass())

    _, base_formula, _ = calculate_mass(mass_ins)
    avg_str = "".join(f"{e}{base_formula[e]}" for e in base_formula if base_formula[e] > 0)
    avg_iso = Iso(formula=avg_str)
    avg_mono = float(avg_iso.getMonoisotopicPeakMass())

    ins_p, ins_n = count_baryons_mono(ins_dict)
    avg_p, avg_n = count_baryons_mono(base_formula)

    ins_bm = average_baryon_mass(exp_mono, ins_p, ins_n)
    avg_bm = average_baryon_mass(avg_mono, avg_p, avg_n)
    total_b = (ins_p + ins_n + avg_p + avg_n) // 2
    delta_e = energy_difference_J(ins_bm, avg_bm, total_b)

    print(f"{'Insulin formula':30}: {ins_formula}")
    print(f"{'Averagine formula':30}: {avg_str}")
    print(f"{'Monoisotopic mass (insulin)':30}: {exp_mono:.4f} Da")
    print(f"{'Monoisotopic mass (averagine)':30}: {avg_mono:.4f} Da")
    print(f"{'Difference':30}: {avg_mono - exp_mono:+.4f} Da")
    print(f"{'Protons (insulin / avg)':30}: {ins_p} / {avg_p}")
    print(f"{'Neutrons (insulin / avg)':30}: {ins_n} / {avg_n}")
    print(f"{'Avg baryon mass (insulin)':30}: {ins_bm:.8f} Da/baryon")
    print(f"{'Avg baryon mass (averagine)':30}: {avg_bm:.8f} Da/baryon")
    print(f"{'Baryon mass defect':30}: {abs(ins_bm - avg_bm):.10f} Da/baryon")
    print(f"{'Energy equivalent':30}: {delta_e:.6e} J")

    out = results_dir / "step05_baryon_analysis.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["property", "insulin", "averagine"])
        w.writerow(["formula", ins_formula, avg_str])
        w.writerow(["monoisotopic_mass_Da", round(exp_mono, 4), round(avg_mono, 4)])
        w.writerow(["protons", ins_p, avg_p])
        w.writerow(["neutrons", ins_n, avg_n])
        w.writerow(["avg_baryon_mass_Da_per_baryon", round(ins_bm, 8), round(avg_bm, 8)])
        w.writerow(["baryon_mass_defect_Da_per_baryon", round(abs(ins_bm - avg_bm), 10), ""])
        w.writerow(["energy_equivalent_J", f"{delta_e:.6e}", ""])
    print(f"  \n  Saved results: {out}")