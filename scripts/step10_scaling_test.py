"""
step10_scaling_test.py

This script evaluates the physical and numerical behavior of the individual
basis components by performing a unit-step sensitivity analysis along the
variance vector (V2) and mass defect vector (V3).

This is the extended, two-scale version of the same unit-step idea introduced
in step09 (there, only tested on the base insulin formula). 
Here it is run on two molecular scales with hydrogen atom tracking
    - 1x insulin  C254H377N65O75S6     (~5.7 kDa)
    - 10x insulin C2540H3770N650O750S60 (~57 kDa)

Outputs:
    figures/step10_scaling_1x.png       1x insulin unit-step comparison
    figures/step10_scaling_10x.png      10x insulin unit-step comparison
    results/step10_scaling_test.csv     Summary metrics, Wasserstein distances, delta_H
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from IsoSpecPy import Iso, IsoTotalProb

from step00_shared_utils import (
    elements, save_fig, results_dir,
    parse_formula_to_dict, formula_dict_to_string, vector_from_formula_dict,
    spectrum_variance, aggregate_close_peaks,
)
from step08_basis_vectors import (
    compute_mass_vector, compute_variance_vector, compute_defect_vector,
    orthogonalize_vectors, get_residue_h,
)


def run_scaling_diagnostic(ins_formula, xlim, tag):
    """
    Run the unit-step sensitivity diagnostic (Mass + 1.0*V2 and Mass + 1.0*V3)
    for a given formula, track the hydrogen adjustment made for each sub-model
    and save the comparison plot.
    """
    print(f"\nRunning unit-step scaling diagnostic for: {ins_formula} ({tag})")

    exp_spec = IsoTotalProb(0.999, formula=ins_formula)
    exp_spec.normalize()
    exp_masses = list(exp_spec.masses)
    exp_probs = list(exp_spec.probs)
    exp_var = spectrum_variance(exp_masses, exp_probs)

    ins_dict = parse_formula_to_dict(ins_formula)
    ins_vec = vector_from_formula_dict(ins_dict)

    # Compute basis and orthogonalise
    mass_vec = compute_mass_vector(elements)
    V_var = compute_variance_vector(elements, ins_dict, exp_var)
    V_def = compute_defect_vector(elements)

    mass_unit, V_var_unit, V_def_unit = orthogonalize_vectors(mass_vec, V_var, V_def)

    k_mass = np.dot(ins_vec, mass_unit)
    print(f"  k_mass: {k_mass:.6f}")

    insulin_mass = Iso(ins_formula).getTheoreticalAverageMass()
    H_mass = float(Iso("H1").getTheoreticalAverageMass())

    # Mass + Variance unit step model
    F_vec_var = k_mass * mass_unit + 1.0 * V_var_unit
    F_dict_var = get_residue_h(F_vec_var, elements, H_mass, insulin_mass)
    formula_var_final = formula_dict_to_string(F_dict_var)
    temp_var = {el: max(0, int(round(F_vec_var[i]))) for i, el in enumerate(elements)}
    delta_h_var = F_dict_var.get("H", 0) - temp_var.get("H", 0)

    # Mass + Defect unit step model
    F_vec_def = k_mass * mass_unit + 1.0 * V_def_unit
    F_dict_def = get_residue_h(F_vec_def, elements, H_mass, insulin_mass)
    formula_def_final = formula_dict_to_string(F_dict_def)
    temp_def = {el: max(0, int(round(F_vec_def[i]))) for i, el in enumerate(elements)}
    delta_h_def = F_dict_def.get("H", 0) - temp_def.get("H", 0)

    # Spectra generation and aggregation
    spec_orig = IsoTotalProb(0.999, formula=ins_formula)
    spec_orig.normalize()
    m_orig, p_orig = aggregate_close_peaks(list(spec_orig.masses), list(spec_orig.probs))

    spec_var = IsoTotalProb(0.999, formula=formula_var_final)
    spec_var.normalize()
    m_var, p_var = aggregate_close_peaks(list(spec_var.masses), list(spec_var.probs))

    spec_def = IsoTotalProb(0.999, formula=formula_def_final)
    spec_def.normalize()
    m_def, p_def = aggregate_close_peaks(list(spec_def.masses), list(spec_def.probs))

    # xlim is centred on the molecule's own mass so the same
    # function works unmodified at any scale
    centre = float(insulin_mass)
    plot_xlim = xlim if xlim is not None else (centre - 150, centre + 150)

    plt.figure(figsize=(12, 6))
    plt.vlines(m_orig, 0, p_orig, color="steelblue", label=f"Reference: {ins_formula}")
    plt.vlines(m_var, 0, p_var, color="pink",
               label=f"Mass + V2×1 reconstruction: {formula_var_final}  (ΔH = {delta_h_var:+d})")
    plt.vlines(m_def, 0, p_def, color="purple",
               label=f"Mass + V3×1 reconstruction: {formula_def_final}  (ΔH = {delta_h_def:+d})")

    plt.xlim(*plot_xlim)
    plt.ylim(0, 0.25)
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.title(f"Unit-step sensitivity ({tag}): reference vs. mass + V2/V3 unit shifts")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    save_fig(f"step10_scaling_{tag}.png")
    plt.close()

    # Wasserstein distance metrics for CSV output
    w_var = spec_var.wassersteinDistance(spec_orig)
    w_def = spec_def.wassersteinDistance(spec_orig)

    return {
        "formula": ins_formula,
        "formula_mass_var_unit": formula_var_final,
        "W_mass_var_unit": round(w_var, 6),
        "delta_H_var_unit": delta_h_var,
        "formula_mass_def_unit": formula_def_final,
        "W_mass_def_unit": round(w_def, 6),
        "delta_H_def_unit": delta_h_def,
    }


r1 = run_scaling_diagnostic("C254H377N65O75S6", xlim=(5700, 5800), tag="1x")
r10 = run_scaling_diagnostic("C2540H3770N650O750S60", xlim=(57275, 57375), tag="10x")

out = results_dir / "step10_scaling_test.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=r1.keys())
    writer.writeheader()
    writer.writerow(r1)
    writer.writerow(r10)

print(f"  Saved results: {out}")