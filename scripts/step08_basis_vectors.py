"""
step08_basis_vectors.py

Core module script. Builds the three-dimensional orthogonal basis {V1, V2, V3}
and provides the residue-H formula reconstruction function.

The three basis vectors:
    V1: mass unit vector (scales total molecular mass)
    V2: variance unit vector (controls isotopic envelope width)
    V3: mass defect unit vector (encodes nuclear binding energy effects)

Outputs (when run directly):
    figures/step08_basis_raw_reconstruction.png  raw reconstruction before H shift
    figures/step08_basis_after_h_search.png      reconstruction after H grid search
    results/step08_basis_vectors.csv             k-coefficients and Wasserstein distances
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from IsoSpecPy import Iso, IsoTotalProb

from step00_shared_utils import (
    elements, ins_formula, save_fig, results_dir,
    parse_formula_to_dict, formula_dict_to_string, vector_from_formula_dict,
    spectrum_variance, aggregate_close_peaks, highest_peak
)


def compute_mass_vector(elements_list):
    """Compute the average atomic mass vector V1 across given elements."""
    vec = np.zeros(len(elements_list))
    for i, el in enumerate(elements_list):
        iso_el = Iso(f"{el}1")
        vec[i] = float(iso_el.getTheoreticalAverageMass())
    return vec


def compute_variance_vector(elements_list, base_dict, base_var):
    """Compute the numerical gradient of spectrum variance V2."""
    vec = np.zeros(len(elements_list))
    for i, el in enumerate(elements_list):
        temp_dict = base_dict.copy()
        temp_dict[el] += 1
        temp_formula = formula_dict_to_string(temp_dict)
        temp_spec = IsoTotalProb(0.999999999, formula=temp_formula)
        temp_spec.normalize()
        temp_masses = list(temp_spec.masses)
        temp_probs = list(temp_spec.probs)
        vec[i] = spectrum_variance(temp_masses, temp_probs) - base_var
    return vec


def compute_defect_vector(elements_list):
    """Compute the probability-weighted mass defect vector V3 per atom."""
    vec = np.zeros(len(elements_list))
    for i, el in enumerate(elements_list):
        iso_el = Iso(el)
        masses = iso_el.isotopeMasses[0]  
        probs  = iso_el.isotopeProbabilities[0]
        md = 0
        for m, p in zip(masses, probs):
            fractional = m - int(m)  
            md += fractional * p 
        vec[i] = md 
    return vec


def orthogonalize_vectors(mass_vec, V_var, V_def):
    """Gram-Schmidt orthogonalization procedure to produce orthonormal V1, V2, V3."""
    mass_unit = mass_vec / np.linalg.norm(mass_vec)
    
    proj_coeff_var = np.dot(V_var, mass_vec) / np.dot(mass_vec, mass_vec)
    V_var_ortho = V_var - proj_coeff_var * mass_vec
    
    V_var_norm = np.linalg.norm(V_var_ortho)
    if V_var_norm > 1e-10:
        V_var_unit = V_var_ortho / V_var_norm
    else:
        V_var_unit = V_var_ortho
    
    proj_coeff_def_mass = np.dot(V_def, mass_vec) / np.dot(mass_vec, mass_vec)
    V_def_temp = V_def - proj_coeff_def_mass * mass_vec
    
    proj_coeff_def_var = np.dot(V_def_temp, V_var_unit) / np.dot(V_var_unit, V_var_unit)
    V_def_ortho = V_def_temp - proj_coeff_def_var * V_var_unit
    
    V_def_norm = np.linalg.norm(V_def_ortho)
    if V_def_norm > 1e-10:
        V_def_unit = V_def_ortho / V_def_norm
    else:
        V_def_unit = V_def_ortho
    
    return mass_unit, V_var_unit, V_def_unit


def get_residue_h(target_vec, elements_list, h_mass, target_mass_val):
    """Round continuous composition vector to integers and compensate mass error via hydrogen count."""
    f_dict = {}
    for i, el in enumerate(elements_list):
        val = int(round(target_vec[i]))
        f_dict[el] = max(0, val)
    
    temp_formula = formula_dict_to_string(f_dict)
    if temp_formula:
        current_mass = Iso(temp_formula).getTheoreticalAverageMass()
    else:
        current_mass = 0.0
    
    mass_diff = target_mass_val - current_mass
    h_correction = int(round(mass_diff / h_mass))
    
    current_h = f_dict.get("H", 0)
    f_dict["H"] = max(0, current_h + h_correction)
    
    return f_dict


if __name__ == "__main__":
    exp_spec = IsoTotalProb(0.999999999, formula=ins_formula)
    exp_spec.normalize()
    exp_masses = list(exp_spec.masses)
    exp_probs = list(exp_spec.probs)
    exp_var = spectrum_variance(exp_masses, exp_probs)

    ins_dict = parse_formula_to_dict(ins_formula)
    ins_vec = vector_from_formula_dict(ins_dict)

    mass_vec = compute_mass_vector(elements)
    V_var = compute_variance_vector(elements, ins_dict, exp_var)
    V_def = compute_defect_vector(elements)

    mass_unit, V_var_unit, V_def_unit = orthogonalize_vectors(mass_vec, V_var, V_def)

    k_mass = np.dot(ins_vec, mass_unit)
    k_var = np.dot(ins_vec, V_var_unit)
    k_def = np.dot(ins_vec, V_def_unit)

    F_vec = k_mass * mass_unit + k_var * V_var_unit + k_def * V_def_unit

    insulin_mass = Iso(ins_formula).getTheoreticalAverageMass()
    H_mass = float(Iso("H1").getTheoreticalAverageMass())

    exp_peak, _, _ = highest_peak(exp_masses, exp_probs)

    temp_dict_before = {}
    for i, el in enumerate(elements):
        temp_dict_before[el] = max(0, int(round(F_vec[i])))
    initial_formula_before = formula_dict_to_string(temp_dict_before)

    initial_spec_before = IsoTotalProb(0.999999999, formula=initial_formula_before)
    initial_spec_before.normalize()
    initial_masses_before = list(initial_spec_before.masses)
    initial_probs_before = list(initial_spec_before.probs)
    wasser_initial_before = initial_spec_before.wassersteinDistance(exp_spec)

    model_peak, _, _ = highest_peak(initial_masses_before, initial_probs_before)
    delta = exp_peak - model_peak

    exp_masses_aggr, exp_probs_aggr = aggregate_close_peaks(exp_masses, exp_probs)
    initial_masses_aggr_before, initial_probs_aggr_before = aggregate_close_peaks(initial_masses_before, 
                                                                                  initial_probs_before)

    # Plot 1: Raw reconstruction before hydrogen shift
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.vlines(exp_masses_aggr, 0, exp_probs_aggr, label=f"Reference: {ins_formula}", color="pink", linewidth=2)
    ax.vlines(initial_masses_aggr_before, 0, initial_probs_aggr_before,
              label=f"Reconstruction (before H shift): {initial_formula_before}",
              color="steelblue", linewidth=1.5)
    ax.set_xlim(5727, 5745)
    ax.set_ylim(0, 0.25)
    ax.set_xlabel("m/z (Da)", fontsize=12)
    ax.set_ylabel("Intensity", fontsize=12)
    ax.set_title("V1+V2+V3 reconstruction before hydrogen shift\n"
                 f"Formula: {initial_formula_before}  |  W = {wasser_initial_before:.4f} Da"
                 , fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    save_fig("step08_basis_raw_reconstruction.png")
    plt.close()

    # Hydrogen grid search optimization (±10 range)
    best = {"wasser": float("inf"), "formula": None, "h_change": 0, "shift": 0.0, "spec": None,
             "masses": None, "probs": None}

    for n in range(-10, 11):
        shift = delta + n
        target_mass_with_shift = insulin_mass + shift
        
        F_test_dict = get_residue_h(F_vec, elements, H_mass, target_mass_with_shift)
        formula_test = formula_dict_to_string(F_test_dict)
        
        if any(F_test_dict[el] < 0 for el in elements):
            continue

        spec_test = IsoTotalProb(0.999999999, formula=formula_test)
        spec_test.normalize()
        w = spec_test.wassersteinDistance(exp_spec)

        if w < best["wasser"]:
            rounded_h = max(0, int(round(F_vec[elements.index("H")])))
            h_change = F_test_dict.get("H", 0) - rounded_h
            
            best.update({
                "wasser": w,
                "formula": formula_test,
                "h_change": h_change,
                "shift": shift,
                "spec": spec_test,
                "masses": list(spec_test.masses) if spec_test else None,
                "probs": list(spec_test.probs) if spec_test else None
            })

    # Plot 2: Optimized result after hydrogen grid search
    if best["masses"] and best["probs"]:
        best_masses_aggr, best_probs_aggr = aggregate_close_peaks(best["masses"], best["probs"])
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.vlines(exp_masses_aggr, 0, exp_probs_aggr, label=f"Reference: {ins_formula}", color="pink", linewidth=2)
        ax.vlines(best_masses_aggr, 0, best_probs_aggr,
                  label=f"Reconstruction (after H shift): {best['formula']}", color="steelblue", linewidth=1.5)
        ax.set_xlim(5727, 5745)
        ax.set_ylim(0, 0.25)
        ax.set_xlabel("m/z (Da)", fontsize=12)
        ax.set_ylabel("Intensity", fontsize=12)
        ax.set_title("V1+V2+V3 reconstruction after hydrogen grid search\n"
                     f"Formula: {best['formula']}  |  Shift = {best['shift']:.4f} Da  |  W = {best['wasser']:.4f} Da", 
                     fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        save_fig("step08_basis_after_h_search.png")
        plt.close()

    out = results_dir / "step08_basis_vectors.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["formula", "model", "reconstructed_formula", "wasserstein_Da", "k_mass", "k_var", "k_def"])
        writer.writerow([ins_formula, "V1+V2+V3_raw", initial_formula_before, round(wasser_initial_before, 6), 
                         round(k_mass, 4), round(k_var, 4), round(k_def, 4)])
        writer.writerow([ins_formula, "V1+V2+V3+Hshift", best["formula"], round(best["wasser"], 6), 
                         round(k_mass, 4), round(k_var, 4), round(k_def, 4)])
    print(f"  Saved results: {out}")