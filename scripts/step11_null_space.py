"""
step11_null_space.py

Extends the basis from 3D to 5D.
Finds V4 and V5 spanning the null spaceof the 3×5 matrix [V1; V2; V3].

Any formula vector F in R5 decomposes as:
    F = k1*V1 + k2*V2 + k3*V3 + alpha*V4 + beta*V5
 
A 45- degree angle rotation is applied, because the orientation of null-space vectors is not unique.
The resulting V4 and V5 vctors complete an orthonormal basis of the full five-dimensional 
formula space.

Outputs:
    figures/step11_null_space_V4V5.png     effect of V4, V5 on insulin
    results/step11_null_space_vectors.csv  V4, V5 components and orthogonality check

"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from IsoSpecPy import Iso, IsoTotalProb

from step00_shared_utils import (
    elements, ins_formula, save_fig, results_dir,
    parse_formula_to_dict, formula_dict_to_string, vector_from_formula_dict,
    spectrum_variance, aggregate_close_peaks,
)
from step08_basis_vectors import (
    compute_mass_vector, compute_variance_vector, compute_defect_vector,
    orthogonalize_vectors, get_residue_h,
)


def complete_ortho(v1, v2, v3):
    """
    Find V4, V5 spanning the null space of [v1; v2; v3] via SVD.

    Parameters:
        v1, v2, v3: mutually orthonormal vectors, shape (5,)

    Returns:
        (v4, v5): orthonormal vectors, shape (5,)
    """
    A = np.array([v1, v2, v3])
    
    # Perform singular value decomposition to extract right singular vectors
    _, _, Vt = np.linalg.svd(A)
    
    v4_orig = Vt[3] 
    v5_orig = Vt[4] 

    # Apply a 45-degree rotation in the null-space plane
    v4 = (v4_orig + v5_orig) / np.sqrt(2)
    v5 = (-v4_orig + v5_orig) / np.sqrt(2)
    
    # Reorthogonalization and normalization 
    v5 = v5 - np.dot(v5, v4) * v4
    v4 = v4 / np.linalg.norm(v4)
    v5 = v5 / np.linalg.norm(v5)

    return v4, v5


if __name__ == "__main__":
    ins_dict = parse_formula_to_dict(ins_formula)
    ins_vec = vector_from_formula_dict(ins_dict)
    ins_mass = float(Iso(ins_formula).getTheoreticalAverageMass())
    H_mass = float(Iso("H1").getTheoreticalAverageMass())

    spec_ref = IsoTotalProb(0.999, formula=ins_formula)
    spec_ref.normalize()
    exp_masses = list(spec_ref.masses)
    exp_probs = list(spec_ref.probs)
    exp_var  = spectrum_variance(exp_masses, exp_probs)

    mass_vec = compute_mass_vector(elements)
    V_var = compute_variance_vector(elements, ins_dict, exp_var)
    V_def = compute_defect_vector(elements)

    mass_unit, V_var_unit, V_def_unit = orthogonalize_vectors(mass_vec, V_var, V_def)

    # Compute null-space basis vectors v4 and v5
    v4, v5 = complete_ortho(mass_unit, V_var_unit, V_def_unit)

    max_alpha = 0
    for alpha in range(1, 20):
        F_vec = ins_vec + alpha * v4
        if np.all(F_vec > 0):
            max_alpha = alpha
        else:
            break

    max_beta = 0
    for beta in range(1, 20):
        F_vec = ins_vec + beta * v5
        if np.all(F_vec > 0):
            max_beta = beta
        else:
            break

    F_vec_v4 = ins_vec + max_alpha * v4
    F_dict_v4 = get_residue_h(F_vec_v4, elements, H_mass, ins_mass)
    formula_v4 = formula_dict_to_string(F_dict_v4)

    spec_v4 = IsoTotalProb(0.999, formula=formula_v4)
    spec_v4.normalize()
    m_v4, p_v4 = aggregate_close_peaks(list(spec_v4.masses), list(spec_v4.probs))

    F_vec_v5 = ins_vec + max_beta * v5
    F_dict_v5 = get_residue_h(F_vec_v5, elements, H_mass, ins_mass)
    formula_v5 = formula_dict_to_string(F_dict_v5)

    spec_v5 = IsoTotalProb(0.999, formula=formula_v5)
    spec_v5.normalize()
    m_v5, p_v5 = aggregate_close_peaks(list(spec_v5.masses), list(spec_v5.probs))

    F_vec_both = ins_vec + max_alpha * v4 + max_beta * v5
    F_dict_both = get_residue_h(F_vec_both, elements, H_mass, ins_mass)
    formula_both = formula_dict_to_string(F_dict_both)

    spec_both = IsoTotalProb(0.999, formula=formula_both)
    spec_both.normalize()
    m_both, p_both = aggregate_close_peaks(list(spec_both.masses), list(spec_both.probs))

    spec_orig = IsoTotalProb(0.999, formula=ins_formula)
    spec_orig.normalize()
    m_orig, p_orig = aggregate_close_peaks(list(spec_orig.masses), list(spec_orig.probs))

    # Plot null-space influence graphs (v4/v5 comparison)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Influence of V4 and V5 on the insulin spectrum\n(α = {max_alpha}, β = {max_beta})', 
                 fontsize=14)

    ax1.vlines(m_orig, 0, p_orig, color='pink', linewidth=2, label=f'Reference: {ins_formula}')
    ax1.vlines(m_v4, 0, p_v4, color='steelblue', alpha=0.7, linewidth=1.5, label=f'V4 only: {formula_v4}')
    ax1.vlines(m_v5, 0, p_v5, color='purple', alpha=0.7, linewidth=1.5, label=f'V5 only: {formula_v5}')

    ax1.set_xlim(5727, 5745)
    ax1.set_ylim(0, 0.25)
    ax1.set_xlabel('m/z (Da)', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title('Reference vs. V4 vs. V5')
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True, alpha=0.2)

    ax2.vlines(m_orig, 0, p_orig, color='pink', linewidth=2, label=f'Reference: {ins_formula}')
    ax2.vlines(m_both, 0, p_both, color='mediumpurple', alpha=0.7, linewidth=1.5, 
               label=f'V4 + V5 reconstruction: {formula_both}')

    ax2.set_xlim(5727, 5745)
    ax2.set_ylim(0, 0.25)
    ax2.set_xlabel('m/z (Da)', fontsize=12)
    ax2.set_ylabel('Intensity', fontsize=12)
    ax2.set_title('Reference vs. V4 and V5 combined')
    ax2.legend(loc='upper right', fontsize='small')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    save_fig("step11_null_space_V4V5.png")
    plt.close()

    print(f"Original formula: {ins_formula}")
    print(f"Formula with v4 (α={max_alpha}): {formula_v4}")
    print(f"Formula with v5 (β={max_beta}): {formula_v5}")
    print(f"Formula with both (α={max_alpha}, β={max_beta}): {formula_both}")

    out = results_dir / "step11_null_space_vectors.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vector"] + elements)
        w.writerow(["V4"] + [round(x, 6) for x in v4])
        w.writerow(["V5"] + [round(x, 6) for x in v5])
        w.writerow([f"V4·V5 = {np.dot(v4,v5):.2e}"])
    print(f"  Saved results: {out}")