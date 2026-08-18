"""
step14_full_analysis.py

Main analysis. Runnins the full 5D orthogonal-basis pipeline over all
proteins in the database and records monoisotopic mass distances.

Outputs:
    results/step14_monoisotopic_distances.txt   per-protein TSV
    results/step14_distance_summary.csv         aggregate statistics
    figures/step14_distance_histogram.png       distribution of distances
"""

import gc
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from IsoSpecPy import Iso, IsoTotalProb

from step00_shared_utils import (
    elements, save_fig, results_dir,
    parse_formula_to_dict, formula_dict_to_string, vector_from_formula_dict,
    spectrum_variance,
)
from step08_basis_vectors import (
    compute_mass_vector, compute_variance_vector, compute_defect_vector,
    orthogonalize_vectors, get_residue_h,
)
from step11_null_space  import complete_ortho
from step13_parse_fasta import load_proteins

# Averagine direction vector (Radziński 2022)
averagine_a  = np.array([4.9245, 7.7724, 1.3555, 1.46, 0.0356])
results_file = results_dir / "step14_monoisotopic_distances.txt"


def get_monoisotopic_mass(formula):
    """Return the exact monoisotopic mass (Da)"""
    return float(Iso(formula=formula).getMonoisotopicPeakMass())


def analyze_one(protein):
    """
    Run the full 5D pipeline for one protein.

    Parameters:
        protein: {uniprot_id, formula, mass}

    Returns:
        {id, orig, new, distance}
    """
    orig = protein["formula"]
    d  = parse_formula_to_dict(orig)
    v  = vector_from_formula_dict(d)
    sp  = IsoTotalProb(0.999, formula=orig)
    sp.normalize()
    var  = spectrum_variance(list(sp.masses), list(sp.probs))

    v1, v2, v3 = orthogonalize_vectors(
        compute_mass_vector(elements),
        compute_variance_vector(elements, d, var),
        compute_defect_vector(elements),
    )
    v4, v5 = complete_ortho(v1, v2, v3)

    k1, k2, k3 = np.dot(v,v1), np.dot(v,v2), np.dot(v,v3)
    K = k1*v1 + k2*v2 + k3*v3

    A = averagine_a
    kv4, av4 = np.dot(K,v4), np.dot(A,v4)
    kv5, av5 = np.dot(K,v5), np.dot(A,v5)
    ka,  aa = np.dot(K,A),  np.dot(A,A)
    denom  = av4**2 + av5**2 - aa
    if abs(denom) < 1e-12:
        alpha = beta = 0.0
    else:
        gamma = (kv4*av4 + kv5*av5 - ka) / denom
        alpha = gamma*av4 - kv4
        beta = gamma*av5 - kv5

    F  = K + alpha*v4 + beta*v5
    H_m  = float(Iso("H1").getTheoreticalAverageMass())
    nd = get_residue_h(F, elements, H_m, protein["mass"])
    new = formula_dict_to_string(nd)

    return {"id": protein["uniprot_id"], "orig": orig, "new": new,
            "distance": get_monoisotopic_mass(new) - get_monoisotopic_mass(orig)}


def run_analysis(input_file=None, max_length=None):
    """
    Analyse all proteins and save the monoisotopic mass distances.

    Parameters:2
        input_file:  path to step13 output TSV; defaults to results_dir path
        max_length:  skip proteins longer than this (amino acids); None = no limit
    """
    if input_file is None:
        input_file = results_dir / "step13_protein_formulas.txt"
    proteins = load_proteins(input_file, max_length=max_length)
    print(f"Proteins to analyse: {len(proteins)}")

    results, skipped = [], 0
    t0 = time.time()

    for i, p in enumerate(proteins):
        if i % 500 == 0:
            print(f"  [{i:>5}/{len(proteins)}]  "
                  f"{(time.time()-t0)/60:.1f} min  ({skipped} skipped)")
        try:
            results.append(analyze_one(p))
        except Exception as e:
            print(f"  skip {p['uniprot_id']}: {e}")
            skipped += 1
        gc.collect()

    elapsed = time.time() - t0
    print(f"\nDone: {len(results)} results, {skipped} skipped")
    print(f"Total time: {elapsed:.1f} s ({elapsed/60:.1f} min)")

    with open(results_file, 'w') as fh:
        fh.write("OriginalFormula\tNewFormula\tDistance(Da)\n")
        for r in results:
            fh.write(f"{r['orig']}\t{r['new']}\t{r['distance']:.6f}\n")
    print(f"  Saved results: {results_file}")
    return results


def plot_histogram():
    """Plot and save the distribution of monoisotopic mass distances."""
    distances = []
    with open(results_file) as fh:
        next(fh)
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                try:
                    distances.append(float(parts[2]))
                except ValueError:
                    pass

    if not distances:
        print("No distances found.")
        return

    mean_abs = np.mean(np.abs(distances))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(distances, bins=50, color='steelblue', edgecolor='black', alpha=0.8)
    ax.axvline(0, color='black',  linestyle='--', linewidth=1, label='0 Da')
    ax.axvline( mean_abs, color='navy', linestyle='--', linewidth=1.5,
                label=f"Mean |d| = {mean_abs:.4f} Da")
    ax.axvline(-mean_abs, color='navy',   linestyle='--', linewidth=1.5)
    ax.set_title(f"Monoisotopic mass distances  (N = {len(distances)})", fontsize=13)
    ax.set_xlabel("Distance (Da)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_fig("step14_distance_histogram.png")
    plt.close()

    arr = np.array(distances)
    out = results_dir / "step14_distance_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_proteins", "mean_abs_distance_Da", "median_abs_Da",
                    "pct_within_0.5Da", "pct_within_1Da"])
        w.writerow([len(arr),
                    round(float(np.mean(np.abs(arr))), 6),
                    round(float(np.median(np.abs(arr))), 6),
                    round(float(np.mean(np.abs(arr) < 0.5)) * 100, 2),
                    round(float(np.mean(np.abs(arr) < 1.0)) * 100, 2)])
    print(f"  Saved summary: {out}")


if __name__ == "__main__":
    run_analysis(max_length=500)
    plot_histogram()