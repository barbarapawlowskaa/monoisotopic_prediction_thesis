"""
step15_envemind_comparison.py

Compares our orthogonal-basis method against Envemind
on all proteins in the analysis results from step14.

For each protein we run Envemind's monoisotopic mass predictor on the
IsoSpec spectra of both the original and the reconstructed formula and
record the distance (mono_new - mono_orig). The two distributions are
plotted for direct comparison in the thesis.

Outputs:
    figures/step15_comparison_histograms.png   side-by-side histograms
    results/step15_comparison_results.csv      per-protein distances (both methods)
    results/step15_comparison_summary.csv      aggregate statistics
"""

import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IsoSpecPy import IsoTotalProb
import envemind as em

from step00_shared_utils import save_fig, results_dir

results_in  = results_dir / "step14_monoisotopic_distances.txt"
results_out = results_dir / "step15_comparison_results.csv"
summary_out = results_dir / "step15_comparison_summary.csv"


def predict_mono_envemind(formula):
    """
    Predict monoisotopic mass of formula using Envemind on its IsoSpec spectrum.
    """
    spec = IsoTotalProb(0.999, formula=formula)
    try:
        return float(em.monoisotopic_mass_prediction(
            experimental_masses=list(spec.masses),
            experimental_intensities=list(spec.probs),
            charge=None,
        ))
    except Exception as e:
        raise RuntimeError(f"Envemind failed for {formula}: {e}") from e


def run_comparison():
    """Run the full comparison and save all outputs.
    """
    df = pd.read_csv(results_in, sep='\t')
    rows = []
    skipped = 0
    t0 = time.time()
    
    print(f"Proteins to analyse with Envemind: {len(df)}")

    for i, row in df.iterrows():
        if i % 500 == 0:
            print(f"  [{i:>5}/{len(df)}]  "
                  f"{(time.time()-t0)/60:.1f} min  ({skipped} skipped)")
        
        orig_f, new_f = row['OriginalFormula'], row['NewFormula']
        our_d  = float(row['Distance(Da)'])
        try:
            em_d = predict_mono_envemind(new_f) - predict_mono_envemind(orig_f)
        except Exception as e:
            skipped += 1
            continue
        
        rows.append({'original_formula': orig_f, 'new_formula': new_f,
                     'our_distance_Da': our_d, 'envemind_distance_Da': em_d})

    elapsed = time.time() - t0
    print(f"\nDone: {len(rows)} results, {skipped} skipped")
    print(f"Total time: {elapsed:.1f} s ({elapsed/60:.1f} min)")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(results_out, index=False)
    print(f"  Saved results: {results_out}")

    _plot(result_df)
    _save_summary(result_df)
    return result_df


def _plot(df):
    """
    Plot and save side-by-side histograms for both methods.
    """
    our = df['our_distance_Da'].values
    emv = df['envemind_distance_Da'].values

    all_v = np.concatenate([our, emv])
    xmn, xmx = all_v.min(), all_v.max()
    margin = (xmx - xmn) * 0.05
    x_range = (xmn - margin, xmx + margin)

    our_mean = np.abs(our).mean()
    em_mean = np.abs(emv).mean()

    h1, _ = np.histogram(our, bins=50, range=x_range)
    h2, _ = np.histogram(emv, bins=50, range=x_range)
    y_max = max(h1.max(), h2.max()) * 1.05

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Monoisotopic mass distances: our method vs Envemind  (N={len(df)})",
        fontsize=13
    )

    for ax, data, mean_abs, label, col, mc in [
        (ax1, our, our_mean, "Orthogonal-basis method", "steelblue", "navy"),
        (ax2, emv, em_mean,  "Envemind", "pink", "deeppink"),
    ]:
        ax.hist(data, bins=50, color=col, edgecolor='black', alpha=0.75, range=x_range)
        ax.axvline(0,color='black', linestyle='--', linewidth=1, label="0 Da")
        ax.axvline(mean_abs, color=mc, linestyle='--', linewidth=1.5,
                    label=f"Mean |d| = {mean_abs:.4f} Da")
        ax.axvline(-mean_abs, color=mc,      linestyle='--', linewidth=1.5)
        ax.set_xlabel("Distance (Da)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(label, fontsize=12)
        ax.set_xlim(x_range)
        ax.set_ylim(0, y_max)
        ax.legend(fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    save_fig("step15_comparison_histograms.png")
    plt.close()


def _save_summary(df):
    """
    Save aggregate statistics for both methods.
    """
    with open(summary_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "our_method", "envemind"])
        for label, fn in [
            ("mean_abs_Da", lambda x: np.mean(np.abs(x))),
            ("median_abs_Da", lambda x: np.median(np.abs(x))),
            ("pct_within_0.5Da", lambda x: np.mean(np.abs(x) < 0.5)*100),
            ("pct_within_1.0Da", lambda x: np.mean(np.abs(x) < 1.0)*100),
        ]:
            w.writerow([label,
                        round(fn(df['our_distance_Da'].values), 4),
                        round(fn(df['envemind_distance_Da'].values), 4)])
    print(f"  Saved summary: {summary_out}")


if __name__ == "__main__":
    run_comparison()