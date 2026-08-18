"""
step00_shared_utils.py

Shared utility functions imported by every other step.
This module has no outputs of its own, you can run it only to verify imports work.

Contains:
    elements                         fixed element order [C, H, N, O, S]
    figures_dir, results_dir         output directory paths
    save_fig(filename)               save current matplotlib figure to figures/
    parse_formula_to_dict(formula)   formula string: {element: count}
    formula_dict_to_string(d)        {element: count}: formula string
    vector_from_formula_dict(d)      {element: count}: np.ndarray shape (5,)
    spectrum_variance(masses, probs) probability-weighted variance (Da²)
    highest_peak(masses, probs)      (mass, prob, index) of most abundant peak
    aggregate_close_peaks(...)       merge nearby peaks for unit-resolution plots
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Output directories 
project_root = Path(__file__).resolve().parent.parent
figures_dir = project_root / "figures"
results_dir = project_root / "results"
figures_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

# Fixed element order used throughout the project
elements = ["C", "H", "N", "O", "S"]

# Reference formula (bovine insulin) used across many steps
ins_formula = "C254H377N65O75S6"


def save_fig(filename, dpi= 300):
    """
    Save the current matplotlib figure to figures_dir/filename.

    Parameters:
        filename: filename including extension, e.g. "step03_averagine_fit.png"
        dpi:      resolution (default 300 for print quality)
    """
    path = figures_dir / filename
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved figure: {path}")


def parse_formula_to_dict(formula):
    """
    Parse a molecular formula string to a count dictionary.
    Only the five elements in elements are tracked, others are ignored.

    Parameters:
        formula: e.g. "C254H377N65O75S6"

    Returns:
        dict {element: count}, missing elements default to 0
    """
    result = {el: 0 for el in elements}
    for el, num in re.findall(r'([CHNOS])(\d+)', formula):
        result[el] = int(num)
    return result


def formula_dict_to_string(dict_1):
    """
    Convert a count dictionary to a molecular formula string.
    Elements with count 0 are omitted. Order follows elements.

    Parameters:
        d: {element: count}

    Returns:
        str, e.g. "C254H377N65O75S6"
    """
    parts = []
    for el in elements:
        count = int(dict_1.get(el, 0))
        if count != 0:
            parts.append(f"{el}{count}")
    return "".join(parts)


def vector_from_formula_dict(dict_2):
    """
    Convert a count dictionary to a numpy vector in elements order.

    Returns:
        np.ndarray, shape (5,), dtype float64
    """
    values = []
    for el in elements:
        values.append(float(dict_2.get(el, 0)))
    return np.array(values)


def spectrum_variance(masses, probs):
    """
    Compute the probability weighted variance of a mass spectrum, in Da squared.
    The probabilities are renormalised internally, so they do not need to sum to one.

    Parameters:
        masses: array-like of peak masses
        probs: array-like of peak probabilities

    Returns:
        float, the variance of the mass distribution in Da squared
    """
    m = np.asarray(masses, dtype=float)
    p = np.asarray(probs, dtype=float)
    p = p / p.sum()
    mu = np.dot(m, p)
    squared_deviations = (m - mu) ** 2
    variance = np.dot(squared_deviations, p)
    return float(variance)


def highest_peak(masses, probs):
    """Return (mass, probability, index) of the most abundant peak."""
    idx = int(np.argmax(probs))
    return masses[idx], probs[idx], idx


def aggregate_close_peaks(masses, probs, max_distance = 0.03):
    """
    Merge isotopologue peaks within max_distance Da into a single peak.

    Parameters:
        masses, probs:  lists of float
        max_distance:   merge threshold in Da (default 0.03)

    Returns:
        (new_masses, new_probs): lists of float
    """
    combined = sorted(zip(masses, probs), key=lambda x: x[0])
    new_masses, new_probs = [], []
    gm, gp = [combined[0][0]], [combined[0][1]]

    for m, p in combined[1:]:
        if abs(m - gm[-1]) <= max_distance:
            gm.append(m)
            gp.append(p)
        else:
            new_masses.append(gm[gp.index(max(gp))])
            new_probs.append(sum(gp))
            gm, gp = [m], [p]

    new_masses.append(gm[gp.index(max(gp))])
    new_probs.append(sum(gp))
    return new_masses, new_probs


if __name__ == "__main__":
    print(f"Elements    : {elements}")
    print(f"figures dir : {figures_dir.resolve()}")
    print(f"results dir : {results_dir.resolve()}")
    d = parse_formula_to_dict(ins_formula)
    print(f"Parsed {ins_formula}: {d}")
    print(f"Back to string: {formula_dict_to_string(d)}")
    print(f"As vector: {vector_from_formula_dict(d)}")