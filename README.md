# Prediction of Monoisotopic Peak Masses in Large Biopolymers

**Master's thesis project**

Barbara Pawłowska (477701)  
Bioinformatics and Systems Biology  
University of Warsaw  

Supervisor: Michał Startek, PhD

## About the Project

This repository contains the computational pipeline developed for the
master's thesis *Prediction of Monoisotopic Peak Masses in Large
Biopolymers*.

The goal is to predict the position of the monoisotopic peak in a
protein mass spectrum, corresponding to a molecule composed entirely of
the lightest stable isotopes of its constituent elements.

For small molecules this peak is directly observable. However, for proteins
above 10–15 kDa it typically falls below the detection limit of most instruments.
For that reason, predicting its position from the observed isotopic envelope is an
important problem in top-down proteomics.

The method represents each protein's elemental composition as a vector
in a five-dimensional space over {C, H, N, O, S} and constructs a
physically motivated orthogonal basis:

| Vector | Interpretation |
|---------|---------|
| **V1** | Total average mass |
| **V2** | Variance of the isotopic envelope |
| **V3** | Mass defect contribution |
| **V4** | Null-space direction |
| **V5** | Null-space direction |

The null-space directions are determined analytically so that the
reconstructed formula lies on the Averagine ray.

Projection onto this basis reconstructs a formula preserving mass,
variance, and mass defect, allowing prediction of the corresponding
monoisotopic peak position.

The method was validated on 6000 proteins from UniProt Swiss-Prot and
benchmarked against the Envemind algorithm. All spectra are generated
with IsoSpecPy, and the Wasserstein distance is used throughout to
compare isotopic distributions.

## Repository Structure

```text
.
├── scripts/          # run_all.py + step00–step16 pipeline scripts
├── data/
│   └── uniprot_sprot.fasta.gz
├── figures/          # created automatically
└── results/          # created automatically
```

Outputs are always written to `figures/` and `results/` at the project
root, regardless of whether `run_all.py` or an individual
`stepNN_*.py` script is executed directly, and regardless of the
directory from which Python is launched.

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Main dependencies include IsoSpecPy, NumPy, matplotlib, pandas, and
`envemind` (Step 15 only).

`envemind` is installed directly from its GitHub repository:

https://github.com/PiotrRadzinski/envemind

The package is installed via:

```bash
git+https://github.com/PiotrRadzinski/envemind.git
```

Therefore, `git` must be available on the system:

https://git-scm.com/downloads

## Data

Step 13 requires the UniProt Swiss-Prot FASTA file
(`uniprot_sprot.fasta.gz`), available from UniProt:

https://www.uniprot.org/downloads#uniprotkb

Place the file in `data/` (see the repository structure above). This is
also the default value of the `--fasta` argument used by `run_all.py`.

## Usage

```bash
# Run the complete pipeline
python scripts/run_all.py

# Run selected steps only
python scripts/run_all.py --steps 07 10 11

# Full dataset analysis (requires the FASTA file)
python scripts/run_all.py --steps 13 14 15 --fasta data/uniprot_sprot.fasta.gz

# Quick test on shorter proteins
python scripts/run_all.py --steps 13 14 --max-length 300
```

Any individual `stepNN_*.py` script can also be executed directly, for
example:

```bash
python scripts/step03_averagine_fit.py
```

The `--max-proteins` (default: 6000) and `--max-length` (default:
500 aa) arguments control dataset size; see:

```bash
python scripts/run_all.py -h
```

The 6000-protein / 500-aa configuration corresponds to the analyses
reported in the thesis and runs in a reasonable time on a standard
laptop. A full Swiss-Prot analysis (`--max-length 0` and larger
`--max-proteins`) is considerably more computationally demanding.

## Pipeline Steps

| Step | Description |
|------|-------------|
| 00 | Shared utilities: constants, formula parsing, spectrum variance, peak aggregation |
| 01 | Averagine scaling model — mass → estimated formula |
| 02 | IsoSpec + Wasserstein distance sanity check (bovine insulin vs. mock spectrum) |
| 03 | Averagine fit to insulin via ±5 H grid search using Wasserstein distance |
| 04 | Per-element fractional mass defect — physical motivation for V3 |
| 05 | Baryon composition analysis: insulin vs. its Averagine approximation |
| 06 | Monte Carlo search for the maximum baryon-defect direction orthogonal to Averagine |
| 07 | Early 2D model (Averagine/mass vector + variance gradient) |
| 08 | **Core module** — constructs V1, V2, and V3 via Gram–Schmidt; includes `get_residue_h` correction |
| 09 | Basis diagnostics: V1+V2 vs. V1+V3 sub-models and unit-step sensitivity |
| 10 | Unit-step sensitivity at 1× and 10× insulin scale, including ΔH tracking |
| 11 | Completion of the basis: V4 and V5 obtained from the SVD null space |
| 12 | Full 5D reconstruction with the closed-form α, β solution |
| 13 | UniProt FASTA parsing and molecular formula generation |
| 14 | Full 5D pipeline over all proteins; monoisotopic mass distance analysis |
| 15 | Benchmark against Envemind |
| 16 | Standalone conceptual 5D basis diagram |

Each script contains a detailed docstring describing its purpose and
implementation.

`step08` serves as the core module and is imported by all subsequent
steps from 09 onward.

## Output Files

All figures (`.png`) are written to `figures/`, while all generated
data files (`.csv`, `.txt`) are written to `results/`.

Files are named according to the step that produced them, for example:

- `step12_full_5D_reconstruction.png`
- `step14_distance_summary.csv`

The main outputs generated by the full pipeline are:

- `step13_protein_formulas.txt` — parsed molecular formulas (up to 6000 proteins)
- `step14_monoisotopic_distances.txt` — per-protein monoisotopic mass distances
- `step14_distance_summary.csv` — aggregate reconstruction statistics
- `step15_comparison_results.csv` — per-protein comparison with Envemind
- `step15_comparison_summary.csv` — benchmark summary statistics

## License

MIT License — see [LICENSE](LICENSE).
