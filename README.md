# Prediction of monoisotopic peak masses in large biopolymers

**Master's Thesis code repository**

Barbara Pawłowska   
*Bioinformatics and Systems Biology*, University of Warsaw  

Supervisor: Michał Startek, PhD

## About the thesis

This repository contains the computational pipeline developed for the
Master's Thesis *Prediction of monoisotopic peak masses in large biopolymers*.

The goal is to predict the position of the monoisotopic peak in a
protein mass spectrum, corresponding to a molecule composed entirely of
the lightest stable isotopes of its constituent elements.

For small molecules, this peak is directly observable. However, for
proteins above 10–15 kDa it typically falls below the detection limit
of most instruments. For that reason, predicting its position from the observed 
isotopic envelope is an important problem in top-down proteomics.

The method represents each protein's elemental composition as a vector
in a five-dimensional space over {C, H, N, O, S} and constructs a
physically motivated orthogonal basis:

| Vector | Interpretation |
|--------|----------------|
| **V1** | Mass vector |
| **V2** | Variance of the isotopic envelope vector |
| **V3** | Mass defect contribution vector |
| **V4** | Null-space direction vector |
| **V5** | Null-space direction vector |

The null-space directions are determined analytically so that the
reconstructed formula lies on the Averagine ray.

Projection onto this basis reconstructs a formula preserving mass,
variance, and mass defect, allowing prediction of the corresponding
monoisotopic peak position.

The method was validated on 6000 proteins from UniProt Swiss-Prot and
benchmarked against the Envemind algorithm. All spectra are generated
with IsoSpecPy, and the Wasserstein distance is used throughout to
compare isotopic distributions.

## Repository structure

```text
.
├── scripts/
│   ├── run_all.py
│   ├── step00_shared_utils.py
│   ├── ...
│   └── step16_visualisation_5d.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

`data/`, `figures/`, and `results/` are not part of the repository.  
`figures/` and `results/` are created automatically when the pipeline
runs.   
`data/` must be created manually (see [Data](#data) below).

All `stepNN_*.py` scripts and `run_all.py` are located in `scripts/`.

Outputs are written to `figures/` and `results/` at the project root. 
This holds whether `run_all.py` is run from the project root or
an individual `stepNN_*.py` script is run directly from inside
`scripts/`. The output location does not depend on the directory from
which Python is launched.

## Installation

Python 3.10 or higher is required.

```bash
pip install -r requirements.txt
```

The main dependencies are IsoSpecPy, NumPy, matplotlib 3.9.2, pandas, and
`envemind` (`step15` only).

`envemind` is installed directly from its
[GitHub repository](https://github.com/PiotrRadzinski/envemind).

It is included in `requirements.txt` as:

```text
git+https://github.com/PiotrRadzinski/envemind.git
```

Therefore, a working `git` installation is required before installing
the dependencies. 

If `git` is not available, it can be installed from [git-scm.com/downloads](https://git-scm.com/downloads).

## Data

The UniProt Swiss-Prot FASTA file required by `step13` is not
included in the repository due to its size. 

Create a `data/` directory at the project root, download the file from
[UniProt](https://www.uniprot.org/downloads#uniprotkb), and place it
at `data/uniprot_sprot.fasta.gz`.  

This is also the default `--fasta` path used by `run_all.py`, 
so no extra argument is needed once the file is in place.

## How to run

Run the complete pipeline:

```bash
python scripts/run_all.py
```

Run only specific steps:

```bash
python scripts/run_all.py --steps 07 10 11
```

Run only the full protein-scale analysis:

```bash
python scripts/run_all.py --steps 13 14 15
```

The FASTA file is automatically read from
`data/uniprot_sprot.fasta.gz`.

Run `step13` on custom FASTA file (e.g. data/my_proteins.fasta.gz):

```bash
python scripts/run_all.py --steps 13 --fasta data/my_proteins.fasta.gz
```

Run a quick test on shorter proteins:

```bash
python scripts/run_all.py --steps 13 14 --max-length 300
```

Run a larger protein-scale analysis with no protein length limit and
and a bigger protein count:

```bash
python scripts/run_all.py --steps 13 14 15 --max-length 0 --max-proteins 20000
```

Any individual `stepNN_*.py` script can also be run directly. For
example:

```bash
python scripts/step03_averagine_fit.py
```

The `--max-proteins` argument controls the maximum number of proteins
analysed and defaults to 6000.

The `--max-length` argument controls the maximum protein length and
defaults to 500 amino acids.

The available options can be displayed with:

```bash
python scripts/run_all.py -h
```

The 6000-protein / 500-aa configuration corresponds to the analysis
reported in the thesis and runs in a reasonable time on a standard
laptop. A full unrestricted Swiss-Prot analysis
(`--max-length 0` with a larger `--max-proteins` value) is
more computationally demanding.

## Pipeline steps

| Step | Description |
|------|-------------|
| 00 | Shared utilities (e.g. formula parsing, spectrum variance, peak aggregation) |
| 01 | Averagine scaling model |
| 02 | IsoSpec & Wasserstein distance check (bovine insulin vs. mock spectrum) and visualization |
| 03 | Averagine fit to insulin via ±5 H grid search using Wasserstein distance |
| 04 | Per-element fractional mass defect (physical motivation for V3) |
| 05 | Baryon composition analysis, insulin vs. its Averagine approximation |
| 06 | Monte Carlo search for the maximum defect direction orthogonal to Averagine |
| 07 | Early 2D model (Averagine/mass + variance gradient) |
| 08 | Core module, constructs V1, V2, and V3 via Gram–Schmidt |
| 09 | Basis diagnostics, V1+V2 vs. V1+V3 submodels and unit-step sensitivity |
| 10 | Unit-step sensitivity at 1x and 10x insulin scale |
| 11 | Completion of the basis with V4 and V5 obtained from the SVD null space |
| 12 | Full 5D reconstruction with the closed-form α & β solution |
| 13 | UniProt FASTA parsing and molecular formula generation |
| 14 | Full 5D pipeline over all proteins with monoisotopic mass distance analysis |
| 15 | Benchmark against Envemind |
| 16 | Standalone conceptual 5D basis diagram |

Each script contains a detailed docstring describing its purpose and
implementation.

`step08` serves as the core module and is imported by all subsequent
steps from `step09` onward.

## Output files

All figures (`.png`) are saved to `figures/`, while all generated data
files (`.csv`, `.txt`) are saved to `results/`.

The main outputs are:

- `step13_protein_formulas.txt`: molecular formulas for up to 6000 proteins
- `step14_monoisotopic_distances.txt`: per-protein monoisotopic mass distances
- `step14_distance_summary.csv`: aggregate reconstruction statistics
- `step15_comparison_results.csv`: per-protein comparison with Envemind
- `step15_comparison_summary.csv`: benchmark summary statistics

The intermediate steps also generate diagnostic figures and data files
corresponding to the analyses described above.

## License

MIT License - see [LICENSE](LICENSE).
