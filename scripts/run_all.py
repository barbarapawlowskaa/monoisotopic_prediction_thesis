"""
run_all.py

Master entry point. Run any subset of the pipeline steps in order.

Step map:
    00  Shared utilities
    01  Averagine model
    02  IsoSpec intro and Wasserstein demo
    03  Averagine fit by Wasserstein optimisation
    04  Mass defect per element (physical motivation for V3)
    05  Baryon analysis (insulin vs Averagine)
    06  Monte Carlo baryon defect search
    07  2D variance experiments - Averagine direction and mass vector as V1
    08  Basis vectors V1/V2/V3  
    09  Basis diagnostics - sub-models, unit-step sensitivity
    10  Scaling test (extension of 09): unit-step on 1x/10x insulin, delta_H
    11  Null-space vectors V4, V5
    12  Full 5D reconstruction with analytical alpha, beta
    13  Parse UniProt FASTA 
    14  Full analysis on all proteins
    15  Benchmark against Envemind 
    16  Conceptual 5D vector space visualisation (standalone)

Output directories (created automatically):
    figures/   all PNG plots
    results/   all CSV and TXT data files
"""

import argparse
import runpy
import sys

from step00_shared_utils import project_root, figures_dir, results_dir

DEFAULT_FASTA = str(project_root / "data" / "uniprot_sprot.fasta.gz")

USAGE_EXAMPLES = """\
examples:
  python scripts/run_all.py                                    all steps, default 500 aa / 6000 protein limits (test run)
  python scripts/run_all.py --steps 08 11 12                   run only specific steps
  python scripts/run_all.py --steps 13 14 --max-length 0       FULL length run: no aa limit, still capped at
                                                               --max-proteins (default 6000)
  python scripts/run_all.py --steps 13 --fasta data/my_proteins.fasta.gz --max-proteins 2000
                                                               step 13 with a custom FASTA file and protein count
  python scripts/run_all.py --steps 13 14 15 --max-length 0 --max-proteins 20000
                                                               larger run including the Envemind benchmark and bigger
                                                               protein count 
                                                        

notes:
  * --max-length affects step 14 (and step 15, which reads its output).
    Default 500 aa. Use 0 for no length limit.
  * --max-proteins affects step 13 only. Entries read from the FASTA file.
    Default is 6000, this thesis's test size. Runtime/memory in step 14 scale
    with it.
  * --fasta affects step 13 only. Default is 'data/uniprot_sprot.fasta.gz'
  * Hardcoded values in each step's own "__main__" block (e.g. max_length=500
    in step14_full_analysis.py) are NOT used via run_all.py, only the flags
    above are.
"""

step_modules = {
    "00": "step00_shared_utils",
    "01": "step01_averagine",
    "02": "step02_isospec_intro",
    "03": "step03_averagine_fit",
    "04": "step04_mass_defect_intro",
    "05": "step05_baryon_analysis",
    "06": "step06_baryon_montecarlo",
    "07": "step07_variance_only",
    "08": "step08_basis_vectors",
    "09": "step09_basis_diagnostics",
    "10": "step10_scaling_test",
    "11": "step11_null_space",
    "12": "step12_projection_analytical",
    "16": "step16_visualisation_5d",
}


def run_step(step, fasta, max_length, max_proteins):
    print()
    print(f"  step {step}")
    print()

    if step in step_modules:
        runpy.run_module(step_modules[step], run_name="__main__", alter_sys=True)

    elif step == "13":
        from step13_parse_fasta import parse_fasta, save_proteins
        from step00_shared_utils import results_dir
        print(f"Parsing {fasta} … (limit: {max_proteins} proteins)")
        proteins = parse_fasta(fasta, max_proteins=max_proteins)
        save_proteins(proteins, results_dir / "step13_protein_formulas.txt")

    elif step == "14":
        from step14_full_analysis import run_analysis, plot_histogram
        run_analysis(max_length=max_length if max_length > 0 else None)
        plot_histogram()

    elif step == "15":
        from step15_envemind_comparison import run_comparison
        run_comparison()

    else:
        print(f"Unknown step '{step}'.", file=sys.stderr)


def main():
    all_steps = [f"{i:02d}" for i in range(17)]

    parser = argparse.ArgumentParser(
        description="Monoisotopic mass prediction pipeline",
        epilog=USAGE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--steps", nargs="+", default=all_steps,
                         help="Steps to run, e.g. --steps 08 11 12")
    parser.add_argument("--fasta", default=DEFAULT_FASTA,
                         help="UniProt FASTA path for step 13 "
                              "(default: data/uniprot_sprot.fasta.gz under the project root)")
    parser.add_argument("--max-length", type=int, default=500,
                         help="Skip proteins longer than this aa in step 14 "
                              "(default: 500; use 0 for no limit / all proteins)")
    parser.add_argument("--max-proteins", type=int, default=6000,
                         help="How many proteins to parse from the FASTA file in "
                              "step 13 (default: 6000). Larger values take longer "
                              "and use more memory in step 14 -- how far you can "
                              "push this depends on your machine.")
    args  = parser.parse_args()
    steps = sorted({s.zfill(2) for s in args.steps})

    for step in steps:
        run_step(step, args.fasta, args.max_length, args.max_proteins)

    print()
    print("  All steps complete.")
    print(f"  Figures: {figures_dir.resolve()}")
    print(f"  Results: {results_dir.resolve()}")
    print()


if __name__ == "__main__":
    main()