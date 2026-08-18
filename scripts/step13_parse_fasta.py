"""
step13_parse_fasta.py

Parses a UniProt FASTA file and computes the molecular formula for each
protein from its amino-acid sequence.

Formula calculation:
    Each protein’s molecular formula is built residue by residue, with 
    elemental compositions summed and one H20 molecule added for the two free 
    ends of the chain. Two hydrogen atoms  are additionally removed for every 
    assumed pair of cysteines, since these can form disulfide bridges.

Outputs:
    results/step13_protein_formulas.txt   protein formula database (TSV)
"""

import gzip
from IsoSpecPy import Iso
from step00_shared_utils import results_dir, project_root

aa_composition = {
    'A': {'C': 3,  'H': 5,  'N': 1, 'O': 1, 'S': 0},
    'R': {'C': 6,  'H': 12, 'N': 4, 'O': 1, 'S': 0},
    'N': {'C': 4,  'H': 6,  'N': 2, 'O': 2, 'S': 0},
    'D': {'C': 4,  'H': 5,  'N': 1, 'O': 3, 'S': 0},
    'C': {'C': 3,  'H': 5,  'N': 1, 'O': 1, 'S': 1},
    'E': {'C': 5,  'H': 7,  'N': 1, 'O': 3, 'S': 0},
    'Q': {'C': 5,  'H': 8,  'N': 2, 'O': 2, 'S': 0},
    'G': {'C': 2,  'H': 3,  'N': 1, 'O': 1, 'S': 0},
    'H': {'C': 6,  'H': 7,  'N': 3, 'O': 1, 'S': 0},
    'I': {'C': 6,  'H': 11, 'N': 1, 'O': 1, 'S': 0},
    'L': {'C': 6,  'H': 11, 'N': 1, 'O': 1, 'S': 0},
    'K': {'C': 6,  'H': 12, 'N': 2, 'O': 1, 'S': 0},
    'M': {'C': 5,  'H': 9,  'N': 1, 'O': 1, 'S': 1},
    'F': {'C': 9,  'H': 9,  'N': 1, 'O': 1, 'S': 0},
    'P': {'C': 5,  'H': 7,  'N': 1, 'O': 1, 'S': 0},
    'S': {'C': 3,  'H': 5,  'N': 1, 'O': 2, 'S': 0},
    'T': {'C': 4,  'H': 7,  'N': 1, 'O': 2, 'S': 0},
    'W': {'C': 11, 'H': 10, 'N': 2, 'O': 1, 'S': 0},
    'Y': {'C': 9,  'H': 9,  'N': 1, 'O': 2, 'S': 0},
    'V': {'C': 5,  'H': 9,  'N': 1, 'O': 1, 'S': 0},
}
water = {'C': 0, 'H': 2, 'N': 0, 'O': 1, 'S': 0}


def calculate_formula(sequence):
    """
    Compute the molecular formula for an amino-acid sequence.

    Parameters:
        sequence: one-letter amino-acid codes (uppercase)

    Returns:
        str, e.g. "C254H377N65O75S6"
    """
    total = {el: 0 for el in ['C', 'H', 'N', 'O', 'S']}
    for aa in sequence:
        if aa in aa_composition:
            for el in total:
                total[el] += aa_composition[aa][el]
    for el in total:
        total[el] += water[el]
    total['H'] -= 2 * (sequence.count('C') // 2)
    return "".join(f"{el}{total[el]}"
                   for el in ['C', 'H', 'N', 'O', 'S'] if total[el] > 0)


def parse_fasta(filepath, max_proteins = 6000):
    """
    Parse a UniProt FASTA file (plain or gzip-compressed).

    UniProt header format: >sp|UNIPROTID|ENTRYNAME Description OS=organism ...

    Parameters:
        filepath:     path to .fasta or .fasta.gz
        max_proteins: stop after this many entries

    Returns:
        list of {uniprot_id, name, formula, length}
    """
    proteins = []
    current_id = current_name = None
    current_seq = []
    open_fn = gzip.open if filepath.endswith('.gz') else open

    with open_fn(filepath, 'rt') as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    seq = ''.join(current_seq)
                    proteins.append({'uniprot_id': current_id, 'name': current_name,
                                     'formula': calculate_formula(seq), 'length': len(seq)})
                    if len(proteins) >= max_proteins:
                        break
                parts = line[1:].split('|')
                if len(parts) >= 3:
                    current_id   = parts[1]
                    current_name = parts[2].split('OS=', 1)[0].strip()
                else:
                    current_id   = parts[0]
                    current_name = 'Unknown'
                current_seq = []
            else:
                current_seq.append(line)

        if current_id and current_seq and len(proteins) < max_proteins:
            seq = ''.join(current_seq)
            proteins.append({'uniprot_id': current_id, 'name': current_name,
                             'formula': calculate_formula(seq), 'length': len(seq)})
    return proteins


def save_proteins(proteins, output_path):
    """Write the protein list to a tab-separated file."""
    with open(output_path, 'w') as fh:
        fh.write("UniProtID\tName\tFormula\tLength\n")
        for i, p in enumerate(proteins):
            safe = p['name'].replace('\t', ' ').replace('\n', ' ')
            fh.write(f"{p['uniprot_id']}\t{safe}\t{p['formula']}\t{p['length']}\n")
            if (i + 1) % 1000 == 0:
                print(f"  Written {i+1} proteins…")
    print(f"  Saved {len(proteins)} proteins: {output_path}")


def load_proteins(filepath, max_length=None):
    """
    Load filed produced by save_proteins() and compute average masses.

    Parameters:
        max_length: skip proteins longer than this (amino acids)
    """

    proteins = []
    with open(filepath) as fh:
        next(fh)
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            length = int(parts[3])
            if max_length is not None and length > max_length:
                continue
            formula = parts[2]
            mass    = float(Iso(formula).getTheoreticalAverageMass())
            proteins.append({'uniprot_id': parts[0], 'formula': formula,
                              'length': length, 'mass': mass})
    return proteins


if __name__ == "__main__":
    FASTA  = str(project_root / "data" / "uniprot_sprot.fasta.gz")
    output = results_dir / "step13_protein_formulas.txt"
    print(f"Parsing {FASTA} …")
    proteins = parse_fasta(FASTA, max_proteins=6000)
    save_proteins(proteins, output)