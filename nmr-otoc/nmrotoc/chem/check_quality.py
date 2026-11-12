# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

from pathlib import Path
import logging
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Structure import Structure
from pdbfixer import PDBFixer

logger = logging.getLogger(__name__)
MAX_MISSING_RESIDUES = 10
MAX_CLASHES = 50


def get_missing_residues(pdb_file_path: Path):
    fixer = PDBFixer(filename=str(pdb_file_path))
    fixer.findMissingResidues()
    return fixer.missingResidues


# Note: this function was largely written by gemini and it's not the most rigorous way to do this.
# A better metric would be distance bewteen vDW radii.
def get_clashes(structure: Structure, cutoff: float = 1.5) -> list[tuple[Atom, Atom]]:
    atoms = list(structure.get_atoms())
    ns = PDB.NeighborSearch(atoms)
    
    clashes = ns.search_all(cutoff, level='A')
    
    real_clashes = []
    for atom1, atom2 in clashes:
        res1 = atom1.get_parent()
        res2 = atom2.get_parent()
        
        # Skip if same residue
        if res1 == res2:
            continue
            
        # Skip if same chain and adjacent residues (likely peptide bond connection)
        if res1.get_parent() == res2.get_parent(): # same chain
            try:
                # We simply check residue ID difference. 
                # Note: This might miss non-standard numbering, but works for most cases.
                id1 = res1.id[1]
                id2 = res2.id[1]
                if abs(id1 - id2) <= 1:
                    continue
            except:
                pass # If IDs aren't integers, we count it as a clash to be safe
                
        real_clashes.append((atom1, atom2))
        
    return real_clashes


def check_quality(pdb_file_path: Path) -> None:
    """Raise an exception if there are too many clashes or too many missing residues."""
    suffix = pdb_file_path.suffix
    parser = PDBParser(QUIET=False) if suffix == ".pdb" else MMCIFParser(QUIET=False)
    structure = parser.get_structure("full_model", pdb_file_path)

    missing_residues = get_missing_residues(pdb_file_path)
    clashes = get_clashes(structure)
    n_missing_residues = len(missing_residues)
    n_clashes = len(clashes)
    
    logger.info(f"Missing residues: {n_missing_residues}")
    logger.info(f"Missing residues: {n_clashes}")
    if any([n_missing_residues > MAX_MISSING_RESIDUES, n_clashes > MAX_CLASHES]):
        raise RuntimeError("Low structure quality")


if __name__ == "__main__":
    name = "6XUU"
    data_dir = Path.cwd() / name
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    check_quality(data_dir / f"{name}.cif")
