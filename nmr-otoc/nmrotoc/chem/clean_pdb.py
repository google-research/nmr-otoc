# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, Superimposer
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure

from openmm.app import ForceField, Modeller, PDBFile, PDBxFile
from pdbfixer import PDBFixer


def filter_inplace(structure: Structure) -> None:
    """Remove heterogens from biopython structure."""
    for model in structure:
        for chain in model:
            residues_to_remove = []
            for residue in chain:
                hetfield, resseq, icode = residue.get_id()
                if hetfield != " ":  # Not a standard protein residue
                    residues_to_remove.append(residue.get_id())

            for res_id in residues_to_remove:
                chain.detach_child(res_id)


def average_structure(structure: Structure) -> Structure:
    """Calculates the average structure from a PDB file containing multiple models.

    Args:
        structure: The structure with multiple models.

    Returns:
        The new structure object with average coordinates.
    """
    atoms_0 = list(structure[0].get_atoms())
    if len(structure) > 1:
        for i, model in enumerate(list(structure)[1:]):
            atoms_i = list(model.get_atoms())
            super_imposer = Superimposer()
            super_imposer.set_atoms(atoms_0, atoms_i)
            print(f"Model {i + 1} RMS: {super_imposer.rms}")
            super_imposer.apply(model.get_atoms())

        atom_coords = {tuple(atom.get_full_id()[-3:]): [] for atom in atoms_0}
        for model in structure:
            for atom in model.get_atoms():
                atom_id = tuple(atom.get_full_id()[-3:])
                atom_coords[atom_id].append(atom.coord)

        average_atoms = {
            atom_id: np.mean(coords, axis=0) for atom_id, coords in atom_coords.items()
        }
    else:
        print("Single model found")
        average_atoms = {}
        for atom in structure[0].get_atoms():
            atom_id = tuple(atom.get_full_id()[-3:])
            average_atoms[atom_id] = atom.coord

    print("Making avg structure")
    avg_structure = structure[0].copy()
    for atom in avg_structure.get_atoms():
        atom_id = tuple(atom.get_full_id()[-3:])
        atom.set_coord(average_atoms[atom_id])

    return avg_structure


def fix_pdb(pdb_file_path: Path, out_file_path: Path) -> None:
    """Fix up the pdb using pdbfixer."""
    suffix = pdb_file_path.suffix
    fixer = PDBFixer(filename=str(pdb_file_path))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)  # we should have already done this
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    # fixer.addSolvent(fixer.topology.getUnitCellDimensions())  # no need to add solvent
    if suffix == ".cif":
        PDBxFile.writeFile(fixer.topology, fixer.positions, out_file_path.open("w"))
    else:
        PDBFile.writeFile(fixer.topology, fixer.positions, out_file_path.open("w"))


def clean(pdb_file_path: Path, avg_file_path: Path, out_file_path: Path) -> None:
    """The function cleans the PDB file in three steps:

        1. Remove any non-protein residues.
        2. If several models are present:
            2.a. Align all models to the first model in the list using all coordinates.
            2.b. Report the RMSD for each alignment.
            2.c. Compute the average coordinates and return the average structure.
        3. Add missing residues and atoms.

    Args:
        pdb_file_path: Path to the pdb file.
        avg_file_path: Path the pdb output with average coordinates.
        out_file_path: Path the cleaned pdb output.
    """
    suffix = pdb_file_path.suffix
    parser = PDBParser(QUIET=False) if suffix == ".pdb" else MMCIFParser(QUIET=False)
    structure = parser.get_structure("full_model", pdb_file_path)

    # step 1
    filter_inplace(structure)

    # step 2
    avg_structure = average_structure(structure)

    io = PDBIO() if suffix == ".pdb" else MMCIFIO()
    io.set_structure(avg_structure)
    io.save(str(avg_file_path))

    # step 3
    fix_pdb(avg_file_path, out_file_path)


if __name__ == "__main__":
    name = "2ayy"
    data_dir = Path.cwd()
    clean(
        data_dir / f"{name}.cif",
        data_dir / f"{name}_avg.cif",
        data_dir / f"{name}_cleaned.cif",
    )
