# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""CLI tool to compute distance statistics between two atoms in a PDB/CIF file."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from nmr_otoc_tools.cli import add_verbose_arg, configure_logging
from nmr_otoc_tools.generate_instances import parse_structure, resolve_atom_index

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute distance statistics between two atoms in a PDB/CIF file "
            "over all frames."
        )
    )
    parser.add_argument("pdb_file", type=str, help="Path to PDB/CIF file")
    parser.add_argument(
        "atom1",
        type=str,
        help="First atom: integer index or chain:resname:resnum:atomlabel",
    )
    parser.add_argument(
        "atom2",
        type=str,
        help="Second atom: integer index or chain:resname:resnum:atomlabel",
    )
    add_verbose_arg(parser)
    args = parser.parse_args()
    configure_logging(verbose=args.verbose)
    return args


def compute_distance_stats(
    pdb_path: Path,
    atom1_spec: str,
    atom2_spec: str,
) -> dict[str, float | int]:
    """Computes distance statistics between two atoms over all frames.

    Args:
        pdb_path: Path to the PDB/CIF file.
        atom1_spec: Specifier for the first atom.
        atom2_spec: Specifier for the second atom.

    Returns:
        dict: A dictionary containing min, max, mean, std, resolved indices, and n_frames.
    """
    # Parse once; the specifiers are resolved against the same structure.
    structure = parse_structure(pdb_path)
    idx1 = resolve_atom_index(structure, atom1_spec)
    idx2 = resolve_atom_index(structure, atom2_spec)

    model_0 = structure[0]
    atoms_0 = list(model_0.get_atoms())
    num_atoms = len(atoms_0)
    if idx1 < 0 or idx1 >= num_atoms:
        msg = f"Atom1 index {idx1} out of bounds (0-{num_atoms-1})"
        raise ValueError(msg)
    if idx2 < 0 or idx2 >= num_atoms:
        msg = f"Atom2 index {idx2} out of bounds (0-{num_atoms-1})"
        raise ValueError(msg)

    atom1_0 = atoms_0[idx1]
    res1 = atom1_0.get_parent()
    chain1 = res1.get_parent()
    path1 = (chain1.get_id(), res1.get_id(), atom1_0.get_name())

    atom2_0 = atoms_0[idx2]
    res2 = atom2_0.get_parent()
    chain2 = res2.get_parent()
    path2 = (chain2.get_id(), res2.get_id(), atom2_0.get_name())

    distances_list = []
    c1, r1, a1 = path1
    c2, r2, a2 = path2
    for model in structure:
        try:
            coord1 = model[c1][r1][a1].coord
            coord2 = model[c2][r2][a2].coord
        except KeyError as e:
            msg = f"Model {model.id} is missing atom {e.args[0]}"
            raise RuntimeError(msg) from e
        dist = np.linalg.norm(coord1 - coord2)
        distances_list.append(dist)

    distances = np.array(distances_list)
    return {
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "idx1": idx1,
        "idx2": idx2,
        "n_frames": len(distances),
    }


def main() -> None:
    """Main entry point."""
    args = parse_args()
    pdb_path = Path(args.pdb_file)
    if not pdb_path.exists():
        logger.error("PDB/CIF file not found: %s", pdb_path)
        sys.exit(1)

    try:
        stats = compute_distance_stats(pdb_path, args.atom1, args.atom2)
    except (ValueError, RuntimeError, argparse.ArgumentTypeError):
        logger.exception("Failed to compute distance stats")
        sys.exit(1)

    sys.stdout.write(
        f"Stats for distance between atom {stats['idx1']} and atom {stats['idx2']} "
        f"over {stats['n_frames']} frames:\n"
    )
    sys.stdout.write(f"  Min: {stats['min']:.4f} A\n")
    sys.stdout.write(f"  Max: {stats['max']:.4f} A\n")
    sys.stdout.write(f"  Average: {stats['mean']:.4f} A\n")
    sys.stdout.write(f"  Std Dev: {stats['std']:.4f} A\n")


if __name__ == "__main__":
    main()
