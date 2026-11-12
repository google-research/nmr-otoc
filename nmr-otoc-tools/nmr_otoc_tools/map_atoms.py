# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""CLI tool to map spin indices back to physical atoms in PDB/CIF files."""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, NamedTuple, cast, get_args

import numpy as np
from Bio.PDB.Atom import Atom

from nmr_otoc_tools.cli import add_verbose_arg, configure_logging
from nmr_otoc_tools.generate_instances import (
    CouplingMode,
    MASOrientation,
    extract_spin_coordinates,
    mean_coupling_matrix,
    parse_structure,
    reference_hydrogen_index,
    select_environment_hydrogen_indices,
    spin_order,
)
from nmr_otoc_tools.mas import MASRecouplingModel
from nmr_otoc_tools.orientations import euler_zyz_from_rotation

logger = logging.getLogger(__name__)
EXPECTED_INDICES_LEN = 2


class InstanceMetadata(NamedTuple):
    """The entries of an instance's ``metadata.json`` needed to rebuild its spin system.

    Attributes:
        c_indices: Indices of the two system spins among the carbons (or fluorines).
        ball_ratio: Ratio defining the radius of the environment-hydrogen selection.
        fluorine: Whether the system spins are fluorine.
        coupling_mode: The coupling model ("dipolar", "mas-simple" or "mas").
        use_unrotated_field: Whether the fields (or MAS orientations) are given in the
            structure frame instead of the molecular reference frame.
        mas: The ``"mas"`` block (MAS recoupling model and grid), if present.
    """

    c_indices: tuple[int, int]
    ball_ratio: float
    fluorine: bool
    coupling_mode: str
    use_unrotated_field: bool
    mas: dict[str, Any] | None


def parse_args() -> argparse.Namespace:
    """Parses command line arguments and configures logging.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Map spin indices from simulation Hamiltonians back to PDB/CIF atoms."
    )
    parser.add_argument("indices_file", type=str, help="Path to ordered_indices{x}.npy file.")
    parser.add_argument("pdb_file", type=str, help="Path to the source PDB/CIF structure file.")
    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        default=None,
        help="Path to metadata.json (defaults to same directory as indices_file).",
    )
    parser.add_argument(
        "-f",
        "--field-file",
        type=str,
        default=None,
        help=(
            "Path to field_direction{x}.npy, read only when the coupling matrix is recomputed "
            "for a static coupling model (defaults to same directory as indices_file)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file (defaults to stdout).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Slice the mapping output to the specified maximum number of spins.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the reconstructed Hamiltonian against the saved dipolar matrix.",
    )
    add_verbose_arg(parser)
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)
    return args


def load_metadata(metadata_path: Path) -> InstanceMetadata:
    """Loads and validates metadata.json.

    Args:
        metadata_path: Path to the metadata file.

    Returns:
        The :class:`InstanceMetadata`. Entries missing from older files default to
        ``ball_ratio=2``, ``fluorine=False``, ``coupling="dipolar"`` and
        ``use_unrotated_field=False``.
    """
    if not metadata_path.exists():
        logger.error("Metadata file not found: %s", metadata_path)
        sys.exit(1)

    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to parse metadata")
        sys.exit(1)

    indices_raw = metadata.get("indices")
    if indices_raw is None or len(indices_raw) != EXPECTED_INDICES_LEN:
        logger.error("Invalid or missing 'indices' in metadata.")
        sys.exit(1)

    try:
        c_ind1, c_ind2 = int(indices_raw[0]), int(indices_raw[1])
    except (ValueError, TypeError):
        logger.exception("Indices in metadata must be integers")
        sys.exit(1)

    return InstanceMetadata(
        c_indices=(c_ind1, c_ind2),
        ball_ratio=metadata.get("config", {}).get("ball_ratio", 2),
        fluorine=bool(metadata.get("fluorine", False)),
        coupling_mode=str(metadata.get("coupling", "dipolar")),
        use_unrotated_field=bool(metadata.get("use_unrotated_field", False)),
        mas=metadata.get("mas"),
    )


def _atom_detail(global_index: int, atom: Atom) -> dict[str, Any]:
    """Returns the JSON description of an atom of model 0."""
    residue = atom.get_parent()
    if residue is None:
        logger.error("Atom has no parent residue.")
        sys.exit(1)
    chain = residue.get_parent()
    if chain is None:
        logger.error("Residue has no parent chain.")
        sys.exit(1)
    return {
        "chain_id": chain.get_id(),
        "residue_name": residue.get_resname().strip(),
        "residue_number": residue.get_id()[1],
        "atom_name": atom.get_name().strip(),
        "element": atom.element,
        "serial_number": atom.serial_number,
        "global_index": global_index,
    }


def extract_atoms_and_coords(
    pdb_path: Path, *, fluorine: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    """Parses PDB/CIF structure and extracts atom details and coordinates.

    Args:
        pdb_path: Path to PDB/CIF file.
        fluorine: Whether to extract fluorines instead of carbons.

    Returns:
        A tuple containing:
            - spin_details (list[dict]): Details of the model-0 carbons (or fluorines).
            - hydrogen_details (list[dict]): Details of the model-0 hydrogens.
            - all_spin_pos_list (list[np.ndarray]): Carbon (or fluorine) coordinates per model.
            - all_h_pos_list (list[np.ndarray]): Hydrogen coordinates per model.
    """
    try:
        structure = parse_structure(pdb_path)
    except (ValueError, RuntimeError):
        logger.exception("Failed to parse PDB structure")
        sys.exit(1)

    if not structure:
        logger.error("Structure is empty.")
        sys.exit(1)

    spin_element = "F" if fluorine else "C"
    spin_details: list[dict[str, Any]] = []
    hydrogen_details: list[dict[str, Any]] = []
    try:
        for i, atom in enumerate(structure[0].get_atoms()):
            element = atom.element.upper()
            if element == spin_element:
                spin_details.append(_atom_detail(i, atom))
            elif element == "H":
                hydrogen_details.append(_atom_detail(i, atom))
        coords = extract_spin_coordinates(structure)
    except Exception:
        logger.exception("Failed to extract details from structure")
        sys.exit(1)

    if not spin_details:
        logger.error("No carbon/fluorine atoms found in PDB structure.")
        sys.exit(1)

    if not hydrogen_details:
        logger.error("No hydrogen atoms found in PDB structure.")
        sys.exit(1)

    return spin_details, hydrogen_details, coords[spin_element], coords["H"]


def _load_array(path: Path, description: str) -> np.ndarray:
    """Loads a .npy file, exiting with an error if it is missing or unreadable.

    Args:
        path: The file to load.
        description: Capitalized description of the file for error messages.

    Returns:
        The loaded array.
    """
    if not path.exists():
        logger.error("%s file not found: %s", description, path)
        sys.exit(1)
    try:
        array = np.load(path)
    except (OSError, ValueError):
        logger.exception("Failed to load %s", description.lower())
        sys.exit(1)
    return array


def load_original_indices(indices_dir: Path, file_id: str, num_indices: int) -> np.ndarray | None:
    """Loads ``original_indices{file_id}.npy``, written when generation pruned the instance.

    Args:
        indices_dir: Directory of the instance files.
        file_id: Instance number.
        num_indices: Number of saved (pruned) indices the file must match.

    Returns:
        The indices of the kept spins in the unpruned system, or None if the file does not exist
        (instances pruned by older versions) or does not match the saved indices.
    """
    path = indices_dir / f"original_indices{file_id}.npy"
    if not path.exists():
        return None
    original_indices = _load_array(path, "Original indices")
    if original_indices.shape != (num_indices,):
        logger.warning(
            "Ignoring %s: shape %s does not match the %d saved indices.",
            path,
            original_indices.shape,
            num_indices,
        )
        return None
    return original_indices


def load_field_direction(field_file: str | None, indices_dir: Path, file_id: str) -> np.ndarray:
    """Loads the field direction of a static (dipolar or mas-simple) instance.

    Args:
        field_file: User-supplied path, or None for ``field_direction{file_id}.npy`` next to
            the indices file.
        indices_dir: Directory of the instance files.
        file_id: Instance number.

    Returns:
        The field direction, shape ``(3,)``.
    """
    field_path = Path(field_file) if field_file else indices_dir / f"field_direction{file_id}.npy"
    field_ref_frame = _load_array(field_path, "Field direction")
    if field_ref_frame.shape != (3,):
        logger.error("Field direction shape must be (3,), got %s.", field_ref_frame.shape)
        sys.exit(1)
    return field_ref_frame


def load_euler_angles(indices_dir: Path, file_id: str) -> np.ndarray:
    """Loads the ZYZ Euler angles of the powder orientation of a MAS instance.

    Uses ``euler_angles.npy`` (written by ``nmr-generate-instances``) if present, and otherwise
    recovers the angles from the rotation in ``orientation{file_id}.npy``.

    Args:
        indices_dir: Directory of the instance files.
        file_id: Instance number, i.e. the index of the orientation in the powder grid.

    Returns:
        The Euler angles ``(alpha, beta, gamma)``.
    """
    euler_path = indices_dir / "euler_angles.npy"
    if euler_path.exists():
        euler_angles = _load_array(euler_path, "Euler angles")
        index = int(file_id)
        if euler_angles.shape[1:] != (3,) or not 0 <= index < len(euler_angles):
            logger.error(
                "%s (shape %s) has no orientation %d.", euler_path, euler_angles.shape, index
            )
            sys.exit(1)
        return euler_angles[index]
    rotation_path = indices_dir / f"orientation{file_id}.npy"
    if not rotation_path.exists():
        logger.error(
            "MAS orientation not found: neither %s nor %s exists.", euler_path, rotation_path
        )
        sys.exit(1)
    return np.array(euler_zyz_from_rotation(_load_array(rotation_path, "Orientation")))


def load_mas_orientation(
    meta: InstanceMetadata, indices_dir: Path, file_id: str
) -> MASOrientation:
    """Loads the MAS recoupling model and the powder orientation of a MAS instance.

    Args:
        meta: The instance metadata, which must contain the ``"mas"`` block.
        indices_dir: Directory of the instance files.
        file_id: Instance number.

    Returns:
        The model and Euler angles of the orientation.
    """
    if meta.mas is None:
        logger.error("The metadata has no 'mas' block, which --coupling mas instances need.")
        sys.exit(1)
    try:
        model = MASRecouplingModel.from_metadata(meta.mas)
    except (KeyError, TypeError, ValueError):
        logger.exception("Invalid 'mas' block in the metadata")
        sys.exit(1)
    return MASOrientation(model=model, euler_zyz=load_euler_angles(indices_dir, file_id))


def recompute_mean_hamiltonian(
    meta: InstanceMetadata,
    all_spin_pos_list: list[np.ndarray],
    reduced_h_pos_list: list[np.ndarray],
    *,
    field_file: str | None,
    indices_dir: Path,
    file_id: str,
) -> np.ndarray:
    """Recomputes the unpruned conformer-averaged coupling matrix of an instance.

    Uses the same code path as ``nmr-generate-instances``: the field direction (or the MAS
    model and orientation) of the instance, ``use_unrotated_field`` and the coupling model.

    Args:
        meta: The instance metadata.
        all_spin_pos_list: System-species positions for every conformer.
        reduced_h_pos_list: Environment-hydrogen positions for every conformer.
        field_file: User-supplied field direction file, or None for the default.
        indices_dir: Directory of the instance files.
        file_id: Instance number.

    Returns:
        The recomputed coupling matrix of the unpruned spin system.
    """
    if meta.coupling_mode not in get_args(CouplingMode):
        logger.error("Unknown coupling mode %r in the metadata.", meta.coupling_mode)
        sys.exit(1)
    coupling_mode = cast("CouplingMode", meta.coupling_mode)
    field_ref_frame = None
    mas_orientation = None
    if coupling_mode == "mas":
        mas_orientation = load_mas_orientation(meta, indices_dir, file_id)
    else:
        field_ref_frame = load_field_direction(field_file, indices_dir, file_id)

    c_ind1, c_ind2 = meta.c_indices
    href_ind = reference_hydrogen_index(reduced_h_pos_list, all_spin_pos_list, c_ind1)
    try:
        mean_hamiltonian = mean_coupling_matrix(
            all_spin_pos_list,
            reduced_h_pos_list,
            c_ind1,
            c_ind2,
            href_ind,
            fluorine=meta.fluorine,
            coupling_mode=coupling_mode,
            use_unrotated_field=meta.use_unrotated_field,
            field_ref_frame=field_ref_frame,
            mas_orientation=mas_orientation,
        )
    except ValueError:
        logger.exception("Failed to recompute the coupling matrix")
        sys.exit(1)
    return mean_hamiltonian


def verify_dipolar_matrix(
    mean_hamiltonian: np.ndarray,
    mapping: np.ndarray,
    *,
    pruned: bool,
    indices_dir: Path,
    file_id: str,
) -> None:
    """Verifies the recomputed coupling matrix against the saved dipolar matrix.

    Args:
        mean_hamiltonian: Recomputed coupling matrix of the unpruned system.
        mapping: Indices (in the unpruned system) of the saved spins.
        pruned: Whether the saved instance was pruned, i.e. holds only the ``mapping`` block.
        indices_dir: Directory containing saved dipolar matrix.
        file_id: Identifier of the indices file.
    """
    saved_dipolar_matrix = _load_array(
        indices_dir / f"dipolar_matrix{file_id}.npy", "Dipolar matrix"
    )
    reconstructed_matrix = (
        mean_hamiltonian[np.ix_(mapping, mapping)] if pruned else mean_hamiltonian
    )

    if reconstructed_matrix.shape != saved_dipolar_matrix.shape or not np.allclose(
        reconstructed_matrix, saved_dipolar_matrix, atol=1e-5
    ):
        logger.error(
            "Verification failed: Reconstructed matrix does not match saved dipolar matrix."
        )
        sys.exit(1)
    logger.info("Verification passed successfully.")


def write_json_output(output_data: list[dict[str, Any]], output_path: str | None) -> None:
    """Writes JSON output data to stdout or to a specified file.

    Args:
        output_data: List of atom details.
        output_path: Optional output file path.
    """
    json_str = json.dumps(output_data, indent=2)

    if output_path:
        try:
            Path(output_path).write_text(json_str, encoding="utf-8")
        except OSError:
            logger.exception("Failed to write output to file")
            sys.exit(1)
    else:
        sys.stdout.write(json_str + "\n")


def instance_file_id(indices_path: Path) -> str:
    """Returns the instance number of an ``ordered_indices{i}.npy`` file.

    Args:
        indices_path: Path of the indices file.

    Returns:
        The digits after ``ordered_indices`` (or, for renamed files, at the end of the file
        stem); "0" if there are none.
    """
    match = re.search(r"ordered_indices(\d+)", indices_path.name) or re.search(
        r"(\d+)$", indices_path.stem
    )
    if match is None:
        logger.warning("Could not infer the instance id from %s; assuming 0.", indices_path.name)
        return "0"
    return match.group(1)


def validate_input_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    """Validates input paths and returns verified paths and file ID."""
    indices_path = Path(args.indices_file)
    if not indices_path.exists():
        logger.error("Indices file not found: %s", indices_path)
        sys.exit(1)

    file_id = instance_file_id(indices_path)

    pdb_path = Path(args.pdb_file)
    if not pdb_path.exists():
        logger.error("PDB/CIF file not found: %s", pdb_path)
        sys.exit(1)

    return indices_path, pdb_path, file_id


def main() -> None:
    """Main CLI entry point for mapping spin indices back to physical atoms."""
    args = parse_args()
    if args.max_size is not None and args.max_size <= 0:
        logger.error("--max-size must be a positive integer.")
        sys.exit(1)

    indices_path, pdb_path, file_id = validate_input_paths(args)
    indices_dir = indices_path.parent

    # 1. Load metadata
    metadata_path = Path(args.metadata) if args.metadata else indices_dir / "metadata.json"
    meta = load_metadata(metadata_path)
    c_ind1, c_ind2 = meta.c_indices

    # 2. Parse PDB/CIF structure and extract details
    spin_details, hydrogen_details, all_spin_pos_list, all_h_pos_list = (
        extract_atoms_and_coords(pdb_path, fluorine=meta.fluorine)
    )
    num_spin_atoms = len(spin_details)
    if not (0 <= c_ind1 < num_spin_atoms and 0 <= c_ind2 < num_spin_atoms):
        logger.error(
            "Indices [%d, %d] in metadata are out of bounds for the number of "
            "parsed carbon/fluorine atoms (%d).",
            c_ind1,
            c_ind2,
            num_spin_atoms,
        )
        sys.exit(1)

    # 3. Rebuild the unpruned spin system: the system pair, then the environment hydrogens
    reduced_h_inds = select_environment_hydrogen_indices(
        all_h_pos_list, all_spin_pos_list, c_ind1, c_ind2, meta.ball_ratio
    )
    unpruned_details = [spin_details[c_ind1], spin_details[c_ind2]] + [
        hydrogen_details[h_idx] for h_idx in reduced_h_inds
    ]
    num_unpruned = len(unpruned_details)

    # 4. Find the unpruned spins that the saved indices refer to
    saved_indices = _load_array(indices_path, "Indices")
    if saved_indices.ndim != 1:
        logger.error("Indices array must be 1D, got shape %s.", saved_indices.shape)
        sys.exit(1)
    num_indices = len(saved_indices)
    pruned = num_indices < num_unpruned
    original_indices = load_original_indices(indices_dir, file_id, num_indices) if pruned else None
    mapping = saved_indices if original_indices is None else original_indices
    # Instances pruned by older versions saved 0..k-1 without original_indices{i}.npy.
    legacy_pruned = (
        original_indices is None
        and pruned
        and np.array_equal(saved_indices, np.arange(num_indices))
    )

    if legacy_pruned or args.verify:
        mean_hamiltonian = recompute_mean_hamiltonian(
            meta,
            all_spin_pos_list,
            [x[reduced_h_inds, :] for x in all_h_pos_list],
            field_file=args.field_file,
            indices_dir=indices_dir,
            file_id=file_id,
        )
        if legacy_pruned:
            logger.debug("No original_indices%s.npy; recomputing the pruned spin order.", file_id)
            mapping = spin_order(mean_hamiltonian)[0][:num_indices]
        if args.verify:
            verify_dipolar_matrix(
                mean_hamiltonian, mapping, pruned=pruned, indices_dir=indices_dir, file_id=file_id
            )

    # 5. Apply max-size if specified
    if args.max_size is not None:
        mapping = mapping[: args.max_size]

    # 6. Output details to JSON
    output_data = []
    for idx in mapping:
        if idx < 0 or idx >= num_unpruned:
            logger.error(
                "Index %d is out of bounds for the reconstructed spin system size %d.",
                idx,
                num_unpruned,
            )
            sys.exit(1)
        output_data.append(unpruned_details[idx])

    write_json_output(output_data, args.output)


if __name__ == "__main__":
    main()
