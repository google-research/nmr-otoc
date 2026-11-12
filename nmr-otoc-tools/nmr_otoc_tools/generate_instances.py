# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Generate instances of appropriate dipolar Hamiltonians given the input files."""

import argparse
import json
import logging
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Structure import Structure
from nmrotoc.chem.hamiltonians import GYRO_FRACTIONS, CONST_HH_kHz_A
from nmrotoc.chem.manipulation import (
    get_molecule_reference_frame,
    get_separated_pair,
)
from nmrotoc.math.graphs import get_indices_sorted_by_classical_time_approx
from numpy.typing import NDArray

from nmr_otoc_tools.cli import MIN_SPINS, add_verbose_arg, configure_logging, log_duration

from .mas import (
    MAGIC_ANGLE_RAD,
    MASRecouplingModel,
    effective_coupling_matrix,
    spatial_tensor_components,
)
from .orientations import (
    SUPPORTED_LEBEDEV_SIZES,
    OrientationGrid,
    lebedev_grid,
    random_so3_orientations,
    so3_grid,
)

logger = logging.getLogger(__name__)

CouplingMode = Literal["dipolar", "mas-simple", "mas"]

SpinCoordinates = dict[str, list[NDArray]]
"""Coordinates of the "C", "H" and "F" atoms of every model, see extract_spin_coordinates."""

_SPIN_ELEMENTS = ("C", "H", "F")
_ATOM_SPECIFIER_FIELDS = 4


def _nuclei_labels(num_hydrogens: int, *, fluorine: bool) -> list[str]:
    """Returns the spin labels in coupling-matrix order: two system spins, then the hydrogens."""
    system = "F" if fluorine else "C"
    return [system] * 2 + ["H"] * num_hydrogens


def _gyro_fractions(labels: Sequence[str]) -> NDArray:
    """Returns the gyromagnetic-ratio fractions (``GYRO_FRACTIONS``) of the labelled nuclei."""
    return np.array([GYRO_FRACTIONS[label] for label in labels])


def _require_distinct_system_spins(spin_pos: NDArray, c_ind1: int, c_ind2: int) -> None:
    """Raises a ValueError unless the system spins are different atoms at different positions.

    Coincident system spins make the molecular reference frame (and hence every coupling) NaN.

    Args:
        spin_pos: Positions of the system-species atoms.
        c_ind1: Index of the first system spin in ``spin_pos``.
        c_ind2: Index of the second system spin in ``spin_pos``.

    Raises:
        ValueError: If the indices are equal or the two atoms share a position.
    """
    if c_ind1 == c_ind2 or np.array_equal(spin_pos[c_ind1], spin_pos[c_ind2]):
        msg = (
            "The two system spins must be different atoms at different positions; got indices "
            f"{c_ind1} and {c_ind2}."
        )
        raise ValueError(msg)


def _pair_geometry(all_pos: NDArray) -> tuple[NDArray, NDArray]:
    """Returns the pairwise distances and unit vectors of a set of nuclei.

    Args:
        all_pos: Positions of the nuclei, shape ``(N, 3)``.

    Returns:
        A tuple ``(r, unit_vectors)``: the distances ``r_ij`` of shape ``(N, N)`` with the
        diagonal set to 1 (so that self-pairs never divide by zero), and the unit vectors
        ``(pos_i - pos_j) / r_ij`` of shape ``(N, N, 3)``, zero on the diagonal. Both keep the
        precision of ``all_pos``.

    Raises:
        ValueError: If two different nuclei share a position.
    """
    diff = all_pos[:, np.newaxis, :] - all_pos[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(r, 1.0)
    coincident = np.argwhere(r == 0.0)
    if coincident.size:
        i, j = coincident[0]
        msg = f"Nuclei {i} and {j} share the position {all_pos[i]}."
        raise ValueError(msg)
    return r, diff / r[:, :, np.newaxis]


def _angular_factor(cos_theta: NDArray, coupling_mode: CouplingMode) -> NDArray:
    """Returns the orientation dependence of a static coupling model.

    Args:
        cos_theta: Cosines of the angles between the internuclear vectors and the field.
        coupling_mode: "dipolar" for ``(3 cos^2(theta) - 1) / 2`` or the deprecated
            "mas-simple" for ``|sin(2 theta)|``.

    Returns:
        The angular factor, with the shape of ``cos_theta``.

    Raises:
        ValueError: If ``coupling_mode`` is not a static coupling model.
    """
    if coupling_mode == "dipolar":
        return cos_theta**2 * 1.5 - 0.5
    if coupling_mode == "mas-simple":
        warnings.warn(
            "coupling_mode='mas-simple' is deprecated: it keeps only the magnitude "
            "|sin(2*theta)| of the m = +-1 rotor component, dropping the rotor phase, the "
            "relative signs and the Wigner prefactors. Use coupling_mode='mas' for the full "
            "MAS effective Hamiltonian with an SO(3) powder average.",
            DeprecationWarning,
            stacklevel=3,
        )
        # |sin(2 theta)| = 2 |cos(theta)| sqrt(1 - cos^2(theta))
        return np.abs(cos_theta) * np.sqrt(1.0 - np.clip(cos_theta**2, 0.0, 1.0)) * 2.0
    msg = (
        f"Unknown coupling_mode for the static coupling kernel: {coupling_mode}. "
        "Use 'dipolar' or 'mas-simple' here; 'mas' is handled by compute_pair_geometry."
    )
    raise ValueError(msg)


def _compute_coupling_matrix(
    all_pos: NDArray,
    nuclei_labels: list[str],
    field_direction: NDArray,
    coupling_mode: CouplingMode,
) -> NDArray:
    """Computes the N x N static coupling matrix of a set of nuclei.

    Entry ``(i, j)`` is ``CONST_HH_kHz_A * g_i * g_j / r_ij^3 * f(cos(theta_ij))``, where
    ``theta_ij`` is the angle between the internuclear vector and the field and ``f`` is the
    angular factor of ``coupling_mode``.

    Args:
        all_pos: Positions of all nuclei, shape (N, 3).
        nuclei_labels: List of N element labels (e.g. "C", "H", "F").
        field_direction: Magnetic field direction in the frame of ``all_pos``, shape (3,).
        coupling_mode: Angular factor formula:
            "dipolar" -- 3/2 cos^2(theta) - 1/2
            "mas-simple" -- |sin(2*theta)| (deprecated)

    Returns:
        Symmetric coupling matrix of shape (N, N) with zeros on the diagonal, in the precision of
        ``all_pos`` (float32 for coordinates read with Bio.PDB).

    Raises:
        ValueError: If the field is the zero vector, two nuclei share a position, or
            ``coupling_mode`` is not a static coupling model.
    """
    field_norm = np.linalg.norm(field_direction)
    if field_norm == 0.0:
        msg = "field_direction must be a non-zero vector."
        raise ValueError(msg)
    r, unit_vectors = _pair_geometry(all_pos)
    cos_theta = np.einsum("k,ijk->ij", field_direction / field_norm, unit_vectors)
    angular_factor = _angular_factor(cos_theta, coupling_mode)

    # The in-place products round every factor to the precision of the coordinates, exactly as
    # the instances have always been generated, which keeps regenerated instances bit-identical.
    coupling = np.power(r, -3)
    coupling *= angular_factor
    coupling *= CONST_HH_kHz_A
    gyro_arr = _gyro_fractions(nuclei_labels)
    coupling *= gyro_arr[:, np.newaxis]
    coupling *= gyro_arr[np.newaxis, :]
    np.fill_diagonal(coupling, 0.0)
    return coupling


def prune_lebedev_grid(fields: NDArray, weights: NDArray) -> tuple[NDArray, NDArray]:
    """Prunes a Lebedev grid by removing inversion-symmetric pairs.

    For each pair of fields (f1, f2) such that f1 = -f2, only one is kept
    and its weight is doubled.

    Args:
        fields: Array of field vectors of shape (N, 3).
        weights: Array of weights of shape (N,).

    Returns:
        A tuple (pruned_fields, pruned_weights) containing the pruned grid.

    Raises:
        ValueError: If a field has no inversion-symmetric partner.
    """
    pruned_fields = []
    pruned_weights = []
    n = len(fields)
    used = np.zeros(n, dtype=bool)

    for i in range(n):
        if used[i]:
            continue

        # Find the inversion-symmetric partner
        partner_found = False
        for j in range(i + 1, n):
            if not used[j] and np.allclose(fields[i], -fields[j]):
                pruned_fields.append(fields[i])
                pruned_weights.append(weights[i] * 2.0)
                used[i] = True
                used[j] = True
                partner_found = True
                break

        if not partner_found:
            msg = f"Field {fields[i]} has no inversion-symmetric partner."
            raise ValueError(msg)

    return np.array(pruned_fields), np.array(pruned_weights)


def get_dipolar_hamiltonian_hbulk_cc(
    c_pos: NDArray,
    h_pos: NDArray,
    c_ind1: int,
    c_ind2: int,
    href_ind: int,
    field_ref_frame: NDArray,
    *,
    fluorine: bool,
    coupling_mode: CouplingMode = "dipolar",
    use_unrotated_field: bool = False,
) -> NDArray:
    """Calculates the static coupling matrix of a system pair and its environment hydrogens.

    Uses vectorized NumPy operations to compute all pairwise couplings in a single
    batch rather than element-by-element.

    Args:
        c_pos: Positions of the system-species atoms (carbons, or fluorines if ``fluorine``).
        h_pos: Positions of the environment hydrogens.
        c_ind1: Index of the first system spin in ``c_pos``.
        c_ind2: Index of the second system spin in ``c_pos``.
        href_ind: Index of the reference hydrogen in ``h_pos``; with the two system spins it
            defines the molecular reference frame.
        field_ref_frame: Magnetic field direction in the molecular reference frame (in the
            structure frame if ``use_unrotated_field``).
        fluorine: Whether the system spins are fluorine rather than carbon.
        coupling_mode: Which coupling model to use. Either "dipolar" (default,
            3/2 cos^2(theta) - 1/2) or the deprecated "mas-simple" (|sin(2*theta)|).
        use_unrotated_field: If True, use ``field_ref_frame`` directly in the coordinate frame
            of the structure file (PDB/CIF) instead of mapping it from the molecular reference
            frame.

    Returns:
        The coupling matrix (Hamiltonian) of shape (N, N), where N is 2 + len(h_pos): the two
        system spins followed by the hydrogens.

    Raises:
        ValueError: If the system spins coincide, two nuclei share a position, the field is
            the zero vector, or ``coupling_mode`` is not a static coupling model.
    """
    _require_distinct_system_spins(c_pos, c_ind1, c_ind2)
    if use_unrotated_field:
        field_struct_frame = field_ref_frame
    else:
        # The molecular axes are given in structure coordinates, so this maps the field from the
        # molecular reference frame into the frame of c_pos and h_pos.
        x_hat, y_hat, z_hat = get_molecule_reference_frame(
            c_pos[c_ind1], c_pos[c_ind2], h_pos[href_ind]
        )
        field_struct_frame = (
            field_ref_frame[0] * x_hat + field_ref_frame[1] * y_hat + field_ref_frame[2] * z_hat
        )
    nuclei_labels = _nuclei_labels(len(h_pos), fluorine=fluorine)
    all_pos = np.concatenate([c_pos[[c_ind1, c_ind2], :], h_pos], axis=0)
    return _compute_coupling_matrix(all_pos, nuclei_labels, field_struct_frame, coupling_mode)


@dataclass(frozen=True)
class PairGeometry:
    """Orientation-independent pair data for the MAS coupling model.

    Attributes:
        spatial: Rank-2 spherical tensor components of every pair in the *reference* frame,
            shape ``(N, N, 5)``.
        prefactors: Dipolar coupling constants ``b_ij = CONST_HH_kHz_A * g_i * g_j / r^3``,
            shape ``(N, N)`` with a zero diagonal.
        same_species: Boolean mask marking homonuclear (double-quantum) pairs, shape ``(N, N)``.
        labels: The nuclear labels used, length ``N``.
    """

    spatial: NDArray
    prefactors: NDArray
    same_species: NDArray
    labels: list[str]


def compute_pair_geometry(
    c_pos: NDArray,
    h_pos: NDArray,
    c_ind1: int,
    c_ind2: int,
    href_ind: int,
    *,
    fluorine: bool,
    use_unrotated_field: bool = False,
) -> PairGeometry:
    """Computes the orientation-independent geometry of all spin pairs.

    The internuclear unit vectors are expressed in the molecular reference frame defined by
    :func:`get_molecule_reference_frame` (the same convention the static models use for the
    magnetic field direction), so that a powder orientation can afterwards be applied as a
    plain rotation of that frame into the rotor frame.

    Args:
        c_pos: Array of carbon (or fluorine) positions.
        h_pos: Array of hydrogen positions.
        c_ind1: Index of the first system spin in ``c_pos``.
        c_ind2: Index of the second system spin in ``c_pos``.
        href_ind: Index of the reference hydrogen used to define the molecular frame.
        fluorine: Whether the system spins are fluorine rather than carbon.
        use_unrotated_field: If True, keep the raw structure-file frame instead of rotating
            into the molecular reference frame.

    Returns:
        The :class:`PairGeometry` for this conformer.

    Raises:
        ValueError: If the system spins coincide or two nuclei share a position.
    """
    _require_distinct_system_spins(c_pos, c_ind1, c_ind2)
    nuclei_labels = _nuclei_labels(len(h_pos), fluorine=fluorine)
    all_pos = np.concatenate([c_pos[[c_ind1, c_ind2], :], h_pos], axis=0)

    if use_unrotated_field:
        frame = np.eye(3)
    else:
        x_hat, y_hat, z_hat = get_molecule_reference_frame(
            c_pos[c_ind1], c_pos[c_ind2], h_pos[href_ind]
        )
        frame = np.stack([x_hat, y_hat, z_hat], axis=1)

    r, unit_vectors = _pair_geometry(all_pos)
    # Express the pair directions in the molecular reference frame (frame columns are the
    # reference axes), matching the static convention cos(theta) = f_ref . (frame^T n).
    unit_vectors = unit_vectors @ frame

    gyro_arr = _gyro_fractions(nuclei_labels)
    prefactors = CONST_HH_kHz_A * gyro_arr[:, np.newaxis] * gyro_arr[np.newaxis, :] * r**-3
    np.fill_diagonal(prefactors, 0.0)

    labels = np.array(nuclei_labels)
    same_species = labels[:, np.newaxis] == labels[np.newaxis, :]

    return PairGeometry(
        spatial=spatial_tensor_components(unit_vectors),
        prefactors=prefactors,
        same_species=same_species,
        labels=nuclei_labels,
    )


def get_mas_hamiltonian_hbulk_cc(
    geometry: PairGeometry,
    model: MASRecouplingModel,
    euler_zyz: NDArray | tuple[float, float, float],
) -> NDArray:
    """Returns the complex MAS effective coupling matrix for a single powder orientation.

    Args:
        geometry: The orientation-independent pair geometry.
        model: The MAS recoupling model (eta factors and rotor angle).
        euler_zyz: ZYZ Euler angles of the molecular-to-rotor rotation.

    Returns:
        Complex symmetric coupling matrix of shape ``(N, N)``.
    """
    return effective_coupling_matrix(
        geometry.spatial,
        geometry.prefactors,
        geometry.same_species,
        model,
        euler_zyz,
    )


def parse_structure(pdb_path: Path | str) -> Structure:
    """Parses a PDB or CIF structure file.

    Args:
        pdb_path: Path to the PDB/CIF file.

    Returns:
        The parsed Bio.PDB.Structure object.

    Raises:
        ValueError: If the file suffix is unrecognized.
        RuntimeError: If parsing fails.
    """
    pdb_path = Path(pdb_path)
    suffix = pdb_path.suffix
    parser: MMCIFParser | PDBParser
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        if suffix == ".cif":
            parser = MMCIFParser(QUIET=True)
        elif suffix == ".pdb":
            parser = PDBParser(QUIET=True)
        else:
            msg = f"Unrecognized suffix: {suffix}"
            raise ValueError(msg)
        structure_id = pdb_path.stem
        structure = parser.get_structure(structure_id, str(pdb_path))
    if structure is None:
        msg = f"Failed to parse structure: {pdb_path}"
        raise RuntimeError(msg)
    return structure


def _as_structure(source: Path | Structure) -> Structure:
    """Returns ``source`` if it is a parsed structure, otherwise parses the file it names."""
    if isinstance(source, (str, PathLike)):
        return parse_structure(Path(source))
    return source


def get_carbon_indices(
    source: Path | Structure, ind1: int, ind2: int, *, fluorine: bool
) -> tuple[int, int]:
    """Converts global atom indices to indices among the system-species atoms of model 0.

    Args:
        source: Path to the PDB/CIF file, or the already parsed structure.
        ind1: Global 0-based index of the first atom in model 0.
        ind2: Global 0-based index of the second atom in model 0.
        fluorine: If True, both atoms must be fluorine ("F") and are counted among the
            fluorines; otherwise they must be carbon ("C") and are counted among the carbons.

    Returns:
        A tuple (c_ind1, c_ind2) of indices into the carbon (or fluorine) coordinates.

    Raises:
        ValueError: If the file suffix is unrecognized, an index is out of range, or an atom is
            not of the required element.
        RuntimeError: If the structure cannot be parsed.
    """
    structure = _as_structure(source)
    element_wanted = "F" if fluorine else "C"
    atoms = list(structure[0].get_atoms())  # only the first model defines the indices
    c_ind1 = -1
    c_ind2 = -1
    count = 0
    for i, atom in enumerate(atoms):
        element = atom.element.upper()
        if i in (ind1, ind2) and element != element_wanted:
            msg = f"Atom {i} is {element!r}, but the system spins must be {element_wanted!r}."
            raise ValueError(msg)
        if ind1 == i:
            c_ind1 = count
        if ind2 == i:
            c_ind2 = count
        if element == element_wanted:
            count += 1
    if c_ind1 < 0 or c_ind2 < 0:
        msg = f"Atom indices ({ind1}, {ind2}) out of range for {len(atoms)} atoms in model 0."
        raise ValueError(msg)
    return c_ind1, c_ind2


class AtomSpecifier(NamedTuple):
    """A structured identifier for an atom in a PDB/CIF file."""

    chain: str
    resname: str
    resnum: int
    atomlabel: str


def parse_atom_specifier(value: str) -> int | AtomSpecifier:
    """Parse a CLI value as either an integer index or a chain:resname:resnum:atomlabel specifier.

    Args:
        value: The raw string from the command line.

    Returns:
        An integer if the value is a plain integer, or an AtomSpecifier otherwise.

    Raises:
        argparse.ArgumentTypeError: If the value is neither a valid integer nor a
            correctly formatted specifier string.
    """
    try:
        return int(value)
    except ValueError:
        pass
    parts = value.split(":")
    if len(parts) != _ATOM_SPECIFIER_FIELDS:
        msg = (
            f"Atom specifier must be an integer index or "
            f"chain:resname:resnum:atomlabel, got: '{value}'"
        )
        raise argparse.ArgumentTypeError(msg)
    chain, resname, resnum_str, atomlabel = parts
    try:
        resnum = int(resnum_str)
    except ValueError:
        msg = f"Residue number must be an integer, got: '{resnum_str}'"
        raise argparse.ArgumentTypeError(msg) from None
    return AtomSpecifier(chain=chain, resname=resname, resnum=resnum, atomlabel=atomlabel)


def resolve_atom_specifier(source: Path | Structure, spec: AtomSpecifier) -> int:
    """Resolve an AtomSpecifier to a global 0-based atom index in the structure.

    Args:
        source: Path to the PDB/CIF file, or the already parsed structure.
        spec: The AtomSpecifier to resolve.

    Returns:
        The global 0-based index of the matching atom.

    Raises:
        ValueError: If the file suffix is unrecognized or no matching atom is found.
    """
    structure = _as_structure(source)
    model = structure[0]
    for i, atom in enumerate(model.get_atoms()):
        residue = atom.get_parent()
        chain = residue.get_parent()
        res_id = residue.get_id()  # (hetfield, resseq, icode)
        if (
            chain.get_id() == spec.chain
            and residue.get_resname().strip() == spec.resname
            and res_id[1] == spec.resnum
            and atom.get_name().strip() == spec.atomlabel
        ):
            return i
    msg = f"No atom matching specifier {spec} found in {source}"
    raise ValueError(msg)


def resolve_atom_index(source: Path | Structure, raw: str) -> int:
    """Resolves a raw CLI atom argument to a global 0-based atom index.

    Args:
        source: Path to the PDB/CIF file, or the already parsed structure. It is only read
            (and a path only parsed) if ``raw`` is a specifier.
        raw: An integer index or a ``chain:resname:resnum:atomlabel`` specifier.

    Returns:
        The global 0-based atom index.
    """
    spec = parse_atom_specifier(raw)
    return resolve_atom_specifier(source, spec) if isinstance(spec, AtomSpecifier) else spec


def resolve_optional_atom_index(source: Path | Structure, raw: str | None) -> int | None:
    """Resolves a raw CLI atom argument to a global atom index, if it was provided.

    Args:
        source: Path to the PDB/CIF file, or the already parsed structure.
        raw: The raw command-line value, either an integer index, a
            ``chain:resname:resnum:atomlabel`` specifier, or None.

    Returns:
        The resolved global 0-based atom index, or None if ``raw`` is None.
    """
    return None if raw is None else resolve_atom_index(source, raw)


def extract_spin_coordinates(structure: Structure) -> SpinCoordinates:
    """Extracts the carbon, hydrogen and fluorine coordinates of every model (conformer).

    Atoms keep their order in the structure, so entry ``k`` of the "C" arrays is the ``k``-th
    carbon of every model (the carbon indices stored in the instance metadata).

    Args:
        structure: The parsed structure.

    Returns:
        ``{"C": [...], "H": [...], "F": [...]}`` with one coordinate array of shape ``(n, 3)``
        per model (``np.array([])`` if a model has no atom of that element).
    """
    coords: SpinCoordinates = {element: [] for element in _SPIN_ELEMENTS}
    for model in structure:
        model_coords: dict[str, list[NDArray]] = {element: [] for element in _SPIN_ELEMENTS}
        for atom in model.get_atoms():
            element = atom.element.upper()
            if element in model_coords:
                model_coords[element].append(atom.get_coord())
        for element, element_coords in model_coords.items():
            coords[element].append(np.array(element_coords))
    return coords


def load_spin_coordinates(
    pdb_path: Path, structure: Structure | None = None
) -> tuple[Structure, SpinCoordinates]:
    """Parses a structure file (unless it is already parsed) and extracts its coordinates.

    Args:
        pdb_path: Path to the PDB/CIF file.
        structure: The already parsed structure of ``pdb_path``, if available.

    Returns:
        The structure and its :func:`extract_spin_coordinates`.
    """
    if structure is None:
        structure = parse_structure(pdb_path)
    coords = extract_spin_coordinates(structure)
    logger.info(
        "Structure %s: %d model(s); model 0 has %s.",
        pdb_path,
        len(coords["H"]),
        ", ".join(
            f"{len(arrays[0]) if arrays else 0} {element}" for element, arrays in coords.items()
        ),
    )
    return structure, coords


def select_environment_hydrogen_indices(
    all_h_pos_list: list[NDArray],
    all_spin_pos_list: list[NDArray],
    c_ind1: int,
    c_ind2: int,
    ball_ratio: float,
) -> list[int]:
    """Selects the hydrogens close enough to the system pair to act as environment spins.

    A hydrogen is kept if, in some conformer, it is closer to either system spin than
    ``ball_ratio * (d_max + 0.5)``, where ``d_max`` is the largest system-pair separation over
    the conformers.

    Args:
        all_h_pos_list: Hydrogen positions for every conformer.
        all_spin_pos_list: System-species positions for every conformer.
        c_ind1: Index of the first system spin.
        c_ind2: Index of the second system spin.
        ball_ratio: Multiple of the (padded) pair separation defining the inclusion radius.

    Returns:
        The indices of the selected hydrogens, in structure order. If no hydrogen falls inside
        the ball, the hydrogen closest to the first system spin in the first conformer.
    """
    max_distance = max(
        np.linalg.norm(all_spin_pos[c_ind1] - all_spin_pos[c_ind2])
        for all_spin_pos in all_spin_pos_list
    )
    distance_lim = max_distance + 0.5

    def get_min_hc_dist(hind: int, cind1: int, cind2: int) -> np.floating[Any]:
        dist1 = min(
            np.linalg.norm(h_pos[hind] - c_pos[cind1])
            for h_pos, c_pos in zip(all_h_pos_list, all_spin_pos_list, strict=True)
        )
        dist2 = min(
            np.linalg.norm(h_pos[hind] - c_pos[cind2])
            for h_pos, c_pos in zip(all_h_pos_list, all_spin_pos_list, strict=True)
        )
        return min((dist1, dist2))

    reduced_h_inds = [
        h_ind
        for h_ind in range(all_h_pos_list[0].shape[0])
        if get_min_hc_dist(h_ind, c_ind1, c_ind2) < distance_lim * ball_ratio
    ]
    if not reduced_h_inds:
        # Fallback to the closest hydrogen if none are within the ball radius
        closest_h_ind = min(
            range(all_h_pos_list[0].shape[0]),
            key=lambda x: np.linalg.norm(all_h_pos_list[0][x] - all_spin_pos_list[0][c_ind1]),
        )
        reduced_h_inds = [closest_h_ind]
    return reduced_h_inds


def select_environment_hydrogens(
    all_h_pos_list: list[NDArray],
    all_spin_pos_list: list[NDArray],
    c_ind1: int,
    c_ind2: int,
    ball_ratio: float,
) -> list[NDArray]:
    """Returns the positions of the environment hydrogens in every conformer.

    Args:
        all_h_pos_list: Hydrogen positions for every conformer.
        all_spin_pos_list: System-species positions for every conformer.
        c_ind1: Index of the first system spin.
        c_ind2: Index of the second system spin.
        ball_ratio: Multiple of the (padded) pair separation defining the inclusion radius.

    Returns:
        The positions of the hydrogens chosen by :func:`select_environment_hydrogen_indices`,
        for every conformer.
    """
    reduced_h_inds = select_environment_hydrogen_indices(
        all_h_pos_list, all_spin_pos_list, c_ind1, c_ind2, ball_ratio
    )
    return [x[reduced_h_inds, :] for x in all_h_pos_list]


def reference_hydrogen_index(
    reduced_h_pos_list: list[NDArray], all_spin_pos_list: list[NDArray], c_ind1: int
) -> int:
    """Returns the environment hydrogen that, with the system spins, fixes the molecular frame.

    Args:
        reduced_h_pos_list: Environment-hydrogen positions for every conformer.
        all_spin_pos_list: System-species positions for every conformer.
        c_ind1: Index of the first system spin.

    Returns:
        The index (into the environment hydrogens) of the hydrogen closest to the first system
        spin in the first conformer.
    """
    return min(
        range(reduced_h_pos_list[0].shape[0]),
        key=lambda x: np.linalg.norm(reduced_h_pos_list[0][x] - all_spin_pos_list[0][c_ind1]),
    )


def conformer_pair_geometries(
    all_spin_pos_list: Sequence[NDArray],
    reduced_h_pos_list: Sequence[NDArray],
    c_ind1: int,
    c_ind2: int,
    href_ind: int,
    *,
    fluorine: bool,
    use_unrotated_field: bool = False,
) -> list[PairGeometry]:
    """Returns the :func:`compute_pair_geometry` of every conformer.

    Args:
        all_spin_pos_list: System-species positions for every conformer.
        reduced_h_pos_list: Environment-hydrogen positions for every conformer.
        c_ind1: Index of the first system spin.
        c_ind2: Index of the second system spin.
        href_ind: Index of the reference hydrogen among the environment hydrogens.
        fluorine: Whether the system spins are fluorine rather than carbon.
        use_unrotated_field: Whether to keep the structure frame instead of the molecular one.

    Returns:
        One :class:`PairGeometry` per conformer.
    """
    return [
        compute_pair_geometry(
            all_spin_pos,
            reduced_h_pos,
            c_ind1,
            c_ind2,
            href_ind,
            fluorine=fluorine,
            use_unrotated_field=use_unrotated_field,
        )
        for all_spin_pos, reduced_h_pos in zip(all_spin_pos_list, reduced_h_pos_list, strict=True)
    ]


class MASOrientation(NamedTuple):
    """One powder orientation of the MAS coupling model.

    Attributes:
        model: The MAS recoupling model.
        euler_zyz: ZYZ Euler angles of the molecular-to-rotor rotation.
        geometries: Pair geometries of every conformer; computed on demand if None.
    """

    model: MASRecouplingModel
    euler_zyz: NDArray
    geometries: Sequence[PairGeometry] | None = None


def mean_coupling_matrix(
    all_spin_pos_list: Sequence[NDArray],
    reduced_h_pos_list: Sequence[NDArray],
    c_ind1: int,
    c_ind2: int,
    href_ind: int,
    *,
    fluorine: bool,
    coupling_mode: CouplingMode,
    use_unrotated_field: bool = False,
    field_ref_frame: NDArray | None = None,
    mas_orientation: MASOrientation | None = None,
    log_level: int = logging.DEBUG,
) -> NDArray:
    """Averages the coupling matrix of one field direction or MAS orientation over conformers.

    Every conformer matrix is divided by the number of conformers and the results are summed in
    conformer order, so a recomputed matrix reproduces the saved instance exactly.

    Args:
        all_spin_pos_list: System-species positions for every conformer.
        reduced_h_pos_list: Environment-hydrogen positions for every conformer.
        c_ind1: Index of the first system spin.
        c_ind2: Index of the second system spin.
        href_ind: Index of the reference hydrogen among the environment hydrogens.
        fluorine: Whether the system spins are fluorine rather than carbon.
        coupling_mode: "dipolar", the deprecated "mas-simple", or "mas".
        use_unrotated_field: Whether the field direction (or MAS orientation) is given in the
            structure frame instead of the molecular reference frame.
        field_ref_frame: The field direction; required for the static coupling models.
        mas_orientation: The powder orientation; required for ``coupling_mode="mas"``.
        log_level: Logging level of the per-conformer progress messages.

    Returns:
        The conformer-averaged coupling matrix (complex for ``coupling_mode="mas"``).

    Raises:
        ValueError: If the field or orientation required by ``coupling_mode`` is missing, the
            structure has no conformers, or a coupling cannot be computed.
    """
    num_conformers = len(all_spin_pos_list)
    if coupling_mode == "mas":
        if mas_orientation is None:
            msg = "coupling_mode='mas' requires a MAS orientation."
            raise ValueError(msg)
        geometries = mas_orientation.geometries
        if geometries is None:
            geometries = conformer_pair_geometries(
                all_spin_pos_list,
                reduced_h_pos_list,
                c_ind1,
                c_ind2,
                href_ind,
                fluorine=fluorine,
                use_unrotated_field=use_unrotated_field,
            )
        conformer_matrices = (
            get_mas_hamiltonian_hbulk_cc(geometry, mas_orientation.model, mas_orientation.euler_zyz)
            for geometry in geometries
        )
    else:
        if field_ref_frame is None:
            msg = f"coupling_mode={coupling_mode!r} requires a field direction."
            raise ValueError(msg)
        conformer_matrices = (
            get_dipolar_hamiltonian_hbulk_cc(
                all_spin_pos,
                reduced_h_pos,
                c_ind1,
                c_ind2,
                href_ind,
                field_ref_frame,
                fluorine=fluorine,
                coupling_mode=coupling_mode,
                use_unrotated_field=use_unrotated_field,
            )
            for all_spin_pos, reduced_h_pos in zip(
                all_spin_pos_list, reduced_h_pos_list, strict=True
            )
        )

    mean_hamiltonian: NDArray | None = None
    for count, conformer_matrix in enumerate(conformer_matrices):
        tmp_hamiltonian = conformer_matrix / num_conformers
        if mean_hamiltonian is None:
            mean_hamiltonian = tmp_hamiltonian
        else:
            mean_hamiltonian += tmp_hamiltonian
        logger.log(log_level, "  Done with trajectory instance %d", count + 1)
    if mean_hamiltonian is None:
        msg = "No conformers available."
        raise ValueError(msg)
    return mean_hamiltonian


def spin_order(mean_hamiltonian: NDArray) -> tuple[NDArray, int]:
    """Orders the spins by the approximate classical time from the measured spin.

    Args:
        mean_hamiltonian: The (real or complex) coupling matrix. The path heuristic only uses
            coupling strengths, which is the right proxy when double-quantum couplings carry a
            rotor phase.

    Returns:
        A tuple ``(indices, num_spins_min)``: the spin order (the two system spins first) and
        the minimum number of spins required by the approximation.
    """
    # Divide-by-zero warnings are expected.
    with np.errstate(divide="ignore"):
        _pl, indices, num_spins_min = get_indices_sorted_by_classical_time_approx(
            np.abs(mean_hamiltonian) if np.iscomplexobj(mean_hamiltonian) else mean_hamiltonian
        )
    return indices, num_spins_min


def _save_instance_matrices(
    output_folder: Path, index: int, mean_hamiltonian: NDArray, max_size: int | None
) -> int:
    """Orders the spins of one instance, prunes it to ``max_size`` spins and saves it.

    Writes ``dipolar_matrix{index}.npy`` and ``ordered_indices{index}.npy``. If the system is
    pruned, the saved matrix is the reordered ``max_size x max_size`` block, the saved order is
    ``0, ..., max_size - 1``, and ``original_indices{index}.npy`` records which spins of the
    unpruned system were kept (``nmr-map-atoms`` uses it to identify the atoms).

    Args:
        output_folder: Folder of the instance files.
        index: Index of the field direction or orientation.
        mean_hamiltonian: Conformer-averaged coupling matrix of the unpruned system.
        max_size: Maximum number of spins to keep, or None to keep all of them.

    Returns:
        The minimum number of spins required by the approximation (at most ``max_size``).
    """
    indices, num_spins_min = spin_order(mean_hamiltonian)
    original_indices_path = output_folder / f"original_indices{index}.npy"
    if max_size is not None and len(indices) > max_size:
        pruned_indices = indices[:max_size]
        np.save(original_indices_path, pruned_indices)
        mean_hamiltonian = mean_hamiltonian[np.ix_(pruned_indices, pruned_indices)]
        indices = np.arange(max_size)
        num_spins_min = min(num_spins_min, max_size)
    else:
        original_indices_path.unlink(missing_ok=True)  # never leave a stale mapping behind
    np.save(output_folder / f"dipolar_matrix{index}.npy", mean_hamiltonian)
    np.save(output_folder / f"ordered_indices{index}.npy", indices)
    return num_spins_min


def generate_spin_system(
    config: dict[str, Any],
    pdb_path: Path,
    output_folder: Path,
    idx_generator: np.random.Generator | np.random.RandomState | tuple[int, int],
    fields: list[NDArray] | NDArray,
    *,
    fluorine: bool = False,
    coupling_mode: CouplingMode = "dipolar",
    max_size: int | None = None,
    use_unrotated_field: bool = False,
    orientation_grid: OrientationGrid | None = None,
    mas_model: MASRecouplingModel | None = None,
    structure: Structure | None = None,
) -> tuple[tuple[int, int], int, float]:
    """Generates a spin system and calculates Hamiltonians for specified fields/orientations.

    Args:
        config: Configuration dictionary containing parameters like 'distance_min',
            'distance_max', and 'ball_ratio'.
        pdb_path: Path to the PDB file.
        output_folder: Directory to save the generated Hamiltonians and indices.
        idx_generator: Random generator for selecting pairs or a fixed tuple
            of global atom indices (ind1, ind2).
        fields: List of magnetic field vectors. Ignored (and replaced by the rotor axes of
            ``orientation_grid``) when ``coupling_mode`` is "mas".
        fluorine: Whether the system spins are fluorine atoms.
        coupling_mode: Which coupling model to use: "dipolar" (default), the deprecated
            "mas-simple", or "mas" (full MAS effective Hamiltonian, complex output).
        max_size: Maximum size of the spin network to keep after generation.
        use_unrotated_field: Whether to skip rotating into the molecular reference frame.
        orientation_grid: Powder grid of molecular-to-rotor orientations. Required for "mas".
        mas_model: MAS recoupling model (eta factors, rotor angle). Required for "mas".
        structure: The already parsed structure of ``pdb_path``; parsed here if None.

    Returns:
        A tuple containing:
            - indices: The carbon-only indices of the selected pair.
            - max_num_spins_min: The maximum number of spins required for approximation.
            - distance: The distance between the selected carbon pair in Angstroms.

    Raises:
        ValueError: If the structure has no hydrogens, the pair is invalid, or the MAS inputs
            are missing.
        TypeError: If ``idx_generator`` has an unsupported type.
    """
    structure, coords = load_spin_coordinates(pdb_path, structure)
    all_h_pos_list = coords["H"]
    if len(all_h_pos_list) == 0 or all_h_pos_list[0].shape[0] == 0:
        msg = (
            f"No hydrogen atoms found in structure '{pdb_path}'. Hydrogens are required as "
            "environment spins."
        )
        raise ValueError(msg)
    all_spin_pos_list = coords["F"] if fluorine else coords["C"]

    if isinstance(idx_generator, (np.random.Generator, np.random.RandomState)):
        c_ind1, c_ind2 = get_separated_pair(
            all_spin_pos_list[0],
            idx_generator,
            distance_min=config["distance_min"],
            distance_max=config["distance_max"],
        )
    elif isinstance(idx_generator, tuple):
        ind1, ind2 = idx_generator
        c_ind1, c_ind2 = get_carbon_indices(structure, ind1, ind2, fluorine=fluorine)
    else:
        msg = f"Unrecognized generator type {type(idx_generator)}"
        raise TypeError(msg)
    distance = np.linalg.norm(all_spin_pos_list[0][c_ind1] - all_spin_pos_list[0][c_ind2])
    reduced_h_pos_list = select_environment_hydrogens(
        all_h_pos_list, all_spin_pos_list, c_ind1, c_ind2, config["ball_ratio"]
    )
    href_ind = reference_hydrogen_index(reduced_h_pos_list, all_spin_pos_list, c_ind1)

    mas_orientations: list[MASOrientation] | None = None
    if coupling_mode == "mas":
        if orientation_grid is None or mas_model is None:
            msg = "coupling_mode='mas' requires both orientation_grid and mas_model."
            raise ValueError(msg)
        geometries = conformer_pair_geometries(
            all_spin_pos_list,
            reduced_h_pos_list,
            c_ind1,
            c_ind2,
            href_ind,
            fluorine=fluorine,
            use_unrotated_field=use_unrotated_field,
        )
        mas_orientations = [
            MASOrientation(mas_model, euler_zyz, geometries)
            for euler_zyz in orientation_grid.euler_zyz
        ]
        fields = orientation_grid.rotor_axes

    max_num_spins_min = 0
    for i, field_ref_frame in enumerate(fields):
        with log_duration(f"  Hamiltonian instance {i} generation", logger):
            logger.info("Running Hamiltonian instance %d ...", i)
            np.save(output_folder / f"field_direction{i}.npy", field_ref_frame)
            if orientation_grid is not None:
                np.save(output_folder / f"orientation{i}.npy", orientation_grid.rotations[i])

            mean_hamiltonian = mean_coupling_matrix(
                all_spin_pos_list,
                reduced_h_pos_list,
                c_ind1,
                c_ind2,
                href_ind,
                fluorine=fluorine,
                coupling_mode=coupling_mode,
                use_unrotated_field=use_unrotated_field,
                field_ref_frame=field_ref_frame,
                mas_orientation=None if mas_orientations is None else mas_orientations[i],
                log_level=logging.INFO,
            )
            logger.info("Done with Hamiltonian instance %d", i)

            num_spins_min = _save_instance_matrices(output_folder, i, mean_hamiltonian, max_size)
            max_num_spins_min = max(max_num_spins_min, num_spins_min)

    return (int(c_ind1), int(c_ind2)), int(max_num_spins_min), float(distance)


def _has_atom_specifier(*raw_values: str | None) -> bool:
    """Returns whether a raw CLI atom argument is a chain:resname:resnum:atomlabel specifier."""
    return any(
        raw is not None and isinstance(parse_atom_specifier(raw), AtomSpecifier)
        for raw in raw_values
    )


def _resolve_atom_pair(
    pdb_path: Path, raw1: str | None, raw2: str | None
) -> tuple[Structure | None, tuple[int, int] | None]:
    """Resolves --cidx1 / --cidx2 to global atom indices, parsing the structure only if needed.

    Exits with an error if both arguments name the same atom (e.g. a specifier and an index).

    Args:
        pdb_path: Path to the PDB/CIF file.
        raw1: The raw --cidx1 value, or None.
        raw2: The raw --cidx2 value, or None.

    Returns:
        The structure, if a specifier required parsing it (None otherwise), and the global atom
        indices of the pair (None if no pair was given).
    """
    if raw1 is None or raw2 is None:
        return None, None
    structure = parse_structure(pdb_path) if _has_atom_specifier(raw1, raw2) else None
    source = pdb_path if structure is None else structure
    pair = (resolve_atom_index(source, raw1), resolve_atom_index(source, raw2))
    if pair[0] == pair[1]:
        logger.error("--cidx1 and --cidx2 resolve to the same atom (index %d).", pair[0])
        sys.exit(1)
    return structure, pair


def _validate_atom_pair(parser: argparse.ArgumentParser, raw1: str, raw2: str) -> None:
    """Rejects malformed, negative, or identical --cidx1 / --cidx2 values."""
    try:
        spec1, spec2 = parse_atom_specifier(raw1), parse_atom_specifier(raw2)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))
    for spec in (spec1, spec2):
        if isinstance(spec, int) and spec < 0:
            parser.error(f"Atom indices must be non-negative, got {spec}.")
    if spec1 == spec2:
        parser.error("--cidx1 and --cidx2 must name two different atoms.")


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Rejects invalid argument values and combinations before any output is written."""
    if args.instance_id == 0:
        parser.error(
            "instance_id must be non-zero: N > 0 draws N random orientations and -L selects the "
            "L-point Lebedev grid."
        )
    if args.instance_id < 0 and -args.instance_id not in SUPPORTED_LEBEDEV_SIZES:
        sizes = ", ".join(str(n) for n in SUPPORTED_LEBEDEV_SIZES)
        parser.error(
            f"A negative instance_id -L selects the L-point Lebedev grid; L must be one of "
            f"{sizes}, got {-args.instance_id}."
        )
    if (args.cidx1 is None) != (args.cidx2 is None):
        parser.error("Either both --cidx1 and --cidx2 must be specified, or neither.")
    if args.cidx1 is not None:
        _validate_atom_pair(parser, args.cidx1, args.cidx2)
    if args.max_size is not None and args.max_size < MIN_SPINS:
        parser.error(f"--max-size must be at least {MIN_SPINS}.")

    mas_only_flags = {
        "--n-twist": args.n_twist != parser.get_default("n_twist"),
        "--rotor-angle-deg": args.rotor_angle_deg != parser.get_default("rotor_angle_deg"),
        "--mas-dq-order": args.mas_dq_order != parser.get_default("mas_dq_order"),
        "--mas-zz-order": args.mas_zz_order != parser.get_default("mas_zz_order"),
    }
    if args.coupling != "mas":
        used = [flag for flag, changed in mas_only_flags.items() if changed]
        if used:
            parser.error(f"{', '.join(used)} require(s) --coupling mas.")
    else:
        if args.use_symmetry:
            parser.error(
                "--use-symmetry is not valid with --coupling mas: the inversion pruning of the "
                "Lebedev grid is not a symmetry of the SO(3) orientation grid and would bias "
                "the powder average."
            )
        if args.n_twist < 1:
            parser.error("--n-twist must be a positive integer.")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate conformer-averaged coupling-matrix instances for a pair of system spins "
            "and their environment hydrogens."
        )
    )
    parser.add_argument(
        "pdb_id", type=str, help="PDB ID; the structure is read from protein_files/<pdb_id>/."
    )
    parser.add_argument(
        "instance_id",
        type=int,
        help=(
            "Powder sampling: N > 0 draws N random field directions (Haar-random orientations "
            "with --coupling mas); -L uses the L-point Lebedev grid (tensored with --n-twist "
            "rotor twists for --coupling mas), e.g. -6, -14, -26, -38 or -50."
        ),
    )
    parser.add_argument(
        "--cidx1",
        type=str,
        required=False,
        default=None,
        help=(
            "First system atom (carbon, or fluorine with --fluorine): global 0-based atom index "
            "in model 0, or chain:resname:resnum:atomlabel"
        ),
    )
    parser.add_argument(
        "--cidx2",
        type=str,
        required=False,
        default=None,
        help=(
            "Second system atom (carbon, or fluorine with --fluorine): global 0-based atom index "
            "in model 0, or chain:resname:resnum:atomlabel"
        ),
    )
    parser.add_argument(
        "--filename",
        type=str,
        required=False,
        default=None,
        help="Structure file in protein_files/<pdb_id>/ (default: <pdb_id lowercased>_out.cif)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=False,
        default=None,
        help=(
            "Name of the output subdirectory of data/generated_instances/<pdb_id>_<instance_id> "
            "(<pdb_id>_L<L> for a Lebedev grid). Default: d_5 for a random pair, "
            "inds_<cidx1>_<cidx2> for a given pair."
        ),
    )
    parser.add_argument(
        "--fluorine",
        required=False,
        default=False,
        action="store_true",
        help="Use two fluorine atoms instead of two carbons as the system spins.",
    )
    parser.add_argument(
        "--use-symmetry",
        required=False,
        default=False,
        action="store_true",
        help="Use inversion symmetry to prune the Lebedev grid.",
    )
    parser.add_argument(
        "--coupling",
        choices=["dipolar", "mas-simple", "mas"],
        default="dipolar",
        help=(
            "Coupling model: 'dipolar' (default, (3cos^2(theta)-1)/2), 'mas-simple' "
            "(deprecated, |sin(2*theta)|), or 'mas' (full MAS effective Hamiltonian with an "
            "SO(3) powder average; writes complex coupling matrices)."
        ),
    )
    parser.add_argument(
        "--n-twist",
        type=int,
        required=False,
        default=8,
        help=(
            "MAS only: number of uniform rotor-twist angles tensored with the Lebedev grid "
            "(total orientations = n_lebedev * n_twist). Default: 8."
        ),
    )
    parser.add_argument(
        "--rotor-angle-deg",
        type=float,
        required=False,
        default=float(np.degrees(MAGIC_ANGLE_RAD)),
        help="MAS only: angle between the rotor axis and B0, in degrees. Default: the magic angle.",
    )
    parser.add_argument(
        "--mas-dq-order",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help=(
            "MAS only: |m| of the spatial Fourier component recoupled to the homonuclear "
            "double-quantum term (0 disables it). Default: 1 (gamma-encoded)."
        ),
    )
    parser.add_argument(
        "--mas-zz-order",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help=(
            "MAS only: |m| of the spatial Fourier component recoupled to the heteronuclear "
            "Z_p Z_i term (0 disables it). Default: 1."
        ),
    )
    parser.add_argument(
        "--max-size",
        type=int,
        required=False,
        default=None,
        help="Maximum size of the spin network to keep after generation.",
    )
    parser.add_argument(
        "--use-unrotated-field",
        required=False,
        default=False,
        action="store_true",
        help=(
            "Interpret the field directions (MAS: the orientations) in the coordinate frame of "
            "the structure file instead of the molecular reference frame defined by the two "
            "system spins and their nearest hydrogen."
        ),
    )
    add_verbose_arg(parser)
    args = parser.parse_args()
    _validate_args(parser, args)
    return args


def get_instance_group_folder(pdb_id: str, instance_id: int, nleb: int | None) -> Path:
    """Creates and returns the instance group folder path."""
    folder_name = (
        (pdb_id + "_" + str(instance_id)) if nleb is None else (pdb_id + "_L" + str(-instance_id))
    )
    instance_group_folder = Path.cwd().resolve() / "data" / "generated_instances" / folder_name
    instance_group_folder.mkdir(parents=True, exist_ok=True)
    return instance_group_folder


def get_fields_and_weights(
    instance_id: int,
    rng_field: np.random.RandomState,
    nleb: int | None,
    *,
    use_symmetry: bool = False,
) -> tuple[list[NDArray] | NDArray, NDArray]:
    """Generates the field directions and quadrature weights of the static powder average.

    Args:
        instance_id: Number of random field directions (used when ``nleb`` is None).
        rng_field: Random generator for the random field directions.
        nleb: Lebedev grid size, or None for random field directions.
        use_symmetry: Keep one field of every inversion-symmetric Lebedev pair, doubling its
            weight.

    Returns:
        A tuple ``(fields, weights)``. Lebedev grids are copies of the cached grid, so callers
        may modify them.

    Raises:
        ValueError: If ``instance_id`` is not positive for random fields, or ``nleb`` is not a
            supported Lebedev grid size.
    """
    if nleb is None:
        if instance_id < 1:
            msg = f"The number of random field directions must be positive, got {instance_id}."
            raise ValueError(msg)
        weights = np.full(instance_id, 1 / instance_id)
        random_fields = []
        for _ in range(instance_id):
            field_ref_frame = rng_field.normal(loc=0, scale=1, size=3)
            field_ref_frame /= np.linalg.norm(field_ref_frame)
            random_fields.append(field_ref_frame)
        return random_fields, weights
    lgrid = lebedev_grid(nleb)
    fields, weights = lgrid[:, :3], lgrid[:, 3]
    if use_symmetry:
        fields, weights = prune_lebedev_grid(fields, weights)
    return fields, weights


def get_orientation_grid(
    instance_id: int,
    rng_field: np.random.RandomState,
    nleb: int | None,
    n_twist: int,
) -> OrientationGrid:
    """Builds the powder grid of molecular-to-rotor orientations for the MAS model.

    Args:
        instance_id: The CLI instance id; positive values request that many random
            (Haar-distributed) orientations.
        rng_field: Random generator used for the random-orientation case.
        nleb: Lebedev grid size, or None to use random orientations.
        n_twist: Number of uniform rotor-twist angles to tensor with the Lebedev grid.

    Returns:
        The :class:`OrientationGrid` to average over.
    """
    if nleb is None:
        return random_so3_orientations(instance_id, rng_field)
    return so3_grid(nleb, n_twist)


def build_mas_model(args: argparse.Namespace) -> MASRecouplingModel:
    """Builds the MAS recoupling model from parsed command-line arguments.

    The model only selects which spatial components are recoupled (with unit weight) and the
    rotor angle. As for the static coupling models, the RF scaling of the pulse sequence is not
    stored in the coupling matrices; it is applied at simulation time via ``--eta_hom`` /
    ``--eta_het``.

    Args:
        args: The parsed arguments namespace.

    Returns:
        The configured :class:`MASRecouplingModel`.
    """
    return MASRecouplingModel.single_component(
        dq_order=args.mas_dq_order,
        zz_order=args.mas_zz_order,
        rotor_angle=float(np.radians(args.rotor_angle_deg)),
    )


def setup_powder_average(
    args: argparse.Namespace,
    instance_id: int,
    rng_field: np.random.RandomState,
    nleb: int | None,
) -> tuple[list[NDArray] | NDArray, NDArray, OrientationGrid | None, MASRecouplingModel | None]:
    """Builds the powder sampling (field directions or SO(3) orientations) for a run.

    Args:
        args: The parsed command-line arguments.
        instance_id: The CLI instance id.
        rng_field: Random generator for the random-sampling case.
        nleb: Lebedev grid size, or None for random sampling.

    Returns:
        A tuple ``(fields, weights, orientation_grid, mas_model)``; the last two entries are
        ``None`` unless the MAS coupling model was requested.
    """
    if args.coupling != "mas":
        fields, weights = get_fields_and_weights(
            instance_id, rng_field, nleb, use_symmetry=args.use_symmetry
        )
        if args.coupling == "mas-simple":
            logger.warning(
                "The 'mas-simple' coupling model is deprecated; use --coupling mas for the "
                "full MAS effective Hamiltonian with an SO(3) powder average."
            )
        return fields, weights, None, None

    orientation_grid = get_orientation_grid(instance_id, rng_field, nleb, args.n_twist)
    mas_model = build_mas_model(args)
    logger.info(
        "MAS powder grid: %d orientations (%d Lebedev x %d twist)",
        len(orientation_grid),
        orientation_grid.n_lebedev,
        orientation_grid.n_twist,
    )
    return orientation_grid.rotor_axes, orientation_grid.weights, orientation_grid, mas_model


def save_mas_grid(
    dist_folder: Path, orientation_grid: OrientationGrid, mas_model: MASRecouplingModel
) -> dict[str, Any]:
    """Saves the MAS orientation grid and returns its metadata block.

    Args:
        dist_folder: Output folder of this instance.
        orientation_grid: The powder grid that was used.
        mas_model: The MAS recoupling model that was used.

    Returns:
        A JSON-serializable description of the grid and the recoupling model.
    """
    np.save(dist_folder / "orientations.npy", orientation_grid.rotations)
    np.save(dist_folder / "euler_angles.npy", orientation_grid.euler_zyz)
    return {
        "n_orientations": len(orientation_grid),
        "n_lebedev": orientation_grid.n_lebedev,
        "n_twist": orientation_grid.n_twist,
        "grid": "lebedev_x_uniform_twist_about_rotor_axis",
        "storage_convention": (
            "H_hom = sum_{i<j} [Re(d) (YY - XX) + Im(d) (XY + YX)]; "
            "heteronuclear entries are real Z_p Z_i couplings; "
            "RF scaling (eta) is not included, it is applied at simulation time via "
            "--eta_hom / --eta_het"
        ),
        **mas_model.to_metadata(),
    }


def main() -> None:
    """Main function to parse arguments and generate NMR OTOC instances."""
    args = parse_args()
    configure_logging(verbose=args.verbose)

    pdb_id = args.pdb_id
    instance_id = args.instance_id

    cwd = Path.cwd().resolve()
    filename = (pdb_id.lower() + "_out.cif") if args.filename is None else args.filename
    pdb_path = cwd / "protein_files" / pdb_id / filename
    instance_seed = 0  # this is for choosing the pair of spins

    nleb = -instance_id if instance_id < 0 else None

    structure, atom_pair = _resolve_atom_pair(pdb_path, args.cidx1, args.cidx2)
    target_distance: float | None = None
    if atom_pair is None:
        # This is for choosing the actual spins, use a fixed seed for reproducibility
        pair_generator: np.random.RandomState | tuple[int, int] = np.random.RandomState(
            instance_seed
        )
        target_distance = 5.0
        default_output_name = f"d_{int(target_distance)}"
    else:
        pair_generator = atom_pair
        default_output_name = f"inds_{atom_pair[0]}_{atom_pair[1]}"
    output_name = default_output_name if args.output_name is None else args.output_name

    # This is for choosing the field
    rng_field = np.random.RandomState()

    logger.info("Generating Instance Group %d", instance_id)
    instance_group_folder = get_instance_group_folder(pdb_id, instance_id, nleb)

    dist_folder = instance_group_folder / output_name
    dist_folder.mkdir(exist_ok=True)
    metadata_file = dist_folder / "metadata.json"

    if metadata_file.exists():
        logger.info("  Skipping d=%s A (already exists).", target_distance)
        return

    logger.info("  Processing target distance: %s A", target_distance)

    config = {
        "ball_ratio": 2,
        "root_dir": str(cwd),
        "target_distance": target_distance,
    }
    if target_distance is not None:
        config["distance_min"] = target_distance - 0.25
        config["distance_max"] = target_distance + 0.25

    coupling_mode: CouplingMode = args.coupling
    fields, weights, orientation_grid, mas_model = setup_powder_average(
        args, instance_id, rng_field, nleb
    )

    indices, num_spins, distance = generate_spin_system(
        config,
        pdb_path,
        dist_folder,
        pair_generator,
        fields,
        fluorine=args.fluorine,
        coupling_mode=coupling_mode,
        max_size=args.max_size,
        use_unrotated_field=args.use_unrotated_field,
        orientation_grid=orientation_grid,
        mas_model=mas_model,
        structure=structure,
    )
    np.save(dist_folder / "weights.npy", weights)

    metadata: dict[str, Any] = {
        "pdb_id": pdb_id,
        "config": config,
        "indices": indices,
        "num_spins_min": num_spins,
        "distance": float(distance),
        "coupling": args.coupling,
        "coupling_format": "complex" if coupling_mode == "mas" else "real",
        "fluorine": args.fluorine,
        "use_unrotated_field": args.use_unrotated_field,
    }
    if orientation_grid is not None and mas_model is not None:
        metadata["mas"] = save_mas_grid(dist_folder, orientation_grid, mas_model)

    # The metadata is written last: its presence marks the instance as complete.
    metadata_file.write_text(json.dumps(metadata))

    logger.info("    Actual distance: %s", distance)
    logger.info("    Success at d=%s with %s", target_distance, pdb_id)

    logger.info("Instance Group %d generation complete for %s.", instance_id, pdb_id)


if __name__ == "__main__":
    main()
