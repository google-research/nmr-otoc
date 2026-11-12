# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Functions and classes for generating and manipulating spin Hamiltonians.

Contains functions for converting molecules defined in nmrotoc.chem.molecule
into matrices that describe dipolar coupling matrices, and further functions
for converting these matrices into many-body operators.
"""

from typing import Tuple, List, Optional
import numpy as np
import re
from openfermion import QubitOperator
from nmrotoc.chem.molecule import Molecule, Vector, CircleVector, RandomVariable
from nmrotoc.math.graphs import dijkstra

CONST_HH_kHz_A = -120.12016082711953
GYROS = {  # https://www.acadiau.ca/~bellis/resources/nmr/isotopes.html
    "H": 26.75105,  # H-1
    "C": 6.72804,  # C-13, 1.1% natural abundance
    "N": -2.71171,  # N-15  0.4% natural abundance
    "O": -36.264,  # O-17, v. low abundance, taken from wiki
    "F": 25.18034,  # F-19,
    "P": 10.84015,  # P-31
    "Cl": 2.62401,  # Cl-35 Spin 3/2, 75% natural abundance
}
GYRO_FRACTIONS = {key: GYROS[key] / GYROS["H"] for key in GYROS}


def symmetrize_matrix(
    mat: np.ndarray, indices: List[int], relative_to: Optional[List[int]] = None
):
    """Symmetrizes a matrix across a set of indices.

    Given a set S of indices and a matrix M, this enforces the condition M_ij=M_ik and
    M_ji=M_ki for any i in R, and any j,k in S. We do this by replacing
    M_ij -> 1/|S| sum_{k in S / {i}} M_ik

    mat: matrix to symmetrize over
    indices: set S described above.
    relative_to: set R described above
    """
    if relative_to is None:
        relative_to = range(mat.shape[0])
    for i in relative_to:
        new_val = np.mean([mat[i, j] for j in indices if j != i])
        for j in indices:
            mat[i, j] = new_val
        new_val = np.mean([mat[j, i] for j in indices if j != i])
        for j in indices:
            mat[j, i] = new_val
    return mat


def calculate_hamiltonian_and_pathing_approx(
    positions, field_direction, msmt_ind=0, bfly_ind=1
):
    """Calculates Hamiltonian and classical path weights for set of positions"""
    ham = np.array(
        [
            [
                calculate_coupling(
                    pos1,
                    pos2,
                    "H",
                    "H",
                    field_flag=True,
                    field_direction=field_direction,
                )
                for pos1 in positions
            ]
            for pos2 in positions
        ]
    )
    weight_mat = 1 / abs(ham)
    paths_left = dijkstra(weight_mat, msmt_ind)
    paths_right = dijkstra(weight_mat, bfly_ind)
    paths_full = paths_left + paths_right
    indices_full = np.argsort(paths_full)
    paths_full = np.sort(paths_full)
    ind0 = list(indices_full).index(msmt_ind)
    indices_full[0], indices_full[ind0] = indices_full[ind0], indices_full[0]
    ind1 = list(indices_full).index(bfly_ind)
    indices_full[1], indices_full[ind1] = indices_full[ind1], indices_full[1]
    minpath = paths_full[0]
    num_spins_min = len([x for x in paths_full if abs(x - minpath) < 1e-6])

    return ham, paths_full, indices_full, num_spins_min


def make_coupling_matrices(
    molecule: Molecule,
    labels: List,
    order_parameters: np.ndarray,
    num_samples: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates the heterogeneous and homogeneous coupling matrices from a Molecule object.

    Args:
        molecule: The Molecule object containing atomic coordinates.
        num_samples: The number of samples to average over for dynamic coordinates.
        order_parameters:  Order parameters for coupling calculations.

    Returns:
        A tuple containing the homogeneous and heterogeneous coupling matrices.
    """

    num_atoms = len(labels)
    hom = np.zeros((num_atoms, num_atoms))
    het = np.zeros((num_atoms, num_atoms))

    if molecule.parameters:  # Dynamic structure
        for _ in range(num_samples):
            molecule.reroll()  # Reroll dynamic parameters for each sample
            hom_sample, het_sample = _calculate_coupling_matrices(
                molecule, labels, order_parameters
            )
            hom += hom_sample
            het += het_sample
        hom /= num_samples
        het /= num_samples

    else:  # Static structure
        hom, het = _calculate_coupling_matrices(molecule, labels, order_parameters)

    return hom, het


def _calculate_coupling_matrices(
    molecule: Molecule, labels: List, order_parameters
) -> Tuple[np.ndarray, np.ndarray]:

    num_atoms = len(labels)
    hom = np.zeros((num_atoms, num_atoms))
    het = np.zeros((num_atoms, num_atoms))

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):

            key1 = labels[i]
            key2 = labels[j]

            label1 = key1[: re.search(r"\d", key1).start()]
            label2 = key2[: re.search(r"\d", key2).start()]

            pos1 = molecule[key1]
            pos2 = molecule[key2]

            coupling = calculate_coupling(
                pos1, pos2, label1, label2, order_parameters=order_parameters
            )

            if label1 == label2:  # Homogeneous coupling
                hom[i, j] = hom[j, i] = coupling
            else:  # Heterogeneous coupling
                het[i, j] = het[j, i] = coupling

    return hom, het


def calculate_coupling(
    pos1,
    pos2,
    label1,
    label2,
    field_flag=False,
    order_parameters=np.array([-0.5, -0.5, 1]),
    field_direction=np.array([0, 0, 1]),
    cutoff=1e-6,
):
    r12 = np.linalg.norm(pos1 - pos2)
    if r12 < cutoff:
        return 0
    n12 = (pos1 - pos2) / r12
    const = CONST_HH_kHz_A * GYRO_FRACTIONS[label1] * GYRO_FRACTIONS[label2]
    if field_flag is True:
        field_direction /= np.linalg.norm(field_direction)
        d_factor = 3 / 2 * np.dot(field_direction, n12) ** 2 - 1 / 2
        coupling = const * (r12**-3) * d_factor
    elif len(order_parameters) == 3:
        coupling = const * (r12**-3) * np.dot(n12 * n12, order_parameters)
    else:
        coupling = const * (r12**-3) * np.dot(np.kron(n12, n12), order_parameters)
    return coupling


def of_nmr_ham_from_matrices(het, hom):
    """
    Constructs the NMR Hamiltonian as an OpenFermion QubitOperator from homogeneous and heterogeneous coupling matrices.

    Args:
        het: The heterogeneous coupling matrix.
        hom: The homogeneous coupling matrix.

    Returns:
        An OpenFermion QubitOperator representing the NMR Hamiltonian.
    """

    num_atoms = het.shape[0]
    hamiltonian = QubitOperator()

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            # Add homogeneous coupling terms
            coupling = hom[i, j]
            if coupling != 0:
                hamiltonian += (
                    2 * QubitOperator(f"Z{i} Z{j}", coupling)
                    - QubitOperator(f"X{i} X{j}", coupling)
                    - QubitOperator(f"Y{i} Y{j}", coupling)
                )
            # Add heterogeneous coupling terms
            coupling = het[i, j]
            if coupling != 0:
                hamiltonian += 2 * QubitOperator(f"Z{i} Z{j}", coupling)

    return hamiltonian


def of_dq_ham_from_matrices(het, hom):
    """
    Constructs the NMR Hamiltonian as an OpenFermion QubitOperator from homogeneous and heterogeneous coupling matrices.

    Args:
        het: The heterogeneous coupling matrix.
        hom: The homogeneous coupling matrix.

    Returns:
        An OpenFermion QubitOperator representing the NMR Hamiltonian.
    """

    num_atoms = het.shape[0]
    hamiltonian = QubitOperator()

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            # Add homogeneous coupling terms
            coupling = hom[i, j]
            if coupling != 0:
                hamiltonian += QubitOperator(f"X{i} X{j}", coupling) - QubitOperator(
                    f"Y{i} Y{j}", coupling
                )
            # Add heterogeneous coupling terms
            coupling = het[i, j]
            if coupling != 0:
                hamiltonian += 2 * QubitOperator(f"Z{i} Z{j}", coupling)

    return hamiltonian
