# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Optional
import itertools
import numpy as np
import scipy
from openfermion import QubitOperator, get_sparse_operator


I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


def split_into_number_conserving_sectors(mat):
    num_qubits = int(np.log2(mat.shape[0]))
    assert 2**num_qubits == mat.shape[0]
    split_indices = [
        np.array([index for index in range(2**num_qubits) if int.bit_count(index) == n])
        for n in range(num_qubits + 1)
    ]
    split_mats = [mat[np.ix_(indices, indices)] for indices in split_indices]
    return split_mats


def construct_dq_hamiltonian_of(
    hom: np.ndarray,
    het: np.ndarray,
    z_vector: Optional[np.ndarray] = None,
    xyyx_flag: Optional[bool] = False,
):
    """Constructs the Hamiltonian matrix using OpenFermion."""
    num_qubits = hom.shape[0]
    assert het.shape == (num_qubits, num_qubits)
    H = QubitOperator()
    for i in range(num_qubits):
        if z_vector is not None:
            H += z_vector[i] * QubitOperator(f"Z{i}")
        for j in range(i + 1, num_qubits):
            if xyyx_flag is True:
                H += hom[i, j] * (
                    QubitOperator(f"X{i} Y{j}") + QubitOperator(f"Y{i} X{j}")
                )
            else:
                H += hom[i, j] * (
                    QubitOperator(f"Y{i} Y{j}") - QubitOperator(f"X{i} X{j}")
                )
            H += het[i, j] * QubitOperator(f"Z{i} Z{j}")
    return H

def construct_dq_hamiltonian_matrix(
    hom: np.ndarray,
    het: np.ndarray,
    z_vector: Optional[np.ndarray] = None,
    xyyx_flag: Optional[bool] = False,
):
    num_qubits = hom.shape[0]
    H = construct_dq_hamiltonian_of(
        hom=hom, het=het, z_vector=z_vector, xyyx_flag=xyyx_flag
    )
    H_mat = get_sparse_operator(H, n_qubits=num_qubits).todense()
    return H_mat


def construct_nmr_hamiltonian_of(
    hom: np.ndarray,
    het: np.ndarray,
    z_vector: Optional[np.ndarray] = None,
    J_hom: Optional[np.ndarray] = None,
    J_het: Optional[np.ndarray] = None,
):
    """Constructs the Hamiltonian matrix using OpenFermion."""
    num_qubits = hom.shape[0]
    assert het.shape == (num_qubits, num_qubits)
    H = QubitOperator()
    for i in range(num_qubits):
        if z_vector is not None:
            H += z_vector[i] * QubitOperator(f"Z{i}")
        for j in range(i + 1, num_qubits):
            H += hom[i, j] * (
                2 * QubitOperator(f"Z{i} Z{j}")
                - QubitOperator(f"X{i} X{j}")
                - QubitOperator(f"Y{i} Y{j}")
            )
            H += 2 * het[i, j] * QubitOperator(f"Z{i} Z{j}")
            if J_hom is not None:
                H += J_hom[i, j] * (
                    QubitOperator(f"Z{i} Z{j}")
                    + QubitOperator(f"X{i} X{j}")
                    + QubitOperator(f"Y{i} Y{j}")
                )
            if J_het is not None:
                H += J_het[i, j] * QubitOperator(f"Z{i} Z{j}")
    return H


def construct_nmr_hamiltonian_matrix(
    hom: np.ndarray,
    het: np.ndarray,
    z_vector: Optional[np.ndarray] = None,
    J_hom: Optional[np.ndarray] = None,
    J_het: Optional[np.ndarray] = None,
):
    """Constructs the Hamiltonian matrix using OpenFermion."""
    H = construct_nmr_hamiltonian_of(
        hom=hom,
        het=het,
        z_vector=z_vector,
        J_hom=J_hom,
        J_het=J_het
    )
    num_qubits = hom.shape[0]
    H_mat = get_sparse_operator(H, n_qubits=num_qubits).todense()
    return H_mat


def calculate_true_unitary(H_mat: np.ndarray, time: float):
    """Calculates the true unitary evolution."""
    U_true = scipy.linalg.expm(1j * time * H_mat)
    return U_true


def calculate_infidelity(U_true: np.ndarray, U_trotter: np.ndarray, num_qubits: int):
    """Calculates the infidelity between the true and Trotter unitaries."""
    infidelity = 1 - (abs(np.trace(U_true.conj().T @ U_trotter)) / (2**num_qubits)) ** 2
    return infidelity


def calculate_expectation_value_error(
    U_true: np.ndarray,
    U_trotter: np.ndarray,
    initial_state: np.ndarray,
    operator: np.ndarray,
):
    """Calculates the error in the expectation value of an operator
       between a true unitary and a Trotterized unitary.

    Args:
        U_true: The true unitary matrix.
        U_trotter: The Trotterized unitary matrix.
        initial_state: The initial state vector.
        operator: The operator matrix.

    Returns:
        float: The absolute difference between the expectation values.
    """

    # Ensure the initial state is a column vector
    if initial_state.ndim == 1:
        initial_state = initial_state.reshape(-1, 1)

    final_state_true = U_true @ initial_state
    expectation_true = (final_state_true.conj().T @ operator @ final_state_true)[
        0, 0
    ].real

    final_state_trotter = U_trotter @ initial_state
    expectation_trotter = (
        final_state_trotter.conj().T @ operator @ final_state_trotter
    )[0, 0].real

    error = abs(expectation_true - expectation_trotter)

    return error


def calculate_otoc_exact(unitary, butterfly_op, measurement_op, num_qubits):
    """Calculates the exact infinite-temperature OTOC.

    Args:
        unitary: The unitary matrix.
        butterfly_op: The butterfly operator matrix.
        measurement_op: The measurement operator matrix.
        num_qubits: The number of qubits in the system.

    Returns:
        float: The infinite-temperature OTOC.
    """

    otoc_matrix = (
        unitary.conj().T
        @ butterfly_op
        @ unitary
        @ measurement_op
        @ unitary.conj().T
        @ butterfly_op.conj().T
        @ unitary
        @ measurement_op
    )
    otoc = np.trace(otoc_matrix).real / (2**num_qubits)

    return otoc


def calculate_otoc_error(U_true, U_trotter, butterfly_op, measurement_op, num_qubits):
    """Calculates the error in estimating the infinite-temperature OTOC.

    Args:
        U_true: The true unitary matrix.
        U_trotter: The Trotterized unitary matrix.
        butterfly_op: The butterfly operator matrix.
        measurement_op: The measurement operator matrix.
        num_qubits: The number of qubits in the system.

    Returns:
        float: The absolute difference between the OTOCs.
    """

    otoc_true = calculate_otoc_exact(U_true, butterfly_op, measurement_op, num_qubits)
    otoc_trotter = calculate_otoc_exact(
        U_trotter, butterfly_op, measurement_op, num_qubits
    )
    error = abs(otoc_true - otoc_trotter)

    return error

