# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

import cirq
import numpy as np
import scipy
import pickle
from scipy.optimize import minimize
from openfermion import QubitOperator, get_sparse_operator, eigenspectrum
import pytest
import unittest
import tempfile
from openfermion import commutator

from nmrotoc.math.operators import (
    construct_dq_hamiltonian_matrix,
    calculate_infidelity,
    calculate_expectation_value_error,
    calculate_otoc_error,
    calculate_participation_ratio,
    calculate_average_spacing_ratio,
    calculate_otoc_exact,
    get_schrieffer_wolff_generator,
    calculate_trace_with_pauli_string,
    matrix_to_qubitoperator,
    get_schrieffer_wolff_generator_matrix_transform,
)

num_qubits = 2
hom = np.array([[0, 1], [1, 0]])
het = np.array([[0, 2], [2, 0]])

def test_construct_dq_hamiltonian_matrix():
    # Test case 1: Basic test with 2 qubits
    hom = np.array([[0, 1], [1, 0]])
    het = np.array([[0, 0], [0, 0]])
    z_vector = np.array([0, 0])
    expected_matrix = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    calculated_matrix = construct_dq_hamiltonian_matrix(
        hom, het, z_vector, xyyx_flag=False
    )
    assert np.allclose(calculated_matrix, expected_matrix)

    # Test case 2: 3 qubits with xyyx_flag = False
    num_qubits = 3
    hom = np.random.rand(num_qubits, num_qubits)
    hom = (hom + hom.T) / 2  # Ensure symmetry
    het = np.random.rand(num_qubits, num_qubits)
    het = (het + het.T) / 2  # Ensure symmetry
    z_vector = np.random.rand(num_qubits)

    # Construct the Hamiltonian using QubitOperator for comparison
    expected_H = QubitOperator()
    for i in range(num_qubits):
        expected_H += z_vector[i] * QubitOperator(f"Z{i}")
        for j in range(i + 1, num_qubits):
            expected_H += hom[i, j] * (
                QubitOperator(f"X{i} Y{j}") + QubitOperator(f"Y{i} X{j}")
            )
            expected_H += het[i, j] * QubitOperator(f"Z{i} Z{j}")
    expected_matrix = expected_H.get_matrix(num_qubits=num_qubits)
    calculated_matrix = construct_dq_hamiltonian_matrix(
        hom, het, z_vector, xyyx_flag=False
    )
    assert np.allclose(calculated_matrix, expected_matrix)

    # Test case 3: 3 qubits with xyyx_flag = True
    expected_H = QubitOperator()
    for i in range(num_qubits):
        expected_H += z_vector[i] * QubitOperator(f"Z{i}")
        for j in range(i + 1, num_qubits):
            expected_H += hom[i, j] * (
                QubitOperator(f"X{i} X{j}") - QubitOperator(f"Y{i} Y{j}")
            )
            expected_H += het[i, j] * QubitOperator(f"Z{i} Z{j}")
    expected_matrix = expected_H.get_matrix(num_qubits=num_qubits)
    calculated_matrix = construct_dq_hamiltonian_matrix(
        hom, het, z_vector, xyyx_flag=True
    )
    assert np.allclose(calculated_matrix, expected_matrix)

    # Test case 4: Test with zero matrices
    hom = np.zeros((2, 2))
    het = np.zeros((2, 2))
    calculated_matrix = construct_dq_hamiltonian_matrix(hom, het, xyyx_flag=True)
    assert np.allclose(calculated_matrix, np.zeros((4, 4)))



def test_construct_dq_hamiltonian_matrix():
    H_mat = construct_dq_hamiltonian_matrix(hom, het)
    expected_H = (
        QubitOperator("X0 X1") - QubitOperator("Y0 Y1") + 2 * QubitOperator("Z0 Z1")
    )
    expected_H_mat = get_sparse_operator(expected_H, n_qubits=num_qubits).todense()
    np.testing.assert_array_almost_equal(H_mat, expected_H_mat)


def test_calculate_infidelity():
    U_true = np.array([[1, 0], [0, 1]])
    U_trotter = np.array([[0, 1j], [1j, 0]])
    infidelity = calculate_infidelity(U_true, U_trotter, num_qubits)
    assert infidelity == pytest.approx(1.0)


def test_calculate_expectation_value_error():
    """Basic test for calculate_expectation_value_error"""

    # set up sample input matrices
    num_qubits = 2
    U_true = np.eye(2**num_qubits, dtype=np.complex128)
    U_trotter = np.eye(2**num_qubits, dtype=np.complex128)
    U_trotter[0, 0] = np.exp(1j * 0.1)
    U_trotter[2, 2] = np.exp(1j * 0.7)
    initial_state = np.zeros(2**num_qubits, dtype=np.complex128)
    initial_state[0] = 1
    op = cirq.Circuit(cirq.Z(cirq.LineQubit(0))).unitary(
        qubit_order=[cirq.LineQubit(0), cirq.LineQubit(1)]
    )

    # call and test function
    error = calculate_expectation_value_error(U_true, U_trotter, initial_state, op)
    assert abs(error) < 1e-8


def test_calculate_otoc_error():

    num_qubits = 2
    U_true = np.eye(2**num_qubits, dtype=np.complex128)
    U_trotter = np.eye(2**num_qubits, dtype=np.complex128)
    U_trotter[0, 0] = np.exp(1j * 0.1)
    U_trotter[2, 2] = np.exp(1j * 0.8)
    butterfly_op = cirq.Circuit(cirq.Z(cirq.LineQubit(0))).unitary(
        qubit_order=[cirq.LineQubit(0), cirq.LineQubit(1)]
    )
    measurement_op = cirq.Circuit(cirq.Z(cirq.LineQubit(0))).unitary(
        qubit_order=[cirq.LineQubit(0), cirq.LineQubit(1)]
    )
    error = calculate_otoc_error(
        U_true, U_trotter, butterfly_op, measurement_op, num_qubits
    )

    assert abs(error) < 1e-8


class TestCalculateOtocExact(unittest.TestCase):

    def test_calculate_otoc_exact_identity(self):
        num_qubits = 2
        unitary = np.eye(2**num_qubits)
        qubits = cirq.LineQubit.range(num_qubits)
        butterfly_op = cirq.Circuit(cirq.Z(cirq.LineQubit(0))).unitary(
            qubit_order=qubits
        )
        measurement_op = cirq.Circuit(cirq.Z(cirq.LineQubit(0))).unitary(
            qubit_order=qubits
        )

        otoc = calculate_otoc_exact(unitary, butterfly_op, measurement_op, num_qubits)
        self.assertAlmostEqual(otoc, 1)

    def test_calculate_otoc_exact_Hadamard(self):

        num_qubits = 1
        unitary = cirq.unitary(cirq.H(cirq.LineQubit(0)))

        qubits = cirq.LineQubit.range(num_qubits)
        butterfly_op = cirq.Circuit(cirq.Z(cirq.LineQubit(0))).unitary(
            qubit_order=qubits
        )
        measurement_op = cirq.Circuit(cirq.Z(cirq.LineQubit(0))).unitary(
            qubit_order=qubits
        )

        otoc = calculate_otoc_exact(unitary, butterfly_op, measurement_op, num_qubits)
        self.assertAlmostEqual(otoc, -1)

