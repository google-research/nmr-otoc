# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

import cirq
import numpy as np
import scipy
import pytest
import unittest

from nmrotoc.circuits.gates import dipolar_gate, dq_gate_fsim

rng = np.random.RandomState(42)

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

XX = np.kron(X, X)
YY = np.kron(Y, Y)
ZZ = np.kron(Z, Z)
IZ = np.kron(I, Z)
ZI = np.kron(Z, I)


def test_dipolar_gate():
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)
    dt = 1
    for _ in range(10):
        coupling = rng.uniform(-2 * np.pi, 2 * np.pi)
        J_coupling = rng.uniform(-2 * np.pi, 2 * np.pi)
        onsite_0 = rng.uniform(-2 * np.pi, 2 * np.pi)
        onsite_1 = rng.uniform(-2 * np.pi, 2 * np.pi)

        circuit = dipolar_gate(
            dt,
            q0,
            q1,
            coupling=coupling,
            J_coupling=J_coupling,
            onsite_0=onsite_0,
            onsite_1=onsite_1,
        )

        unitary = circuit.unitary(qubit_order=[q1, q0])
        dipolar_term = coupling * (ZZ - 0.5 * XX - 0.5 * YY)
        H_term = J_coupling * (XX + YY + ZZ)
        onsite_term = onsite_0 * IZ + onsite_1 * ZI
        H = dipolar_term + H_term + onsite_term

        U_target = scipy.linalg.expm(1j * H)
        U_target = U_target / U_target[0, 0]

        assert abs(np.trace(unitary.conj().T @ U_target) - 4) < 1e-6


def test_dq_gate_fsim():
    """Tests the my_gate function against the expected matrix."""
    dt = 0.3
    het = 1.0
    hom = 2.0

    # Define Pauli operators
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    # Test with swap = False
    q0, q1 = cirq.LineQubit.range(2)
    circuit = dq_gate_fsim(dt, q0, q1, het, hom, swap=False)
    XX = np.kron(X, X)
    YY = np.kron(Y, Y)
    ZZ = np.kron(Z, Z)
    expected_matrix = scipy.linalg.expm(1j * dt * hom * (XX - YY)) @ scipy.linalg.expm(
        1j * dt * het * ZZ
    )
    expected_matrix = expected_matrix * np.exp(-1j * np.angle(expected_matrix[0, 0]))
    actual_matrix = cirq.unitary(circuit)
    actual_matrix = actual_matrix * np.exp(-1j * np.angle(actual_matrix[0, 0]))
    np.testing.assert_array_almost_equal(actual_matrix, expected_matrix)

    # Test with swap = True
    circuit = dq_gate_fsim(dt, q0, q1, het, hom, swap=True)
    expected_matrix = scipy.linalg.expm(1j * dt * hom * (XX - YY)) @ scipy.linalg.expm(
        1j * dt * het * ZZ
    )
    # Apply SWAP to the expected matrix
    swap_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    expected_matrix = swap_matrix @ expected_matrix
    expected_matrix = expected_matrix * np.exp(-1j * np.angle(expected_matrix[0, 0]))
    actual_matrix = cirq.unitary(circuit)
    actual_matrix = actual_matrix * np.exp(-1j * np.angle(actual_matrix[0, 0]))
    np.testing.assert_array_almost_equal(actual_matrix, expected_matrix)
