# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

import unittest
import numpy as np
import scipy.linalg
import cirq
from openfermion import QubitOperator, get_sparse_operator, ops

# Assuming the functions are in this module
from nmrotoc.circuits.time_evolution import (
    TimeEvolutionGate,
    pulse_with_time_evolution,
    make_pulsed_scheme_circuit,
    dream_1_sequence_CH,
    dream_1_inverse_sequence_CH,
    TimeEvolutionGate,
    make_pulse_generators_DREAM_2,  # Import generator function
    make_pulse_generators_DREAM_2_inverse,  # Import generator function
    dream_1_inverse_sequence_CH_new,
    dream_2_sequence_CH,  # Import function to test
    dream_2_inverse_sequence_CH,  # Import function to test
    dream_1_sequence_CH_with_background,
    dream_2_sequence_CH_with_background,
    dream_2_inverse_sequence_CH_with_background,
    dream_1_inverse_sequence_CH_with_background,
    dream_1_inverse_sequence_CH_new_with_background,
)
from nmrotoc.math.operators import (
    construct_nmr_hamiltonian_of,
    construct_dq_hamiltonian_matrix,
    construct_dq_hamiltonian_of,
)
from nmrotoc.math.operators import matrix_to_qubitoperator

# Define Pauli matrices for convenience
I = np.identity(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Define some 2-qubit operators
ZZ = np.kron(Z, Z)
XX = np.kron(X, X)
ZI = np.kron(Z, I)
IX = np.kron(I, X)
IZ = np.kron(I, Z)


class TestDream1InverseSequence(unittest.TestCase):

    def run_inverse_check(
        self,
        labels,
        qubits,
        pulse_time,
        background_H,
        fractional_pulse_error,
        expected_decimal,
        expected_error_fwd,
        sequence,
        inverse_sequence,
    ):
        """Helper function to run the core logic of the inverse test."""
        nq = len(qubits)

        # Generate the forward sequence
        circuit_fwd = sequence(
            labels, qubits, pulse_time, background_H, fractional_pulse_error
        )
        U_fwd = circuit_fwd.unitary(qubit_order=qubits)
        assert np.abs(np.trace(U_fwd)) < 4 - expected_error_fwd

        # Generate the inverse sequence components
        circuit_inv_block, circuit_start, circuit_end = inverse_sequence(
            labels, qubits, pulse_time, background_H, fractional_pulse_error
        )

        # Construct the full inverse circuit: U_start then U_block then U_end
        # The unitary of this is U_end @ U_block @ U_start
        full_inverse_circuit = circuit_start + circuit_inv_block + circuit_end

        # The combined circuit (Forward @ FullInverse) should be Identity
        # Total circuit: C_fwd + C_start + C_block + C_end
        # Total unitary: U_end @ U_block @ U_start @ U_fwd
        total_circuit = full_inverse_circuit + circuit_fwd
        print(total_circuit)

        U_total = total_circuit.unitary(qubit_order=qubits)
        assert np.abs(np.trace(U_total)) > 4 - 10 ** (-expected_decimal)
        # print(U_total.round(3))

    def run_inverse_check_with_background(
        self,
        labels,
        qubits,
        pulse_time,
        background_H,
        fractional_pulse_error,
        expected_decimal,
        expected_error_fwd,
        sequence,
        inverse_sequence,
        z_angle,
    ):
        """Helper function to run the core logic of the inverse test."""
        nq = len(qubits)

        # Generate the forward sequence
        circuit_fwd = sequence(
            labels, qubits, pulse_time, background_H, fractional_pulse_error, z_angle
        )
        U_fwd = circuit_fwd.unitary(qubit_order=qubits)

        total_circuit = circuit_fwd
        print(total_circuit)

        U_total = total_circuit.unitary(qubit_order=qubits)
        # ---Check total unitary !== Identity when Background H applied---
        assert np.abs(np.trace(U_total)) < 4 - expected_error_fwd

        # Generate the inverse sequence components
        circuit_inv_block, circuit_start, circuit_end = inverse_sequence(
            labels, qubits, pulse_time, background_H, fractional_pulse_error, z_angle
        )

        # Construct the full inverse circuit: U_start then U_block then U_end
        # The unitary of this is U_end @ U_block @ U_start
        full_inverse_circuit = circuit_start + circuit_inv_block + circuit_end

        # The combined circuit (Forward @ FullInverse) should be Identity
        # Total circuit: C_fwd + C_start + C_block + C_end
        # Total unitary: U_end @ U_block @ U_start @ U_fwd
        total_circuit = full_inverse_circuit + circuit_fwd
        print(total_circuit)

        U_total = total_circuit.unitary(qubit_order=qubits)
        assert np.abs(np.trace(U_total)) > 4 - 10 ** (-expected_decimal)
        # print(U_total.round(3))

    def test_dream1_inverse_is_approximate_inverse_of_dream1_case_1(self):
        """Tests that dream_1_inverse_sequence_CH is an approximate inverse of dream_1_sequence_CH."""
        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(len(labels))
        pulse_time = 0.1

        # Case 1: Ideal conditions (zero error, zero background)
        self.run_inverse_check(
            labels,
            qubits,
            pulse_time,
            background_H=QubitOperator(),
            fractional_pulse_error=0.0,
            expected_decimal=2,
            expected_error_fwd=-1e-9,  # No background Hamiltonian, so this should be I.
            sequence=dream_1_sequence_CH,
            inverse_sequence=dream_1_inverse_sequence_CH,
        )

    def test_dream1_inverse_with_background_is_approximate_inverse_of_dream1_case_1(
        self,
    ):
        """Tests that dream_1_inverse_sequence_CH is an approximate inverse of dream_1_sequence_CH."""
        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(len(labels))
        pulse_time = 0.1

        # Case 1: Ideal conditions (zero error, zero background)
        self.run_inverse_check_with_background(
            labels,
            qubits,
            pulse_time,
            background_H=QubitOperator(),
            fractional_pulse_error=0.0,
            expected_decimal=2,
            expected_error_fwd=-1e-9,  # No background Hamiltonian, so this should be I.
            sequence=dream_1_sequence_CH_with_background,
            inverse_sequence=dream_1_inverse_sequence_CH_with_background,
            z_angle=0,
        )

    def test_dream1_inverse_with_background_03_is_inverse_of_dream1_with_background_03(
        self,
    ):
        """Tests that dream_1_sequence_CH_with_background is not identity when Background H applied."""
        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(len(labels))
        pulse_time = 0.05
        background_H_nonzero = QubitOperator("Z0 Z1")

        # Case 1: Ideal conditions (zero error, zero background)
        self.run_inverse_check_with_background(
            labels,
            qubits,
            pulse_time,
            background_H=background_H_nonzero,
            fractional_pulse_error=0.0,
            expected_decimal=1.5,
            expected_error_fwd=0.05,
            sequence=dream_1_sequence_CH_with_background,
            inverse_sequence=dream_1_inverse_sequence_CH_with_background,
            z_angle=0.3,
        )

    def test_dream1_inverse_new_with_background_03_is_inverse_of_dream1_with_background_03(
        self,
    ):
        """Tests that dream_1_sequence_CH_with_background is not identity when Background H applied."""
        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(len(labels))
        pulse_time = 0.05
        background_H_nonzero = QubitOperator("Z0 Z1")

        # Case 1: Ideal conditions (zero error, zero background)
        self.run_inverse_check_with_background(
            labels,
            qubits,
            pulse_time,
            background_H=background_H_nonzero,
            fractional_pulse_error=0.0,
            expected_decimal=1.5,
            expected_error_fwd=0.05,
            sequence=dream_1_sequence_CH_with_background,
            inverse_sequence=dream_1_inverse_sequence_CH_new_with_background,
            z_angle=0.2,
        )

    def test_dream1_inverse_is_approximate_inverse_of_dream1_case_2(self):
        # Case 2: With background Hamiltonian, no pulse error

        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(len(labels))
        pulse_time = 0.05

        background_H_nonzero = QubitOperator("Z0 Z1")
        self.run_inverse_check(
            labels,
            qubits,
            pulse_time,
            background_H=background_H_nonzero,
            fractional_pulse_error=0.0,
            expected_decimal=1.5,  # Expect less precision due to background
            expected_error_fwd=0.05,
            sequence=dream_1_sequence_CH,
            inverse_sequence=dream_1_inverse_sequence_CH,
        )

    def test_dream1_inverse_with_background_0_is_inverse_of_dream1_case2(
        self,
    ):
        """Tests that dream_1_sequence_CH_with_background is not identity when Background H applied."""
        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(len(labels))
        pulse_time = 0.05
        background_H_nonzero = QubitOperator("Z0 Z1")

        # Case 1: Ideal conditions (zero error, zero background)
        self.run_inverse_check_with_background(
            labels,
            qubits,
            pulse_time,
            background_H=background_H_nonzero,
            fractional_pulse_error=0.0,
            expected_decimal=1.5,
            expected_error_fwd=0.05,
            sequence=dream_1_sequence_CH_with_background,
            inverse_sequence=dream_1_inverse_sequence_CH_with_background,
            z_angle=0,
        )

    def test_dream1_with_error_and_background_is_not_identity(self):
        """Tests that dream_1_sequence_CH_with_background is not identity when Background H applied."""
        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(len(labels))
        pulse_time = 0.05
        background_H_nonzero = QubitOperator("Z0 Z1")

        # Case 1: Ideal conditions (zero error, zero background)
        self.run_inverse_check_with_background(
            labels,
            qubits,
            pulse_time,
            background_H=background_H_nonzero,
            fractional_pulse_error=0.01,
            expected_decimal=1.5,
            expected_error_fwd=0.05,
            sequence=dream_1_sequence_CH_with_background,
            inverse_sequence=dream_1_inverse_sequence_CH_with_background,
            z_angle=0,
        )

    def test_dream1_inverse_is_approximate_inverse_of_dream1_case_3(self):
        # Case 3: With pulse error, no background

        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(len(labels))
        pulse_time = 0.05
        self.run_inverse_check(
            labels,
            qubits,
            pulse_time,
            background_H=QubitOperator(),
            fractional_pulse_error=0.01,  # 1% pulse error
            expected_decimal=3,  # Expect less precision due to pulse error
            expected_error_fwd=0,
            sequence=dream_1_sequence_CH,
            inverse_sequence=dream_1_inverse_sequence_CH,
        )

    def test_dream1_inverse_is_approximate_inverse_of_dream1_case_4(self):
        # Case 4: With both background and pulse error
        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(len(labels))
        pulse_time = 0.05
        background_H_nonzero = QubitOperator("Z0 Z1")
        self.run_inverse_check(
            labels,
            qubits,
            pulse_time,
            background_H=background_H_nonzero,
            fractional_pulse_error=0.01,
            expected_decimal=1.5,  # Expect least precision
            expected_error_fwd=0.05,
            sequence=dream_1_sequence_CH,
            inverse_sequence=dream_1_inverse_sequence_CH,
        )

    def test_dream1_inverse_new_is_approximate_inverse_of_dream1_case_1(self):
        """Tests that dream_1_inverse_sequence_CH_new is an approximate inverse of dream_1_sequence_CH."""
        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(len(labels))
        pulse_time = 0.1

        # Case 1: Ideal conditions (zero error, zero background)
        self.run_inverse_check(
            labels,
            qubits,
            pulse_time,
            background_H=QubitOperator(),
            fractional_pulse_error=0.0,
            expected_decimal=6,
            expected_error_fwd=-1e-9,
            sequence=dream_1_sequence_CH,
            inverse_sequence=dream_1_inverse_sequence_CH_new,
        )

    def test_dream1_inverse_new_is_approximate_inverse_of_dream1_case_2(self):
        # Case 2: With background Hamiltonian, no pulse error

        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(len(labels))
        pulse_time = 0.05

        background_H_nonzero = QubitOperator("Z0 Z1")
        self.run_inverse_check(
            labels,
            qubits,
            pulse_time,
            background_H=background_H_nonzero,
            fractional_pulse_error=0.0,
            expected_decimal=1.5,  # Expect less precision due to background
            expected_error_fwd=0.05,
            sequence=dream_1_sequence_CH,
            inverse_sequence=dream_1_inverse_sequence_CH_new,
        )

    def test_dream2_inverse_is_approximate_inverse_of_dream2_case_1(self):
        """Tests that dream_1_inverse_sequence_CH is an approximate inverse of dream_1_sequence_CH."""
        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(len(labels))
        pulse_time = 0.1

        # Case 1: Ideal conditions (zero error, zero background)
        self.run_inverse_check(
            labels,
            qubits,
            pulse_time,
            background_H=QubitOperator(),
            fractional_pulse_error=0.0,
            expected_decimal=6,
            expected_error_fwd=-1e-9,
            sequence=dream_2_sequence_CH,
            inverse_sequence=dream_2_inverse_sequence_CH,
        )

    def test_dream2_with_background_is_not_identity_case_2(self):
        """Tests that dream_2_sequence_CH_with_background is not identity when Background H applied."""
        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(len(labels))
        pulse_time = 0.05
        background_H_nonzero = QubitOperator("Z0 Z1")

        # Case 1: Ideal conditions (zero error, zero background)
        self.run_inverse_check_with_background(
            labels,
            qubits,
            pulse_time,
            background_H=background_H_nonzero,
            fractional_pulse_error=0.0,
            expected_decimal=1.5,
            expected_error_fwd=0.05,
            sequence=dream_2_sequence_CH_with_background,
            inverse_sequence=dream_2_inverse_sequence_CH_with_background,
            z_angle=0,
        )

    def test_dream2_inverse_is_approximate_inverse_of_dream2_case_2(self):
        # Case 2: With background Hamiltonian, no pulse error

        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(len(labels))
        pulse_time = 0.05

        background_H_nonzero = QubitOperator("Z0 Z1")
        self.run_inverse_check(
            labels,
            qubits,
            pulse_time,
            background_H=background_H_nonzero,
            fractional_pulse_error=0.0,
            expected_decimal=1.5,  # Expect less precision due to background
            expected_error_fwd=0.05,
            sequence=dream_2_sequence_CH,
            inverse_sequence=dream_2_inverse_sequence_CH,
        )

    def test_dream2_inverse_is_approximate_inverse_of_dream2_case_3(self):
        # Case 3: With pulse error, no background

        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(len(labels))
        pulse_time = 0.05
        self.run_inverse_check(
            labels,
            qubits,
            pulse_time,
            background_H=QubitOperator(),
            fractional_pulse_error=0.01,  # 1% pulse error
            expected_decimal=2,  # Expect less precision due to pulse error
            expected_error_fwd=0,
            sequence=dream_2_sequence_CH,
            inverse_sequence=dream_2_inverse_sequence_CH,
        )

    def test_dream2_inverse_is_approximate_inverse_of_dream2_case_4(self):
        # Case 4: With both background and pulse error
        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(len(labels))
        pulse_time = 0.05
        background_H_nonzero = QubitOperator("Z0 Z1")
        self.run_inverse_check(
            labels,
            qubits,
            pulse_time,
            background_H=background_H_nonzero,
            fractional_pulse_error=0.01,
            expected_decimal=1,  # Expect least precision
            expected_error_fwd=0.05,
            sequence=dream_2_sequence_CH,
            inverse_sequence=dream_2_inverse_sequence_CH,
        )


class TestTimeEvolutionGate(unittest.TestCase):

    def test_make_circuit_get_unitary(self):
        qubits = cirq.LineQubit.range(2)
        hamiltonian = QubitOperator("Z0 Z1")
        dt = 1.28
        g1 = TimeEvolutionGate(hamiltonian, dt, 2).on(*qubits)
        g2 = TimeEvolutionGate(hamiltonian, dt * 2, 2).on(*qubits)
        circuit = cirq.Circuit([g1, g2])
        unitary = circuit.unitary(qubit_order=qubits)
        target_unitary = scipy.linalg.expm(
            -1j * dt * 3 * get_sparse_operator(hamiltonian).todense()
        )
        np.testing.assert_almost_equal(
            np.abs(np.trace(unitary.conj().T @ target_unitary)), 4
        )


class TestPulseWithTimeEvolution(unittest.TestCase):

    def test_zero_background_zero_pulse(self):
        """Test with zero background and zero pulse -> Identity."""
        H0 = np.zeros((2, 2))
        V = np.zeros((2, 2))
        tp = 1.0
        theta = 0.0
        expected_U = np.identity(2)
        actual_U = pulse_with_time_evolution(H0, V, tp, theta)
        np.testing.assert_array_almost_equal(actual_U, expected_U)

    def test_zero_background_simple_pulse(self):
        """Test with zero background and a simple X pulse."""
        H0 = np.zeros((2, 2))
        V = X
        tp = np.pi / 2
        theta = np.pi / 2  # Amplitude = 1
        # Expected: exp(-i * tp * (H0 + (theta/tp)*V)) = exp(-i * (pi/2) * X) = -iX
        expected_U = scipy.linalg.expm(-1j * (np.pi / 2) * X)
        actual_U = pulse_with_time_evolution(H0, V, tp, theta)
        np.testing.assert_array_almost_equal(actual_U, expected_U)

    def test_simple_background_zero_pulse(self):
        """Test with simple Z background and zero pulse."""
        H0 = Z
        V = np.zeros((2, 2))
        tp = np.pi
        theta = 0.0  # Amplitude = 0
        # Expected: exp(-i * tp * H0) = exp(-i * pi * Z)
        expected_U = scipy.linalg.expm(-1j * np.pi * Z)
        actual_U = pulse_with_time_evolution(H0, V, tp, theta)
        np.testing.assert_array_almost_equal(actual_U, expected_U)

    def test_background_and_pulse(self):
        """Test with both background and pulse active (commuting)."""
        H0 = Z
        V = Z
        tp = 0.5
        theta = np.pi / 4  # Amplitude = pi/2
        # Expected: exp(-i * tp * (H0 + (theta/tp)*V)) = exp(-i * 0.5 * (Z + (pi/2)*Z))
        # = exp(-i * 0.5 * (1 + pi/2) * Z)
        expected_H_eff = Z + (np.pi / 2) * Z
        expected_U = scipy.linalg.expm(-1j * tp * expected_H_eff)
        actual_U = pulse_with_time_evolution(H0, V, tp, theta)
        np.testing.assert_array_almost_equal(actual_U, expected_U)

    def test_background_and_pulse_non_commuting(self):
        """Test with non-commuting background and pulse."""
        H0 = Z
        V = X
        tp = 0.7
        theta = np.pi / 3  # Amplitude = pi / (3 * 0.7)
        amplitude = theta / tp
        expected_H_eff = H0 + amplitude * V
        expected_U = scipy.linalg.expm(-1j * tp * expected_H_eff)
        actual_U = pulse_with_time_evolution(H0, V, tp, theta)
        np.testing.assert_array_almost_equal(actual_U, expected_U)

    def test_two_qubit_case(self):
        """Test with 2-qubit operators."""
        H0 = ZZ
        V = XX
        tp = 0.2
        theta = np.pi / 5
        amplitude = theta / tp
        expected_H_eff = H0 + amplitude * V
        expected_U = scipy.linalg.expm(-1j * tp * expected_H_eff)
        actual_U = pulse_with_time_evolution(H0, V, tp, theta)
        np.testing.assert_array_almost_equal(actual_U, expected_U)

    def test_shape_mismatch(self):
        """Test that TypeError is raised for mismatched shapes."""
        H0 = np.zeros((4, 4))
        V = np.zeros((2, 2))
        with self.assertRaises(TypeError):
            pulse_with_time_evolution(H0, V, 1.0, 1.0)

    def test_very_small_pulse_time(self):
        """Test behavior with very small pulse time."""
        H0 = Z
        V = X
        tp = 1e-9
        theta = np.pi / 2
        amplitude = theta / tp
        expected_H_eff = H0 + amplitude * V
        expected_U = scipy.linalg.expm(-1j * tp * expected_H_eff)
        actual_U = pulse_with_time_evolution(H0, V, tp, theta)
        # The effective Hamiltonian is dominated by the pulse term
        # U approx exp(-i * tp * amp * V) = exp(-i * theta * V)
        expected_U_approx = scipy.linalg.expm(-1j * theta * V)
        # Check that the actual result is close to the exact calculation
        np.testing.assert_array_almost_equal(actual_U, expected_U)
        # Also check it's close to the approximation where H0 is ignored
        np.testing.assert_array_almost_equal(actual_U, expected_U_approx, decimal=5)


class TestMakePulsedSchemeCircuit(unittest.TestCase):

    def test_single_pulse_no_name(self):
        """Test circuit creation with a single pulse, no name."""
        qubits = cirq.LineQubit.range(1)
        H0 = QubitOperator("Z0")
        Vs = [QubitOperator("X0")]
        tps = [0.5]
        thetas = [np.pi / 4]

        circuit = make_pulsed_scheme_circuit(qubits, H0, Vs, tps, thetas)

        self.assertEqual(len(circuit), 1)
        op = list(circuit.all_operations())[0]

        # Check the matrix of the gate
        expected_U_pulse = pulse_with_time_evolution(Z, X, tps[0], thetas[0])
        actual_U_gate = cirq.unitary(op.gate)
        np.testing.assert_array_almost_equal(actual_U_gate, expected_U_pulse)
        self.assertEqual(op.qubits, tuple(qubits))

    def test_multiple_pulses(self):
        """Test circuit creation with multiple pulses."""
        qubits = cirq.LineQubit.range(2)
        H0 = QubitOperator("Z0 Z1")
        Vs = [QubitOperator("X0 X1"), QubitOperator("Z0")]
        tps = [0.2, 0.3]
        thetas = [np.pi / 5, np.pi / 6]
        names = ["XX_p", "ZI_p"]

        circuit = make_pulsed_scheme_circuit(qubits, H0, Vs, tps, thetas, names)

        self.assertEqual(len(circuit), 2)  # Two moments, one for each pulse gate

        # Calculate expected total unitary
        U1 = pulse_with_time_evolution(ZZ, XX, tps[0], thetas[0])
        U2 = pulse_with_time_evolution(ZZ, ZI, tps[1], thetas[1])
        # Circuit applies gates left-to-right: U_total = U2 @ U1
        expected_U_total = U2 @ U1

        # Get circuit unitary
        actual_U_total = circuit.unitary(qubit_order=qubits)

        np.testing.assert_array_almost_equal(actual_U_total, expected_U_total)

    def test_empty_pulse_list(self):
        """Test circuit creation with empty pulse lists."""
        qubits = cirq.LineQubit.range(1)
        H0 = QubitOperator("Z0")
        Vs = []
        tps = []
        thetas = []

        circuit = make_pulsed_scheme_circuit(qubits, H0, Vs, tps, thetas)
        self.assertEqual(len(circuit), 0)  # Empty circuit


class TestDream1Sequence(unittest.TestCase):

    def test_dream1_structure_and_length(self):
        """Test the basic structure and length of the DREAM-1 circuit."""
        label_list = ["H", "C"]
        qubits = cirq.LineQubit.range(2)
        pulse_time = 0.1
        background_H = QubitOperator()

        circuit = dream_1_sequence_CH(
            label_list, qubits, pulse_time, background_H, fractional_pulse_error=0
        )

        # DREAM-1 has 24 pulses
        self.assertEqual(len(circuit), 24)
        all_ops = list(circuit.all_operations())
        self.assertEqual(len(all_ops), 24)

        # Check that all gates are MatrixGates acting on the correct qubits
        for op in all_ops:
            self.assertIsInstance(op.gate, cirq.Gate)
            self.assertEqual(op.qubits, tuple(qubits))
            self.assertEqual(op.gate.num_qubits(), 2)

        # Test unitary generation
        unitary = circuit.unitary(qubit_order=qubits)

    def test_dream1_structure_and_length_with_background(self):
        """Test the basic structure and length of the DREAM-1 circuit."""
        label_list = ["H", "C"]
        qubits = cirq.LineQubit.range(2)
        pulse_time = 0.1
        background_H = QubitOperator()

        circuit = dream_1_sequence_CH_with_background(
            label_list, qubits, pulse_time, background_H, fractional_pulse_error=0
        )

        # DREAM-1 has 26 pulses
        self.assertEqual(len(circuit), 26)
        all_ops = list(circuit.all_operations())
        self.assertEqual(len(all_ops), 26)

        # Check that all gates are MatrixGates acting on the correct qubits
        for op in all_ops:
            self.assertIsInstance(op.gate, cirq.Gate)
            self.assertEqual(op.qubits, tuple(qubits))
            self.assertEqual(op.gate.num_qubits(), 2)

        # Test unitary generation
        unitary = circuit.unitary(qubit_order=qubits)

    def test_dream1_no_error_zero_background(self):
        """Test DREAM-1 with no pulse error and zero background Hamiltonian."""
        label_list = ["H", "C"]
        nq = len(label_list)
        qubits = cirq.LineQubit.range(nq)
        pulse_time = 0.05
        background_H = QubitOperator()
        fractional_pulse_error = 0

        # --- Manually calculate the expected unitary ---
        # Define generators as in the function
        XX_gen = np.zeros((2**nq, 2**nq), dtype=np.complex128)
        XXbar_gen = np.zeros((2**nq, 2**nq), dtype=np.complex128)
        YY_gen = np.zeros((2**nq, 2**nq), dtype=np.complex128)
        YYbar_gen = np.zeros((2**nq, 2**nq), dtype=np.complex128)
        for j, label in enumerate(label_list):
            X_big = np.kron(
                np.identity(2**j), np.kron(X, np.identity(2 ** (nq - j - 1)))
            )
            Y_big = np.kron(
                np.identity(2**j), np.kron(Y, np.identity(2 ** (nq - j - 1)))
            )
            XX_gen += X_big / 2
            YY_gen += Y_big / 2
            if label == "H":
                XXbar_gen += X_big / 2
                YYbar_gen += Y_big / 2
            elif label == "C":
                XXbar_gen -= X_big / 2
                YYbar_gen -= Y_big / 2

        Ra = [XX_gen, YYbar_gen, XX_gen]
        Rb = [XXbar_gen, YY_gen, XXbar_gen]

        pulse_generators = Ra + Ra + Rb + Rb + Rb + Rb + Ra + Ra
        pulse_times = [pulse_time] * 24
        pulse_angles = [np.pi / 2 * (1 + fractional_pulse_error)] * 12 + [
            -np.pi / 2 * (1 + fractional_pulse_error)
        ] * 12

        # Calculate expected unitary by multiplying individual pulse unitaries
        expected_U = np.identity(2**nq, dtype=complex)
        for V, t, theta in zip(pulse_generators, pulse_times, pulse_angles):
            U_step = pulse_with_time_evolution(
                np.zeros((4, 4), dtype=complex), V, t, theta
            )
            expected_U = (
                U_step @ expected_U
            )  # Apply in reverse order for matrix multiplication

        # --- Get the circuit from the function ---
        circuit = dream_1_sequence_CH(
            label_list, qubits, pulse_time, background_H, fractional_pulse_error
        )
        actual_U = circuit.unitary(qubit_order=qubits)

        # --- Compare ---
        # Compare up to a global phase
        phase_diff = np.angle(expected_U[0, 0] / actual_U[0, 0])
        np.testing.assert_array_almost_equal(
            actual_U * np.exp(1j * phase_diff), expected_U
        )


class TestDream2Sequence(unittest.TestCase):

    def test_dream2_structure_and_length(self):
        """Test the basic structure and length of the DREAM-2 circuit."""
        label_list = ["H", "C"]
        qubits = cirq.LineQubit.range(2)
        pulse_time = 0.1
        background_H = QubitOperator()

        circuit = dream_2_sequence_CH(
            label_list, qubits, pulse_time, background_H, fractional_pulse_error=0
        )

        # DREAM-2 has 36 pulses
        self.assertEqual(len(circuit), 36)
        all_ops = list(circuit.all_operations())
        self.assertEqual(len(all_ops), 36)

        # Check that all gates are MatrixGates acting on the correct qubits
        for op in all_ops:
            self.assertIsInstance(op.gate, cirq.Gate)
            self.assertEqual(op.qubits, tuple(qubits))
            self.assertEqual(op.gate.num_qubits(), 2)

    def test_dream2_structure_and_length_with_background(self):
        """Test the basic structure and length of the DREAM-2 circuit."""
        label_list = ["H", "C"]
        qubits = cirq.LineQubit.range(2)
        pulse_time = 0.1
        background_H = QubitOperator()

        circuit = dream_2_sequence_CH_with_background(
            label_list, qubits, pulse_time, background_H, fractional_pulse_error=0
        )

        # DREAM-2 has 39 pulses with background
        self.assertEqual(len(circuit), 39)
        all_ops = list(circuit.all_operations())
        self.assertEqual(len(all_ops), 39)

        # Check that all gates are MatrixGates acting on the correct qubits
        for op in all_ops:
            self.assertIsInstance(op.gate, cirq.Gate)
            self.assertEqual(op.qubits, tuple(qubits))
            self.assertEqual(op.gate.num_qubits(), 2)

    def test_dream2_no_error_zero_background(self):
        """Test DREAM-2 with no pulse error and zero background Hamiltonian."""
        label_list = ["H", "C"]
        nq = len(label_list)
        qubits = cirq.LineQubit.range(nq)
        pulse_time = 0.05
        background_H = QubitOperator()
        fractional_pulse_error = 0

        # --- Manually calculate the expected unitary ---
        # Define generators as in the function
        II_gen, XX_gen, XI_gen, IY_gen, XYbar_gen, IXbar_gen, ZI_gen = [
            get_sparse_operator(op, n_qubits=2).todense()
            for op in make_pulse_generators_DREAM_2(label_list)
        ]

        # Construct subsequences (as defined in dream_2_sequence_CH)
        Pines = [XI_gen, II_gen, XI_gen] * 2
        DR = [XX_gen, IY_gen, XX_gen, XYbar_gen, IXbar_gen, XYbar_gen]

        pulse_generators = Pines + Pines + DR + DR + Pines + Pines
        pulse_times = [pulse_time] * 36
        pulse_angles = (
            [np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [-np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [-np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [-np.pi / 2 * (1 + fractional_pulse_error)] * 6
        )

        # Calculate expected unitary by multiplying individual pulse unitaries
        expected_U = np.identity(2**nq, dtype=complex)
        for V, t, theta in zip(pulse_generators, pulse_times, pulse_angles):
            U_step = pulse_with_time_evolution(np.zeros((4, 4)), V, t, theta)
            expected_U = U_step @ expected_U  # Apply in reverse order

        # --- Get the circuit from the function ---
        circuit = dream_2_sequence_CH(
            label_list, qubits, pulse_time, background_H, fractional_pulse_error
        )
        actual_U = circuit.unitary(qubit_order=qubits)

        # --- Compare ---
        # Compare up to a global phase
        phase_diff = np.angle(expected_U[0, 0] / actual_U[0, 0])
        np.testing.assert_array_almost_equal(
            actual_U * np.exp(1j * phase_diff), expected_U
        )

    def test_dream2_with_error_and_background(self):
        """Test DREAM-2 with pulse error and a non-zero background Hamiltonian."""
        label_list = ["H", "C"]
        nq = len(label_list)
        qubits = cirq.LineQubit.range(nq)
        pulse_time = 0.07
        background_H = 0.4 * QubitOperator("Z0 Z1") + 0.1 * QubitOperator("Z0")
        fractional_pulse_error = -0.02  # -2% error

        # --- Manually calculate the expected unitary ---
        II_gen, XX_gen, XI_gen, IY_gen, XYbar_gen, IXbar_gen, ZI_gen = [
            get_sparse_operator(op, n_qubits=2).todense()
            for op in make_pulse_generators_DREAM_2(label_list)
        ]
        Pines = [XI_gen, II_gen, XI_gen] * 2
        DR = [XX_gen, IY_gen, XX_gen, XYbar_gen, IXbar_gen, XYbar_gen]

        pulse_generators = Pines + Pines + DR + DR + Pines + Pines
        pulse_times = [pulse_time] * 36
        pulse_angles = (
            [np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [-np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [-np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [-np.pi / 2 * (1 + fractional_pulse_error)] * 6
        )

        expected_U = np.identity(2**nq, dtype=complex)
        for V, t, theta in zip(pulse_generators, pulse_times, pulse_angles):
            U_step = pulse_with_time_evolution(0.4 * ZZ + 0.1 * ZI, V, t, theta)
            expected_U = U_step @ expected_U  # Apply in reverse order

        # --- Get the circuit from the function ---
        circuit = dream_2_sequence_CH(
            label_list, qubits, pulse_time, background_H, fractional_pulse_error
        )
        actual_U = circuit.unitary(qubit_order=qubits)

        # --- Compare ---
        phase_diff = np.angle(expected_U[0, 0] / actual_U[0, 0])
        np.testing.assert_array_almost_equal(
            actual_U * np.exp(1j * phase_diff), expected_U
        )

    def test_dream2_with_error_and_background(self):
        """Test DREAM-2 with pulse error and a non-zero background Hamiltonian."""
        label_list = ["H", "C"]
        nq = len(label_list)
        qubits = cirq.LineQubit.range(nq)
        pulse_time = 0.07
        background_H = 0.4 * QubitOperator("Z0 Z1") + 0.1 * QubitOperator("Z0")
        fractional_pulse_error = -0.02  # -2% error

        # --- Get the circuit from the function ---
        circuit = dream_2_sequence_CH_with_background(
            label_list, qubits, pulse_time, background_H, fractional_pulse_error
        )
        actual_U = circuit.unitary(qubit_order=qubits)
        assert actual_U.shape == (4, 4)


class TestDream2InverseSequence(unittest.TestCase):

    def test_dream2_inverse_structure_and_length(self):
        """Test the basic structure and length of the DREAM-2 inverse circuit."""
        label_list = ["H", "C"]
        qubits = cirq.LineQubit.range(2)
        pulse_time = 0.1
        background_H = QubitOperator()

        unitary_block, start_pulse, end_pulse = dream_2_inverse_sequence_CH(
            label_list, qubits, pulse_time, background_H, fractional_pulse_error=0
        )

        # Main block has 36 pulses
        self.assertEqual(len(unitary_block), 36)
        all_ops_main = list(unitary_block.all_operations())
        self.assertEqual(len(all_ops_main), 36)
        for op in all_ops_main:
            self.assertIsInstance(op.gate, cirq.Gate)
            self.assertEqual(op.qubits, tuple(qubits))
            self.assertEqual(op.gate.num_qubits(), 2)

        # Start pulse has 1 pulse
        self.assertEqual(len(start_pulse), 1)
        op_start = list(start_pulse.all_operations())[0]
        self.assertIsInstance(op_start.gate, cirq.Gate)
        self.assertEqual(op_start.qubits, tuple(qubits))

        # End pulse has 1 pulse
        self.assertEqual(len(end_pulse), 1)
        op_end = list(end_pulse.all_operations())[0]
        self.assertIsInstance(op_end.gate, cirq.Gate)
        self.assertEqual(op_end.qubits, tuple(qubits))

    def test_dream2_inverse_structure_and_length(self):
        """Test the basic structure and length of the DREAM-2 inverse circuit."""
        label_list = ["H", "C"]
        qubits = cirq.LineQubit.range(2)
        pulse_time = 0.1
        background_H = QubitOperator()

        unitary_block, start_pulse, end_pulse = (
            dream_2_inverse_sequence_CH_with_background(
                label_list, qubits, pulse_time, background_H, fractional_pulse_error=0
            )
        )

        # Main block has 39 pulses
        self.assertEqual(len(unitary_block), 39)
        all_ops_main = list(unitary_block.all_operations())
        self.assertEqual(len(all_ops_main), 39)
        for op in all_ops_main:
            self.assertIsInstance(op.gate, cirq.Gate)
            self.assertEqual(op.qubits, tuple(qubits))
            self.assertEqual(op.gate.num_qubits(), 2)

        # Start pulse has 1 pulse
        self.assertEqual(len(start_pulse), 1)
        op_start = list(start_pulse.all_operations())[0]
        self.assertIsInstance(op_start.gate, cirq.Gate)
        self.assertEqual(op_start.qubits, tuple(qubits))

        # End pulse has 1 pulse
        self.assertEqual(len(end_pulse), 1)
        op_end = list(end_pulse.all_operations())[0]
        self.assertIsInstance(op_end.gate, cirq.Gate)
        self.assertEqual(op_end.qubits, tuple(qubits))

    def test_dream2_inverse_no_error_zero_background(self):
        """Test DREAM-2 inverse with no pulse error and zero background."""
        label_list = ["H", "C"]
        nq = len(label_list)
        qubits = cirq.LineQubit.range(nq)
        pulse_time = 0.06
        background_H = QubitOperator()
        fractional_pulse_error = 0

        # --- Manually calculate the expected unitaries ---
        II_gen, YX_gen, YI_gen, IY_gen, YYbar_gen, IXbar_gen, ZI_gen = [
            get_sparse_operator(op, n_qubits=2).todense()
            for op in make_pulse_generators_DREAM_2_inverse(label_list)
        ]

        # Main Block
        Pines = [YI_gen, II_gen, YI_gen] * 2
        DR = [YX_gen, IY_gen, YX_gen, YYbar_gen, IXbar_gen, YYbar_gen]
        pulse_generators = Pines + Pines + DR + DR + Pines + Pines
        pulse_times = [pulse_time] * 36
        pulse_angles = (
            [np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [-np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [-np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [-np.pi / 2 * (1 + fractional_pulse_error)] * 6
        )

        expected_U_main = np.identity(2**nq, dtype=complex)
        for V, t, theta in zip(pulse_generators, pulse_times, pulse_angles):
            U_step = pulse_with_time_evolution(np.zeros((4, 4)), V, t, theta)
            expected_U_main = U_step @ expected_U_main

        # Start Pulse
        expected_U_start = pulse_with_time_evolution(
            np.zeros((4, 4)), IY_gen, pulse_time, np.pi * (1 + fractional_pulse_error)
        )

        # End Pulse
        expected_U_end = pulse_with_time_evolution(
            np.zeros((4, 4)), IY_gen, pulse_time, -np.pi * (1 + fractional_pulse_error)
        )

        # --- Get the circuits from the function ---
        actual_circuit_main, actual_circuit_start, actual_circuit_end = (
            dream_2_inverse_sequence_CH(
                label_list, qubits, pulse_time, background_H, fractional_pulse_error
            )
        )
        actual_U_main = actual_circuit_main.unitary(qubit_order=qubits)
        actual_U_start = actual_circuit_start.unitary(qubit_order=qubits)
        actual_U_end = actual_circuit_end.unitary(qubit_order=qubits)

        print(actual_U_start)
        print(expected_U_start)

        # --- Compare ---
        phase_diff_main = np.angle(expected_U_main[0, 0] / actual_U_main[0, 0])
        np.testing.assert_array_almost_equal(
            actual_U_main * np.exp(1j * phase_diff_main), expected_U_main
        )
        np.testing.assert_almost_equal(
            np.abs(np.trace(expected_U_start.conj().T @ actual_U_start)),
            expected_U_start.shape[0],
        )

        np.testing.assert_almost_equal(
            np.abs(np.trace(expected_U_end.conj().T @ actual_U_end)),
            expected_U_end.shape[0],
        )

    def test_dream2_inverse_with_error_and_background(self):
        """Test DREAM-2 inverse with pulse error and non-zero background."""
        label_list = ["H", "C"]
        nq = len(label_list)
        qubits = cirq.LineQubit.range(nq)
        pulse_time = 0.1
        background_H_mat = 0.2 * ZZ - 0.3 * IZ  # Example background
        background_H = 0.2 * QubitOperator("Z0 Z1") - 0.3 * QubitOperator("Z1")
        fractional_pulse_error = 0.03  # 3% error

        # --- Manually calculate the expected unitaries ---
        II_gen, YX_gen, YI_gen, IY_gen, YYbar_gen, IXbar_gen, ZI_gen = [
            get_sparse_operator(op, n_qubits=2).todense()
            for op in make_pulse_generators_DREAM_2_inverse(label_list)
        ]

        # Main Block
        Pines = [YI_gen, II_gen, YI_gen] * 2
        DR = [YX_gen, IY_gen, YX_gen, YYbar_gen, IXbar_gen, YYbar_gen]
        pulse_generators = Pines + Pines + DR + DR + Pines + Pines
        pulse_times = [pulse_time] * 36
        pulse_angles = (
            [np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [-np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [-np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [np.pi / 2 * (1 + fractional_pulse_error)] * 6
            + [-np.pi / 2 * (1 + fractional_pulse_error)] * 6
        )

        expected_U_main = np.identity(2**nq, dtype=complex)
        for V, t, theta in zip(pulse_generators, pulse_times, pulse_angles):
            U_step = pulse_with_time_evolution(background_H_mat, V, t, theta)
            expected_U_main = U_step @ expected_U_main

        # Start Pulse
        expected_U_start = pulse_with_time_evolution(
            background_H_mat,
            IY_gen,
            2 * pulse_time,
            np.pi * (1 + fractional_pulse_error),
        )

        # End Pulse
        expected_U_end = pulse_with_time_evolution(
            background_H_mat,
            IY_gen,
            2 * pulse_time,
            -np.pi * (1 + fractional_pulse_error),
        )

        # --- Get the circuits from the function ---
        actual_circuit_main, actual_circuit_start, actual_circuit_end = (
            dream_2_inverse_sequence_CH(
                label_list, qubits, pulse_time, background_H, fractional_pulse_error
            )
        )
        actual_U_main = actual_circuit_main.unitary(qubit_order=qubits)
        actual_U_start = actual_circuit_start.unitary(qubit_order=qubits)
        actual_U_end = actual_circuit_end.unitary(qubit_order=qubits)

        # --- Compare ---
        phase_diff_main = np.angle(expected_U_main[0, 0] / actual_U_main[0, 0])
        np.testing.assert_array_almost_equal(
            actual_U_main * np.exp(1j * phase_diff_main), expected_U_main
        )

        np.testing.assert_almost_equal(
            np.abs(np.trace(expected_U_start.conj().T @ actual_U_start)),
            expected_U_start.shape[0],
        )

        np.testing.assert_almost_equal(
            np.abs(np.trace(expected_U_end.conj().T @ actual_U_end)),
            expected_U_end.shape[0],
        )


def test_approximate_ham_dream_2_tiny_dt():
    _test_approximate_ham_dream_2(2, 1e-6, 1e-6, 4.5e-2)
    _test_approximate_ham_dream_2(4, 1e-6, 1e-6, 4.5e-2)
    _test_approximate_ham_dream_2(5, 1e-6, 1e-6, 0.25)


def test_approximate_ham_dream_2_small_dt_small_N_small_CH():
    _test_approximate_ham_dream_2(2, 0.001, 0.001, CH_coupling=-100)


def test_approximate_ham_dream_2_small_dt_small_N():
    _test_approximate_ham_dream_2(2, 0.001, 0.005)


def test_approximate_ham_dream_2_small_dt_medium_N_small_CH():
    _test_approximate_ham_dream_2(3, 0.001, 0.001, 0.045, CH_coupling=0)
    _test_approximate_ham_dream_2(5, 0.001, 0.001, 0.25, CH_coupling=-100)


def test_approximate_ham_dream_2_small_dt_medium_N():
    _test_approximate_ham_dream_2(4, 0.001, 0.005, 0.045)
    _test_approximate_ham_dream_2(5, 0.001, 0.005, 0.25)


def test_approximate_ham_dream_2_large_dt_small_N_small_CH():
    _test_approximate_ham_dream_2(2, 0.01, 0.01, CH_coupling=-10)


def test_approximate_ham_dream_2_large_dt_small_N():
    _test_approximate_ham_dream_2(2, 0.01, 0.3)


def test_approximate_ham_dream_2_large_dt_medium_N_small_CH():
    _test_approximate_ham_dream_2(3, 0.01, 0.01, 0.045, CH_coupling=0)
    _test_approximate_ham_dream_2(5, 0.01, 0.01, 0.25, CH_coupling=-100)


def test_approximate_ham_dream_2_large_dt_medium_N():
    _test_approximate_ham_dream_2(4, 0.01, 0.3)
    _test_approximate_ham_dream_2(5, 0.01, 0.3)


def test_approximate_ham_dream_2_tiny_dt_with_background():
    _test_approximate_ham_dream_2(2, 1e-6, 1e-6, 4.5e-2, z_coupling=0.1)
    _test_approximate_ham_dream_2(4, 1e-6, 1e-6, 4.5e-2, z_coupling=0.1)
    _test_approximate_ham_dream_2(5, 1e-6, 1e-6, 0.25, z_coupling=0.1)


def test_approximate_ham_dream_2_tiny_dt_with_large_background():
    _test_approximate_ham_dream_2(2, 1e-6, 1e-6, 4.5e-2, z_coupling=1)
    _test_approximate_ham_dream_2(4, 1e-6, 1e-6, 4.5e-2, z_coupling=1)
    _test_approximate_ham_dream_2(5, 1e-6, 1e-6, 0.25, z_coupling=1)


def _test_approximate_ham_dream_2(
    nq,
    pulse_time,
    target_error,
    target_inverse_error=None,
    CH_coupling=None,
    z_coupling=0,
):
    # Using the real toluene data to make things easier
    # H-H coupling matrix (would contain C-C couplings too but there are none)
    hom = (
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -242.4, -242.4, -196.7, -196.7, -100.9, -100.9, -100.9],
                [0, -242.4, 0, 26.6, -1412.4, -30.6, -126.9, -126.9, -126.9],
                [0, -242.4, 26.6, 0, -30.6, -1412.4, -126.9, -126.9, -126.9],
                [0, -196.7, -1412.4, -30.6, 0, 27.3, -405.2, -405.2, -405.2],
                [0, -196.7, -30.6, -1412.4, 27.3, 0, -405.2, -405.2, -405.2],
                [0, -100.9, -126.9, -126.9, -405.2, -405.2, 0, 1860.2, 1860.2],
                [0, -100.9, -126.9, -126.9, -405.2, -405.2, 1860.2, 0, 1860.2],
                [0, -100.9, -126.9, -126.9, -405.2, -405.2, 1860.2, 1860.2, 0],
            ]
        )
        / 1000  # Convert to KHz
    )

    # C-H coupling matrix
    if CH_coupling is None:
        CH_coupling = -3841.6
    if z_coupling > 0:
        z_vector = np.array([0] + [z_coupling] * 8)
    else:
        z_vector = None
    het = np.zeros((9, 9))
    het[0, :] = (
        np.array([0, CH_coupling, 50.6, 50.6, -73.9, -73.9, -44.6, -44.6, -44.6]) / 1000
    )  # Convert to KHz
    het += het.T
    hom = hom[:nq, :nq]
    het = het[:nq, :nq]
    ham = construct_nmr_hamiltonian_of(hom / 4, het / 4, z_vector=z_vector) * 2 * np.pi
    print(ham)

    target_inverse_error = target_inverse_error or target_error

    labels = ["C"] + (["H"] * (nq - 1))
    qubits = cirq.LineQubit.range(nq)
    if z_coupling > 0:
        z_angle = z_coupling * (2 * np.pi * pulse_time) / 3
        circuit = dream_2_sequence_CH_with_background(
            labels=labels,
            qubits=qubits,
            pulse_time=pulse_time,
            background_H=ham,
            fractional_pulse_error=0,
            z_angle=z_angle,
        )
    else:
        circuit = dream_2_sequence_CH(
            labels=labels,
            qubits=qubits,
            pulse_time=pulse_time,
            background_H=ham,
            fractional_pulse_error=0,
        )
    unitary_block = circuit.unitary(qubit_order=qubits)
    if unitary_block.shape[0] < 10:
        print(unitary_block.round(3))
    unitary_block = unitary_block * np.exp(
        -1j * np.angle(np.sum(np.diag(unitary_block)))
    )
    block_time = pulse_time * 36
    gen = 1j * scipy.linalg.logm(unitary_block) / block_time

    homogeneous_coupling_const = np.pi / 2
    heterogeneous_coupling_const = np.pi / 9
    target_hop = construct_dq_hamiltonian_of(
        hom=hom * homogeneous_coupling_const,
        het=het * heterogeneous_coupling_const,
        xyyx_flag=False,
    )

    gen_op = matrix_to_qubitoperator(gen, tol=1e-3, max_k=2)
    print("Target H | Obtained H")
    for key in target_hop.terms:
        print(key, target_hop.terms[key], np.real(gen_op.terms[key]))
    for key in target_hop.terms:
        assert abs(target_hop.terms[key] - gen_op.terms[key]) < target_error
    print()

    circuit, start_pulse, end_pulse = dream_2_inverse_sequence_CH(
        labels=labels,
        qubits=qubits,
        pulse_time=pulse_time,
        background_H=ham,
        fractional_pulse_error=0,
    )

    num_blocks = 1
    inverse_unitary_block = cirq.Circuit(
        [start_pulse, [circuit for k in range(num_blocks)], end_pulse]
    ).unitary(qubit_order=qubits)
    inverse_unitary_block = inverse_unitary_block * np.exp(
        -1j * np.angle(np.sum(np.diag(inverse_unitary_block)))
    )

    target_hop = -target_hop

    gen = 1j * scipy.linalg.logm(inverse_unitary_block) / (block_time * num_blocks)
    gen_op = matrix_to_qubitoperator(gen, tol=1e-3, max_k=2)
    print("Target H inverse | Obtained H inverse")
    for key in target_hop.terms:
        print(key, target_hop.terms[key], np.real(gen_op.terms[key]))
    for key in target_hop.terms:
        assert abs(target_hop.terms[key] - gen_op.terms[key]) < target_inverse_error
    print()


def test_approximate_ham_dream_1_tiny_dt():
    _test_approximate_ham_dream_1(2, 1e-5, 1e-5, 1e-3)
    _test_approximate_ham_dream_1(4, 1e-5, 1e-5, 1e-3)
    _test_approximate_ham_dream_1(5, 1e-5, 1e-5, 1e-3)


def test_approximate_ham_dream_1_small_dt_small_N():
    _test_approximate_ham_dream_1(2, 0.001, 0.001, CH_coupling=-100)


def test_approximate_ham_dream_1_small_dt_small_N():
    _test_approximate_ham_dream_1(2, 0.001, 0.001)


def test_approximate_ham_dream_1_small_dt_medium_N_small_CH():
    _test_approximate_ham_dream_1(3, 0.001, 0.001, 0.01, CH_coupling=0)
    _test_approximate_ham_dream_1(5, 0.001, 0.001, 0.01, CH_coupling=-100)


def test_approximate_ham_dream_1_small_dt_medium_N():
    _test_approximate_ham_dream_1(4, 0.001, 0.001)
    _test_approximate_ham_dream_1(5, 0.001, 0.001)


def test_approximate_ham_dream_1_large_dt_small_N_small_CH():
    _test_approximate_ham_dream_1(2, 0.01, 0.01, CH_coupling=-100)


def test_approximate_ham_dream_1_large_dt_small_N():
    _test_approximate_ham_dream_1(2, 0.01, 0.07)


def test_approximate_ham_dream_1_large_dt_medium_N_small_CH():
    _test_approximate_ham_dream_1(3, 0.01, 0.01, CH_coupling=0)
    _test_approximate_ham_dream_1(5, 0.01, 0.01, CH_coupling=-100)


def test_approximate_ham_dream_1_large_dt_medium_N():
    _test_approximate_ham_dream_1(4, 0.01, 0.07)
    _test_approximate_ham_dream_1(5, 0.01, 0.07)


def test_approximate_ham_dream_1_tiny_dt_z_coupling():
    _test_approximate_ham_dream_1(2, 1e-5, 1e-5, 1e-3, z_coupling=0.1)
    _test_approximate_ham_dream_1(4, 1e-5, 1e-5, 1e-3, z_coupling=0.1)
    _test_approximate_ham_dream_1(5, 1e-5, 1e-5, 1e-3, z_coupling=0.1)


def test_approximate_ham_dream_1_tiny_dt_large_z_coupling():
    _test_approximate_ham_dream_1(2, 1e-5, 1e-5, 1e-3, z_coupling=1)
    _test_approximate_ham_dream_1(4, 1e-5, 1e-5, 1e-3, z_coupling=1)
    _test_approximate_ham_dream_1(5, 1e-5, 1e-5, 1e-3, z_coupling=1)


def _test_approximate_ham_dream_1(
    nq,
    pulse_time,
    target_error,
    target_inverse_error=None,
    CH_coupling=None,
    z_coupling=0,
):
    # Using the real toluene data to make things easier
    # H-H coupling matrix (would contain C-C couplings too but there are none)
    hom = (
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -242.4, -242.4, -196.7, -196.7, -100.9, -100.9, -100.9],
                [0, -242.4, 0, 26.6, -1412.4, -30.6, -126.9, -126.9, -126.9],
                [0, -242.4, 26.6, 0, -30.6, -1412.4, -126.9, -126.9, -126.9],
                [0, -196.7, -1412.4, -30.6, 0, 27.3, -405.2, -405.2, -405.2],
                [0, -196.7, -30.6, -1412.4, 27.3, 0, -405.2, -405.2, -405.2],
                [0, -100.9, -126.9, -126.9, -405.2, -405.2, 0, 1860.2, 1860.2],
                [0, -100.9, -126.9, -126.9, -405.2, -405.2, 1860.2, 0, 1860.2],
                [0, -100.9, -126.9, -126.9, -405.2, -405.2, 1860.2, 1860.2, 0],
            ]
        )
        / 1000  # Convert to KHz
    )

    # C-H coupling matrix
    if CH_coupling is None:
        CH_coupling = -3841.6
    het = np.zeros((9, 9))
    het[0, :] = (
        np.array([0, CH_coupling, 50.6, 50.6, -73.9, -73.9, -44.6, -44.6, -44.6]) / 1000
    )  # Convert to KHz
    het += het.T
    hom = hom[:nq, :nq]
    het = het[:nq, :nq]
    if z_coupling > 0:
        z_vector = np.array([0] + [z_coupling] * 8)
    else:
        z_vector = None
    ham = construct_nmr_hamiltonian_of(hom / 4, het / 4, z_vector) * 2 * np.pi

    target_inverse_error = target_inverse_error or target_error

    labels = ["C"] + (["H"] * (nq - 1))
    qubits = cirq.LineQubit.range(nq)
    if z_coupling > 0:
        z_angle = z_coupling * (2 * np.pi * pulse_time) / 2
        circuit = dream_1_sequence_CH_with_background(
            labels=labels,
            qubits=qubits,
            pulse_time=pulse_time,
            background_H=ham,
            fractional_pulse_error=0,
            z_angle=z_angle,
        )
    else:
        circuit = dream_1_sequence_CH(
            labels=labels,
            qubits=qubits,
            pulse_time=pulse_time,
            background_H=ham,
            fractional_pulse_error=0,
        )
    unitary_block = circuit.unitary(qubit_order=qubits)
    print(unitary_block.round(3))
    block_time = pulse_time * 24
    gen = 1j * scipy.linalg.logm(unitary_block) / block_time

    homogeneous_coupling_const = 1 / 2
    heterogeneous_coupling_const = np.pi / 3
    target_hop = construct_dq_hamiltonian_of(
        hom=hom * homogeneous_coupling_const,
        het=het * heterogeneous_coupling_const,
        xyyx_flag=True,
    )

    gen_op = matrix_to_qubitoperator(gen, tol=1e-3, max_k=2)
    print("Target H | Obtained H")
    for key in target_hop.terms:
        print(key, target_hop.terms[key], np.real(gen_op.terms[key]))
    for key in target_hop.terms:
        assert abs(target_hop.terms[key] - gen_op.terms[key]) < target_error
    print()

    circuit, start_pulse, end_pulse = dream_1_inverse_sequence_CH(
        labels=labels,
        qubits=qubits,
        pulse_time=pulse_time,
        background_H=ham,
        fractional_pulse_error=0,
    )

    num_blocks = 1
    inverse_unitary_block = cirq.Circuit(
        [start_pulse, [circuit for _ in range(num_blocks)], end_pulse]
    ).unitary(qubit_order=qubits)

    target_hop = -target_hop

    gen = 1j * scipy.linalg.logm(inverse_unitary_block) / (block_time * num_blocks)
    gen_op = matrix_to_qubitoperator(gen, tol=1e-3, max_k=2)
    print("op | Target H inverse | Obtained H inverse | Background H")
    for key in target_hop.terms:
        if key in ham.terms:
            print(
                key, target_hop.terms[key], np.real(gen_op.terms[key]), ham.terms[key]
            )
        else:
            print(key, target_hop.terms[key], 0, np.real(gen_op.terms[key]))
    for key in target_hop.terms:
        assert abs(target_hop.terms[key] - gen_op.terms[key]) < target_inverse_error
    print()

    circuit, start_pulse, end_pulse = dream_1_inverse_sequence_CH_new(
        labels=labels,
        qubits=qubits,
        pulse_time=pulse_time,
        background_H=ham,
        fractional_pulse_error=0,
    )
    inverse_unitary_block = cirq.Circuit(
        [start_pulse, [circuit for k in range(num_blocks)], end_pulse]
    ).unitary(qubit_order=qubits)

    gen = 1j * scipy.linalg.logm(inverse_unitary_block) / (block_time * num_blocks)
    gen_op = matrix_to_qubitoperator(gen, tol=1e-3, max_k=2)
    print("Target H inverse | Obtained H inverse")
    for key in target_hop.terms:
        print(key, target_hop.terms[key], np.real(gen_op.terms[key]))
    for key in target_hop.terms:
        assert abs(target_hop.terms[key] - gen_op.terms[key]) < target_inverse_error
    print()


class TestFractionalPulseErrorC(unittest.TestCase):
    def test_fractional_pulse_error_C_different_circuits(self):
        """Test that setting fractional_pulse_error_C to a non-zero value produces different circuits."""
        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(2)
        pulse_time = 0.1
        background_H = QubitOperator()

        # Case 1: fractional_pulse_error_C is not set (implicitly equal to fractional_pulse_error)
        circuit1 = dream_1_sequence_CH(
            labels=labels,
            qubits=qubits,
            pulse_time=pulse_time,
            background_H=background_H,
            fractional_pulse_error=0.01,
        )

        # Case 2: fractional_pulse_error_C is set to a different value
        circuit2 = dream_1_sequence_CH(
            labels=labels,
            qubits=qubits,
            pulse_time=pulse_time,
            background_H=background_H,
            fractional_pulse_error=0.01,
            fractional_pulse_error_C=0.02,
        )

        # Ensure that the circuits are different
        self.assertNotEqual(circuit1, circuit2)

    def test_fractional_pulse_error_C_different_circuits_inverse_sequence(self):
        """Test that setting fractional_pulse_error_C to a non-zero value produces different inverse circuits."""
        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(2)
        pulse_time = 0.1
        background_H = QubitOperator()

        # Case 1: fractional_pulse_error_C is not set (implicitly equal to fractional_pulse_error)
        circuit1, _, _ = dream_1_inverse_sequence_CH(
            labels=labels,
            qubits=qubits,
            pulse_time=pulse_time,
            background_H=background_H,
            fractional_pulse_error=0.01,
        )

        # Case 2: fractional_pulse_error_C is set to a different value
        circuit2, _, _ = dream_1_inverse_sequence_CH(
            labels=labels,
            qubits=qubits,
            pulse_time=pulse_time,
            background_H=background_H,
            fractional_pulse_error=0.01,
            fractional_pulse_error_C=0.02,
        )

        # Ensure that the circuits are different
        self.assertNotEqual(circuit1, circuit2)

    def test_dream2_fractional_pulse_error_C_different_circuits(self):
        """Test that setting fractional_pulse_error_C to a non-zero value produces different DREAM2 circuits."""
        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(2)
        pulse_time = 0.1
        background_H = QubitOperator()

        # Case 1: fractional_pulse_error_C is not set (implicitly equal to fractional_pulse_error)
        circuit1 = dream_2_sequence_CH(
            labels=labels,
            qubits=qubits,
            pulse_time=pulse_time,
            background_H=background_H,
            fractional_pulse_error=0.01,
        )

        # Case 2: fractional_pulse_error_C is set to a different value
        circuit2 = dream_2_sequence_CH(
            labels=labels,
            qubits=qubits,
            pulse_time=pulse_time,
            background_H=background_H,
            fractional_pulse_error=0.01,
            fractional_pulse_error_C=0.02,
        )

        # Ensure that the circuits are different
        self.assertNotEqual(circuit1, circuit2)

    def test_dream2_fractional_pulse_error_C_different_circuits_inverse_sequence(self):
        """Test that setting fractional_pulse_error_C to a non-zero value produces different inverse DREAM2 circuits."""
        labels = ["H", "C"]
        qubits = cirq.LineQubit.range(2)
        pulse_time = 0.1
        background_H = QubitOperator()

        # Case 1: fractional_pulse_error_C is not set (implicitly equal to fractional_pulse_error)
        circuit1, _, _ = dream_2_inverse_sequence_CH(
            labels=labels,
            qubits=qubits,
            pulse_time=pulse_time,
            background_H=background_H,
            fractional_pulse_error=0.01,
        )

        # Case 2: fractional_pulse_error_C is set to a different value
        circuit2, _, _ = dream_2_inverse_sequence_CH(
            labels=labels,
            qubits=qubits,
            pulse_time=pulse_time,
            background_H=background_H,
            fractional_pulse_error=0.01,
            fractional_pulse_error_C=0.02,
        )

        # Ensure that the circuits are different
        self.assertNotEqual(circuit1, circuit2)


class TestTimeEvolutionGateDecompose(unittest.TestCase):
    def test_decompose_converges_for_small_dt_and_large_ntrotter(self):
        """Tests that the _decompose_ method of TimeEvolutionGate works
        correctly for a small dt and sufficiently large n_trotter.
        """
        num_qubits = 2
        dt = 0.2
        n_trotter = 10

        # Create a Hamiltonian from random 2-qubit interactions
        hamiltonian = ops.QubitOperator()
        rng = np.random.RandomState(42)  # for reproducibility
        for _ in range(5):  # add a few random terms
            term = tuple(
                (int(i), rng.choice(["X", "Y", "Z"]))
                for i in rng.choice(range(num_qubits), size=2, replace=False)
            )
            hamiltonian += ops.QubitOperator(term, rng.uniform(-1, 1))

        # Create a TimeEvolutionGate
        gate = TimeEvolutionGate(hamiltonian, dt, num_qubits, n_trotter=n_trotter)
        qubits = cirq.LineQubit.range(num_qubits)

        # Decompose the gate
        decomposed_circuit = cirq.Circuit(gate.approximately_decompose(qubits))

        # Calculate the unitary of the decomposed circuit
        decomposed_unitary = decomposed_circuit.unitary(qubit_order=qubits)

        # Calculate the target unitary using exact diagonalization
        ham_mat = get_sparse_operator(hamiltonian, n_qubits=num_qubits).todense()
        print(hamiltonian)
        target_unitary = scipy.linalg.expm(1j * dt * ham_mat)

        assert np.abs(np.abs(np.trace(target_unitary)) - target_unitary.shape[0]) > 1e-2
        assert (
            np.abs(
                np.abs(np.trace(target_unitary @ decomposed_unitary.conj().T))
                - target_unitary.shape[0]
            )
            < 1e-2
        )

    def test_decompose_converges_for_small_dt_and_large_ntrotter_3qubits(self):
        """Tests that the _decompose_ method of TimeEvolutionGate works
        correctly for a small dt and sufficiently large n_trotter for 3 qubits.
        """
        num_qubits = 3
        dt = 0.2
        n_trotter = 10

        # Create a Hamiltonian from random 2-qubit interactions
        hamiltonian = ops.QubitOperator()
        rng = np.random.RandomState(42)  # for reproducibility
        for _ in range(5):  # add a few random terms
            term = tuple(
                (int(i), rng.choice(["X", "Y", "Z"]))
                for i in rng.choice(range(num_qubits), size=2, replace=False)
            )
            hamiltonian += ops.QubitOperator(term, rng.uniform(-1, 1))

        # Create a TimeEvolutionGate
        gate = TimeEvolutionGate(hamiltonian, dt, num_qubits, n_trotter=n_trotter)
        qubits = cirq.LineQubit.range(num_qubits)

        # Decompose the gate
        decomposed_circuit = cirq.Circuit(gate.approximately_decompose(qubits))

        # Calculate the unitary of the decomposed circuit
        decomposed_unitary = decomposed_circuit.unitary(qubit_order=qubits)

        # Calculate the target unitary using exact diagonalization
        ham_mat = get_sparse_operator(hamiltonian, n_qubits=num_qubits).todense()
        target_unitary = scipy.linalg.expm(1j * dt * ham_mat)

        assert np.abs(np.abs(np.trace(target_unitary)) - target_unitary.shape[0]) > 1e-2
        assert (
            np.abs(
                np.abs(np.trace(target_unitary @ decomposed_unitary.conj().T))
                - target_unitary.shape[0]
            )
            < 1e-2
        )


if __name__ == "__main__":
    unittest.main()
