# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Callable, Dict, List, Optional, Tuple
import cirq
import numpy as np
import scipy
import pytest
import unittest
from openfermion import QubitOperator, get_sparse_operator

from nmrotoc.dataclasses import (
    NMROTOCExperiment,
    DoubleQuantumSwapNetworkObj,
    DoubleQuantumXZYObj,
    OTOCCircuitObj,
    NestedProductFormulaObj,
    ProductFormulaObj,
)
from nmrotoc.circuits.trotter import (
    first_order_trotterize_dq_hamiltonian,
    first_order_trotterize_xzy_dq_hamiltonian,
    mixed_times_trotterize_xzy_dq_hamiltonian,
    make_product_formula_swap_network_dq_hamiltonian,
    make_product_formula_xzy_dq_hamiltonian,
    sorting_circuit,
    compile_nested_product_formula,
    _create_swap_network_dq_step,
    _create_xzy_dq_step,
)
from nmrotoc.circuits.otocs import make_butterfly_circuit
from nmrotoc.scripts.run_scripts import calculate_unitary_trotter_error

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

XX = np.kron(X, X)
YY = np.kron(Y, Y)
ZZ = np.kron(Z, Z)
IZ = np.kron(I, Z)
ZI = np.kron(Z, I)

rng = np.random.RandomState(42)


class TestZStepMats(unittest.TestCase):
    def test_z_step_mats_functionality(self):
        """Test that z_step_mats functionality works correctly."""
        num_qubits = 3
        qubits = cirq.LineQubit.range(num_qubits)
        time = 1
        num_steps = 2

        # Create a NMROTOCExperiment with only onsite terms
        for onsite_vector in [
            np.array([np.pi / 4, 0, 0]),
            np.array([0, np.pi / 4, 0]),
            np.array([0, 0, np.pi / 4]),
            np.array([np.pi / 6, np.pi / 8, np.pi / 4]),
            np.array([1.7, 2.5, 0.45]),
        ]:
            experiment = NMROTOCExperiment(
                t_range=np.array([time]),
                onsite_vector=onsite_vector,
                homogeneous_coupling_mat=np.zeros((num_qubits, num_qubits)),
                heterogeneous_coupling_mat=np.zeros((num_qubits, num_qubits)),
            )

            # Trotterize the Hamiltonian
            trotter_obj = first_order_trotterize_dq_hamiltonian(
                experiment, time, num_steps
            )

            # Construct the circuit
            circuit, qubit_id_refs = make_product_formula_swap_network_dq_hamiltonian(
                trotter_obj, qubits
            )

            # Calculate the exact unitary
            h_exact = QubitOperator()
            for i in range(num_qubits):
                h_exact += QubitOperator(f"Z{i}", onsite_vector[i])
            h_exact_mat = get_sparse_operator(h_exact, n_qubits=num_qubits).todense()
            exact_unitary = scipy.linalg.expm(1j * time * h_exact_mat)
            print(
                (exact_unitary * np.exp(-1j * np.angle(exact_unitary[0, 0]))).round(3)
            )

            # Calculate the trotterized unitary
            trotter_unitary = circuit.unitary(qubit_order=qubits)
            print(
                (trotter_unitary * np.exp(-1j * np.angle(trotter_unitary[0, 0]))).round(
                    3
                )
            )

            # Check if the unitaries are equal
            assert (
                np.abs(
                    np.abs(np.trace(trotter_unitary.conj().T @ exact_unitary))
                    - 2**num_qubits
                )
                < 1e-6
            )


class TestNestedProductFormulaObj(unittest.TestCase):
    def test_compile_nested_product_formula_single_step(self):
        # Define a simple generating function for testing
        def simple_generating_function(obj, qubits, return_pairs):
            return cirq.Circuit(cirq.X(qubits[0])), {
                x: y for j, q in enumerate(qubits) for x, y in [(j, q), (q, j)]
            }

        # Create a ProductFormulaObj
        sub_term = ProductFormulaObj(generating_function=simple_generating_function)

        # Create a NestedProductFormulaObj with one sub-term
        nested_pf_obj = NestedProductFormulaObj(sub_terms=[sub_term], qid_list=[[0]])

        # Define qubits
        qubits = cirq.LineQubit.range(1)

        # Compile the nested product formula
        compiled_circuit, _ = compile_nested_product_formula(nested_pf_obj, qubits)

        # Check if the compiled circuit is correct
        expected_circuit = cirq.Circuit(cirq.X(qubits[0]))
        compiled_unitary = compiled_circuit.unitary(qubit_order=qubits)
        expected_unitary = expected_circuit.unitary(qubit_order=qubits)
        self.assertAlmostEqual(
            np.abs(np.trace(compiled_unitary.conj().T @ expected_unitary)),
            2 ** len(qubits)
        )

    def test_compile_nested_product_formula_return_pairs(self):
        # Define a generating function that returns known pairs
        def simple_generating_function_with_pairs(obj, qubits, return_pairs):
            circuit = cirq.Circuit(cirq.X(qubits[0]))
            # Placeholder relabeling dic
            relabeling_dic = {
                x: y for j, q in enumerate(qubits) for x, y in [(j, q), (q, j)]
            }
            pairs = [(0, 1)] if len(qubits) > 1 else [(0, 0)] # Example pairs
            if return_pairs:
                return circuit, relabeling_dic, pairs
            return circuit, relabeling_dic

        sub_term = ProductFormulaObj(generating_function=simple_generating_function_with_pairs)
        nested_pf_obj = NestedProductFormulaObj(sub_terms=[sub_term], num_steps=2, qid_list=[[0]])
        qubits = cirq.LineQubit.range(1)

        _, _, returned_pairs = compile_nested_product_formula(nested_pf_obj, qubits, return_pairs=True)
        expected_pairs = [(0, 0), (0, 0)] # Pairs from each step
        self.assertEqual(returned_pairs, expected_pairs)

    def test_compile_nested_product_formula_multiple_steps(self):
        # Define a simple generating function for testing
        def simple_generating_function(obj, qubits, return_pairs):
            return cirq.Circuit(cirq.X(qubits[0])), {
                x: y for j, q in enumerate(qubits) for x, y in [(j, q), (q, j)]
            }

        # Create a ProductFormulaObj
        sub_term = ProductFormulaObj(generating_function=simple_generating_function)

        # Create a NestedProductFormulaObj with multiple sub-terms
        nested_pf_obj = NestedProductFormulaObj(
            sub_terms=[sub_term, sub_term], qid_list=[[0], [0]]
        )

        # Define qubits
        qubits = cirq.LineQubit.range(1)

        # Compile the nested product formula
        compiled_circuit, _ = compile_nested_product_formula(nested_pf_obj, qubits)

        # Check if the compiled circuit is correct
        expected_circuit = cirq.Circuit(cirq.X(qubits[0]), cirq.X(qubits[0]))
        compiled_unitary = compiled_circuit.unitary(qubit_order=qubits)
        expected_unitary = expected_circuit.unitary(qubit_order=qubits)
        self.assertAlmostEqual(
            np.abs(np.trace(compiled_unitary.conj().T @ expected_unitary)),
            2 ** len(qubits)
        )

    def test_compile_nested_product_formula_multiple_iterations(self):
        # Define a simple generating function for testing
        def simple_generating_function(obj, qubits, return_pairs):
            return cirq.Circuit(cirq.X(qubits[0])), {
                x: y for j, q in enumerate(qubits) for x, y in [(j, q), (q, j)]
            }

        # Create a ProductFormulaObj
        sub_term = ProductFormulaObj(generating_function=simple_generating_function)

        # Create a NestedProductFormulaObj with multiple sub-terms
        nested_pf_obj = NestedProductFormulaObj(
            sub_terms=[sub_term], num_steps=2, qid_list=[[0]]
        )

        # Define qubits
        qubits = cirq.LineQubit.range(1)

        # Compile the nested product formula
        compiled_circuit, _ = compile_nested_product_formula(nested_pf_obj, qubits)

        # Check if the compiled circuit is correct
        expected_circuit = cirq.Circuit(cirq.X(qubits[0]), cirq.X(qubits[0]))
        compiled_unitary = compiled_circuit.unitary(qubit_order=qubits)
        expected_unitary = expected_circuit.unitary(qubit_order=qubits)
        self.assertAlmostEqual(
            np.abs(np.trace(compiled_unitary.conj().T @ expected_unitary)),
            2 ** len(qubits)
        )

    def test_compile_nested_product_formula_complex(self):
        # Define a more complex generating function for testing
        def complex_generating_function(obj, qubits, return_pairs):
            return cirq.Circuit(
                cirq.X(qubits[0]), cirq.CNOT(qubits[0], qubits[1]), cirq.Y(qubits[1])
            ), {x: y for j, q in enumerate(qubits) for x, y in [(j, q), (q, j)]}

        # Create a ProductFormulaObj
        sub_term1 = ProductFormulaObj(generating_function=complex_generating_function)
        sub_term2 = ProductFormulaObj(generating_function=complex_generating_function)

        # Create a NestedProductFormulaObj with multiple sub-terms and iterations
        nested_pf_obj = NestedProductFormulaObj(
            sub_terms=[sub_term1, sub_term2], num_steps=3, qid_list=[[0, 1], [0, 1]]
        )

        # Define qubits
        qubits = cirq.LineQubit.range(2)

        # Compile the nested product formula
        compiled_circuit, _ = compile_nested_product_formula(nested_pf_obj, qubits)
        compiled_unitary = compiled_circuit.unitary(qubit_order=qubits)

        # Check if the compiled circuit is correct
        expected_circuit = cirq.Circuit(
            cirq.X(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.Y(qubits[1]),
            cirq.X(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.Y(qubits[1]),
            cirq.X(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.Y(qubits[1]),
            cirq.X(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.Y(qubits[1]),
            cirq.X(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.Y(qubits[1]),
            cirq.X(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.Y(qubits[1]),
        )
        expected_unitary = expected_circuit.unitary(qubit_order=qubits)
        self.assertAlmostEqual(
            np.trace(np.abs(compiled_unitary.conj().T @ expected_unitary)), 4
        )


class TestCreateSwapNetworkDQStep(unittest.TestCase):
    def test_create_swap_network_dq_step_return_pairs(self):
        """Tests if _create_swap_network_dq_step returns the correct pairs."""
        num_qubits = 4
        qubits = cirq.LineQubit.range(num_qubits)
        het_mat = np.ones((num_qubits, num_qubits))
        hom_mat = np.ones((num_qubits, num_qubits)) * 0.5

        # Initial qubit_id_refs
        qubit_id_refs = {q: i for i, q in enumerate(qubits)}
        for i, q in enumerate(qubits):
            qubit_id_refs[i] = q

        # Define layers for a 4-qubit swap network
        layers = [
            [(qubits[0], qubits[1]), (qubits[2], qubits[3])], # Layer 0 (even pairs)
            [(qubits[1], qubits[2])],                         # Layer 1 (odd pairs)
        ]

        # Call the function with return_pairs=True
        _, returned_pairs = _create_swap_network_dq_step(
            qubits=qubits,
            layers=layers,
            qubit_id_refs=qubit_id_refs.copy(), # Pass a copy to avoid modifying original
            het_mat=het_mat,
            hom_mat=hom_mat,
            num_layers=num_qubits, # Full swap network
            return_pairs=True,
        )
        expected_pairs = [
            (0, 1), (2, 3), # j=0
            (0, 3),         # j=1
            (1, 3), (0, 2), # j=2
            (1, 2)          # j=3
        ]

        self.assertEqual(returned_pairs, expected_pairs)

    def test_make_product_formula_swap_network_dq_hamiltonian_return_pairs(self):
        """Tests if make_product_formula_swap_network_dq_hamiltonian returns pairs."""
        num_qubits = 3
        qubits = cirq.LineQubit.range(num_qubits)
        trotter_obj = DoubleQuantumSwapNetworkObj(
            homogeneous_step_mats=[np.ones((num_qubits, num_qubits))] * 2,
            heterogeneous_step_mats=[np.ones((num_qubits, num_qubits))] * 2,
            num_steps=2,
        )
        _, _, returned_pairs = make_product_formula_swap_network_dq_hamiltonian(trotter_obj, qubits, return_pairs=True)
        expected_pairs = [(0,1), (0,2), (1,2), (1,0), (2,0), (2,1)]
        self.assertEqual(returned_pairs, expected_pairs)


class TestSortingCircuit(unittest.TestCase):
    def test_sorting_circuit_no_sort(self):
        """Tests sorting when qubits are already ordered."""
        num_qubits = 4
        qubits = cirq.LineQubit.range(num_qubits)
        init_qubit_id_refs = {q: i for i, q in enumerate(qubits)}
        qubit_id_refs = {}
        for q in init_qubit_id_refs:
            qubit_id_refs[init_qubit_id_refs[q]] = q
            qubit_id_refs[q] = init_qubit_id_refs[q]
        circuit = sorting_circuit(qubits, qubit_id_refs)

        # No swaps should be added
        self.assertEqual(len(circuit), 0)
        # Check id references
        for i in range(num_qubits):
            self.assertEqual(qubit_id_refs[qubits[i]], i)
            self.assertEqual(qubit_id_refs[i], qubits[i])

    def test_sorting_circuit_reversed(self):
        """Tests sorting when qubits are in reversed order."""

        num_qubits = 4
        qubits = cirq.LineQubit.range(num_qubits)
        init_qubit_id_refs = {q: num_qubits - i - 1 for i, q in enumerate(qubits)}
        qubit_id_refs = {}
        for q in init_qubit_id_refs:
            qubit_id_refs[init_qubit_id_refs[q]] = q
            qubit_id_refs[q] = init_qubit_id_refs[q]

        circuit = sorting_circuit(qubits, qubit_id_refs)
        print(circuit)
        print(qubit_id_refs)

        # Check the circuit
        # Expect a ladder of SWAPs
        expected_ops = [
            cirq.SWAP(cirq.LineQubit(0), cirq.LineQubit(3)),
            cirq.SWAP(cirq.LineQubit(1), cirq.LineQubit(2)),
        ]
        self.assertEqual(len(list(circuit.all_operations())), len(expected_ops))

        for op, expected_op in zip(circuit.all_operations(), expected_ops):
            self.assertEqual(op, expected_op)
        # Check id references
        for i in range(num_qubits):
            self.assertEqual(qubit_id_refs[qubits[i]], i)
            self.assertEqual(qubit_id_refs[i], qubits[i])

    def test_sorting_circuit_random(self):
        """Tests sorting with a random qubit order."""
        num_qubits = 4
        qubits = cirq.LineQubit.range(num_qubits)
        qubit_id_refs = {}

        # Shuffle the qubit order in the dictionary
        ids = list(range(num_qubits))
        np.random.shuffle(ids)
        for i, q in enumerate(qubits):
            qubit_id_refs[q] = ids[i]
            qubit_id_refs[ids[i]] = q
        circuit = sorting_circuit(qubits, qubit_id_refs)
        # Check id references
        for i in range(num_qubits):
            self.assertEqual(qubit_id_refs[qubits[i]], i)
            self.assertEqual(qubit_id_refs[i], qubits[i])


class TestFirstOrderTrotterize(unittest.TestCase):
    def test_first_order_trotterize_dq_hamiltonian(self):
        """Tests the first_order_trotterize_dq_hamiltonian function."""

        num_qubits = 3
        time = 1.0
        num_steps = 4

        # Create a sample NMROTOCExperiment object
        homogeneous_coupling_mat = np.random.rand(num_qubits, num_qubits)
        heterogeneous_coupling_mat = np.random.rand(num_qubits, num_qubits)
        bfly_gate = cirq.X
        bfly_ids = [0, 1]
        experiment = NMROTOCExperiment(
            homogeneous_coupling_mat=homogeneous_coupling_mat,
            heterogeneous_coupling_mat=heterogeneous_coupling_mat,
            bfly_gate=bfly_gate,
            bfly_ids=bfly_ids,
        )

        # Call the function
        qubits = [cirq.LineQubit(j) for j in range(num_qubits)]
        trotter_circuit_obj = first_order_trotterize_dq_hamiltonian(
            experiment, time, num_steps
        )
        unitary, qubit_id_refs = make_product_formula_swap_network_dq_hamiltonian(
            trotter_circuit_obj, qubits
        )
        butterfly = make_butterfly_circuit(experiment, qubit_id_refs)
        assert len(list(butterfly.all_operations())) == 2
        assert (
            len(
                [
                    moment
                    for moment in unitary
                    if len(list(moment.operations)[0].qubits) == 2
                ]
            )
            == num_qubits * num_steps
        )


class TestMakeTrotterStepsArb(unittest.TestCase):
    def _check_gate_ordering(self, num_qubits: int, alternate_order: bool):

        # Settings for test
        num_steps = 2
        homogeneous_coupling_mat = np.ones((num_qubits, num_qubits)) * 0.2
        heterogeneous_coupling_mat = np.ones((num_qubits, num_qubits)) * 0.1
        bfly_gate = cirq.X
        bfly_ids = [0]

        trotter_circuit_obj = DoubleQuantumSwapNetworkObj(
            homogeneous_step_mats=[homogeneous_coupling_mat, homogeneous_coupling_mat],
            heterogeneous_step_mats=[
                heterogeneous_coupling_mat,
                heterogeneous_coupling_mat,
            ],
            num_steps=num_steps,
            alternate_order=alternate_order,
        )

        qubits = cirq.LineQubit.range(num_qubits)
        unitary, _ = make_product_formula_swap_network_dq_hamiltonian(
            trotter_circuit_obj, qubits
        )

        # Extract gates from the two Trotter steps
        twoq_gate_layers = [
            moment for moment in unitary if len(list(moment.operations)[0].qubits) == 2
        ]
        for layer in twoq_gate_layers:
            print(layer)
        assert len(twoq_gate_layers) == num_qubits * 2
        gates_step_0 = twoq_gate_layers[num_qubits - 1]
        gates_step_1 = twoq_gate_layers[num_qubits]

        # Test
        if alternate_order == True:
            for gate in gates_step_0:
                assert gate in gates_step_1
            for gate in gates_step_1:
                assert gate in gates_step_0
        else:
            for gate in gates_step_0:
                assert gate not in gates_step_1
            for gate in gates_step_1:
                assert gate not in gates_step_0
        return

    def test_make_product_formula_swap_network_dq_hamiltonian_even_qubits_alternate_order(
        self,
    ):
        """Tests gate order for even qubits with alternate_order=True."""
        self._check_gate_ordering(num_qubits=4, alternate_order=True)

    def test_make_product_formula_swap_network_dq_hamiltonian_odd_qubits_alternate_order(
        self,
    ):
        """Tests gate order for odd qubits with alternate_order=True."""
        self._check_gate_ordering(num_qubits=3, alternate_order=True)

    def test_make_product_formula_swap_network_dq_hamiltonian_even_qubits_no_alternate_order(
        self,
    ):
        """Tests gate order for even qubits with alternate_order=False."""
        self._check_gate_ordering(num_qubits=4, alternate_order=False)

    def test_make_product_formula_swap_network_dq_hamiltonian_odd_qubits_no_alternate_order(
        self,
    ):
        """Tests gate order for odd qubits with alternate_order=False."""
        self._check_gate_ordering(num_qubits=3, alternate_order=False)

    def _compare_make_product_formula_swap_network_dq_hamiltonian_two_steps(
        self, num_qubits, alternate_order
    ):
        """Tests equivalence of 2 Trotter steps vs. 1 Trotter step with reversed qubits."""
        # Example coupling matrices (can be random)
        homogeneous_coupling_mat = np.random.rand(num_qubits, num_qubits)
        homogeneous_coupling_mat += homogeneous_coupling_mat.T
        heterogeneous_coupling_mat = np.random.rand(num_qubits, num_qubits)
        heterogeneous_coupling_mat += heterogeneous_coupling_mat.T

        # Create DoubleQuantumSwapNetworkObj for 2 Trotter steps
        trotter_circuit_obj_2steps = DoubleQuantumSwapNetworkObj(
            homogeneous_step_mats=[
                homogeneous_coupling_mat / 2,
                homogeneous_coupling_mat / 2,
            ],
            heterogeneous_step_mats=[
                heterogeneous_coupling_mat / 2,
                heterogeneous_coupling_mat / 2,
            ],
            num_steps=2,
            alternate_order=alternate_order,
        )

        qubits = cirq.LineQubit.range(num_qubits)
        circuit_2steps, _ = make_product_formula_swap_network_dq_hamiltonian(
            trotter_circuit_obj_2steps, qubits
        )
        unitary_2steps = circuit_2steps.unitary(qubit_order=qubits)

        # Create DoubleQuantumSwapNetworkObj for 1 Trotter step
        trotter_circuit_obj_1step = DoubleQuantumSwapNetworkObj(
            homogeneous_step_mats=[homogeneous_coupling_mat / 2],
            heterogeneous_step_mats=[heterogeneous_coupling_mat / 2],
            num_steps=1,
        )

        # Call make_product_formula_swap_network_dq_hamiltonian twice with reversed qubits
        circuit_1step, _ = make_product_formula_swap_network_dq_hamiltonian(
            trotter_circuit_obj_1step, qubits
        )

        qubits_reversed = qubits[::-1]
        trotter_circuit_obj_1step.reverse = alternate_order

        circuit_1step_reversed, _ = make_product_formula_swap_network_dq_hamiltonian(
            trotter_circuit_obj_1step, qubits_reversed
        )

        circuit_1step.append(circuit_1step_reversed)
        unitary_1step = circuit_1step.unitary(qubit_order=qubits)

        # Compare the unitaries
        self.assertAlmostEqual(
            np.abs(np.trace(unitary_1step.conj().T @ unitary_2steps)), 2**num_qubits
        )

    def test_make_product_formula_swap_network_dq_hamiltonian_two_steps_4q(self):
        self._compare_make_product_formula_swap_network_dq_hamiltonian_two_steps(
            4, False
        )
        self._compare_make_product_formula_swap_network_dq_hamiltonian_two_steps(
            4, True
        )

    def test_make_product_formula_swap_network_dq_hamiltonian_two_steps_5q(self):
        self._compare_make_product_formula_swap_network_dq_hamiltonian_two_steps(
            5, False
        )
        self._compare_make_product_formula_swap_network_dq_hamiltonian_two_steps(
            5, True
        )


class TestMakeProductFormulaSwapNetworkDQHamiltonian(unittest.TestCase):
    def test_make_product_formula_swap_network_dq_hamiltonian_randomized_qubits(
        self,
    ):
        """Tests Trotter error with randomized qubits in each step."""

        num_qubits = 4
        time = 0.1
        num_steps = 100
        tolerance = 0.02  # Adjust tolerance as needed

        # Create a sample NMROTOCExperiment
        homogeneous_coupling_mat = np.random.rand(num_qubits, num_qubits)
        heterogeneous_coupling_mat = np.random.rand(num_qubits, num_qubits)

        experiment = NMROTOCExperiment(
            homogeneous_coupling_mat=homogeneous_coupling_mat,
            heterogeneous_coupling_mat=heterogeneous_coupling_mat,
            bfly_gate=cirq.X,  # Placeholder gate
            bfly_ids=[0, 1],
            msmt_gate=cirq.Z,
            msmt_id=0,
        )

        # Create ProductFormulaObj
        trotter_circuit_obj = first_order_trotterize_dq_hamiltonian(
            experiment, time, num_steps
        )

        circuit_obj = OTOCCircuitObj(
            qubits=cirq.LineQubit.range(num_qubits),
            measurement_diagonalization=cirq.I,
            measurement_qubit=cirq.LineQubit(0),
        )

        # Calculate error
        error_no_perms = calculate_unitary_trotter_error(
            experiment, trotter_circuit_obj, circuit_obj, time
        )

        # Create randomized qubits for each step
        perms = [list(range(num_qubits)) for _ in range(num_steps)]
        rng = np.random.RandomState(42)  # Use a seed for reproducibility
        for q in perms:
            rng.shuffle(q)
        trotter_circuit_obj.permutations = perms

        # Calculate error
        error_with_perms = calculate_unitary_trotter_error(
            experiment, trotter_circuit_obj, circuit_obj, time
        )

        self.assertNotAlmostEqual(error_with_perms, error_no_perms)
        self.assertLessEqual(error_with_perms, tolerance)


class TestXZYProductFormula(unittest.TestCase):
    def test_single_step_xzy_unitary_exact_match(self):
        """Tests that a single XZY Trotter step matches the exact matrix product U_Y * U_Z * U_X."""
        num_qubits = 4
        qubits = cirq.LineQubit.range(num_qubits)
        rng_local = np.random.RandomState(123)
        full_ham = rng_local.randn(num_qubits, num_qubits) * 0.2
        full_ham = np.triu(full_ham, 1)
        full_ham = full_ham + full_ham.T

        hom = np.zeros_like(full_ham)
        het = np.zeros_like(full_ham)
        hom[2:, 2:] = full_ham[2:, 2:]
        hom[:2, :2] = full_ham[:2, :2]
        het[2:, :2] = full_ham[2:, :2]
        het[:2, 2:] = full_ham[:2, 2:]

        experiment = NMROTOCExperiment(
            homogeneous_coupling_mat=hom,
            heterogeneous_coupling_mat=het,
            bfly_gate=cirq.X,
            bfly_ids=[1],
            msmt_gate=cirq.X,
            msmt_id=0,
        )
        dt = 0.35
        trotter_obj = first_order_trotterize_xzy_dq_hamiltonian(
            experiment, time=dt, num_steps=1
        )
        circuit, qubit_id_refs = make_product_formula_xzy_dq_hamiltonian(
            trotter_obj, qubits
        )
        u_circuit = circuit.unitary(qubit_order=qubits)

        # Build exact matrix product: U_Y(dt) @ U_Z(dt) @ U_X(dt)
        h_x = QubitOperator()
        h_z = QubitOperator()
        h_y = QubitOperator()
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                h_x += hom[i, j] * QubitOperator(f"X{i} X{j}")
                h_z += het[i, j] * QubitOperator(f"Z{i} Z{j}")
                h_y += hom[i, j] * QubitOperator(f"Y{i} Y{j}")

        u_x = scipy.linalg.expm(
            1j * dt * get_sparse_operator(h_x, n_qubits=num_qubits).todense()
        )
        u_z = scipy.linalg.expm(
            -1j * dt * get_sparse_operator(h_z, n_qubits=num_qubits).todense()
        )
        u_y = scipy.linalg.expm(
            -1j * dt * get_sparse_operator(h_y, n_qubits=num_qubits).todense()
        )
        u_expected = u_y @ u_z @ u_x

        np.testing.assert_allclose(u_circuit, u_expected, atol=1e-12)
        for i, q in enumerate(qubits):
            self.assertEqual(qubit_id_refs[i], q)
            self.assertEqual(qubit_id_refs[q], i)

    def test_xzy_trotter_convergence_to_exact_hamiltonian(self):
        """Tests that first-order XZY Trotter converges to exp(-i H t) as O(dt)."""
        num_qubits = 4
        qubits = cirq.LineQubit.range(num_qubits)
        rng_local = np.random.RandomState(456)
        hom = np.triu(rng_local.randn(num_qubits, num_qubits) * 0.15, 1)
        hom = hom + hom.T
        het = np.triu(rng_local.randn(num_qubits, num_qubits) * 0.15, 1)
        het = het + het.T

        experiment = NMROTOCExperiment(
            homogeneous_coupling_mat=hom,
            heterogeneous_coupling_mat=het,
            bfly_gate=cirq.X,
            bfly_ids=[1],
            msmt_gate=cirq.X,
            msmt_id=0,
        )
        time = 0.4
        h_op = QubitOperator()
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                h_op += hom[i, j] * (
                    QubitOperator(f"Y{i} Y{j}") - QubitOperator(f"X{i} X{j}")
                )
                h_op += het[i, j] * QubitOperator(f"Z{i} Z{j}")
        h_mat = get_sparse_operator(h_op, n_qubits=num_qubits).todense()
        u_exact = scipy.linalg.expm(-1j * time * h_mat)

        errors = []
        for num_steps in [5, 10, 20]:
            trotter_obj = first_order_trotterize_xzy_dq_hamiltonian(
                experiment, time=time, num_steps=num_steps
            )
            circuit, _ = make_product_formula_xzy_dq_hamiltonian(trotter_obj, qubits)
            u_trotter = circuit.unitary(qubit_order=qubits)
            errors.append(np.linalg.norm(u_trotter - u_exact, ord=2))

        self.assertLess(errors[1], errors[0] * 0.6)
        self.assertLess(errors[2], errors[1] * 0.6)

    def test_xzy_reverse_and_return_pairs(self):
        """Tests reverse order, alternate_order, mixed_times, and return_pairs in XZY."""
        num_qubits = 3
        qubits = cirq.LineQubit.range(num_qubits)
        hom = np.array([[0.0, 0.1, 0.0], [0.1, 0.0, 0.2], [0.0, 0.2, 0.0]])
        het = np.array([[0.0, 0.0, 0.3], [0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
        experiment = NMROTOCExperiment(
            homogeneous_coupling_mat=hom,
            heterogeneous_coupling_mat=het,
            bfly_gate=cirq.X,
            bfly_ids=[1],
            msmt_gate=cirq.X,
            msmt_id=0,
        )
        trotter_obj = mixed_times_trotterize_xzy_dq_hamiltonian(
            experiment, dts=[0.1, 0.2]
        )
        trotter_obj.alternate_order = True
        circuit, qubit_id_refs, pairs = make_product_formula_xzy_dq_hamiltonian(
            trotter_obj, qubits, return_pairs=True
        )
        # Each step has 2 hom pairs in X, 1 het pair in Z, 2 hom pairs in Y -> 5 pairs per step
        self.assertEqual(len(pairs), 10)
        butterfly = make_butterfly_circuit(experiment, qubit_id_refs)
        self.assertEqual(list(butterfly.all_operations()), [cirq.X(qubits[1])])
        self.assertEqual(circuit.unitary(qubit_order=qubits).shape, (8, 8))

    def test_xzy_rejects_xyyx_flag(self):
        """Tests that the XZY builders refuse DREAM-1 (XY + YX) experiments instead of silently using XX/YY."""
        experiment = NMROTOCExperiment(
            homogeneous_coupling_mat=np.zeros((3, 3)),
            heterogeneous_coupling_mat=np.zeros((3, 3)),
            bfly_gate=cirq.X,
            bfly_ids=[1],
            msmt_gate=cirq.X,
            msmt_id=0,
            xyyx_flag=True,
        )
        with self.assertRaises(NotImplementedError):
            first_order_trotterize_xzy_dq_hamiltonian(experiment, time=1.0, num_steps=2)
        with self.assertRaises(NotImplementedError):
            mixed_times_trotterize_xzy_dq_hamiltonian(experiment, dts=[0.1, 0.2])

