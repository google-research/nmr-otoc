# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

import unittest
import cirq
import numpy as np
from nmrotoc.math.graphs import find_hamiltonian_paths


class TestFindHamiltonianPaths(unittest.TestCase):

    def test_empty_input(self):
        self.assertEqual(find_hamiltonian_paths([]), [])

    def test_single_qubit(self):
        qubits = [cirq.GridQubit(0, 0)]
        expected_path = [qubits]
        self.assertEqual(find_hamiltonian_paths(qubits), expected_path)

    def test_two_connected_qubits(self):
        qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]
        expected_paths = [
            [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)],
            [cirq.GridQubit(0, 1), cirq.GridQubit(0, 0)],
        ]
        self.assertCountEqual(
            find_hamiltonian_paths(qubits), expected_paths
        )  # order doesn't matter here.

    def test_two_disconnected_qubits(self):
        qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 2)]
        self.assertEqual(find_hamiltonian_paths(qubits), [])

    def test_line_of_qubits(self):
        qubits = [cirq.GridQubit(0, i) for i in range(3)]
        expected_paths = [
            [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)],
            [cirq.GridQubit(0, 2), cirq.GridQubit(0, 1), cirq.GridQubit(0, 0)],
        ]
        self.assertCountEqual(find_hamiltonian_paths(qubits), expected_paths)

    def test_square_of_qubits(self):
        qubits = [cirq.GridQubit(i, j) for i in range(2) for j in range(2)]
        # There are 8 Hamiltonian paths in a 2x2 square grid.  Listing them all is tedious, but we can
        # at least check that we get the correct number.
        paths = find_hamiltonian_paths(qubits)
        self.assertEqual(len(paths), 8)
        # Check that all paths are valid (visit each qubit exactly once) and of correct length
        for path in paths:
            self.assertEqual(len(path), len(qubits))
            self.assertCountEqual(path, qubits)
