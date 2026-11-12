# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

import unittest
import pickle
import numpy as np
import cirq
import tempfile

from nmrotoc.dataclasses import (
    NMROTOCExperiment,
    OTOCCircuitObj,
    DoubleQuantumSwapNetworkObj,
    hash_ndarray,
)


def test_hash_ndarray():
    """Tests the hash_ndarray function."""

    array1 = np.array([1, 2, 3])
    array2 = np.array([1, 2, 3])
    array3 = np.array([3, 2, 1])

    hash1 = hash_ndarray(array1)
    hash2 = hash_ndarray(array2)
    hash3 = hash_ndarray(array3)

    assert hash1 == hash2
    assert hash1 != hash3

    # test different dtypes hash to different values
    array4 = np.array([1, 2, 3], dtype=np.float32)
    hash4 = hash_ndarray(array4)
    assert hash1 != hash4


class TestDataClassPickling(unittest.TestCase):
    def test_trotter_circuit_obj_pickling(self):
        """Tests pickling and unpickling of TrotterCircuitObj."""
        num_qubits = 3
        num_steps = 2
        homogeneous_step_mats = [
            np.random.rand(num_qubits, num_qubits) for _ in range(num_steps)
        ]
        heterogeneous_step_mats = [
            np.random.rand(num_qubits, num_qubits) for _ in range(num_steps)
        ]
        obj = DoubleQuantumSwapNetworkObj(
            homogeneous_step_mats=homogeneous_step_mats,
            heterogeneous_step_mats=heterogeneous_step_mats,
            num_steps=num_steps,
        )

        pickled_obj = pickle.dumps(obj)
        unpickled_obj = pickle.loads(pickled_obj)

        self.assertEqual(obj.num_steps, unpickled_obj.num_steps)
        for step in range(obj.num_steps):
            self.assertTrue(
                np.all(
                    obj.homogeneous_step_mats[step]
                    == unpickled_obj.homogeneous_step_mats[step]
                )
            )
            self.assertTrue(
                np.all(
                    obj.heterogeneous_step_mats[step]
                    == unpickled_obj.heterogeneous_step_mats[step]
                )
            )

    def test_nmrotoc_experiment_pickling(self):
        """Tests pickling and unpickling of NMROTOCExperiment."""

        t_range = np.array([0.0, 0.1, 0.2])
        bfly_ids = [0, 1]
        bfly_gate = cirq.X
        msmt_id = 0
        msmt_gate = cirq.Z
        num_qubits = 3
        homogeneous_coupling_mat = np.random.rand(num_qubits, num_qubits)
        heterogeneous_coupling_mat = np.random.rand(num_qubits, num_qubits)

        obj = NMROTOCExperiment(
            t_range=t_range,
            bfly_ids=bfly_ids,
            bfly_gate=bfly_gate,
            msmt_id=msmt_id,
            msmt_gate=msmt_gate,
            homogeneous_coupling_mat=homogeneous_coupling_mat,
            heterogeneous_coupling_mat=heterogeneous_coupling_mat,
        )

        pickled_obj = pickle.dumps(obj)
        unpickled_obj = pickle.loads(pickled_obj)

        self.assertTrue(np.all(obj.t_range == unpickled_obj.t_range))
        self.assertEqual(obj.bfly_ids, unpickled_obj.bfly_ids)
        self.assertEqual(obj.bfly_gate, unpickled_obj.bfly_gate)
        self.assertEqual(obj.msmt_id, unpickled_obj.msmt_id)  # Expanded check
        self.assertEqual(obj.msmt_gate, unpickled_obj.msmt_gate)  # Expanded check
        self.assertTrue(
            np.all(
                obj.homogeneous_coupling_mat == unpickled_obj.homogeneous_coupling_mat
            )
        )
        self.assertTrue(
            np.all(
                obj.heterogeneous_coupling_mat
                == unpickled_obj.heterogeneous_coupling_mat
            )
        )

    def test_otoc_circuit_obj_pickling(self):
        """Tests pickling and unpickling of OTOCCircuitObj."""
        qubits = cirq.LineQubit.range(3)
        unitary = cirq.Circuit(cirq.H.on_each(qubits))
        butterfly = cirq.Circuit(cirq.X.on_each(qubits))
        measurement_diagonalization = cirq.H
        measurement_qubit = qubits[0]

        obj = OTOCCircuitObj(
            unitary=unitary,
            butterfly=butterfly,
            measurement_diagonalization=measurement_diagonalization,
            measurement_qubit=measurement_qubit,
            qubits=qubits,
        )

        pickled_obj = pickle.dumps(obj)
        unpickled_obj = pickle.loads(pickled_obj)

        self.assertEqual(str(obj.unitary), str(unpickled_obj.unitary))
        self.assertEqual(str(obj.butterfly), str(unpickled_obj.butterfly))
        self.assertEqual(
            obj.measurement_diagonalization, unpickled_obj.measurement_diagonalization
        )
        self.assertEqual(obj.measurement_qubit, unpickled_obj.measurement_qubit)
        self.assertEqual(obj.qubits, unpickled_obj.qubits)


class TestDataClassSavingLoading(unittest.TestCase):
    def test_trotter_circuit_obj(self):
        """Tests saving and loading of TrotterCircuitObj."""
        num_qubits = 3
        num_steps = 2
        homogeneous_step_mats = [
            np.random.rand(num_qubits, num_qubits) for _ in range(num_steps)
        ]
        heterogeneous_step_mats = [
            np.random.rand(num_qubits, num_qubits) for _ in range(num_steps)
        ]
        bfly_ids = [0, 1]
        bfly_gate = cirq.X

        obj = DoubleQuantumSwapNetworkObj(
            homogeneous_step_mats=homogeneous_step_mats,
            heterogeneous_step_mats=heterogeneous_step_mats,
            num_steps=num_steps,
        )

        with tempfile.NamedTemporaryFile(delete=False) as f:
            filename = f.name
        obj.save(filename)
        loaded_obj = DoubleQuantumSwapNetworkObj.load(filename)

        self.assertEqual(obj.num_steps, loaded_obj.num_steps)
        for step in range(obj.num_steps):
            self.assertTrue(
                np.all(
                    obj.homogeneous_step_mats[step]
                    == loaded_obj.homogeneous_step_mats[step]
                )
            )
            self.assertTrue(
                np.all(
                    obj.heterogeneous_step_mats[step]
                    == loaded_obj.heterogeneous_step_mats[step]
                )
            )

    def test_nmrotoc_experiment(self):
        """Tests saving and loading of NMROTOCExperiment."""

        t_range = np.array([0.0, 0.1, 0.2])
        bfly_ids = [0, 1]
        bfly_gate = cirq.X
        msmt_id = 0
        msmt_gate = cirq.Z
        num_qubits = 3
        homogeneous_coupling_mat = np.random.rand(num_qubits, num_qubits)
        heterogeneous_coupling_mat = np.random.rand(num_qubits, num_qubits)

        obj = NMROTOCExperiment(
            t_range=t_range,
            bfly_ids=bfly_ids,
            bfly_gate=bfly_gate,
            msmt_id=msmt_id,
            msmt_gate=msmt_gate,
            homogeneous_coupling_mat=homogeneous_coupling_mat,
            heterogeneous_coupling_mat=heterogeneous_coupling_mat,
        )
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filename = f.name

        obj.save(filename)
        loaded_obj = NMROTOCExperiment.load(filename)

        self.assertTrue(np.all(obj.t_range == loaded_obj.t_range))
        self.assertEqual(obj.bfly_ids, loaded_obj.bfly_ids)
        self.assertEqual(obj.bfly_gate, loaded_obj.bfly_gate)
        self.assertEqual(obj.msmt_id, loaded_obj.msmt_id)  # Expanded check
        self.assertEqual(obj.msmt_gate, loaded_obj.msmt_gate)  # Expanded check
        self.assertTrue(
            np.all(obj.homogeneous_coupling_mat == loaded_obj.homogeneous_coupling_mat)
        )
        self.assertTrue(
            np.all(
                obj.heterogeneous_coupling_mat == loaded_obj.heterogeneous_coupling_mat
            )
        )

    def test_otoc_circuit_obj(self):
        """Tests saving and loading of OTOCCircuitObj."""
        qubits = cirq.LineQubit.range(3)
        unitary = cirq.Circuit(cirq.H.on_each(qubits))
        butterfly = cirq.Circuit(cirq.X.on_each(qubits))
        measurement_diagonalization = cirq.H
        measurement_qubit = qubits[0]

        obj = OTOCCircuitObj(
            unitary=unitary,
            butterfly=butterfly,
            measurement_diagonalization=measurement_diagonalization,
            measurement_qubit=measurement_qubit,
            qubits=qubits,
        )

        with tempfile.NamedTemporaryFile(delete=False) as f:
            filename = f.name
        obj.save(filename)
        loaded_obj = OTOCCircuitObj.load(filename)

        self.assertEqual(str(obj.unitary), str(loaded_obj.unitary))
        self.assertEqual(str(obj.butterfly), str(loaded_obj.butterfly))
        self.assertEqual(
            obj.measurement_diagonalization, loaded_obj.measurement_diagonalization
        )
        self.assertEqual(obj.measurement_qubit, loaded_obj.measurement_qubit)
        self.assertEqual(obj.qubits, loaded_obj.qubits)
