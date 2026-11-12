# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Dict, Optional
import cirq
from nmrotoc.dataclasses import NMROTOCExperiment, OTOCCircuitObj


def make_butterfly_circuit(
    experiment: NMROTOCExperiment,
    qubit_id_refs: Dict,
) -> cirq.Circuit:
    """Makes butterfly circuit for OTOC experiment, accounting for qubit rearrangement

    Args:
        experiment: data container for OTOC experiment. Needs to have bfly_ids
            set, and either bfly_gate or bfly_gate_list
        qubit_id_refs: double_ended dictionary mapping between spin indices and
            the qubits they are mapped to.
    """
    if experiment.bfly_gate_list is not None:
        butterfly = cirq.Circuit(
            [
                gate(qubit_id_refs[idx])
                for gate, idx in zip(experiment.bfly_gate_list, experiment.bfly_ids)
            ]
        )
    elif experiment.bfly_gate is not None:
        butterfly = cirq.Circuit(
            [experiment.bfly_gate(qubit_id_refs[idx]) for idx in experiment.bfly_ids]
        )
    else:
        raise ValueError("Either bfly_gate or bfly_gate_list needs to be defined")
    return butterfly


def make_OTOC_circuit(
    circuit_obj: OTOCCircuitObj, prep_circuit: Optional[cirq.Circuit] = None
):
    """Constructs the full OTOC circuit.

    This function assembles the complete Out-of-Time-Order Correlator (OTOC)
    circuit, including state preparation, unitary evolution, butterfly
    perturbation, and measurement basis change.

    Args:
        circuit_obj: An OTOCCircuitObj containing the unitary evolution,
            butterfly perturbation, measurement diagonalization, and
            measurement qubit.
        prep_circuit: An optional circuit for state preparation. If not
            provided, it defaults to the `prep_circuit` in `circuit_obj`.
            If `circuit_obj.prep_circuit` is also None, no state preparation
            is performed.

    Returns:
        A cirq.Circuit representing the full OTOC circuit.
    """

    if circuit_obj.inverse_unitary is not None:
        inverse_unitary = circuit_obj.inverse_unitary
    else:
        inverse_unitary = cirq.inverse(circuit_obj.unitary)

    if prep_circuit is None:
        if circuit_obj.prep_circuit is not None:
            prep_circuit = circuit_obj.prep_circuit
        else:
            prep_circuit = []

    OTOC_circuit = cirq.Circuit(
        [
            [prep_circuit if prep_circuit else []],
            circuit_obj.measurement_diagonalization.on(
                circuit_obj.measurement_qubit
            ),  # Change to appropriate basis so we get a signal
            circuit_obj.unitary,
            circuit_obj.butterfly,
            inverse_unitary,
            cirq.inverse(circuit_obj.measurement_diagonalization).on(
                circuit_obj.measurement_qubit
            ),  # Change to appropriate basis so we get a signal
        ]
    )
    return OTOC_circuit
