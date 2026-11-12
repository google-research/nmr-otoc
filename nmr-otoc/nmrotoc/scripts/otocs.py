# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

import cirq.circuits
import numpy as np
import qsimcirq
import cirq
from typing import Optional

from tqdm import tqdm
from nmrotoc.dataclasses import OTOCCircuitObj, SimulatorWrapper
from nmrotoc.circuits.misc import prep_basis_state
from nmrotoc.circuits.otocs import make_OTOC_circuit


def get_inf_temp_OTOC(circuit_details: OTOCCircuitObj, simulator: SimulatorWrapper):
    """Calculates the infinite temperature OTOC for a given experiment.

    Args:
        circuit_details: OTOCCircuit dataclass containing experiment details.

    Returns:
        Tuple[float, float]: Mean OTOC value and standard error.
    """

    OTOCs_single_state = []
    num_states = 2 ** len(circuit_details.qubits)
    if circuit_details.inverse_unitary is not None:
        inverse_unitary = circuit_details.inverse_unitary
    else:
        inverse_unitary = cirq.inverse(circuit_details.unitary)
    for j in tqdm(np.arange(simulator.num_states_to_average)):
        if j >= num_states:
            break
        elif simulator.num_states_to_average >= num_states:
            prep_circuit = prep_basis_state(circuit_details.qubits, j)
        else:
            prep_circuit = prep_basis_state(
                circuit_details.qubits,
                simulator.rng.randint(num_states),
            )
        if circuit_details.measurement_qubit in prep_circuit.all_qubits():
            sign = -1
        else:
            sign = 1

        OTOC_circuit = make_OTOC_circuit(circuit_details, prep_circuit)

        if simulator.noise_model is not None:
            OTOC_circuit = simulator.noise_model(OTOC_circuit)
        OTOCs_single_state.append(
            np.real(
                simulator.simulator.simulate_expectation_values(
                    OTOC_circuit,
                    observables=[cirq.Z(circuit_details.measurement_qubit)],
                )[0]
            )
            * sign
        )
    mean_OTOC = np.mean(OTOCs_single_state)
    if simulator.num_states_to_average < num_states:
        stderr_OTOC = np.std(OTOCs_single_state) / np.sqrt(
            simulator.num_states_to_average
        )
    else:
        stderr_OTOC = 0
    return mean_OTOC, stderr_OTOC
