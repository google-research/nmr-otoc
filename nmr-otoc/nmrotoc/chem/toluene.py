# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np
import cirq
from tqdm import tqdm
import scipy
from typing import List, Optional
from nmrotoc.chem.molecule import (
    Molecule,
    StaticVector,
    CircleVector,
    RandomVariable,
    ContinuousIntegrator,
)
from nmrotoc.chem.hamiltonians import make_coupling_matrices
from nmrotoc.math.operators import construct_nmr_hamiltonian_matrix
from nmrotoc.circuits.time_evolution import (
    dream_2_sequence_CH,
    dream_2_inverse_sequence_CH,
)


def make_toluene_molecule(rng: np.random.RandomState) -> Molecule:
    """Creates a Molecule object representing the structure of toluene with dynamic methyl group.

    Args:
        rng: A NumPy random number generator for the methyl rotation.

    Returns:
        A Molecule object.
    """
    angle = RandomVariable(rng)
    v6 = np.array([-0.5150, -0.8919, 5.7510])
    v7 = np.array([-0.5150, 0.8919, 5.7510])
    v8 = np.array([1.0299, 0, 5.7510])
    center = np.array([0, 0, 5.7510])

    cs_methyl_group6 = CircleVector(v6, center, v7, angle)
    cs_methyl_group7 = CircleVector(v7, center, v8, angle)
    cs_methyl_group8 = CircleVector(v8, center, v6, angle)

    coordinates = {
        "C0": StaticVector(0, 0, 1.0866),
        "H1": StaticVector(0, 0, 0),
        "H2": StaticVector(0, 2.1486, 1.2417),
        "H3": StaticVector(0, -2.1486, 1.2417),
        "H4": StaticVector(0, 2.1290, 3.7051),
        "H5": StaticVector(0, -2.1290, 3.7051),
        "H6": cs_methyl_group6,
        "H7": cs_methyl_group7,
        "H8": cs_methyl_group8,
    }

    molecule = Molecule(coordinates=coordinates, parameters=[angle])
    return molecule


def make_toluene_molecule_cts_integration(num_points: int) -> Molecule:
    """Creates a Molecule object representing the structure of toluene with dynamic methyl group.

    Args:
        rng: A NumPy random number generator for the methyl rotation.

    Returns:
        A Molecule object.
    """
    points = np.linspace(0, 1, num_points)
    angle = ContinuousIntegrator(points)
    v6 = np.array([-0.5150, -0.8919, 5.7510])
    v7 = np.array([-0.5150, 0.8919, 5.7510])
    v8 = np.array([1.0299, 0, 5.7510])
    center = np.array([0, 0, 5.7510])

    cs_methyl_group6 = CircleVector(v6, center, v7, angle)
    cs_methyl_group7 = CircleVector(v7, center, v8, angle)
    cs_methyl_group8 = CircleVector(v8, center, v6, angle)

    coordinates = {
        "C0": StaticVector(0, 0, 1.0866),
        "H1": StaticVector(0, 0, 0),
        "H2": StaticVector(0, 2.1486, 1.2417),
        "H3": StaticVector(0, -2.1486, 1.2417),
        "H4": StaticVector(0, 2.1290, 3.7051),
        "H5": StaticVector(0, -2.1290, 3.7051),
        "H6": cs_methyl_group6,
        "H7": cs_methyl_group7,
        "H8": cs_methyl_group8,
    }

    molecule = Molecule(coordinates=coordinates, parameters=[angle])
    return molecule


def generate_shifted_hamiltonians_toluene(
    shifts: np.ndarray,
    saupe_tensor: np.ndarray = np.array([-0.1758 + 0.0176, -0.0176, 0.1758]),
    v1: np.ndarray = np.array((0, 0, 1)),  # Example shift vector 1 (adjust as needed)
    l1: List[str] = ["C0"],  # Example list of labels 1
    v2: np.ndarray = np.array((0, 0, 1)),  # Example shift vector 2
    l2: List[str] = ["H4", "H5", "H6", "H7", "H8"],  # Example List of labels 2
    num_samples: int = 10000,
    rng_seed: int = 42,
):
    """Generates a grid of shifted Hamiltonians for toluene and saves them to a file.

    Args:
        num_shifts: The number of shifts to apply in each direction (default: 21).
        order_parameters: Order parameters for coupling calculations.
    """

    rng = np.random.RandomState(rng_seed)
    molecule = make_toluene_molecule(rng)
    labels = ["C0", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"]

    hamiltonians = {}

    for d1 in shifts:
        print(d1)
        for d2 in tqdm(shifts):
            shifted_molecule = molecule.subset(labels)  # Create a copy to modify

            for label in l1:
                shifted_molecule.coordinates[label] += d1 * v1
            for label in l2:
                shifted_molecule.coordinates[label] += d2 * v2

            hom, het = make_coupling_matrices(
                shifted_molecule,
                labels,
                order_parameters=saupe_tensor,
                num_samples=num_samples,
            )

            hamiltonians[(d1, d2)] = (hom, het)
    return hamiltonians


def run_dream_2_otoc(
    msmt_gate: cirq.Gate,
    msmt_ids: List[int],
    bfly_gate: cirq.Gate,
    bfly_ids: List[int],
    hom: np.ndarray,
    het: np.ndarray,
    onsite_vec: np.ndarray,
    J_hom: np.ndarray,
    rng: np.random.RandomState,
    num_pulse_range: np.ndarray,
    pulse_time: float = 0.0105,
    rf_std: Optional[float] = 0,
    num_repetitions=1,
):
    """Runs the DREAM-2 sequence to generate an OTOC

    We calculate Trace[M U* B U M U* B U] where U is the evolution generated
    by the DREAM-2 sequence - this is approximated by time evolution

    U = \prod_j exp(i H_j t), where H_j = H + V_j for V_j an RF pulse.

    ***Be careful with units with Hamiltonian data, this assumes
    Pauli operators instead of spin operators and rad s^-1 units, so compared
    to e.g. the Berkeley data coupling terms should be multiplied by (2 * pi) / 4
    and onsite terms by (2 * pi) / 2

    Args:
        msmt_gate: M in the above equation acting on a single qubit
        msmt_ids: spins that M acts on (assume everything else is the identity)
        bfly_gate: B in the above equation acting on a single qubit
        bfly_ids: spins that B acts on
        hom: homogeneous coupling matrix in units rad s^-1
        het: heterogeneous coupling matrix in units rad s^-1
        onsite_vec: chemical shifts in units rad s^-1
        J_hom: homogeneous J-couplings in units rad s^-1 (heterogeneous J-couplings
            can be added to het)
        num_pulse_range: range of values k to use (the experiment returns a set of
            k * dt where dt is pulse_time * 36)
        pulse_time: dt in the above
        rf_std: standard deviation to draw RF noise from
        num_repetitions: number of times to repeat the experiment for averaging over
            RF inhomogeneity
        rng: random number generator

    Returns:
        otocs: list of lists of otocs for different draws of the RF inhomogeneity

    """
    labels = ["C"] + (["H"] * 8)
    qubits = cirq.LineQubit.range(9)
    M_circuit = cirq.Circuit([msmt_gate(qubits[qid]) for qid in msmt_ids])
    M = M_circuit.unitary(qubit_order=qubits)
    B_circuit = cirq.Circuit([bfly_gate(qubits[qid]) for qid in bfly_ids])
    B = B_circuit.unitary(qubit_order=qubits)

    ham = construct_nmr_hamiltonian_matrix(hom, het, onsite_vec, J_hom)

    otocs = []
    for _ in tqdm(np.arange(num_repetitions)):
        if abs(rf_std) < 1e-12:
            frac_error = 0
        else:
            frac_error = rng.normal(0, rf_std)
        otocs.append([])

        circuit = dream_2_sequence_CH(
            labels=labels,
            qubits=qubits,
            pulse_time=pulse_time,
            background_H=ham,
            fractional_pulse_error=frac_error,
        )
        inv_circuit, start_pulse, end_pulse = dream_2_inverse_sequence_CH(
            labels=labels,
            qubits=qubits,
            pulse_time=pulse_time,
            background_H=ham,
            fractional_pulse_error=frac_error,
        )
        unitary_block = circuit.unitary(qubit_order=qubits)
        inv_unitary_block = inv_circuit.unitary(qubit_order=qubits)
        start_pulse_block = start_pulse.unitary(qubit_order=qubits)
        end_pulse_block = end_pulse.unitary(qubit_order=qubits)

        for k in num_pulse_range:

            if k == 0:
                otocs[-1].append(1)
                continue

            U = np.linalg.matrix_power(unitary_block, k)
            Udag = (
                end_pulse_block
                @ np.linalg.matrix_power(inv_unitary_block, k)
                @ start_pulse_block
            )

            otoc_matrix = Udag @ B @ U @ M @ U.conj().T @ B.conj().T @ Udag.conj().T @ M
            otoc = np.trace(otoc_matrix).real / (2 ** len(qubits))
            otocs[-1].append(otoc)

    return otocs
