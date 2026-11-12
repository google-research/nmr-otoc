# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Callable, Dict, List, Optional, Tuple
import cirq
import numpy as np

from nmrotoc.circuits.gates import dq_gate_fsim
from nmrotoc.circuits.compilation import (
    safe_concatenate_circuits,
    check_separated_single_and_two_qubit_layers,
)
from nmrotoc.dataclasses import (
    NMROTOCExperiment,
    DoubleQuantumSwapNetworkObj,
    DoubleQuantumOrderedProductObj,
    DoubleQuantumXZYObj,
    SwapNetworkProductFormulaObj,
    NestedProductFormulaObj,
)

I = np.identity(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
P = {"X": X, "Y": Y, "Z": Z, "I": I}
P2 = {p1 + p2: np.kron(P[p1], P[p2]) for p1 in P for p2 in P}
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def swap_network(qubits: List[cirq.GridQubit]):
    """Makes a single round of a swap network for a linear chain"""
    layers = [
        [(q0, q1) for q0, q1 in zip(qubits[::2], qubits[1::2])],
        [(q0, q1) for q0, q1 in zip(qubits[1::2], qubits[2::2])],
    ]
    swap_network = cirq.Circuit()
    for j in range(len(qubits)):
        for q0, q1 in layers[j % 2]:
            gate = dq_gate_fsim(1, q0, q1, 0, 0, swap=True)
            swap_network.append(gate)
    swap_network = cirq.merge_single_qubit_gates_to_phxz(swap_network)
    swap_network = cirq.align_right(swap_network)
    return swap_network


def sorting_circuit(qubits: list[cirq.Qid], qubit_id_refs: Dict):
    """Performs a series of swaps to sort a list of qubits

    Args:
        qubits: list of qubits
        qubit_id_refs: double-ended dictionary linking qubits to
            a corresponding spin id (which needs to come from an
            ordered set). These will be shuffled
            via swap gates so that the ordering on the spins
            corresponds to the ordering on qubits
    """
    spins = list(sorted([qubit_id_refs[qubit] for qubit in qubits]))
    sorting_circuit = cirq.Circuit()
    for spin, qubit in zip(spins, qubits):
        # Check whether spin is in the appropriate qubit
        if qubit_id_refs[spin] == qubit:
            continue

        qubit2 = qubit_id_refs[spin]
        spin2 = qubit_id_refs[qubit]
        assert spin2 > spin
        sorting_circuit.append(cirq.SWAP(qubit, qubit2))

        # Update ids
        qubit_id_refs[spin] = qubit
        qubit_id_refs[qubit] = spin
        qubit_id_refs[spin2] = qubit2
        qubit_id_refs[qubit2] = spin2
    return sorting_circuit


def get_swap_network_ordering(indices):
    """Returns ordering of gates in swap network"""
    indices = list(indices)
    ordering = {}
    idx = 0
    for j in range(len(indices)):
        for k in range(j % 2, len(indices), 2):
            if k == len(indices) - 1:
                continue
            ordering[(indices[k], indices[k + 1])] = idx
            ordering[(indices[k + 1], indices[k])] = idx
            indices[k], indices[k + 1] = indices[k + 1], indices[k]
            idx += 1
    return ordering


def _create_swap_network_dq_step(
    qubits: List[cirq.Qid],
    layers: List[List[Tuple[cirq.Qid, cirq.Qid]]],
    qubit_id_refs: Dict,
    het_mat: np.ndarray,
    hom_mat: np.ndarray,
    z_mat: Optional[np.ndarray] = None,
    num_layers: Optional[int] = None,
    return_pairs: Optional[bool] = False,
    concatenate: Optional[bool] = True,
    xyyx_flag: Optional[bool] = False,
):
    """Creates a single step of a swap network for a double quantum Hamiltonian.

    This function constructs a circuit representing one step of a swap network,
    where each layer of the network consists of `dq_gate_fsim` gates applied
    to pairs of qubits. It also handles qubit reordering and optional single-qubit
    Z rotations.

    Args:
        qubits: A list of `cirq.Qid` objects representing the qubits in the system.
        layers: A pair of lists; each list contains tuples of
            `(cirq.Qid, cirq.Qid)` pairs representing the qubit pairs for each layer
            of the swap network.
        qubit_id_refs: A dictionary mapping `cirq.Qid` objects to their corresponding
            integer indices and vice versa. This dictionary is updated in place to
            reflect the qubit reordering caused by the swap network.
        het_mat: A NumPy array representing the heterogeneous coupling matrix.
        hom_mat: A NumPy array representing the homogeneous coupling matrix.
        z_mat: Single-qubit Z rotation angles.
            If provided, z-rotations are applied before two-qubit interactions.
        return_pairs: flag whether to return an ordered list of pairs of interactions.
        concatenate: flag whether to append or concatenate subsequent layers
    Returns:
        A `cirq.Circuit` object representing a single step of the swap network.
        The `qubit_id_refs` dictionary is updated in place to reflect the qubit
        reordering.
    """
    # Set up storage
    step_circuit = cirq.Circuit()
    if return_pairs is True:
        pairs = []

    # A full swap network has length equal to the number of qubits N.
    if num_layers is None:
        num_layers = len(qubits)
    for j in np.arange(num_layers):

        # We make each step separately to make compilation easier.
        single_step = cirq.Circuit()
        for q0, q1 in layers[j % 2]:

            # Find interaction terms and coefficients
            i0 = qubit_id_refs[q0]
            i1 = qubit_id_refs[q1]
            if return_pairs is True:
                pairs.append((i0, i1))
            het = het_mat[i0, i1]
            hom = hom_mat[i0, i1]

            # Add z-phases if present
            if z_mat is not None:
                z0, z1 = z_mat[i0, i1]
                gate0 = cirq.rz(-z0 * 2).on(q0)
                gate1 = cirq.rz(-z1 * 2).on(q1)
                single_step.append([gate0, gate1])

            if xyyx_flag:
                single_step.append(cirq.rz(-np.pi / 2).on(q0))

            # Make gate (n.b. this is a composite gate) and add to circuit
            gate = dq_gate_fsim(1, q0, q1, het, -hom, swap=True)
            single_step.append(gate)

            if xyyx_flag:
                single_step.append(cirq.rz(np.pi / 2).on(q1))

            # Update qubit-spin mapping
            qubit_id_refs[q0] = i1
            qubit_id_refs[q1] = i0
            qubit_id_refs[i0] = q1
            qubit_id_refs[i1] = q0

        # Compile circuit before adding to larger circuit in a way that preserves
        # single and two-qubit layer structure.
        single_step = cirq.drop_empty_moments(
            cirq.align_right(cirq.merge_single_qubit_gates_to_phxz(single_step))
        )
        check_separated_single_and_two_qubit_layers([single_step], strict=False)
        if concatenate:
            step_circuit = safe_concatenate_circuits(step_circuit, single_step)
        else:
            step_circuit.append(
                single_step, strategy=cirq.InsertStrategy.NEW_THEN_INLINE
            )

    # Check that circuit is correctly compiled.
    check_separated_single_and_two_qubit_layers([step_circuit], strict=False)
    if return_pairs:
        return step_circuit, pairs
    return step_circuit


def make_butterfly_circuit(
    experiment: NMROTOCExperiment,
    qubit_id_refs: Dict,
):
    """Makes butterfly circuit for the final step, accounting for qubit rearrangement"""
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
        butterfly = None
    return butterfly


def compile_nested_product_formula(
    pf_obj: NestedProductFormulaObj,
    qubits: List[cirq.Qid],
    return_pairs: Optional[bool] = False,
):
    """Compiles a nested product formula by looping over terms.

    Args:
        pf_obj: object defining the product formula. Need to have
            pf.sub_terms containing a list of ProductFormulaObj with
            sub_term.callable defined, and pf.qid_list containing a list
            of qids that are passed to each sub-term.
        qubits: qubits to act on
    """
    # Generate list for editing in-place (we allow subunitaries to rearrange qubits)
    qubits = list(qubits)

    # unitary will hold the circuit as we create it
    unitary = cirq.Circuit()
    if return_pairs is True:
        pairs = []
    for _ in range(pf_obj.num_steps):
        for sub_term, qids in zip(pf_obj.sub_terms, pf_obj.qid_list):

            # Pull the qubits that this term will act on
            sub_qubit_list = [qubits[q] for q in qids]

            # Generate the sub-unitary and add it to the larger circuit
            data_out = sub_term.generating_function(
                sub_term, sub_qubit_list, return_pairs=return_pairs
            )
            sub_unitary = data_out[0]
            relabeling_dic = data_out[1]
            if pf_obj.concatenate:
                unitary = safe_concatenate_circuits(unitary, sub_unitary)
            else:
                unitary.append(
                    sub_unitary, strategy=cirq.InsertStrategy.NEW_THEN_INLINE
                )

            # Concatenate pairs if needed
            if return_pairs is True:
                pairs += data_out[2]

            # We allow the sub-unitary to rearrange qubits; this is captured
            # in the relabeling dic, which we update here.
            for j, qid in enumerate(qids):
                qubits[qid] = relabeling_dic[j]

    # We pass the qubit list as a double-ended dictionary for ease of access
    # and backwards compatibility
    qubit_id_dic = {
        x: y for j, qubit in enumerate(qubits) for x, y in [(j, qubit), (qubit, j)]
    }
    if return_pairs is True:
        return unitary, qubit_id_dic, pairs
    return unitary, qubit_id_dic


def make_product_formula_swap_network_dq_hamiltonian(
    trotter_obj: DoubleQuantumSwapNetworkObj,
    qubits: List[cirq.Qid],
    return_pairs: Optional[bool] = False,
):
    """Compiles K Trotter steps for a dipolar Hamiltonian on a linear chain.

    Each step utilizes a swap network, and implements
    prod_{j,k}exp(iH_{r,j,k}c_{r,j,k}), where H_{r,j,k} is a dipolar Hamiltonian
    and the coefficient c_{r,j,k} is found in trotter_obj.homogeneous_step_mats
    and trotter_obj.heterogeneous_step_mats.

    Butterfly qubit indices are tracked, and the final butterfly is returned at the
    end as well.

    Args:
        trotter_obj: DoubleQuantumSwapNetworkObj dataclass containing details of trotterization.
            See nmrotoc.dataclasses for more details
        qubits: either a list of qubits for all steps to act on, or a list of lists
            of qubits for each step. The order matters here as it defines the 1D
            chain that the swap network acts on.
        return_ids: flag for whether to return a dictionary mapping qubits to their
            corresponding original indices.
    """
    if trotter_obj.z_step_mats is None:
        trotter_obj.z_step_mats = [None] * trotter_obj.num_steps

    # Broadcast qubits to cover permutations
    permutations = trotter_obj.permutations or [
        range(len(qubits)) for _ in range(trotter_obj.num_steps)
    ]
    assert len(permutations) == trotter_obj.num_steps
    q_perms = [[qubits[j] for j in perm] for perm in permutations]

    if not (isinstance(trotter_obj.alternate_order, list)):
        trotter_obj.alternate_order = [
            trotter_obj.alternate_order for _ in range(trotter_obj.num_steps)
        ]

    # Make double-ended index dictionary
    qubit_id_refs = {}
    for j, qubit in enumerate(qubits):
        qubit_id_refs[qubit] = j
        qubit_id_refs[j] = qubit

    unitary = cirq.Circuit()
    if return_pairs is True:
        pairs = []
    for k in range(trotter_obj.num_steps):

        # Order parity determines whether to flip layers every second product step
        # The swap network flips the order of the spins in the qubit chain each
        # time. Thus, if the number of qubits is even, using the same layers
        # for each step will perform gates between spins in the same order for each
        # step, whereas if the number of qubits is odd, the gates will be applied in
        # reverse order every second step --- we call this "alternate order".
        order_parity = int(trotter_obj.alternate_order[k]) ^ (len(qubits) % 2)

        # Set up layers of gates for swap network.
        layers = [
            [(q0, q1) for q0, q1 in zip(q_perms[k][::2], q_perms[k][1::2])],
            [(q0, q1) for q0, q1 in zip(q_perms[k][1::2], q_perms[k][2::2])],
        ]

        # Switch the order of layers if needed
        if trotter_obj.reverse:
            layers = [layers[1], layers[0]]
        if order_parity == 1 and k % 2 == 1:
            layers = [layers[1], layers[0]]

        # Construct single swap network using subroutine
        trotter_step = _create_swap_network_dq_step(
            qubits,
            layers,
            qubit_id_refs,
            trotter_obj.heterogeneous_step_mats[k],
            trotter_obj.homogeneous_step_mats[k],
            trotter_obj.z_step_mats[k],
            return_pairs=return_pairs,
            concatenate=trotter_obj.concatenate,
            xyyx_flag=getattr(trotter_obj, "xyyx_flag", False),
        )
        if return_pairs is True:
            pairs += trotter_step[1]
            trotter_step = trotter_step[0]
        if trotter_obj.concatenate:
            unitary = safe_concatenate_circuits(unitary, trotter_step)
        else:
            unitary.append(trotter_step, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    if return_pairs is True:
        return unitary, qubit_id_refs, pairs
    return unitary, qubit_id_refs


def make_product_formula_swap_network_two_term_split(
    trotter_obj: DoubleQuantumSwapNetworkObj,
    qubits: List[cirq.Qid],
    return_pairs: Optional[bool] = False,
):
    """Compiles K Trotter steps for a dipolar Hamiltonian on a linear chain.

    Each step utilizes a swap network, and implements
    prod_{j,k}exp(iH_{r,j,k}c_{r,j,k}), where H_{r,j,k} is a dipolar Hamiltonian
    and the coefficient c_{r,j,k} is found in trotter_obj.homogeneous_step_mats
    and trotter_obj.heterogeneous_step_mats.

    Butterfly qubit indices are tracked, and the final butterfly is returned at the
    end as well.

    Args:
        trotter_obj: DoubleQuantumSwapNetworkObj dataclass containing details of trotterization.
            See nmrotoc.dataclasses for more details
        qubits: either a list of qubits for all steps to act on, or a list of lists
            of qubits for each step. The order matters here as it defines the 1D
            chain that the swap network acts on.
        return_ids: flag for whether to return a dictionary mapping qubits to their
            corresponding original indices.
    """
    # Broadcast qubits to cover permutations
    permutations = trotter_obj.permutations or [
        range(len(qubits)) for _ in range(trotter_obj.num_steps)
    ]
    assert len(permutations) == trotter_obj.num_steps
    q_perms = [[qubits[j] for j in perm] for perm in permutations]

    # Make double-ended index dictionary
    qubit_id_refs = {}
    for j, qubit in enumerate(qubits):
        qubit_id_refs[qubit] = j
        qubit_id_refs[j] = qubit

    # Order parity determines whether to flip layers every second product step
    # The swap network flips the order of the spins in the qubit chain each
    # time. Thus, if the number of qubits is even, using the same layers
    # for each step will perform gates between spins in the same order for each step,
    # whereas if the number of qubits is odd, the gates will be applied in
    # reverse order every second step --- we call this "alternate order".
    order_parity = int(trotter_obj.alternate_order) ^ (len(qubits) % 2)

    trotter_steps = []
    if return_pairs is True:
        pairs = []
    for k in range(trotter_obj.num_steps):

        # Set up layers of gates for swap network.
        layers = [
            [(q0, q1) for q0, q1 in zip(q_perms[k][::2], q_perms[k][1::2])],
            [(q0, q1) for q0, q1 in zip(q_perms[k][1::2], q_perms[k][2::2])],
        ]

        # Switch the order of layers if needed
        reverse_flag = trotter_obj.reverse ^ (order_parity == 1 and k % 2 == 1)
        if reverse_flag:
            layers = [layers[1], layers[0]]

        # Construct single swap network for Z step
        step_circuit_z = _create_swap_network_dq_step(
            qubits,
            layers,
            qubit_id_refs,
            trotter_obj.heterogeneous_step_mats[k],
            np.zeros_like(trotter_obj.homogeneous_step_mats[k]),
            return_pairs=return_pairs,
        )
        if return_pairs is True:
            pairs += step_circuit_z[1]
            step_circuit_z = step_circuit_z[0]

        # Construct single swap network for X step
        step_circuit_x = cirq.Circuit(
            [cirq.ry(-np.pi / 2).on(qubit) for qubit in qubits]
        )
        middle_circuit_x = _create_swap_network_dq_step(
            qubits,
            layers,
            qubit_id_refs,
            trotter_obj.homogeneous_step_mats[k],
            np.zeros_like(trotter_obj.homogeneous_step_mats[k]),
            return_pairs=return_pairs,
        )
        if return_pairs is True:
            pairs += middle_circuit_x[1]
            middle_circuit_x = middle_circuit_x[0]
        if trotter_obj.concatenate:
            step_circuit_x = safe_concatenate_circuits(step_circuit_x, middle_circuit_x)
            step_circuit_x = safe_concatenate_circuits(
                step_circuit_x,
                cirq.Circuit([cirq.ry(np.pi / 2).on(qubit) for qubit in qubits]),
            )
        else:
            step_circuit_x.append(
                middle_circuit_x, strategy=cirq.InsertStrategy.NEW_THEN_INLINE
            )
            step_circuit_x.append(
                cirq.Circuit([cirq.ry(np.pi / 2).on(qubit) for qubit in qubits]),
                strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
            )

        # Construct single swap network
        step_circuit_y = cirq.Circuit(
            [cirq.rx(np.pi / 2).on(qubit) for qubit in qubits]
        )
        middle_circuit_y = _create_swap_network_dq_step(
            qubits,
            layers,
            qubit_id_refs,
            -trotter_obj.homogeneous_step_mats[k],
            np.zeros_like(trotter_obj.homogeneous_step_mats[k]),
            return_pairs=return_pairs,
        )
        if return_pairs is True:
            pairs += middle_circuit_y[1]
            middle_circuit_y = middle_circuit_y[0]
        if trotter_obj.concatenate:
            step_circuit_y = safe_concatenate_circuits(step_circuit_y, middle_circuit_y)
            step_circuit_y = safe_concatenate_circuits(
                step_circuit_y,
                cirq.Circuit([cirq.rx(-np.pi / 2).on(qubit) for qubit in qubits]),
            )
        else:
            step_circuit_y.append(
                middle_circuit_y, strategy=cirq.InsertStrategy.NEW_THEN_INLINE
            )
            step_circuit_y.append(
                cirq.Circuit([cirq.rx(-np.pi / 2).on(qubit) for qubit in qubits]),
                strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
            )

        # Tidy up circuit and add to list
        if (
            (trotter_obj.alternate_order is True and k % 2 == 1) ^ trotter_obj.reverse
        ) is True:
            trotter_steps.append(step_circuit_y)
            trotter_steps.append(step_circuit_x)
            trotter_steps.append(step_circuit_z)
        else:
            trotter_steps.append(step_circuit_z)
            trotter_steps.append(step_circuit_x)
            trotter_steps.append(step_circuit_y)

    # Compile circuit
    unitary = cirq.Circuit()
    for step in trotter_steps:
        if trotter_obj.concatenate:
            unitary = safe_concatenate_circuits(unitary, step)
        else:
            unitary.append(step, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    if return_pairs:
        return unitary, qubit_id_refs, pairs
    return unitary, qubit_id_refs


def first_order_trotterize_dq_hamiltonian(
    experiment: NMROTOCExperiment,
    time: float,
    num_steps: int,
) -> DoubleQuantumSwapNetworkObj:
    """
    Generates a DoubleQuantumSwapNetworkObj describing a first-order Trotterization
    of the given Hamiltonian for a fixed time and number of Trotter steps.

    Args:
        experiment: NMROTOCExperiment object containing the Hamiltonian information.
        time: The total evolution time.
        num_steps: The number of Trotter steps to use.
        qubits: a list of cirq.GridQubit to act on.

    Returns:
        A DoubleQuantumSwapNetworkObj object containing the Trotterized evolution parameters.
    """

    num_qubits = experiment.homogeneous_coupling_mat.shape[0]

    if experiment.onsite_vector is None:
        experiment.onsite_vector = np.zeros(num_qubits)

    dt = time / num_steps
    homogeneous_step_mats = [
        experiment.homogeneous_coupling_mat * dt for _ in range(num_steps)
    ]
    heterogeneous_step_mats = [
        experiment.heterogeneous_coupling_mat * dt for _ in range(num_steps)
    ]
    # We split the onsite terms across all N-1 couplings in a single
    z_step_mats = [
        np.array(
            [
                [
                    (
                        [
                            experiment.onsite_vector[i0] * dt / (num_qubits - 1),
                            experiment.onsite_vector[i1] * dt / (num_qubits - 1),
                        ]
                        if i1 != i0
                        else [0, 0]
                    )
                    for i1 in range(num_qubits)
                ]
                for i0 in range(num_qubits)
            ]
        )
        for _ in range(num_steps)
    ]

    trotter_circuit_obj = DoubleQuantumSwapNetworkObj(
        homogeneous_step_mats=homogeneous_step_mats,
        heterogeneous_step_mats=heterogeneous_step_mats,
        z_step_mats=z_step_mats,
        num_steps=num_steps,
        xyyx_flag=experiment.xyyx_flag,
    )

    return trotter_circuit_obj


def mixed_times_trotterize_dq_hamiltonian(
    experiment: NMROTOCExperiment, dts: List[float]
) -> DoubleQuantumSwapNetworkObj:
    """
    Generates a DoubleQuantumSwapNetworkObj describing a first-order Trotterization
    of the given Hamiltonian for a fixed time and number of Trotter steps.

    Args:
        experiment: NMROTOCExperiment object containing the Hamiltonian information.
        dts: List of different dts to use at every step.

    Returns:
        A DoubleQuantumSwapNetworkObj object containing the Trotterized evolution parameters.
    """

    homogeneous_step_mats = [experiment.homogeneous_coupling_mat * dt for dt in dts]
    heterogeneous_step_mats = [experiment.heterogeneous_coupling_mat * dt for dt in dts]

    trotter_circuit_obj = DoubleQuantumSwapNetworkObj(
        homogeneous_step_mats=homogeneous_step_mats,
        heterogeneous_step_mats=heterogeneous_step_mats,
        num_steps=len(dts),
        xyyx_flag=experiment.xyyx_flag,
    )

    return trotter_circuit_obj


def make_trotter_steps_ordered(
    trotter_circuit_obj: DoubleQuantumOrderedProductObj,
    qubits: List[cirq.Qid],
    return_pairs: Optional[bool] = False,
) -> cirq.Circuit:
    """Constructs a Trotterized time evolution circuit.

    This function takes a DoubleQuantumOrderedProductObj and a list of qubits,
    and generates a Trotterized time evolution circuit by iterating over the
    gate_ordering and calling dq_gate_fsim with the appropriate parameters
    from homogeneous_step_mats and heterogeneous_step_mats.

    Args:
        trotter_circuit_obj: DoubleQuantumOrderedProductObj containing Trotter step matrices, etc.
        qubits: List of cirq.Qid objects representing the qubits.

    Returns:
        the Trotterized unitary circuit
    """

    trotter_steps = []
    if return_pairs:
        pairs = []
    for k in range(trotter_circuit_obj.num_steps):
        step_circuit = cirq.Circuit()
        for i0, i1 in trotter_circuit_obj.gate_ordering[k]:
            q0 = qubits[i0]
            q1 = qubits[i1]
            if return_pairs is True:
                pairs.append((i0, i1))

            hom = trotter_circuit_obj.homogeneous_step_mats[k][i0, i1]
            het = trotter_circuit_obj.heterogeneous_step_mats[k][i0, i1]

            step_circuit.append(dq_gate_fsim(1, q0, q1, het, hom, swap=False))

        step_circuit = cirq.merge_single_qubit_gates_to_phxz(step_circuit)
        step_circuit = cirq.align_right(step_circuit)
        trotter_steps.append(step_circuit)

    unitary = cirq.Circuit(trotter_steps)

    if return_pairs:
        return unitary, pairs
    return unitary


def first_order_randomized_trotterize(
    experiment: NMROTOCExperiment, time: float, num_steps: int, seed: int
) -> DoubleQuantumOrderedProductObj:
    """
    Generates a DoubleQuantumOrderedProductObj for a first-order Trotter formula
    with randomized term ordering in each layer.

    Args:
        experiment: NMROTOCExperiment object containing Hamiltonian info.
        time: Total evolution time.
        num_steps: Number of Trotter steps.
        seed: Seed for random number generation.

    Returns:
        DoubleQuantumOrderedProductObj with randomized Trotter parameters.
    """

    rng = np.random.RandomState(seed)
    num_qubits = experiment.homogeneous_coupling_mat.shape[0]
    dt = time / num_steps

    homogeneous_step_mats = [
        experiment.homogeneous_coupling_mat * dt for _ in range(num_steps)
    ]
    heterogeneous_step_mats = [
        experiment.heterogeneous_coupling_mat * dt for _ in range(num_steps)
    ]

    # Generate randomized gate orderings
    gate_ordering = []
    for _ in range(num_steps):
        all_pairs = [
            (i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)
        ]

        rng.shuffle(all_pairs)
        gate_ordering.append(all_pairs)

    trotter_circuit_obj = DoubleQuantumOrderedProductObj(
        homogeneous_step_mats=homogeneous_step_mats,
        heterogeneous_step_mats=heterogeneous_step_mats,
        gate_ordering=gate_ordering,
        num_steps=num_steps,
    )
    return trotter_circuit_obj


def _check_xzy_supported(experiment: NMROTOCExperiment):
    """Raises if the experiment requests a Hamiltonian the XZY product formula cannot represent."""
    if getattr(experiment, "xyyx_flag", False):
        raise NotImplementedError(
            "The XZY product formula implements the DREAM-2 homonuclear term "
            "D_ij (Y_i Y_j - X_i X_j); xyyx_flag=True (DREAM-1, X_i Y_j + Y_i X_j) "
            "is not supported."
        )


def _create_xzy_dq_step(
    qubits: List[cirq.Qid],
    qubit_id_refs: Dict,
    het_mat: np.ndarray,
    hom_mat: np.ndarray,
    z_mat: Optional[np.ndarray] = None,
    reverse: Optional[bool] = False,
    return_pairs: Optional[bool] = False,
    concatenate: Optional[bool] = True,
    float_tol: float = 1e-12,
):
    """Creates a single step of an XZY product formula for a double quantum Hamiltonian.

    Implements the single-step unitary:
        [ prod_{i<j} exp(-i * hom_mat[i,j] * Y_i Y_j) ]
        * [ prod_{i<j} exp(-i * het_mat[i,j] * Z_i Z_j) ]
        * [ prod_{i<j} exp(+i * hom_mat[i,j] * X_i X_j) ]
    (or in reversed Y-Z-X order when `reverse=True`), so that in Cirq circuit order
    the XX layer acts first on the state vector, followed by the ZZ layer, followed
    by the YY layer.

    Args:
        qubits: A list of `cirq.Qid` objects representing the qubits in the system.
        qubit_id_refs: A dictionary mapping `cirq.Qid` objects to their corresponding
            integer spin indices and vice versa.
        het_mat: A NumPy array representing the heterogeneous coupling step matrix (dt * D_het).
        hom_mat: A NumPy array representing the homogeneous coupling step matrix (dt * D_hom).
        z_mat: Optional single-qubit Z rotation angles for the step.
        reverse: Flag whether to apply the layers in reversed (Y-Z-X) order.
        return_pairs: Flag whether to return an ordered list of interacting spin pairs.
        concatenate: Flag whether to merge layers using `safe_concatenate_circuits`.
        float_tol: Threshold below which coupling angles are treated as zero.

    Returns:
        A `cirq.Circuit` object representing a single XZY Trotter step (and `pairs` if
        `return_pairs=True`).
    """
    num_qubits = len(qubits)
    if return_pairs is True:
        pairs = []

    ops_x = []
    ops_z = []
    ops_y = []
    pairs_x = []
    pairs_z = []
    pairs_y = []

    for i0 in range(num_qubits):
        q0 = qubit_id_refs[i0]
        for i1 in range(i0 + 1, num_qubits):
            q1 = qubit_id_refs[i1]
            hom = hom_mat[i0, i1]
            het = het_mat[i0, i1]

            if abs(hom) > float_tol:
                # exp(+i * hom * X_i X_j) = XXPowGate(exponent = -2 * hom / pi, global_shift = -0.5)
                ops_x.append(
                    cirq.XXPowGate(
                        exponent=-2.0 * float(hom) / np.pi, global_shift=-0.5
                    ).on(q0, q1)
                )
                # exp(-i * hom * Y_i Y_j) = YYPowGate(exponent = +2 * hom / pi, global_shift = -0.5)
                ops_y.append(
                    cirq.YYPowGate(
                        exponent=2.0 * float(hom) / np.pi, global_shift=-0.5
                    ).on(q0, q1)
                )
                if return_pairs is True:
                    pairs_x.append((i0, i1))
                    pairs_y.append((i0, i1))

            if abs(het) > float_tol:
                # exp(-i * het * Z_i Z_j) = ZZPowGate(exponent = +2 * het / pi, global_shift = -0.5)
                ops_z.append(
                    cirq.ZZPowGate(
                        exponent=2.0 * float(het) / np.pi, global_shift=-0.5
                    ).on(q0, q1)
                )
                if return_pairs is True:
                    pairs_z.append((i0, i1))

    step_circuit_x = cirq.Circuit(ops_x)
    step_circuit_z = cirq.Circuit(ops_z)
    step_circuit_y = cirq.Circuit(ops_y)

    # Include optional single-qubit onsite Z rotations in the Z layer if non-zero
    if z_mat is not None and np.any(np.abs(z_mat) > float_tol):
        z_ops = []
        if z_mat.ndim == 1:
            for i0 in range(num_qubits):
                if abs(z_mat[i0]) > float_tol:
                    z_ops.append(cirq.rz(2.0 * float(z_mat[i0])).on(qubit_id_refs[i0]))
        else:
            for i0 in range(num_qubits):
                z_angle = 0.0
                for i1 in range(i0 + 1, num_qubits):
                    z0, _ = z_mat[i0, i1]
                    z_angle += float(z0)
                for i1 in range(i0):
                    _, z1 = z_mat[i1, i0]
                    z_angle += float(z1)
                if abs(z_angle) > float_tol:
                    z_ops.append(cirq.rz(2.0 * z_angle).on(qubit_id_refs[i0]))
        if z_ops:
            z_1q_circuit = cirq.Circuit(z_ops)
            if concatenate:
                step_circuit_z = safe_concatenate_circuits(z_1q_circuit, step_circuit_z)
            else:
                step_circuit_z = z_1q_circuit + step_circuit_z

    if reverse:
        ordered_layers = [step_circuit_y, step_circuit_z, step_circuit_x]
        if return_pairs is True:
            pairs = pairs_y + pairs_z + pairs_x
    else:
        ordered_layers = [step_circuit_x, step_circuit_z, step_circuit_y]
        if return_pairs is True:
            pairs = pairs_x + pairs_z + pairs_y

    step_circuit = cirq.Circuit()
    for layer in ordered_layers:
        if len(layer) == 0:
            continue
        if (
            concatenate
            and len(step_circuit) > 0
            and any(len(op.qubits) == 1 for op in step_circuit[-1])
            and any(len(op.qubits) == 1 for op in layer[0])
        ):
            step_circuit = safe_concatenate_circuits(step_circuit, layer)
        else:
            step_circuit += layer

    if return_pairs is True:
        return step_circuit, pairs
    return step_circuit


def make_product_formula_xzy_dq_hamiltonian(
    trotter_obj: DoubleQuantumXZYObj,
    qubits: List[cirq.Qid],
    return_pairs: Optional[bool] = False,
):
    """Compiles K Trotter steps for a double quantum Hamiltonian in the XZY format.

    Each step k implements:
        [ prod_{i<j} exp(-i * hom_step_mats[k][i,j] * Y_i Y_j) ]
        * [ prod_{i<j} exp(-i * het_step_mats[k][i,j] * Z_i Z_j) ]
        * [ prod_{i<j} exp(+i * hom_step_mats[k][i,j] * X_i X_j) ]
    without a swap network, preserving the identity mapping between spin indices
    and qubits in `qubit_id_refs`.

    Args:
        trotter_obj: DoubleQuantumXZYObj dataclass containing details of the Trotterization.
        qubits: List of `cirq.Qid` objects for all steps to act on.
        return_pairs: Flag for whether to return the ordered list of interacting pairs.

    Returns:
        Tuple `(unitary, qubit_id_refs)` (or `(unitary, qubit_id_refs, pairs)` if
        `return_pairs=True`).
    """
    if trotter_obj.z_step_mats is None:
        z_step_mats = [None] * trotter_obj.num_steps
    else:
        z_step_mats = trotter_obj.z_step_mats

    if isinstance(trotter_obj.alternate_order, list):
        alternate_order = trotter_obj.alternate_order
    else:
        alternate_order = [
            bool(trotter_obj.alternate_order) for _ in range(trotter_obj.num_steps)
        ]

    # Make double-ended index dictionary
    qubit_id_refs = {}
    for j, qubit in enumerate(qubits):
        qubit_id_refs[qubit] = j
        qubit_id_refs[j] = qubit

    unitary = cirq.Circuit()
    if return_pairs is True:
        pairs = []

    for k in range(trotter_obj.num_steps):
        reverse_step = bool(trotter_obj.reverse) ^ (
            bool(alternate_order[k]) and (k % 2 == 1)
        )
        trotter_step = _create_xzy_dq_step(
            qubits,
            qubit_id_refs,
            trotter_obj.heterogeneous_step_mats[k],
            trotter_obj.homogeneous_step_mats[k],
            z_step_mats[k],
            reverse=reverse_step,
            return_pairs=return_pairs,
            concatenate=trotter_obj.concatenate,
        )
        if return_pairs is True:
            pairs += trotter_step[1]
            trotter_step = trotter_step[0]
        if (
            trotter_obj.concatenate
            and len(unitary) > 0
            and len(trotter_step) > 0
            and any(len(op.qubits) == 1 for op in unitary[-1])
            and any(len(op.qubits) == 1 for op in trotter_step[0])
        ):
            unitary = safe_concatenate_circuits(unitary, trotter_step)
        else:
            unitary += trotter_step

    if return_pairs is True:
        return unitary, qubit_id_refs, pairs
    return unitary, qubit_id_refs


def first_order_trotterize_xzy_dq_hamiltonian(
    experiment: NMROTOCExperiment,
    time: float,
    num_steps: int,
) -> DoubleQuantumXZYObj:
    """Generates a DoubleQuantumXZYObj describing a first-order XZY Trotterization
    of the given double quantum Hamiltonian for a fixed time and number of Trotter steps.

    Convention: the resulting circuit approximates exp(-i H t) with
    H = sum_hom D_ij (Y_i Y_j - X_i X_j) + sum_het D_ij Z_i Z_j. Note that
    nmrotoc.math.operators.calculate_true_unitary uses exp(+i H t) and the swap-network
    dq_gate_fsim implements exp(+i dt [hom (XX - YY) + het ZZ]); these conventions give
    identical X/X OTOCs but differ at the level of unitaries and states.

    Args:
        experiment: NMROTOCExperiment object containing the Hamiltonian information.
        time: The total evolution time.
        num_steps: The number of Trotter steps to use.

    Returns:
        A DoubleQuantumXZYObj object containing the Trotterized evolution parameters.

    Raises:
        NotImplementedError: if experiment.xyyx_flag is set (XY + YX homonuclear terms).
    """
    _check_xzy_supported(experiment)
    num_qubits = experiment.homogeneous_coupling_mat.shape[0]

    if experiment.onsite_vector is None:
        experiment.onsite_vector = np.zeros(num_qubits)

    dt = time / num_steps
    homogeneous_step_mats = [
        experiment.homogeneous_coupling_mat * dt for _ in range(num_steps)
    ]
    heterogeneous_step_mats = [
        experiment.heterogeneous_coupling_mat * dt for _ in range(num_steps)
    ]
    z_step_mats = [
        np.array(
            [
                [
                    (
                        [
                            experiment.onsite_vector[i0] * dt / (num_qubits - 1),
                            experiment.onsite_vector[i1] * dt / (num_qubits - 1),
                        ]
                        if i1 != i0
                        else [0, 0]
                    )
                    for i1 in range(num_qubits)
                ]
                for i0 in range(num_qubits)
            ]
        )
        for _ in range(num_steps)
    ]

    trotter_circuit_obj = DoubleQuantumXZYObj(
        homogeneous_step_mats=homogeneous_step_mats,
        heterogeneous_step_mats=heterogeneous_step_mats,
        z_step_mats=z_step_mats,
        num_steps=num_steps,
        generating_function=make_product_formula_xzy_dq_hamiltonian,
    )

    return trotter_circuit_obj


def mixed_times_trotterize_xzy_dq_hamiltonian(
    experiment: NMROTOCExperiment, dts: List[float]
) -> DoubleQuantumXZYObj:
    """Generates a DoubleQuantumXZYObj describing a first-order XZY Trotterization
    of the given double quantum Hamiltonian with a custom list of step sizes `dts`.

    Uses the same exp(-i H t) convention as first_order_trotterize_xzy_dq_hamiltonian.

    Args:
        experiment: NMROTOCExperiment object containing the Hamiltonian information.
        dts: List of different dts to use at every step.

    Returns:
        A DoubleQuantumXZYObj object containing the Trotterized evolution parameters.

    Raises:
        NotImplementedError: if experiment.xyyx_flag is set (XY + YX homonuclear terms).
    """
    _check_xzy_supported(experiment)
    homogeneous_step_mats = [experiment.homogeneous_coupling_mat * dt for dt in dts]
    heterogeneous_step_mats = [experiment.heterogeneous_coupling_mat * dt for dt in dts]

    trotter_circuit_obj = DoubleQuantumXZYObj(
        homogeneous_step_mats=homogeneous_step_mats,
        heterogeneous_step_mats=heterogeneous_step_mats,
        num_steps=len(dts),
        generating_function=make_product_formula_xzy_dq_hamiltonian,
    )

    return trotter_circuit_obj

