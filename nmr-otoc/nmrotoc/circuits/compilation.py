# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

import cirq
import numpy as np
from typing import List, Type, Union, Optional


def fsimgate_to_three_czs(gate, check=False):
    """Converts a cirq.FSimGate to three CZ gates and Cirq.PhasedXZGates"""
    # Pull out parameters
    q0, q1 = gate.qubits
    phi = gate.gate.phi
    theta = gate.gate.theta
    z = phi / 4
    x = theta / 2
    y = theta / 2

    def phxz(x, a, z):
        return cirq.PhasedXZGate(x_exponent=x, axis_phase_exponent=a, z_exponent=z)

    # Implement np.exp(1j * [rx * XX + ry * YY + rz * ZZ])
    weyl_piece = cirq.Circuit(
        [
            cirq.S(q1),
            cirq.H(q0),
            cirq.CZ(q0, q1),
            cirq.H(q0),
            cirq.Z(q0) ** (z * 2 / np.pi - 0.5),
            cirq.Y(q1) ** (x * 2 / np.pi - 0.5),
            cirq.H(q1),
            cirq.CZ(q0, q1),
            cirq.H(q1),
            cirq.Y(q1) ** (0.5 - y * 2 / np.pi),
            cirq.H(q0),
            cirq.CZ(q0, q1),
            cirq.H(q0),
            cirq.S(q0) ** -1,
        ]
    )
    # Add single-qubit rotations from phi
    G0R = cirq.rz(-phi / 2).on(q0)
    G1R = cirq.rz(-phi / 2).on(q1)

    # Compile and compress
    full_gate = cirq.Circuit([weyl_piece, G0R, G1R])
    full_gate = cirq.align_right(cirq.merge_single_qubit_gates_to_phxz(full_gate))

    # Test if needed
    if check:
        full_unitary = full_gate.unitary([q0, q1])
        target_full_unitary = cirq.Circuit([gate]).unitary([q0, q1])
        assert (
            np.abs(np.trace(full_unitary.conj().T @ target_full_unitary)) - 4
        ) < 1e-6
    return full_gate


def get_moment_type(moment: cirq.Moment):
    """Determines if a moment is a single qubit or two qubit moment"""
    test_set = set([len(gate.qubits) for gate in moment])
    if len(test_set) != 1:
        print(moment)
        raise ValueError("Circuit not in correct form")
    (moment_type,) = test_set
    return moment_type


def circuit_to_cz_gates(circuit):
    """Converts a circuit consisting of FSimGates and single qubit rotations to CZs"""
    new_circuit = cirq.Circuit()
    for moment in circuit:
        moment_type = get_moment_type(moment)
        if moment_type == 1:
            new_circuit = safe_concatenate_circuits(new_circuit, cirq.Circuit(moment))
        else:
            expanded_moment = cirq.Circuit(
                [fsimgate_to_three_czs(gate) for gate in moment]
            )
            expanded_moment = cirq.align_left(
                cirq.merge_single_qubit_gates_to_phxz(expanded_moment)
            )
            new_circuit = safe_concatenate_circuits(new_circuit, expanded_moment)
    return new_circuit


def phxz_to_pi2_rotation_only(gate):
    """Converts a single phxz gate into two phxz gates with pi/2 angles only"""
    (q,) = gate.qubits
    x = gate.gate.x_exponent
    z = gate.gate.z_exponent
    a = gate.gate.axis_phase_exponent
    g1 = cirq.PhasedXZGate(
        x_exponent=-0.5, z_exponent=x, axis_phase_exponent=a + 0.5
    ).on(q)
    g2 = cirq.PhasedXZGate(
        x_exponent=0.5, z_exponent=z, axis_phase_exponent=a + 0.5
    ).on(q)
    return g1, g2


def convert_rotations_in_circuit_to_pi2_only(circuit, x_exponents=None, tol=1e-3):
    """Converts single-qubit rotations in a circuit to pi/2 MW rotations only"""

    # Default set of rotations is just pi and pi/2
    if x_exponents is None:
        x_exponents = [0, 0.5, 1, 1.5, 2]

    # Iterate over moments
    new_circuit = cirq.Circuit()
    for moment in circuit:

        # Determine if moment is 1Q or 2Q
        moment_type = get_moment_type(moment)

        # No change needed for 2Q gates
        if moment_type == 2:
            new_circuit.append(moment, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
            continue

        # Make new moments
        m1 = []
        m2 = []
        for gate in moment:
            # Don't change gate if it's not needed
            if min([abs((gate.gate.x_exponent % 2) - y) for y in x_exponents]) < tol:
                m1.append(gate)
            else:
                g1, g2 = phxz_to_pi2_rotation_only(gate)
                m1.append(g1)
                m2.append(g2)

        new_circuit.append(m1, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        # Don't insert moment if we haven't changed any gates.
        if len(m2) > 0:
            new_circuit.append(m2, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    return new_circuit


def _round_to_nearest(
    value: Union[float, complex], rounding_error: float
) -> Union[float, complex]:
    """Rounds a value to the nearest multiple of rounding_error.

    Note: behaviour near intervals of rounding_error / 2 is undefined
    due to floating point error (i.e. we may round up or down).
    """
    if (
        not isinstance(value, float)
        and not isinstance(value, complex)
        and not (value.is_number)
    ):
        return value
    if rounding_error <= 0:
        raise ValueError("rounding_error must be positive.")
    if isinstance(value, complex):
        # Round real and imaginary parts separately
        real_part = np.round(value.real / rounding_error) * rounding_error
        imag_part = np.round(value.imag / rounding_error) * rounding_error
        # Handle potential -0.0 cases from rounding near zero
        if abs(real_part) < rounding_error / 2:
            real_part = 0.0
        if abs(imag_part) < rounding_error / 2:
            imag_part = 0.0
        return complex(real_part, imag_part)
    else:
        # Handle potential -0.0 cases from rounding near zero
        rounded_val = np.round(value / rounding_error) * rounding_error
        return 0.0 if abs(rounded_val) < rounding_error / 2 else rounded_val


def round_circuit_angles_to_error(
    circuit: cirq.Circuit, rounding_error: float, gate_types_to_exclude: Optional[List[type]] = [],
) -> cirq.Circuit:
    """
    Iterates through a Cirq circuit and rounds the angles/exponents
    of known parameterized gates to the nearest multiple of a given error.

    Args:
        circuit: The input cirq.Circuit.
        rounding_error: The interval to round to (e.g., 0.05).
                        Parameters will be rounded to the nearest
                        multiple of this value. Must be positive.

    Returns:
        A new cirq.Circuit with gate parameters rounded.

    Raises:
        ValueError: If rounding_error is not positive.
    """
    if rounding_error <= 0:
        raise ValueError("rounding_error must be positive.")
    
    gate_types_to_exclude = gate_types_to_exclude or []

    new_moments = []
    for moment in circuit:
        new_ops = []
        for op in moment:
            gate = op.gate
            if any([isinstance(gate, gate_type) for gate_type in gate_types_to_exclude]):
                continue
            new_gate = gate  # Default to original gate

            # --- Handle power gates (EigenGates) ---
            pow_gate_types: List[Type[cirq.EigenGate]] = [
                cirq.XPowGate,
                cirq.YPowGate,
                cirq.ZPowGate,
                cirq.HPowGate,
                cirq.CXPowGate,
                cirq.CZPowGate,  # cirq.CNotPowGate is CXPowGate
                cirq.SwapPowGate,
                cirq.ISwapPowGate,
                cirq.XXPowGate,
                cirq.YYPowGate,
                cirq.ZZPowGate,
            ]

            # --- Handle single-qubit rotation gates ---
            if isinstance(gate, (cirq.Rx, cirq.Ry, cirq.Rz)):
                new_angle = _round_to_nearest(gate._rads, rounding_error)
                new_gate = type(gate)(rads=new_angle)

            elif any(isinstance(gate, T) for T in pow_gate_types):
                if hasattr(gate, "exponent") and hasattr(gate, "global_shift"):
                    # Exponent can be complex for some gates, handle appropriately
                    new_exponent = _round_to_nearest(gate.exponent, rounding_error)
                    new_gate = type(gate)(
                        exponent=new_exponent, global_shift=gate.global_shift
                    )

            # --- Handle FSimGate ---
            elif isinstance(gate, cirq.FSimGate):
                new_theta = _round_to_nearest(gate.theta, rounding_error)
                new_phi = _round_to_nearest(gate.phi, rounding_error)
                new_gate = cirq.FSimGate(theta=new_theta, phi=new_phi)

            # --- Handle PhasedXPowGate ---
            elif isinstance(gate, cirq.PhasedXPowGate):
                if (
                    hasattr(gate, "phase_exponent")
                    and hasattr(gate, "exponent")
                    and hasattr(gate, "global_shift")
                ):
                    new_phase_exponent = _round_to_nearest(
                        gate.phase_exponent, rounding_error
                    )
                    new_exponent = _round_to_nearest(gate.exponent, rounding_error)
                    new_gate = cirq.PhasedXPowGate(
                        phase_exponent=new_phase_exponent,
                        exponent=new_exponent,
                        global_shift=gate.global_shift,
                    )

            # --- Handle PhasedXZGate ---
            elif isinstance(gate, cirq.PhasedXZGate):
                new_x_exponent = _round_to_nearest(gate.x_exponent, rounding_error)
                new_z_exponent = _round_to_nearest(gate.z_exponent, rounding_error)
                new_axis_phase_exponent = _round_to_nearest(
                    gate.axis_phase_exponent, rounding_error
                )
                new_gate = cirq.PhasedXZGate(
                    x_exponent=new_x_exponent,
                    z_exponent=new_z_exponent,
                    axis_phase_exponent=new_axis_phase_exponent,
                )

            # --- Handle TwoQubitDiagonalGate ---
            elif isinstance(gate, cirq.TwoQubitDiagonalGate):
                if hasattr(gate, "diag_angles_radians"):
                    new_angles = tuple(
                        _round_to_nearest(a, rounding_error)
                        for a in gate.diag_angles_radians
                    )
                    new_gate = cirq.TwoQubitDiagonalGate(diag_angles_radians=new_angles)

            # --- Add more gate types here if needed ---

            # If the gate was modified, create a new operation
            if new_gate is not gate:
                # Ensure the new gate is valid before creating the operation
                # (e.g., some gates might have constraints on parameters)
                try:
                    # Attempt to create the operation. This might implicitly validate.
                    # For more robust validation, you might need specific checks per gate type.
                    new_op = new_gate.on(*op.qubits)
                    # Optional: Add specific validation if needed, e.g.,
                    # if isinstance(new_gate, MyGateWithConstraints):
                    #     new_gate.validate_parameters()
                    new_ops.append(new_op)
                except ValueError as e:
                    print(
                        f"Warning: Could not create operation for rounded gate {new_gate} "
                        f"on qubits {op.qubits}. Keeping original. Error: {e}"
                    )
                    new_ops.append(
                        op
                    )  # Keep original if rounding leads to invalid gate
            else:
                # Otherwise, keep the original operation
                new_ops.append(op)

        # Add the (potentially modified) operations to the new moment
        if new_ops:
            new_moments.append(cirq.Moment(new_ops))

    return cirq.Circuit(new_moments)


def safe_concatenate_circuits(circuit1, circuit2):
    # Assumes that moments are in alternating 1q and 2q form,
    # outputs a circuit that continues to follow this form
    if len(circuit2) == 0:
        return circuit1[:]
    if len(circuit1) == 0:
        return circuit2[:]
    moment_1r = circuit1[-1]
    moment_2l = circuit2[0]
    test_set_1r = set([len(gate.qubits) for gate in moment_1r])
    test_set_2l = set([len(gate.qubits) for gate in moment_2l])

    if len(test_set_1r) != 1:
        print(moment_1r)
        print(moment_2l)
        raise ValueError("Circuit 1 not in correct form")
    if len(test_set_2l) != 1:
        print(moment_1r)
        print(moment_2l)
        raise ValueError("Circuit 2 not in correct form")
    (gate_type_1r,) = test_set_1r
    (gate_type_2l,) = test_set_2l

    if gate_type_1r == 1 and gate_type_2l == 1:
        new_middle_moment = cirq.Moment()
        for gate_1r in moment_1r:
            if gate_1r.qubits[0] in moment_2l.qubits:
                gate_2l = moment_2l[gate_1r.qubits[0]]
                new_gate = merge_faster(gate_1r, gate_2l)
                if new_gate is not None:
                    new_middle_moment += [new_gate]
            else:
                new_middle_moment += [gate_1r]
        for gate_2l in moment_2l:
            if gate_2l.qubits[0] not in moment_1r.qubits:
                new_middle_moment += [gate_2l]

        circuit = cirq.Circuit()
        for moment in circuit1[:-1]:
            circuit.append(moment, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuit.append(new_middle_moment, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        for moment in circuit2[1:]:
            circuit.append(moment, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    else:
        circuit = cirq.Circuit()
        for moment in circuit1:
            circuit.append(moment, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        for moment in circuit2:
            circuit.append(moment, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    return circuit


def merge_faster(gate1, gate2, tol=1e-6):
    # Merges single qubit gates to a phasedXZ gate. Works faster
    # if the two gates are rz rotations
    if (
        type(gate1.gate) == cirq.PhasedXZGate
        and type(gate2.gate) == cirq.PhasedXZGate
        and abs(gate2.gate.x_exponent) < tol
    ):
        new_gate = cirq.PhasedXZGate(
            axis_phase_exponent=gate1.gate.axis_phase_exponent,
            x_exponent=gate1.gate.x_exponent,
            z_exponent=gate1.gate.z_exponent + gate2.gate.z_exponent,
        )
    else:
        new_unitary = cirq.unitary(gate2.gate) @ cirq.unitary(gate1.gate)
        new_gate = cirq.single_qubit_matrix_to_phxz(new_unitary)
        if new_gate is None:
            return None
    return new_gate.on(gate1.qubits[0])


def check_separated_single_and_two_qubit_layers(
    circuits: List[cirq.Circuit], strict: Optional[bool] = True
):
    """Checks whether a set of circuits satisfy the 1q-2q layer form.

    In the loose 1q-2q layer form, we simply require that each moment in the
    circuit either contains single-qubit or two-qubit gates.

    In the strict 1q-2q layer form, we insist that moments strictly alternate
    (i.e. 1q layer, 2q layer ..., allowing for empty layers.)

    We assume that the last layer contains measurements, and so we ignore it.

    Args:
        circuits [List[cirq.Circuit]]: the circuits to be checked
        strict [bool]: whether to enforce the strict condition above
    """
    for circuit_id, circuit in enumerate(circuits):
        for moment_id, moment in enumerate(circuit[:-1]):
            test_set = set([len(gate.qubits) for gate in moment])
            if len(test_set) > 1:
                raise TypeError(
                    "Circuit: {}, moment: {} is mixed".format(circuit_id, moment_id)
                )
            if strict and (
                len(test_set) == 1 and (next(iter(test_set)) % 2 != (moment_id + 1) % 2)
            ):
                raise TypeError(
                    "Circuit: {}, moment: {} off-position.".format(
                        circuit_id, moment_id
                    )
                )
