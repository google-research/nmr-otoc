# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Optional, List

import cirq
import numpy as np
import scipy

I = np.identity(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
P = {"X": X, "Y": Y, "Z": Z, "I": I}
P2 = {p1 + p2: np.kron(P[p1], P[p2]) for p1 in P for p2 in P}
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def dq_gate_fsim(dt, q0, q1, het, hom, swap=False, tol=1e-5):
    """Implements the gate exp(i * dt * hom * (XX - YY)) * exp(i * dt * het * ZZ)
    using a single FSimGate and single-qubit rotations. Optionally includes a SWAP gate.

    Args:
        q0, q1: The qubits to apply the gate to.
        dt: The time step.
        het: The strength of the ZZ interaction (heterogeneous coupling).
        hom: The strength of the XX-YY interaction (homogeneous coupling).
        swap: A boolean flag. If True, a SWAP gate is compiled into the FSimGate.
    """
    phi = dt * het
    theta = dt * hom

    # Adjust theta and phi for the swap
    if swap:
        theta += np.pi / 4
        phi += np.pi / 4
        qout = q1
    else:
        qout = q0

    # Construct the circuit
    if abs(phi) < tol:
        circuit = cirq.Circuit(
            cirq.ry(np.pi)(q0),  # Convert XX + YY to XX - YY
            cirq.FSimGate(theta=2 * theta, phi=4 * phi)(q0, q1),
            cirq.ry(np.pi)(qout),  # Convert XX + YY to XX - YY
        )
    else:
        circuit = cirq.Circuit(
            cirq.ry(np.pi)(q0),  # Convert XX + YY to XX - YY
            cirq.rz(2 * phi)(q0),  # Correct phi rotation to ZZ rotation
            cirq.rz(2 * phi)(q1),  # Correct phi rotation to ZZ rotation
            cirq.FSimGate(theta=2 * theta, phi=4 * phi)(q0, q1),
            cirq.ry(np.pi)(qout),  # Convert XX + YY to XX - YY
        )
    return circuit

def dipolar_gate_fsim(dt, q0, q1, het, hom, swap=False, tol=1e-5):
    """Implements the gate exp(i * dt * hom * (XX + YY)) * exp(i * dt * het * ZZ)
    using a single FSimGate and single-qubit rotations. Optionally includes a SWAP gate.

    Args:
        q0, q1: The qubits to apply the gate to.
        dt: The time step.
        het: The strength of the ZZ interaction (heterogeneous coupling).
        hom: The strength of the XX-YY interaction (homogeneous coupling).
        swap: A boolean flag. If True, a SWAP gate is compiled into the FSimGate.
    """
    phi = dt * het
    theta = dt * hom

    # Adjust theta and phi for the swap
    if swap:
        theta += np.pi / 4
        phi += np.pi / 4
        qout = q1
    else:
        qout = q0

    # Construct the circuit
    if abs(phi) < tol:
        circuit = cirq.Circuit(
            cirq.FSimGate(theta=2 * theta, phi=4 * phi)(q0, q1),
        )
    else:
        circuit = cirq.Circuit(
            cirq.rz(2 * phi)(q0),  # Correct phi rotation to ZZ rotation
            cirq.rz(2 * phi)(q1),  # Correct phi rotation to ZZ rotation
            cirq.FSimGate(theta=2 * theta, phi=4 * phi)(q0, q1),
        )
    return circuit

