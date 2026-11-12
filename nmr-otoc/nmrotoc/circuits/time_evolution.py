# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Functions to investigate exact time evolution on NMR devices"""

import numpy as np
from matplotlib import pyplot as plt
import scipy
import cirq
from typing import List, Optional, Tuple
from openfermion import QubitOperator, count_qubits, get_sparse_operator

from nmrotoc.math.operators import I, X, Y, Z, get_pauli_tensor


class TimeEvolutionGate(cirq.Gate):
    """Gate defined by time evolution on some set of qubits"""

    def __init__(
        self,
        hamiltonian: QubitOperator,
        dt: float,
        num_qubits: int,
        name: str = "H",
        n_trotter: int = 1,
    ):
        self.hamiltonian = hamiltonian
        self.dt = dt
        self.name = name
        self._nq = num_qubits
        self.n_trotter = n_trotter

    def _num_qubits_(self):
        return self._nq

    def _unitary_(self):
        ham_mat = get_sparse_operator(self.hamiltonian, n_qubits=self.num_qubits())
        unitary = np.array(scipy.sparse.linalg.expm(-1j * self.dt * ham_mat).todense())
        assert unitary.shape == (2 ** self.num_qubits(), 2 ** self.num_qubits())
        return unitary

    def approximately_decompose(self, qubits):
        unique_tuples = set(
            [tuple(sorted(x[1] for x in key)) for key in self.hamiltonian.terms]
        )
        cirq_paulis = {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}

        def make_cirq_paulistring_from_of_term(term, qubits):
            paulistring = cirq.PauliString(
                [cirq_paulis[pauli](qubits[qid]) for qid, pauli in term]
            )
            return paulistring

        hamiltonian_decomposed = {
            tup: {
                key: self.hamiltonian.terms[key]
                for key in self.hamiltonian.terms
                if tuple(sorted(x[1] for x in key)) == tup
            }
            for tup in unique_tuples
        }
        num_divisions = {
            qid: len(
                [
                    term
                    for term in hamiltonian_decomposed
                    if len(term) > 1 and qid in term
                ]
            )
            for qid in range(self._num_qubits_())
        }
        circuit = cirq.Circuit()
        for _ in range(self.n_trotter):
            for tup in hamiltonian_decomposed:

                # We divide up single-qubit terms and insert them through the circuit
                # before other terms.
                if len(tup) == 1 and num_divisions[tup[1]] > 0:
                    continue

                # Now insert first order terms
                for qid in tup:
                    if (qid,) not in hamiltonian_decomposed:
                        continue
                    for term, amp in hamiltonian_decomposed[(qid,)].items():
                        generator = make_cirq_paulistring_from_of_term(term, qubits)
                        time = amp * self.dt / (num_divisions[qid] * self.n_trotter)
                        circuit.append(generator ** (-2 * time / np.pi))
                for term, amp in hamiltonian_decomposed[tup].items():
                    generator = make_cirq_paulistring_from_of_term(term, qubits)
                    time = amp * self.dt / self.n_trotter
                    circuit.append(generator ** (-2 * time / np.pi))
        return circuit

    def _circuit_diagram_info_(self, args):
        return [
            "exp[i*{}*{}]".format(self.name, self.dt) for i in range(self.num_qubits())
        ]


def pulse_with_time_evolution(
    background_H: np.ndarray,
    pulse_generator: np.ndarray,
    pulse_time: float,
    pulse_angle: float,
    pulse_time_tol: float = 1e-10,
):
    """Applies a pulse with constant background time evolution

    Assuming that a pulse intends to apply a rotation exp(i*theta*V)
    in time t_p, this calculates the amplitude a = theta / t_p,
    and then returns the unitary

    U = exp(i * t_p * (H_0 + a * V))

    Args:
        background_H: H_0
        pulse_generator: V
        pulse_time: t_p - if t_p=0 we switch of background_H
        pulse_angle: theta
    """

    if background_H.shape != pulse_generator.shape:
        raise TypeError("background_H and pulse_generator must have same dimension")
    if pulse_time <= pulse_time_tol:
        Ht = pulse_generator * pulse_angle
        unitary = scipy.linalg.expm(-1j * Ht)
    else:
        amplitude = pulse_angle / pulse_time
        H = background_H + pulse_generator * amplitude
        unitary = scipy.linalg.expm(-1j * H * pulse_time)
    return unitary


def make_pulsed_scheme_circuit(
    qubits: List[cirq.Qid],
    background_H: np.ndarray,
    pulse_generators: List[np.ndarray],
    pulse_times: List[float],
    pulse_angles: List[float],
    pulse_names: Optional[List[str]] = None,
):
    """Constructs a cirq.Circuit representation of a realistic pulse scheme

    The circuit takes the form
    prod_j exp(i * t_j * (H_0 + a_j * V_j))

    where H_0 is a background Hamiltonian, t_j is the jth pulse time, V_j is
    the jth pulse generator, and a_j is the jth pulse amplitude, calculated
    by a_j = theta_j / t_j.

    Args:
        qubits: qubits to act the circuit on
        backgroud_H: constant background Hamiltonian H_0
        pulse_generators: list of V_j in the above expression
        pulse_times: list of t_j in the above expression
        pulse_angles: list of theta_j in the above expression
        pulse_names: list of pulse names to add to cirq circuit
    """
    if pulse_names is None:
        pulse_names = ["H"] * len(pulse_generators)

    if len(pulse_generators) != len(pulse_times):
        raise ValueError(
            f"len(pulse_generators) = {len(pulse_generators)}"
            f"!= len(pulse_times) = {len(pulse_times)}"
        )
    if len(pulse_generators) != len(pulse_angles):
        raise ValueError(
            f"len(pulse_generators) = {len(pulse_generators)}"
            f"!= len(pulse_angles) = {len(pulse_angles)}"
        )
    if len(pulse_generators) != len(pulse_names):
        raise ValueError(
            f"len(pulse_generators) = {len(pulse_generators)}"
            f"!= len(pulse_names) = {len(pulse_names)}"
        )

    circuit = cirq.Circuit()
    for pulse_generator, pulse_time, pulse_angle, pulse_name in zip(
        pulse_generators, pulse_times, pulse_angles, pulse_names
    ):
        if pulse_time == 0:
            circuit.append(
                TimeEvolutionGate(
                    hamiltonian=pulse_generator * pulse_angle,
                    dt=1,
                    name=("-" + pulse_name if pulse_angle < 0 else pulse_name),
                    num_qubits=len(qubits),
                ).on(*qubits),
                strategy=cirq.InsertStrategy.EARLIEST,
            )
        else:
            circuit.append(
                TimeEvolutionGate(
                    hamiltonian=background_H
                    + pulse_generator * pulse_angle / pulse_time,
                    dt=pulse_time,
                    name=("-" + pulse_name if pulse_angle < 0 else pulse_name),
                    num_qubits=len(qubits),
                ).on(*qubits),
                strategy=cirq.InsertStrategy.EARLIEST,
            )
    return circuit


def make_pulse_generator(
    channel_ids: List[int],
    phases: List[float],
    amps: List[float],
    z_flags: Optional[List[bool]] = None,
):
    """Makes the generator for an NMR pulse on a set of qubits

    Args:
        labels: List identifying each spin's channel.
        phases: List of rotation phases for each channel in the XY plane

    Returns:
        generator: An openfermion.QubitOperator representation of the generator.
    """

    if z_flags is None:
        z_flags = [False] * len(channel_ids)
    generator = QubitOperator()
    for j, c in enumerate(channel_ids):
        phase = phases[c]
        amp = amps[c]
        if z_flags[c] is True:
            pauli = amp * QubitOperator("Z{}".format(j))
        else:
            pauli = amp * (
                np.cos(phase) * QubitOperator("X{}".format(j))
                - np.sin(phase) * QubitOperator("Y{}".format(j))
            )
        generator += pauli / 2
    return generator


def make_pulse_generators_DREAM_1(labels: List[str]) -> Tuple[np.ndarray, ...]:
    """Generates operators used in a 2-channel DREAM-1 NMR experiment.

    Convention: Generators V are defined such that a pulse is exp(-i * angle * V).
                For a pi/2 pulse around X, angle=pi/2 and V = X/2.

    Args:
        labels: List identifying each spin's channel ('H' or 'C').

    Returns:
        Tuple containing: XX_gen, YY_gen, XXbar_gen, YYbar_gen
        where XX_gen = sum_j X_j/2, YY_gen = sum_j Y_j/2,
        and XXbar/YYbar include signs based on the channel.
    """

    channel_map = {"H": 0, "C": 1}
    channel_ids = [channel_map[label] for label in labels]
    XX_gen = make_pulse_generator(channel_ids, phases=[0, 0], amps=[1, 1])
    YY_gen = make_pulse_generator(
        channel_ids, phases=[np.pi / 2, np.pi / 2], amps=[1, 1]
    )
    XXbar_gen = make_pulse_generator(channel_ids, phases=[0, np.pi], amps=[1, 1])
    YYbar_gen = make_pulse_generator(
        channel_ids, phases=[np.pi / 2, -np.pi / 2], amps=[1, 1]
    )
    ZI_gen = make_pulse_generator(
        channel_ids, phases=[0, 0], amps=[1, 0], z_flags=[True, False]
    )

    return XX_gen, YY_gen, XXbar_gen, YYbar_gen, ZI_gen


def make_pulse_generators_DREAM_1_inverse(labels: List[str]) -> Tuple[np.ndarray, ...]:
    """Generates operators used in a 2-channel DREAM-1 NMR experiment.

    Convention: Generators V are defined such that a pulse is exp(-i * angle * V).
                For a pi/2 pulse around X, angle=pi/2 and V = X/2.

    Args:
        labels: List identifying each spin's channel ('H' or 'C').

    Returns:
        Tuple containing: XX_gen, YY_gen, XXbar_gen, YYbar_gen
        where XX_gen = sum_j X_j/2, YY_gen = sum_j Y_j/2,
        and XXbar/YYbar include signs based on the channel.
    """

    channel_map = {"H": 0, "C": 1}
    channel_ids = [channel_map[label] for label in labels]
    IY_gen = make_pulse_generator(channel_ids, phases=[0, np.pi / 2], amps=[0, 1])
    YX_gen = make_pulse_generator(channel_ids, phases=[np.pi / 2, 0], amps=[1, 1])
    XbarY_gen = make_pulse_generator(
        channel_ids, phases=[np.pi, np.pi / 2], amps=[1, 1]
    )
    YXbar_gen = make_pulse_generator(
        channel_ids, phases=[np.pi / 2, np.pi], amps=[1, 1]
    )
    XbarYbar_gen = make_pulse_generator(
        channel_ids, phases=[np.pi, -np.pi / 2], amps=[1, 1]
    )
    ZI_gen = make_pulse_generator(
        channel_ids, phases=[0, 0], amps=[1, 0], z_flags=[True, False]
    )

    return IY_gen, YX_gen, XbarY_gen, YXbar_gen, XbarYbar_gen, ZI_gen


def make_pulse_generators_DREAM_1_inverse_new(
    labels: List[str],
) -> Tuple[np.ndarray, ...]:
    """Generates operators used in a 2-channel DREAM-1 NMR experiment.

    Convention: Generators V are defined such that a pulse is exp(-i * angle * V).
                For a pi/2 pulse around X, angle=pi/2 and V = X/2.

    Args:
        labels: List identifying each spin's channel ('H' or 'C').

    Returns:
        Tuple containing: XX_gen, YY_gen, XXbar_gen, YYbar_gen
        where XX_gen = sum_j X_j/2, YY_gen = sum_j Y_j/2,
        and XXbar/YYbar include signs based on the channel.
    """

    channel_map = {"H": 0, "C": 1}
    channel_ids = [channel_map[label] for label in labels]

    YI_gen = make_pulse_generator(channel_ids, phases=[np.pi / 2, 0], amps=[1, 0])
    XY_gen = make_pulse_generator(channel_ids, phases=[0, np.pi / 2], amps=[1, 1])
    YXbar_gen = make_pulse_generator(
        channel_ids, phases=[np.pi / 2, np.pi], amps=[1, 1]
    )
    XYbar_gen = make_pulse_generator(channel_ids, phases=[0, -np.pi / 2], amps=[1, 1])
    YX_gen = make_pulse_generator(channel_ids, phases=[np.pi / 2, 0], amps=[1, 1])
    ZI_gen = make_pulse_generator(
        channel_ids, phases=[0, 0], amps=[1, 0], z_flags=[True, False]
    )

    return YI_gen, XY_gen, YXbar_gen, XYbar_gen, YX_gen, ZI_gen


def make_pulse_generators_DREAM_2(labels: List[str]):
    """Generates pi/2 rotations used in a 2-channel DREAM-2 NMR experiment

    Args:
        labels: list of which spin indices correspond to which channel
    """

    channel_map = {"H": 0, "C": 1}
    channel_ids = [channel_map[label] for label in labels]
    II_gen = make_pulse_generator(channel_ids, phases=[0, 0], amps=[0, 0])
    XX_gen = make_pulse_generator(channel_ids, phases=[0, 0], amps=[1, 1])
    XI_gen = make_pulse_generator(channel_ids, phases=[0, 0], amps=[1, 0])
    IY_gen = make_pulse_generator(channel_ids, phases=[0, np.pi / 2], amps=[0, 1])
    XYbar_gen = make_pulse_generator(channel_ids, phases=[0, -np.pi / 2], amps=[1, 1])
    IXbar_gen = make_pulse_generator(channel_ids, phases=[0, np.pi], amps=[0, 1])
    ZI_gen = make_pulse_generator(
        channel_ids, phases=[0, 0], amps=[1, 0], z_flags=[True, False]
    )

    return II_gen, XX_gen, XI_gen, IY_gen, XYbar_gen, IXbar_gen, ZI_gen


def make_pulse_generators_DREAM_2_inverse(labels: List[str]):
    """Generates pi/2 rotations used in a 2-channel DREAM-2 NMR experiment

    Args:
        labels: list of which spin indices correspond to which channel
    """
    channel_map = {"H": 0, "C": 1}
    channel_ids = [channel_map[label] for label in labels]
    II_gen = make_pulse_generator(channel_ids, phases=[0, 0], amps=[0, 0])
    YX_gen = make_pulse_generator(channel_ids, phases=[np.pi / 2, 0], amps=[1, 1])
    YI_gen = make_pulse_generator(channel_ids, phases=[np.pi / 2, 0], amps=[1, 0])
    IY_gen = make_pulse_generator(channel_ids, phases=[0, np.pi / 2], amps=[0, 1])
    YYbar_gen = make_pulse_generator(
        channel_ids, phases=[np.pi / 2, -np.pi / 2], amps=[1, 1]
    )
    IXbar_gen = make_pulse_generator(channel_ids, phases=[0, np.pi], amps=[0, 1])
    ZI_gen = make_pulse_generator(
        channel_ids, phases=[0, 0], amps=[1, 0], z_flags=[True, False]
    )

    return II_gen, YX_gen, YI_gen, IY_gen, YYbar_gen, IXbar_gen, ZI_gen


def make_pulse_generators_blew12(labels: List[str]):
    """Generates pi/2 rotations used in a BLEW12 scheme

    Args:
         labels: list of which spin indices correspond to which channel
    """
    channel_map = {"H": 0, "C": 1}
    channel_ids = [channel_map[label] for label in labels]
    XI_gen = make_pulse_generator(channel_ids, phases=[0, 0], amps=[1, 0])
    YI_gen = make_pulse_generator(channel_ids, phases=[np.pi / 2, 0], amps=[1, 0])
    XbarI_gen = make_pulse_generator(channel_ids, phases=[np.pi, 0], amps=[1, 0])
    YbarI_gen = make_pulse_generator(channel_ids, phases=[-np.pi / 2, 0], amps=[1, 0])
    IY_gen = make_pulse_generator(channel_ids, phases=[0, np.pi / 2], amps=[0, 1])
    return XI_gen, YI_gen, XbarI_gen, YbarI_gen, IY_gen


def dream_1_sequence_CH(
    labels: List[str],
    qubits: List[cirq.Qid],
    pulse_time: float,
    background_H: QubitOperator,
    fractional_pulse_error: Optional[float] = 0,
    fractional_pulse_error_C: Optional[float] = None,
):
    """Constructs the DREAM-1 sequence on a molecule as a cirq.Circuit.

    Args:
        label_list: whether the atom is H or C
        qubits: list of qubits to act on
        pulse_time: time per pulse
        fractional_pulse_error: to model inhomogeneity; we set all pulses to
            have amplitude = target_amp * (1 + fractional_pulse_error)
        background_H: Background NMR Hamiltonian
        z_angle: an optional phase to rotate by
    Returns:
        A cirq.Circuit representing the DREAM-1 sequence.

    Raises:
        ValueError: If the length of labels and qubits do not match.
    """
    if len(qubits) != len(labels):
        raise ValueError("Length of qubits and labels must match.")
    if fractional_pulse_error_C is None:
        fractional_pulse_error_C = fractional_pulse_error

    XX_gen, YY_gen, XXbar_gen, YYbar_gen, _ = make_pulse_generators_DREAM_1(labels)
    for rot_gen in [XX_gen, YY_gen, XXbar_gen, YYbar_gen]:
        for term in rot_gen.terms:
            if labels[term[0][0]] == "C":
                rot_gen.terms[term] *= 1 + fractional_pulse_error_C
            elif labels[term[0][0]] == "H":
                rot_gen.terms[term] *= 1 + fractional_pulse_error

    # The DREAM-1 sequence is made up of 8 subblocks of 3 pulses
    Ra = [XX_gen, YYbar_gen, XX_gen]
    Rb = [XXbar_gen, YY_gen, XXbar_gen]
    # Full sequence: Ra Ra Rb Rb Rb[bar] Rb[bar] Ra[bar] Ra[bar]
    pulse_generators = Ra + Ra + Rb + Rb + Rb + Rb + Ra + Ra

    # Define pulse names for circuit plotting purposes
    Ra_names = ["XX", "YYb", "XX"]
    Rb_names = ["XXb", "YY", "XXb"]
    # Names for the inverted angle pulses (often denoted with a bar)
    Rabar_names = ["XX", "YYb", "XX"]  # Generators are the same, angle flips
    Rbbar_names = ["XXb", "YY", "XXb"]  # Generators are the same, angle flips
    pulse_names = (Ra_names + Ra_names + Rb_names + Rb_names) + (
        Rbbar_names + Rbbar_names + Rabar_names + Rabar_names
    )

    # Define pulse times and angles
    pulse_times = [pulse_time] * 12 + [pulse_time] * 12
    # Angles: First half positive pi/2, second half negative pi/2
    actual_pi2 = np.pi / 2
    pulse_angles = [actual_pi2] * 12 + [-actual_pi2] * 12

    return make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=pulse_generators,
        pulse_times=pulse_times,
        pulse_angles=pulse_angles,
        pulse_names=pulse_names,
    )


def dream_1_sequence_CH_with_background(
    labels: List[str],
    qubits: List[cirq.Qid],
    pulse_time: float,
    background_H: QubitOperator,
    fractional_pulse_error: Optional[float] = 0,
    fractional_pulse_error_C: Optional[float] = None,
    z_angle: Optional[float] = 0,
):
    """Constructs the DREAM-1 sequence on a molecule as a cirq.Circuit.

    Args:
        label_list: whether the atom is H or C
        qubits: list of qubits to act on
        pulse_time: time per pulse
        fractional_pulse_error: to model inhomogeneity; we set all pulses to
            have amplitude = target_amp * (1 + fractional_pulse_error)
        background_H: Background NMR Hamiltonian
        z_angle: an optional phase to rotate by
    Returns:
        A cirq.Circuit representing the DREAM-1 sequence.

    Raises:
        ValueError: If the length of labels and qubits do not match.
    """
    if len(qubits) != len(labels):
        raise ValueError("Length of qubits and labels must match.")
    if fractional_pulse_error_C is None:
        fractional_pulse_error_C = fractional_pulse_error

    XX_gen, YY_gen, XXbar_gen, YYbar_gen, ZI_gen = make_pulse_generators_DREAM_1(labels)
    for rot_gen in [XX_gen, YY_gen, XXbar_gen, YYbar_gen]:
        for term in rot_gen.terms:
            if labels[term[0][0]] == "C":
                rot_gen.terms[term] *= 1 + fractional_pulse_error_C
            elif labels[term[0][0]] == "H":
                rot_gen.terms[term] *= 1 + fractional_pulse_error

    # The DREAM-1 sequence is made up of 8 subblocks of 3 pulses
    Ra = [XX_gen, YYbar_gen, XX_gen]
    Rb = [XXbar_gen, YY_gen, XXbar_gen]
    # Full sequence: Ra Ra Rb Rb Rb[bar] Rb[bar] Ra[bar] Ra[bar]
    pulse_generators = Ra + Ra + Rb + Rb + [ZI_gen] + Rb + Rb + Ra + Ra + [ZI_gen]

    # Define pulse names for circuit plotting purposes
    Ra_names = ["XX", "YYb", "XX"]
    Rb_names = ["XXb", "YY", "XXb"]
    # Names for the inverted angle pulses (often denoted with a bar)
    Rabar_names = ["XX", "YYb", "XX"]  # Generators are the same, angle flips
    Rbbar_names = ["XXb", "YY", "XXb"]  # Generators are the same, angle flips
    pulse_names = (
        (Ra_names + Ra_names + Rb_names + Rb_names)
        + ["ZI"]
        + (Rbbar_names + Rbbar_names + Rabar_names + Rabar_names)
        + ["ZI"]
    )

    # Define pulse times and angles
    pulse_times = [pulse_time] * 12 + [0] + [pulse_time] * 12 + [0]
    # Angles: First half positive pi/2, second half negative pi/2
    actual_pi2 = np.pi / 2
    pulse_angles = [actual_pi2] * 12 + [z_angle] + [-actual_pi2] * 12 + [z_angle]

    return make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=pulse_generators,
        pulse_times=pulse_times,
        pulse_angles=pulse_angles,
        pulse_names=pulse_names,
    )


def dream_1_inverse_sequence_CH(
    labels: List[str],
    qubits: List[cirq.Qid],
    pulse_time: float,
    background_H: QubitOperator,
    fractional_pulse_error: Optional[float] = 0,
    fractional_pulse_error_C: Optional[float] = None,
):
    """Constructs the DREAM-1 sequence on a molecule as a cirq.Circuit.

    Args:
        label_list: whether the atom is H or C
        qubits: list of qubits to act on
        pulse_time: time per pulse
        fractional_pulse_error: to model inhomogeneity; we set all pulses to
            have amplitude = target_amp * (1 + fractional_pulse_error)
        background_H: Background NMR Hamiltonian
        z_angle: an optional phase to rotate by
    Returns:
        A cirq.Circuit representing the DREAM-1 sequence.

    Raises:
        ValueError: If the length of labels and qubits do not match.
    """
    if len(qubits) != len(labels):
        raise ValueError("Length of qubits and labels must match.")
    if fractional_pulse_error_C is None:
        fractional_pulse_error_C = fractional_pulse_error

    IY_gen, YX_gen, XbarY_gen, YXbar_gen, XbarYbar_gen, _ = (
        make_pulse_generators_DREAM_1_inverse(labels)
    )
    for rot_gen in [IY_gen, YX_gen, XbarY_gen, YXbar_gen, XbarYbar_gen]:
        for term in rot_gen.terms:
            if labels[term[0][0]] == "C":
                rot_gen.terms[term] *= 1 + fractional_pulse_error_C
            elif labels[term[0][0]] == "H":
                rot_gen.terms[term] *= 1 + fractional_pulse_error

    # The DREAM-1 sequence is made up of 8 subblocks of 3 pulses
    Ra = [YX_gen, XbarYbar_gen, YX_gen]
    Rb = [YXbar_gen, XbarY_gen, YXbar_gen]
    # Full sequence: Ra Ra Rb Rb Rb[bar] Rb[bar] Ra[bar] Ra[bar]
    pulse_generators = Ra + Ra + Rb + Rb + Rb + Rb + Ra + Ra

    # Define pulse names for circuit plotting purposes
    Ra_names = ["XX", "YYb", "XX"]
    Rb_names = ["XXb", "YY", "XXb"]
    # Names for the inverted angle pulses (often denoted with a bar)
    Rabar_names = ["XX", "YYb", "XX"]  # Generators are the same, angle flips
    Rbbar_names = ["XXb", "YY", "XXb"]  # Generators are the same, angle flips
    pulse_names = (Ra_names + Ra_names + Rb_names + Rb_names) + (
        Rbbar_names + Rbbar_names + Rabar_names + Rabar_names
    )

    # Define pulse times and angles
    pulse_times = [pulse_time] * 12 + [pulse_time] * 12
    # Angles: First half positive pi/2, second half negative pi/2
    actual_pi2 = np.pi / 2
    pulse_angles = [actual_pi2] * 12 + [-actual_pi2] * 12

    # --- Start Pulse (pi rotation around IY) ---
    start_pulse = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=[IY_gen],
        pulse_times=[pulse_time * 2],
        pulse_angles=[np.pi],
        pulse_names=["IY"],
    )

    # --- End Pulse (-pi rotation around IY) ---
    end_pulse = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=[IY_gen],
        pulse_times=[pulse_time * 2],
        pulse_angles=[-np.pi],
        pulse_names=["IY"],
    )

    unitary_block = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=pulse_generators,
        pulse_times=pulse_times,
        pulse_angles=pulse_angles,
        pulse_names=pulse_names,
    )

    return unitary_block, start_pulse, end_pulse


def dream_1_inverse_sequence_CH_with_background(
    labels: List[str],
    qubits: List[cirq.Qid],
    pulse_time: float,
    background_H: QubitOperator,
    fractional_pulse_error: Optional[float] = 0,
    fractional_pulse_error_C: Optional[float] = None,
    z_angle: Optional[float] = 0,
):
    """Constructs the DREAM-1 sequence on a molecule as a cirq.Circuit.

    Args:
        label_list: whether the atom is H or C
        qubits: list of qubits to act on
        pulse_time: time per pulse
        fractional_pulse_error: to model inhomogeneity; we set all pulses to
            have amplitude = target_amp * (1 + fractional_pulse_error)
        background_H: Background NMR Hamiltonian
        z_angle: an optional phase to rotate by
    Returns:
        A cirq.Circuit representing the DREAM-1 sequence.

    Raises:
        ValueError: If the length of labels and qubits do not match.
    """
    if len(qubits) != len(labels):
        raise ValueError("Length of qubits and labels must match.")
    if fractional_pulse_error_C is None:
        fractional_pulse_error_C = fractional_pulse_error

    IY_gen, YX_gen, XbarY_gen, YXbar_gen, XbarYbar_gen, ZI_gen = (
        make_pulse_generators_DREAM_1_inverse(labels)
    )
    for rot_gen in [IY_gen, YX_gen, XbarY_gen, YXbar_gen, XbarYbar_gen]:
        for term in rot_gen.terms:
            if labels[term[0][0]] == "C":
                rot_gen.terms[term] *= 1 + fractional_pulse_error_C
            elif labels[term[0][0]] == "H":
                rot_gen.terms[term] *= 1 + fractional_pulse_error

    # The DREAM-1 sequence is made up of 8 subblocks of 3 pulses
    Ra = [YX_gen, XbarYbar_gen, YX_gen]
    Rb = [YXbar_gen, XbarY_gen, YXbar_gen]
    # Full sequence: Ra Ra Rb Rb Rb[bar] Rb[bar] Ra[bar] Ra[bar]
    pulse_generators = Ra + Ra + Rb + Rb + [ZI_gen] + Rb + Rb + Ra + Ra + [ZI_gen]

    # Define pulse names for circuit plotting purposes
    Ra_names = ["XX", "YYb", "XX"]
    Rb_names = ["XXb", "YY", "XXb"]
    # Names for the inverted angle pulses (often denoted with a bar)
    Rabar_names = ["XX", "YYb", "XX"]  # Generators are the same, angle flips
    Rbbar_names = ["XXb", "YY", "XXb"]  # Generators are the same, angle flips
    pulse_names = (
        (Ra_names + Ra_names + Rb_names + Rb_names)
        + ["ZI"]
        + (Rbbar_names + Rbbar_names + Rabar_names + Rabar_names)
        + ["ZI"]
    )

    # Define pulse times and angles
    pulse_times = [pulse_time] * 12 + [0] + [pulse_time] * 12 + [0]
    # Angles: First half positive pi/2, second half negative pi/2
    actual_pi2 = np.pi / 2
    pulse_angles = [actual_pi2] * 12 + [-z_angle] + [-actual_pi2] * 12 + [-z_angle]

    # --- Start Pulse (pi rotation around IY) ---
    start_pulse = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=[IY_gen],
        pulse_times=[pulse_time * 2],
        pulse_angles=[np.pi],
        pulse_names=["IY"],
    )

    # --- End Pulse (-pi rotation around IY) ---
    end_pulse = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=[IY_gen],
        pulse_times=[pulse_time * 2],
        pulse_angles=[-np.pi],
        pulse_names=["IY"],
    )

    unitary_block = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=pulse_generators,
        pulse_times=pulse_times,
        pulse_angles=pulse_angles,
        pulse_names=pulse_names,
    )

    return unitary_block, start_pulse, end_pulse


def dream_1_inverse_sequence_CH_new_with_background(
    labels: List[str],
    qubits: List[cirq.Qid],
    pulse_time: float,
    background_H: QubitOperator,
    fractional_pulse_error: Optional[float] = 0,
    fractional_pulse_error_C: Optional[float] = None,
    z_angle: Optional[float] = 0,
):
    """Constructs the DREAM-1 sequence on a molecule as a cirq.Circuit.

    Args:
        label_list: whether the atom is H or C
        qubits: list of qubits to act on
        pulse_time: time per pulse
        fractional_pulse_error: to model inhomogeneity; we set all pulses to
            have amplitude = target_amp * (1 + fractional_pulse_error)
        background_H: Background NMR Hamiltonian
    Returns:
        A cirq.Circuit representing the DREAM-1 sequence.

    Raises:
        ValueError: If the length of labels and qubits do not match.
    """
    if len(qubits) != len(labels):
        raise ValueError("Length of qubits and labels must match.")
    if fractional_pulse_error_C is None:
        fractional_pulse_error_C = fractional_pulse_error

    YI_gen, XY_gen, YXbar_gen, XYbar_gen, YX_gen, ZI_gen = (
        make_pulse_generators_DREAM_1_inverse_new(labels)
    )
    for rot_gen in [YI_gen, XY_gen, YXbar_gen, XYbar_gen, YX_gen]:
        for term in rot_gen.terms:
            if labels[term[0][0]] == "C":
                rot_gen.terms[term] *= 1 + fractional_pulse_error_C
            elif labels[term[0][0]] == "H":
                rot_gen.terms[term] *= 1 + fractional_pulse_error

    # The DREAM-1 sequence is made up of 8 subblocks of 3 pulses
    Ra = [XY_gen, YX_gen, XY_gen]
    Rb = [XYbar_gen, YXbar_gen, XYbar_gen]

    # Full sequence: Ra Ra Rb Rb Rb[bar] Rb[bar] Ra[bar] Ra[bar]
    pulse_generators = Ra + Ra + Rb + Rb + [ZI_gen] + Rb + Rb + Ra + Ra + [ZI_gen]

    # Define pulse names for circuit plotting purposes
    Ra_names = ["XY", "YX", "XY"]
    Rb_names = ["XYb", "YXb", "XYb"]
    # Names for the inverted angle pulses (often denoted with a bar)
    Rabar_names = ["XY", "YX", "XY"]  # Generators are the same, angle flips
    Rbbar_names = ["XYb", "YXb", "XYb"]  # Generators are the same, angle flips

    pulse_names = (
        (Ra_names + Ra_names + Rb_names + Rb_names)
        + ["ZI"]
        + (Rbbar_names + Rbbar_names + Rabar_names + Rabar_names)
        + ["ZI"]
    )

    # Define pulse times and angles
    pulse_times = [pulse_time] * 12 + [0] + [pulse_time] * 12 + [0]
    # Angles: First half positive pi/2, second half negative pi/2
    actual_pi2 = np.pi / 2
    pulse_angles = [actual_pi2] * 12 + [z_angle] + [-actual_pi2] * 12 + [z_angle]

    # --- Start Pulse (pi rotation around YI) ---
    start_pulse = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=[YI_gen],
        pulse_times=[pulse_time * 2],
        pulse_angles=[np.pi],
        pulse_names=["YI"],
    )

    # --- End Pulse (-pi rotation around YI) ---
    end_pulse = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=[YI_gen],
        pulse_times=[pulse_time * 2],
        pulse_angles=[-np.pi],
        pulse_names=["YI"],
    )

    unitary_block = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=pulse_generators,
        pulse_times=pulse_times,
        pulse_angles=pulse_angles,
        pulse_names=pulse_names,
    )

    return unitary_block, start_pulse, end_pulse


def dream_1_inverse_sequence_CH_new(
    labels: List[str],
    qubits: List[cirq.Qid],
    pulse_time: float,
    background_H: QubitOperator,
    fractional_pulse_error: Optional[float] = 0,
    fractional_pulse_error_C: Optional[float] = None,
):
    """Constructs the DREAM-1 sequence on a molecule as a cirq.Circuit.

    Args:
        label_list: whether the atom is H or C
        qubits: list of qubits to act on
        pulse_time: time per pulse
        fractional_pulse_error: to model inhomogeneity; we set all pulses to
            have amplitude = target_amp * (1 + fractional_pulse_error)
        background_H: Background NMR Hamiltonian
    Returns:
        A cirq.Circuit representing the DREAM-1 sequence.

    Raises:
        ValueError: If the length of labels and qubits do not match.
    """
    if len(qubits) != len(labels):
        raise ValueError("Length of qubits and labels must match.")
    if fractional_pulse_error_C is None:
        fractional_pulse_error_C = fractional_pulse_error

    YI_gen, XY_gen, YXbar_gen, XYbar_gen, YX_gen, _ = (
        make_pulse_generators_DREAM_1_inverse_new(labels)
    )
    for rot_gen in [YI_gen, XY_gen, YXbar_gen, XYbar_gen, YX_gen]:
        for term in rot_gen.terms:
            if labels[term[0][0]] == "C":
                rot_gen.terms[term] *= 1 + fractional_pulse_error_C
            elif labels[term[0][0]] == "H":
                rot_gen.terms[term] *= 1 + fractional_pulse_error

    # The DREAM-1 sequence is made up of 8 subblocks of 3 pulses
    Ra = [XY_gen, YX_gen, XY_gen]
    Rb = [XYbar_gen, YXbar_gen, XYbar_gen]

    # Full sequence: Ra Ra Rb Rb Rb[bar] Rb[bar] Ra[bar] Ra[bar]
    pulse_generators = Ra + Ra + Rb + Rb + Rb + Rb + Ra + Ra

    # Define pulse names for circuit plotting purposes
    Ra_names = ["XY", "YX", "XY"]
    Rb_names = ["XYb", "YXb", "XYb"]
    # Names for the inverted angle pulses (often denoted with a bar)
    Rabar_names = ["XY", "YX", "XY"]  # Generators are the same, angle flips
    Rbbar_names = ["XYb", "YXb", "XYb"]  # Generators are the same, angle flips

    pulse_names = (Ra_names + Ra_names + Rb_names + Rb_names) + (
        Rbbar_names + Rbbar_names + Rabar_names + Rabar_names
    )

    # Define pulse times and angles
    pulse_times = [pulse_time] * 12 + [pulse_time] * 12
    # Angles: First half positive pi/2, second half negative pi/2
    actual_pi2 = np.pi / 2
    pulse_angles = [actual_pi2] * 12 + [-actual_pi2] * 12

    # --- Start Pulse (pi rotation around YI) ---
    start_pulse = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=[YI_gen],
        pulse_times=[pulse_time * 2],
        pulse_angles=[np.pi],
        pulse_names=["YI"],
    )

    # --- End Pulse (-pi rotation around YI) ---
    end_pulse = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=[YI_gen],
        pulse_times=[pulse_time * 2],
        pulse_angles=[-np.pi],
        pulse_names=["YI"],
    )

    unitary_block = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=pulse_generators,
        pulse_times=pulse_times,
        pulse_angles=pulse_angles,
        pulse_names=pulse_names,
    )

    return unitary_block, start_pulse, end_pulse


def dream_2_sequence_CH(
    labels: List[str],
    qubits: List[cirq.Qid],
    pulse_time: float,
    background_H: QubitOperator,
    fractional_pulse_error: Optional[float] = 0,
    fractional_pulse_error_C: Optional[float] = None,
):
    """Constructs the DREAM-1 sequence on a molecule as a cirq.Circuit.

    Args:
        label_list: whether the atom is H or C
        qubits: list of qubits to act on
        pulse_time: time per pulse
        fractional_pulse_error: to model inhomogeneity; we set all pulses to
            have amplitude = target_amp * (1 + fractional_pulse_error)
        background_H: Background NMR Hamiltonian
    """
    # Make generators
    if len(qubits) != len(labels):
        raise ValueError("Need 1 qubit per label")
    if fractional_pulse_error_C is None:
        fractional_pulse_error_C = fractional_pulse_error

    II_gen, XX_gen, XI_gen, IY_gen, XYbar_gen, IXbar_gen, _ = (
        make_pulse_generators_DREAM_2(labels)
    )
    for rot_gen in [II_gen, XX_gen, XI_gen, IY_gen, XYbar_gen, IXbar_gen]:
        for term in rot_gen.terms:
            if labels[term[0][0]] == "C":
                rot_gen.terms[term] *= 1 + fractional_pulse_error_C
            elif labels[term[0][0]] == "H":
                rot_gen.terms[term] *= 1 + fractional_pulse_error

    # The DREAM-2 sequence is broken up into 6 sub-blocks of 6 pulses
    Pines = [XI_gen, II_gen, XI_gen] * 2  # 6 pulses
    DR = [XX_gen, IY_gen, XX_gen, XYbar_gen, IXbar_gen, XYbar_gen]  # 6 pulses
    # Full sequence: P P[bar] DR DR[bar] P P[bar]
    pulse_generators = Pines + Pines + DR + DR + Pines + Pines

    # Define pulse names for circuit plotting
    Pines_names = ["XI", "II", "XI"] * 2
    DR_names = ["XX", "IY", "XX", "XYb", "IXb", "XYb"]
    # Names for the inverted angle pulses
    Pinesbar_names = ["XI", "II", "XI"] * 2  # Generators are the same
    DRbar_names = ["XX", "IY", "XX", "XYb", "IXb", "XYb"]  # Generators are the same
    pulse_names = (
        (Pines_names + Pinesbar_names)
        + (DR_names + DRbar_names)
        + (Pines_names + Pinesbar_names)
    )

    # Define times and angles
    pulse_times = [pulse_time] * 12 + [pulse_time] * 12 + [pulse_time] * 12
    # Angles: Alternating blocks of positive/negative pi/2
    actual_pi2 = np.pi / 2
    pulse_angles = []
    for i in range(6):  # Iterate through the 6 blocks
        sign = 1 if i % 2 == 0 else -1
        pulse_angles.extend([sign * actual_pi2] * 6)

    return make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=pulse_generators,
        pulse_times=pulse_times,
        pulse_angles=pulse_angles,
        pulse_names=pulse_names,
    )


def dream_2_sequence_CH_with_background(
    labels: List[str],
    qubits: List[cirq.Qid],
    pulse_time: float,
    background_H: QubitOperator,
    fractional_pulse_error: Optional[float] = 0,
    fractional_pulse_error_C: Optional[float] = None,
    z_angle: Optional[float] = 0,
):
    """Constructs the DREAM-1 sequence on a molecule as a cirq.Circuit.

    Args:
        label_list: whether the atom is H or C
        qubits: list of qubits to act on
        pulse_time: time per pulse
        fractional_pulse_error: to model inhomogeneity; we set all pulses to
            have amplitude = target_amp * (1 + fractional_pulse_error)
        background_H: Background NMR Hamiltonian
    """
    # Make generators
    if len(qubits) != len(labels):
        raise ValueError("Need 1 qubit per label")
    if fractional_pulse_error_C is None:
        fractional_pulse_error_C = fractional_pulse_error

    II_gen, XX_gen, XI_gen, IY_gen, XYbar_gen, IXbar_gen, ZI_gen = (
        make_pulse_generators_DREAM_2(labels)
    )
    for rot_gen in [II_gen, XX_gen, XI_gen, IY_gen, XYbar_gen, IXbar_gen]:
        for term in rot_gen.terms:
            if labels[term[0][0]] == "C":
                rot_gen.terms[term] *= 1 + fractional_pulse_error_C
            elif labels[term[0][0]] == "H":
                rot_gen.terms[term] *= 1 + fractional_pulse_error

    # The DREAM-2 sequence is broken up into 6 sub-blocks of 6 pulses
    Pines = [XI_gen, II_gen, XI_gen] * 2  # 6 pulses
    DR = [XX_gen, IY_gen, XX_gen, XYbar_gen, IXbar_gen, XYbar_gen]  # 6 pulses
    # Full sequence: P P[bar] DR DR[bar] P P[bar]
    pulse_generators = (
        Pines + Pines + [ZI_gen] + DR + DR + [ZI_gen] + Pines + Pines + [ZI_gen]
    )

    # Define pulse names for circuit plotting
    Pines_names = ["XI", "II", "XI"] * 2
    DR_names = ["XX", "IY", "XX", "XYb", "IXb", "XYb"]
    # Names for the inverted angle pulses
    Pinesbar_names = ["XI", "II", "XI"] * 2  # Generators are the same
    DRbar_names = ["XX", "IY", "XX", "XYb", "IXb", "XYb"]  # Generators are the same
    pulse_names = (
        (Pines_names + Pinesbar_names + ["IZ"])
        + (DR_names + DRbar_names + ["IZ"])
        + (Pines_names + Pinesbar_names + ["IZ"])
    )

    # Define times and angles
    pulse_times = (
        [pulse_time] * 12 + [0] + [pulse_time] * 12 + [0] + [pulse_time] * 12 + [0]
    )
    # Angles: Alternating blocks of positive/negative pi/2
    actual_pi2 = np.pi / 2
    pulse_angles = []
    for i in range(6):  # Iterate through the 6 blocks
        sign = 1 if i % 2 == 0 else -1
        pulse_angles.extend([sign * actual_pi2] * 6)
        if i % 2 == 1:
            pulse_angles += [-z_angle]

    return make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=pulse_generators,
        pulse_times=pulse_times,
        pulse_angles=pulse_angles,
        pulse_names=pulse_names,
    )


def dream_2_inverse_sequence_CH_with_background(
    labels: List[str],
    qubits: List[cirq.Qid],
    pulse_time: float,
    background_H: QubitOperator,
    fractional_pulse_error: Optional[float] = 0,
    fractional_pulse_error_C: Optional[float] = None,
    z_angle: Optional[float] = 0,
):
    """Constructs the DREAM-1 sequence on a molecule as a cirq.Circuit.

    Args:
        label_list: whether the atom is H or C
        qubits: list of qubits to act on
        pulse_time: time per pulse
        fractional_pulse_error: to model inhomogeneity; we set all pulses to
            have amplitude = target_amp * (1 + fractional_pulse_error)
        background_H: Background NMR Hamiltonian
    """
    # Make generators
    if len(qubits) != len(labels):
        raise ValueError("Need 1 qubit per label")
    if fractional_pulse_error_C is None:
        fractional_pulse_error_C = fractional_pulse_error

    II_gen, YX_gen, YI_gen, IY_gen, YYbar_gen, IXbar_gen, ZI_gen = (
        make_pulse_generators_DREAM_2_inverse(labels)
    )
    for rot_gen in [II_gen, YX_gen, YI_gen, IY_gen, YYbar_gen, IXbar_gen]:
        for term in rot_gen.terms:
            if labels[term[0][0]] == "C":
                rot_gen.terms[term] *= 1 + fractional_pulse_error_C
            elif labels[term[0][0]] == "H":
                rot_gen.terms[term] *= 1 + fractional_pulse_error

    # --- Main Unitary Block ---
    Pines = [YI_gen, II_gen, YI_gen] * 2  # 6 pulses
    DR = [YX_gen, IY_gen, YX_gen, YYbar_gen, IXbar_gen, YYbar_gen]  # 6 pulses
    pulse_generators = (
        Pines + Pines + [ZI_gen] + DR + DR + [ZI_gen] + Pines + Pines + [ZI_gen]
    )

    # Labels for circuit plotting purposes
    Pines_names = ["YI", "II", "YI"] * 2
    DR_names = ["YX", "IY", "YX", "YYb", "IXb", "YYb"]
    Pinesbar_names = ["YI", "II", "YI"] * 2
    DRbar_names = ["YX", "IY", "YX", "YYb", "IXb", "YYb"]
    pulse_names = (
        (Pines_names + Pinesbar_names + ["IZ"])
        + (DR_names + DRbar_names + ["IZ"])
        + (Pines_names + Pinesbar_names + ["IZ"])
    )

    # Define times and angles
    pulse_times = (
        [pulse_time] * 12 + [0] + [pulse_time] * 12 + [0] + [pulse_time] * 12 + [0]
    )
    # Angles: Alternating blocks of positive/negative pi/2
    actual_pi2 = np.pi / 2
    pulse_angles = []
    for i in range(6):  # Iterate through the 6 blocks
        sign = 1 if i % 2 == 0 else -1
        pulse_angles.extend([sign * actual_pi2] * 6)
        if i % 2 == 1:
            pulse_angles += [z_angle]

    unitary_block = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=pulse_generators,
        pulse_times=pulse_times,
        pulse_angles=pulse_angles,
        pulse_names=pulse_names,
    )

    # The inversion of DREAM-2 needs an extra pi rotation on the C spin
    # This is only done once for the entire unitary, so we make this
    # separately and return everything to the user.

    # --- Start Pulse (pi rotation around IY) ---
    start_pulse = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=[IY_gen],
        pulse_times=[pulse_time * 2],
        pulse_angles=[np.pi],
        pulse_names=["IY"],
    )

    # --- End Pulse (-pi rotation around IY) ---
    end_pulse = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=[IY_gen],
        pulse_times=[pulse_time * 2],
        pulse_angles=[-np.pi],
        pulse_names=["IY"],
    )

    return unitary_block, start_pulse, end_pulse


def dream_2_inverse_sequence_CH(
    labels: List[str],
    qubits: List[cirq.Qid],
    pulse_time: float,
    background_H: QubitOperator,
    fractional_pulse_error: Optional[float] = 0,
    fractional_pulse_error_C: Optional[float] = None,
):
    """Constructs the DREAM-1 sequence on a molecule as a cirq.Circuit.

    Args:
        label_list: whether the atom is H or C
        qubits: list of qubits to act on
        pulse_time: time per pulse
        fractional_pulse_error: to model inhomogeneity; we set all pulses to
            have amplitude = target_amp * (1 + fractional_pulse_error)
        background_H: Background NMR Hamiltonian
    """
    # Make generators
    if len(qubits) != len(labels):
        raise ValueError("Need 1 qubit per label")
    if fractional_pulse_error_C is None:
        fractional_pulse_error_C = fractional_pulse_error

    II_gen, YX_gen, YI_gen, IY_gen, YYbar_gen, IXbar_gen, _ = (
        make_pulse_generators_DREAM_2_inverse(labels)
    )
    for rot_gen in [II_gen, YX_gen, YI_gen, IY_gen, YYbar_gen, IXbar_gen]:
        for term in rot_gen.terms:
            if labels[term[0][0]] == "C":
                rot_gen.terms[term] *= 1 + fractional_pulse_error_C
            elif labels[term[0][0]] == "H":
                rot_gen.terms[term] *= 1 + fractional_pulse_error

    # --- Main Unitary Block ---
    Pines = [YI_gen, II_gen, YI_gen] * 2  # 6 pulses
    DR = [YX_gen, IY_gen, YX_gen, YYbar_gen, IXbar_gen, YYbar_gen]  # 6 pulses
    pulse_generators = Pines + Pines + DR + DR + Pines + Pines

    # Labels for circuit plotting purposes
    Pines_names = ["YI", "II", "YI"] * 2
    DR_names = ["YX", "IY", "YX", "YYb", "IXb", "YYb"]
    Pinesbar_names = ["YI", "II", "YI"] * 2
    DRbar_names = ["YX", "IY", "YX", "YYb", "IXb", "YYb"]
    pulse_names = (
        (Pines_names + Pinesbar_names)
        + (DR_names + DRbar_names)
        + (Pines_names + Pinesbar_names)
    )

    # Define times and angles
    pulse_times = [pulse_time] * 12 + [pulse_time] * 12 + [pulse_time] * 12
    # Angles: Alternating blocks of positive/negative pi/2
    actual_pi2 = np.pi / 2
    pulse_angles = []
    for i in range(6):  # Iterate through the 6 blocks
        sign = 1 if i % 2 == 0 else -1
        pulse_angles.extend([sign * actual_pi2] * 6)

    unitary_block = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=pulse_generators,
        pulse_times=pulse_times,
        pulse_angles=pulse_angles,
        pulse_names=pulse_names,
    )

    # The inversion of DREAM-2 needs an extra pi rotation on the C spin
    # This is only done once for the entire unitary, so we make this
    # separately and return everything to the user.

    # --- Start Pulse (pi rotation around IY) ---
    start_pulse = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=[IY_gen],
        pulse_times=[pulse_time * 2],
        pulse_angles=[np.pi],
        pulse_names=["IY"],
    )

    # --- End Pulse (-pi rotation around IY) ---
    end_pulse = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=[IY_gen],
        pulse_times=[pulse_time * 2],
        pulse_angles=[-np.pi],
        pulse_names=["IY"],
    )

    return unitary_block, start_pulse, end_pulse


def blew12_constructor_generation(
    labels: List[str],
    pulse_time: float,
    fractional_pulse_error: Optional[float] = 0,
    fractional_pulse_error_C: Optional[float] = None,
):
    if fractional_pulse_error_C is None:
        fractional_pulse_error_C = fractional_pulse_error
    "Makes the constructors for the blew12 pulse sequence"
    XI_gen, YI_gen, XbarI_gen, YbarI_gen, IY_gen = make_pulse_generators_blew12(labels)

    for rot_gen in [XI_gen, YI_gen, XbarI_gen, YbarI_gen, IY_gen]:
        for term in rot_gen.terms:
            if labels[term[0][0]] == "C":
                rot_gen.terms[term] *= 1 + fractional_pulse_error_C
            elif labels[term[0][0]] == "H":
                rot_gen.terms[term] *= 1 + fractional_pulse_error

    blew_generators = [
        XI_gen,
        YI_gen,
        XbarI_gen,
        XbarI_gen,
        YbarI_gen,
        XbarI_gen,
        XI_gen,
        YI_gen,
        XI_gen,
        XI_gen,
        YbarI_gen,
        XbarI_gen,
    ]
    num_pulses = len(blew_generators)  # Should be 12
    blew_names = [
        "XI",
        "YI",
        "XbI",
        "XbI",
        "YbI",
        "XbI",
        "XI",
        "YI",
        "XI",
        "XI",
        "YbI",
        "XbI",
    ]
    blew_times = [pulse_time] * num_pulses
    target_angle = np.pi / 2
    actual_angle = target_angle
    blew_angles = [actual_angle] * num_pulses
    return blew_generators, blew_angles, blew_times, blew_names


def blew12_pulse_sequence_CH(
    labels: List[str],
    qubits: List[cirq.Qid],
    pulse_time: float,
    background_H: QubitOperator,
    fractional_pulse_error: Optional[float] = 0,
    fractional_pulse_error_C: Optional[float] = None,
):
    """Generates unitaries for the BLEW12 pulse sequence as a cirq.Circuit

    Args:
        label_list: whether the atom is H or C
        qubits: list of qubits to act on
        pulse_time: time per pulse
        fractional_pulse_error: to model inhomogeneity; we set all pulses to
            have amplitude = target_amp * (1 + fractional_pulse_error)
        background_H: Background NMR Hamiltonian
    """
    if fractional_pulse_error_C is None:
        fractional_pulse_error_C = fractional_pulse_error
    blew_generators, blew_angles, blew_times, blew_names = (
        blew12_constructor_generation(
            labels=labels,
            pulse_time=pulse_time,
            fractional_pulse_error=fractional_pulse_error,
            fractional_pulse_error_C=fractional_pulse_error_C,
        )
    )
    blew_block = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=blew_generators,
        pulse_times=blew_times,
        pulse_angles=blew_angles,
        pulse_names=blew_names,
    )
    XI_gen, YI_gen, XbarI_gen, YbarI_gen, IY_gen = make_pulse_generators_blew12(labels)

    for rot_gen in [XI_gen, YI_gen, XbarI_gen, YbarI_gen, IY_gen]:
        for term in rot_gen.terms:
            if labels[term[0][0]] == "C":
                rot_gen.terms[term] *= 1 + fractional_pulse_error_C
            elif labels[term[0][0]] == "H":
                rot_gen.terms[term] *= 1 + fractional_pulse_error

    mid_block = make_pulsed_scheme_circuit(
        qubits=qubits,
        background_H=background_H,
        pulse_generators=[IY_gen],
        pulse_times=[2 * pulse_time],
        pulse_angles=[np.pi],
        pulse_names=["IY"],
    )
    return blew_block, mid_block
