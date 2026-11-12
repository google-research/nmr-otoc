# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""OTOC simulation: experiment construction, Trotterized circuits, and Haar-averaged runs."""

import itertools
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import cirq
import numpy as np
import qsimcirq
from joblib import Parallel, delayed
from nmrotoc.circuits.trotter import (
    first_order_trotterize_dq_hamiltonian,
    make_butterfly_circuit,
    make_product_formula_swap_network_dq_hamiltonian,
)
from nmrotoc.dataclasses import NMROTOCExperiment
from numpy.typing import NDArray

from nmr_otoc_tools.cli import configure_logging, log_duration

logger = logging.getLogger(__name__)


class CircuitType(str, Enum):
    """Circuit construction used for the Trotterized time evolution."""

    SWAP_NETWORK = "swap_network"
    DENSE = "dense"


@dataclass(frozen=True, kw_only=True)
class OTOCSimulation:
    """Parameters for the OTOC simulation.

    Attributes:
        num_spins: Number of spins (qubits) in the simulated system.
        t_range: Time points to simulate. The simulators require the uniform grid
            ``(Δ, 2Δ, ..., KΔ)`` with ``Δ > 0`` (see :func:`trotter_step_counts`).
        n_trotter: Number of Trotter steps per grid interval ``Δ``; time index ``k`` uses
            ``(k + 1) * n_trotter`` steps.
        n_haar: Number of Haar-random states averaged over.
        n_thread: Number of qsim threads per process.
        n_proc: Number of parallel processes (joblib ``n_jobs``: negative values count back from
            the number of CPUs, e.g. ``-1`` uses all cores).
        save_circuit: Whether to write each OTOC circuit to the instance folder as Cirq JSON.
        circuit_type: Circuit construction: ``"swap_network"`` (default) or ``"dense"``.
        eta_hom: Scale factor applied to the homogeneous (same-spin) coupling matrix.
        eta_het: Scale factor applied to the heterogeneous (cross-spin) coupling matrix.
    """

    num_spins: int
    t_range: tuple[float, ...]
    n_trotter: int
    n_haar: int
    n_thread: int
    n_proc: int
    save_circuit: bool
    circuit_type: CircuitType = CircuitType.SWAP_NETWORK
    eta_hom: float = 1.0
    eta_het: float = 1.0

    def __post_init__(self) -> None:
        """Normalizes ``circuit_type`` and validates the parameters."""
        if isinstance(self.circuit_type, str) and not isinstance(self.circuit_type, CircuitType):
            object.__setattr__(self, "circuit_type", CircuitType(self.circuit_type))
        for name in ("n_trotter", "n_haar", "n_thread"):
            value = getattr(self, name)
            if value < 1:
                msg = f"{name} must be at least 1, got {value}."
                raise ValueError(msg)
        if self.n_proc == 0:
            msg = "n_proc must be non-zero (negative values follow joblib: -1 = all cores)."
            raise ValueError(msg)
        if len(self.t_range) == 0:
            msg = "t_range must not be empty."
            raise ValueError(msg)

    def dump(self, in_logger: logging.Logger) -> None:
        """Dumps simulation parameters to the provided logger."""
        for key, value in asdict(self).items():
            in_logger.info("%17s: %s", key, str(value))

    def to_dict(self) -> dict:
        """Converts the simulation parameters to a dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Converts the simulation parameters to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "OTOCSimulation":
        """Creates an OTOCSimulation from a dictionary (the dictionary is not modified)."""
        if "t_range" in d:
            d = {**d, "t_range": tuple(d["t_range"])}
        return cls(**d)

    @classmethod
    def from_json(cls, json_str: str) -> "OTOCSimulation":
        """Creates an OTOCSimulation from a JSON string."""
        return cls.from_dict(json.loads(json_str))


def trotter_step_counts(t_range: Sequence[float] | NDArray, n_trotter: int) -> list[int]:
    """Returns the number of Trotter steps used for each time point.

    The simulators assume the uniform grid ``t_range = (Δ, 2Δ, ..., KΔ)`` with ``Δ > 0`` and use
    ``(k + 1) * n_trotter`` steps for time index ``k``, i.e. a constant step size
    ``Δ / n_trotter`` (which also makes incremental state caching possible).

    Args:
        t_range: The time points to simulate.
        n_trotter: Number of Trotter steps per grid interval ``Δ``.

    Returns:
        The step counts ``[n_trotter, 2 * n_trotter, ..., K * n_trotter]``.

    Raises:
        ValueError: If ``t_range`` is empty, does not start at a positive time, or is not
            proportional to ``(1, 2, ..., K)``.
    """
    t = np.asarray(t_range, dtype=np.float64)
    if t.size == 0 or t[0] <= 0 or not np.allclose(t, t[0] * np.arange(1, t.size + 1)):
        msg = (
            "t_range must be the uniform grid (Δ, 2Δ, ..., KΔ) with Δ > 0, e.g. "
            f"np.linspace(max_time / n_time, max_time, n_time); got {tuple(t.tolist())}."
        )
        raise ValueError(msg)
    return [(k + 1) * n_trotter for k in range(t.size)]


def generate_haar_state(n_qubits: int, rng: np.random.Generator) -> NDArray:
    """Generates a random quantum state vector sampled from the Haar measure.

    Args:
        n_qubits: Number of qubits.
        rng: Random number generator.

    Returns:
        Normalized complex state vector of size 2**n_qubits sampled from the Haar measure.
    """
    dim = 2**n_qubits
    real = rng.standard_normal(dim)
    imag = rng.standard_normal(dim)
    state = real + 1j * imag
    state /= np.linalg.norm(state)
    return state


def make_experiment_fixed_num_spins(
    num_spins: int,
    indices_full: NDArray,
    full_hamiltonian: NDArray,
    eta_hom: float = 1.0,
    eta_het: float = 1.0,
    num_system_spins: int = 2,
) -> NMROTOCExperiment:
    """Creates an NMR OTOC experiment configuration for a fixed number of spins.

    The first ``num_system_spins`` spins form the system and the rest the environment; both
    circuit builders apply every coupling of the resulting partition. Spin 0 is measured and
    spin 1 receives the butterfly.

    Args:
        num_spins: Number of spins (qubits) to include.
        indices_full: Array of sorted indices for the full system.
        full_hamiltonian: The full Hamiltonian matrix.
        eta_hom: Scale factor applied to the homogeneous (same-spin) coupling matrix.
        eta_het: Scale factor applied to the heterogeneous (cross-spin) coupling matrix.
        num_system_spins: The number of spins belonging to the system subsystem (default 2).

    Returns:
        The experiment configuration object with sliced coupling matrices.
    """
    if num_spins < num_system_spins:
        msg = (
            f"num_spins ({num_spins}) must be at least num_system_spins ({num_system_spins}) "
            "to define a system-environment partition."
        )
        raise ValueError(msg)
    indices = indices_full[:num_spins]
    ham = full_hamiltonian[np.ix_(indices, indices)]
    hom = np.zeros(ham.shape)
    het = np.zeros(ham.shape)
    hom[num_system_spins:, num_system_spins:] = ham[num_system_spins:, num_system_spins:]
    hom[:num_system_spins, :num_system_spins] = ham[:num_system_spins, :num_system_spins]
    het[num_system_spins:, :num_system_spins] = ham[num_system_spins:, :num_system_spins]
    het[:num_system_spins, num_system_spins:] = ham[:num_system_spins, num_system_spins:]
    return NMROTOCExperiment(
        bfly_ids=[1],
        msmt_id=0,
        msmt_gate=cirq.X,
        bfly_gate=cirq.X,
        homogeneous_coupling_mat=hom * eta_hom,
        heterogeneous_coupling_mat=het * eta_het,
    )


def make_mas_experiment_fixed_num_spins(
    num_spins: int,
    indices_full: NDArray,
    full_hamiltonian: NDArray,
    eta_hom: float = 1.0,
    eta_het: float = 1.0,
    num_system_spins: int = 2,
) -> NMROTOCExperiment:
    """Creates an NMR OTOC experiment from a complex (MAS) coupling matrix.

    This is the MAS counterpart of :func:`make_experiment_fixed_num_spins`. The homogeneous
    (double-quantum) blocks keep their complex rotor phases, in the storage convention of
    ``nmr-generate-instances --coupling mas``:

    ``H_hom = sum_{i<j} [Re(d_ij) (Y_i Y_j - X_i X_j) + Im(d_ij) (X_i Y_j + Y_i X_j)]``.

    The heterogeneous (``Z_p Z_i``) block must be real. ``eta_hom`` / ``eta_het`` are applied
    exactly as in the static case.

    Args:
        num_spins: Number of spins (qubits) to include.
        indices_full: Array of sorted indices for the full system.
        full_hamiltonian: The full complex symmetric coupling matrix.
        eta_hom: Scale factor applied to the homogeneous (same-spin) coupling matrix.
        eta_het: Scale factor applied to the heterogeneous (cross-spin) coupling matrix.
        num_system_spins: The number of spins belonging to the system subsystem (default 2).

    Returns:
        The experiment configuration with a complex homogeneous and a real heterogeneous
        coupling matrix.

    Raises:
        ValueError: If there are fewer than ``num_system_spins`` spins, if the sliced matrix is
            not symmetric, or if a heterogeneous coupling has a non-zero imaginary part.
    """
    if num_spins < num_system_spins:
        msg = (
            f"num_spins ({num_spins}) must be at least num_system_spins ({num_system_spins}) "
            "to define a system-environment partition."
        )
        raise ValueError(msg)
    indices = indices_full[:num_spins]
    ham = np.asarray(full_hamiltonian[np.ix_(indices, indices)], dtype=np.complex128)
    if not np.allclose(ham, ham.T):
        msg = "MAS coupling matrix must be complex symmetric (d_ij = d_ji)."
        raise ValueError(msg)
    hom_mask = np.zeros(ham.shape, dtype=bool)
    hom_mask[:num_system_spins, :num_system_spins] = True
    hom_mask[num_system_spins:, num_system_spins:] = True
    tol = 1e-12 * max(float(np.abs(ham).max(initial=0.0)), 1.0)
    if np.abs(ham.imag[~hom_mask]).max(initial=0.0) > tol:
        msg = (
            "Heterogeneous (Z_p Z_i) couplings must be real, but the MAS coupling matrix has "
            "complex system-environment entries."
        )
        raise ValueError(msg)
    hom = np.where(hom_mask, ham, 0.0)
    het = np.where(hom_mask, 0.0, ham.real)
    return NMROTOCExperiment(
        bfly_ids=[1],
        msmt_id=0,
        msmt_gate=cirq.X,
        bfly_gate=cirq.X,
        homogeneous_coupling_mat=hom * eta_hom,
        heterogeneous_coupling_mat=het * eta_het,
    )


def create_experiment(
    iavg: int, sim_params: OTOCSimulation, sub_folder: Path, name: str
) -> NMROTOCExperiment | None:
    """Loads data and constructs the NMROTOCExperiment.

    Args:
        iavg: Index of the averaging instance.
        sim_params: Simulation parameters.
        sub_folder: Path to the folder containing Hamiltonian and indices.
        name: Name of the sub-instance (for logging).

    Returns:
        The experiment configuration object, or None if validation fails.
    """
    try:
        hamiltonian = np.load(sub_folder / f"dipolar_matrix{iavg}.npy")
        indices = np.load(sub_folder / f"ordered_indices{iavg}.npy")
    except FileNotFoundError as e:
        msg = (
            f"Required matrix or indices file not found in '{sub_folder}' for instance {iavg}. "
            f"Please ensure instances are generated for '{name}' before running simulations. "
            f"Original error: {e}"
        )
        logger.exception(msg)
        raise FileNotFoundError(msg) from e
    is_mas = np.iscomplexobj(hamiltonian)
    if is_mas and sim_params.circuit_type != CircuitType.DENSE:
        msg = (
            f"Coupling matrix for instance {iavg} in '{sub_folder}' is complex, i.e. it carries "
            "double-quantum rotor phases produced by 'nmr-generate-instances --coupling mas'. "
            "Phased DQ couplings are only supported by the dense circuit builder; rerun with "
            "--circuit_type dense."
        )
        logger.error(msg)
        raise NotImplementedError(msg)
    if np.isnan(hamiltonian).any():
        msg = f"NaN values found in Hamiltonian for averaging instance {iavg}."
        logger.error(msg)
        raise ValueError(msg)

    num_spins = sim_params.num_spins
    if num_spins > len(indices):
        logger.warning(
            "Requested num_spins %d exceeds available indices %d for instance %s. Skipping.",
            num_spins,
            len(indices),
            name,
        )
        return None

    if is_mas:
        return make_mas_experiment_fixed_num_spins(
            num_spins,
            indices,
            hamiltonian,
            eta_hom=sim_params.eta_hom,
            eta_het=sim_params.eta_het,
        )
    return make_experiment_fixed_num_spins(
        num_spins,
        indices,
        hamiltonian,
        eta_hom=sim_params.eta_hom,
        eta_het=sim_params.eta_het,
    )


def _require_qubit0_x_measurement(experiment: NMROTOCExperiment) -> None:
    """Checks the measurement assumptions hard-coded in the OTOC circuits and simulators.

    The protocol prepares ``|0>`` on qubit 0, applies ``H`` to it before and after the
    evolution, and measures ``Z`` on qubit 0, i.e. an ``X`` measurement of spin 0.

    Raises:
        ValueError: If the experiment measures another spin or uses another measurement gate.
    """
    if experiment.msmt_id != 0 or experiment.msmt_gate not in (None, cirq.X):
        msg = (
            "Only an X measurement of spin 0 is supported (msmt_id=0, msmt_gate=X); got "
            f"msmt_id={experiment.msmt_id}, msmt_gate={experiment.msmt_gate}."
        )
        raise ValueError(msg)


def make_otoc_components(
    experiment: NMROTOCExperiment,
    expt_time: float,
    num_steps: int,
    qubits: Sequence[cirq.Qid],
) -> tuple[cirq.Circuit, cirq.Circuit]:
    """Creates the components (forward evolution and butterfly) for an OTOC circuit."""
    if np.iscomplexobj(experiment.homogeneous_coupling_mat):
        msg = (
            "The swap-network circuit builder does not support complex (MAS phased-DQ) "
            "couplings; use circuit_type='dense'."
        )
        raise NotImplementedError(msg)
    trotter_obj = first_order_trotterize_dq_hamiltonian(experiment, expt_time, num_steps)
    trotter_obj.alternate_order = False
    circuit, qubit_id_refs = make_product_formula_swap_network_dq_hamiltonian(trotter_obj, qubits)
    butterfly = make_butterfly_circuit(experiment, qubit_id_refs)
    return circuit, butterfly


def make_otoc_circuit(
    experiment: NMROTOCExperiment,
    expt_time: float,
    num_steps: int,
    qubits: Sequence[cirq.Qid],
) -> cirq.Circuit:
    """Creates the main OTOC circuit for a given time and number of steps.

    Args:
        experiment: The experiment configuration.
        expt_time: The evolution time.
        num_steps: Number of Trotter steps.
        qubits: The qubits to use.

    Returns:
        The complete OTOC circuit.
    """
    _require_qubit0_x_measurement(experiment)
    circuit, butterfly = make_otoc_components(experiment, expt_time, num_steps, qubits)

    return cirq.Circuit(
        cirq.H(qubits[0]), circuit, butterfly, cirq.inverse(circuit), cirq.H(qubits[0])
    )


def _cirq_exponent(coupling: float, dt: float, *, is_negative: bool = False) -> float:
    """Returns the exponent ``a`` with ``P**a = exp(-i sign coupling dt P)`` up to global phase.

    For ``P`` in ``{XX, YY, ZZ}``, Cirq's ``P**a`` equals ``exp(-i pi a P / 2)`` up to a global
    phase, so ``a = 2 sign coupling dt / pi`` with ``sign = -1`` if ``is_negative``.
    """
    sign = -1 if is_negative else 1
    return 2 * sign * (coupling * dt) / (np.pi)


def _split_phase(coupling: complex) -> tuple[float, float]:
    """Splits a homogeneous coupling ``d = |d| exp(i phi)`` into a gate magnitude and a phase.

    Args:
        coupling: The (possibly complex) homogeneous coupling.

    Returns:
        ``(Re d, 0.0)`` for a real coupling, so that real entries need no ``rz`` gates;
        otherwise ``(|d|, arg d)``.
    """
    if np.imag(coupling) == 0.0:
        return float(np.real(coupling)), 0.0
    return float(np.abs(coupling)), float(np.angle(coupling))


def _dq_gates(
    q_i: cirq.Qid, q_j: cirq.Qid, coupling: complex, dt: float
) -> tuple[list[cirq.Operation], list[cirq.Operation]]:
    """Returns the XX and YY parts of ``exp(-i dt [Re d (YY - XX) + Im d (XY + YX)])``.

    For ``d = |d| exp(i phi)`` this equals ``Rz_i(-phi) exp(-i dt |d| (Y_i Y_j - X_i X_j))
    Rz_i(phi)``, so a complex coupling wraps the ``XX`` and ``YY`` gates in ``rz(phi)`` ...
    ``rz(-phi)`` on qubit ``i``.
    """
    magnitude, phase = _split_phase(coupling)
    x_part = [cirq.XX(q_i, q_j) ** _cirq_exponent(magnitude, dt, is_negative=True)]
    y_part = [cirq.YY(q_i, q_j) ** _cirq_exponent(magnitude, dt, is_negative=False)]
    if phase != 0.0:
        x_part = [cirq.rz(phase)(q_i), *x_part, cirq.rz(-phase)(q_i)]
        y_part = [cirq.rz(phase)(q_i), *y_part, cirq.rz(-phase)(q_i)]
    return x_part, y_part


def _real_heteronuclear(het: NDArray) -> NDArray:
    """Returns the heterogeneous coupling matrix as a real array.

    Raises:
        ValueError: If the matrix has non-zero imaginary parts.
    """
    het = np.asarray(het)
    if np.iscomplexobj(het):
        if np.any(np.imag(het) != 0.0):
            msg = "Heterogeneous (Z_p Z_i) couplings must be real."
            raise ValueError(msg)
        het = np.real(het)
    return het


def _require_symmetric(mat: NDArray, name: str) -> None:
    """Raises a ``ValueError`` unless ``mat`` is (complex) symmetric, ``d_ij = d_ji``."""
    if not np.allclose(mat, mat.T):
        msg = f"The {name} coupling matrix must be symmetric (d_ij = d_ji)."
        raise ValueError(msg)


def make_otoc_components2(
    experiment: NMROTOCExperiment,
    expt_time: float,
    num_steps: int,
    qubits: Sequence[cirq.Qid],
) -> tuple[cirq.Circuit, cirq.Circuit]:
    """Creates the dense forward-evolution and butterfly circuits for an OTOC circuit.

    Real and complex (MAS phased double-quantum) couplings are handled alike. Every spin pair
    ``i < j`` with a non-zero homogeneous coupling ``d`` evolves under
    ``exp(-i dt [Re(d) (Y_i Y_j - X_i X_j) + Im(d) (X_i Y_j + Y_i X_j)])`` (see
    :func:`_dq_gates`), and every pair with a non-zero heterogeneous coupling under
    ``exp(-i dt d Z_i Z_j)``. One Trotter step applies all ``XX`` parts, then all ``YY``
    parts, then all ``ZZ`` terms. The pair of system spins (measurement and butterfly) is
    always emitted, even if its coupling is zero.

    Args:
        experiment: The experiment configuration.
        expt_time: The evolution time.
        num_steps: Number of Trotter steps.
        qubits: The qubits to use, indexed by spin.

    Returns:
        The forward evolution circuit and the butterfly circuit.

    Raises:
        ValueError: If a coupling matrix is not symmetric or the heterogeneous couplings have
            non-zero imaginary parts.
    """
    hom = np.asarray(experiment.homogeneous_coupling_mat)
    het = _real_heteronuclear(experiment.heterogeneous_coupling_mat)
    _require_symmetric(hom, "homogeneous")
    _require_symmetric(het, "heterogeneous")
    dt = expt_time / num_steps if num_steps > 0 else 0.0
    system_pair = tuple(sorted((experiment.msmt_id, experiment.bfly_ids[0])))
    pairs = list(itertools.combinations(range(len(qubits)), 2))

    trotter_x = cirq.Circuit()
    trotter_y = cirq.Circuit()
    trotter_z = cirq.Circuit()
    for i, j in pairs:
        if hom[i, j] != 0 or (i, j) == system_pair:
            x_part, y_part = _dq_gates(qubits[i], qubits[j], hom[i, j], dt)
            trotter_x.append(x_part)
            trotter_y.append(y_part)
    for i, j in pairs:
        if het[i, j] != 0:
            trotter_z.append(cirq.ZZ(qubits[i], qubits[j]) ** _cirq_exponent(het[i, j], dt))

    circuit = cirq.Circuit()
    for _ in range(num_steps):
        circuit.append(trotter_x)
        circuit.append(trotter_y)
        circuit.append(trotter_z)

    butterfly = make_butterfly_circuit(experiment, qubits)
    return circuit, butterfly


def make_phased_dq_components(
    experiment: NMROTOCExperiment,
    expt_time: float,
    num_steps: int,
    qubits: Sequence[cirq.Qid],
) -> tuple[cirq.Circuit, cirq.Circuit]:
    """Alias of :func:`make_otoc_components2`, kept for backward compatibility."""
    return make_otoc_components2(experiment, expt_time, num_steps, qubits)


def make_otoc_circuit2(
    experiment: NMROTOCExperiment,
    expt_time: float,
    num_steps: int,
    qubits: Sequence[cirq.Qid],
) -> cirq.Circuit:
    """Creates the main OTOC circuit for a given time and number of steps.

    Args:
        experiment: The experiment configuration.
        expt_time: The evolution time.
        num_steps: Number of Trotter steps.
        qubits: The qubits to use.

    Returns:
        The complete OTOC circuit.
    """
    _require_qubit0_x_measurement(experiment)
    circuit, butterfly = make_otoc_components2(experiment, expt_time, num_steps, qubits)
    return cirq.Circuit(
        cirq.H(qubits[0]), circuit, butterfly, cirq.inverse(circuit), cirq.H(qubits[0])
    )


def run_simulation(
    experiment: NMROTOCExperiment,
    sim_params: OTOCSimulation,
    sub_folder: Path,
    iavg: int,
    name: str,
) -> list[float]:
    """Performs the OTOC simulation without caching/incremental steps."""
    _require_qubit0_x_measurement(experiment)
    num_spins = sim_params.num_spins
    n_thread = sim_params.n_thread
    n_haar = sim_params.n_haar
    t_range = np.asarray(sim_params.t_range)
    step_counts = trotter_step_counts(t_range, sim_params.n_trotter)

    with log_duration(f"  Worker simulation for N={num_spins} iavg={iavg} ({name})", logger):
        logger.info("  Simulating N=%d iavg=%d (%s)...", num_spins, iavg, name)
        options = qsimcirq.QSimOptions(cpu_threads=n_thread)
        simulator = qsimcirq.QSimSimulator(qsim_options=options)

        qubits = cirq.LineQubit.range(num_spins)

        otoc_accum = [0.0] * len(t_range)
        rng_sim = np.random.default_rng()  # for haar state sampling

        # Draw seeds for Haar random states to keep them consistent across timesteps
        haar_seeds = rng_sim.integers(0, 2**32, size=n_haar)

        ti = time.perf_counter()
        for step_id, (expt_time, num_steps) in enumerate(zip(t_range, step_counts, strict=True)):
            with log_duration(
                f"    Trotter circuit construction (N={num_spins}, steps={num_steps})", logger
            ):
                if sim_params.circuit_type == CircuitType.DENSE:
                    main_circuit = make_otoc_circuit2(experiment, expt_time, num_steps, qubits)
                else:
                    main_circuit = make_otoc_circuit(experiment, expt_time, num_steps, qubits)

            if sim_params.save_circuit:
                circuit_path = sub_folder / (
                    f"circ{iavg}_N{num_spins}_t{sim_params.n_trotter}_{step_id}.json"
                )
                circuit_path.write_text(cirq.to_json(main_circuit))

            for h in range(n_haar):
                local_rng = np.random.default_rng(seed=haar_seeds[h])
                psi = np.kron(np.array((1, 0)), generate_haar_state(num_spins - 1, local_rng))
                psi_complex = np.complex64(psi)
                val = simulator.simulate_expectation_values(
                    main_circuit,
                    observables=[cirq.Z(qubits[0])],
                    initial_state=psi_complex,
                    qubit_order=qubits,
                )[0]
                otoc_accum[step_id] += float(np.real(val))
        tf = time.perf_counter()
        logger.debug("    Ran %d haar states over %d steps in %s s.", n_haar, len(t_range), tf - ti)

        otoc = [accum_val / n_haar for accum_val in otoc_accum]
        if np.isnan(otoc).sum() > 0:
            msg = "Found NaNs"
            raise RuntimeError(msg)
    return otoc


def _cached_simulation_circuits(
    experiment: NMROTOCExperiment,
    sim_params: OTOCSimulation,
    qubits: Sequence[cirq.Qid],
    sub_folder: Path,
    iavg: int,
) -> list[tuple[bool, cirq.Circuit, cirq.Circuit]]:
    """Builds the per-time-point circuits for :func:`run_simulation_cached`.

    For time index ``k`` the forward evolution is one dense Trotter step (step size
    ``dt_k = t_k / num_steps_k``) repeated ``num_steps_k`` times, so the step and its inverse
    are built once per time point instead of building and inverting the full evolution.

    Returns:
        ``(can_incremental, delta_forward, eval_circuit)`` for every time point.
        ``delta_forward`` advances the cached state from the previous time point (or prepares
        the evolved state from scratch, including the first ``H``, when ``can_incremental`` is
        False); ``eval_circuit`` applies the butterfly, the inverse evolution and the final
        ``H``.
    """
    num_spins = sim_params.num_spins
    t_range = np.asarray(sim_params.t_range)
    step_counts = trotter_step_counts(t_range, sim_params.n_trotter)

    circuits = []
    prev_dt = None
    prev_num_steps = 0
    for step_id, (expt_time, num_steps) in enumerate(zip(t_range, step_counts, strict=True)):
        dt = expt_time / num_steps
        with log_duration(
            f"    Trotter circuit construction (N={num_spins}, steps={num_steps})", logger
        ):
            step, butterfly = make_otoc_components2(experiment, dt, 1, qubits)
            inv_forward = cirq.inverse(step) * num_steps

            can_incremental = prev_dt is not None and bool(
                np.isclose(dt, prev_dt, rtol=0.0, atol=1e-12)
            )
            if can_incremental:
                delta_steps = num_steps - prev_num_steps
                delta_step, _ = make_otoc_components2(
                    experiment, (delta_steps * dt) / delta_steps, 1, qubits
                )
                delta_forward = delta_step * delta_steps
            else:
                delta_forward = cirq.Circuit(cirq.H(qubits[0]), step * num_steps)

            eval_circuit = cirq.Circuit(butterfly, inv_forward, cirq.H(qubits[0]))

        if sim_params.save_circuit:
            main_circuit = cirq.Circuit(
                cirq.H(qubits[0]), step * num_steps, butterfly, inv_forward, cirq.H(qubits[0])
            )
            circuit_path = sub_folder / (
                f"circ{iavg}_N{num_spins}_t{sim_params.n_trotter}_{step_id}.json"
            )
            circuit_path.write_text(cirq.to_json(main_circuit))

        circuits.append((can_incremental, delta_forward, eval_circuit))
        prev_dt = dt
        prev_num_steps = num_steps
    return circuits


def run_simulation_cached(
    experiment: NMROTOCExperiment,
    sim_params: OTOCSimulation,
    sub_folder: Path,
    iavg: int,
    name: str,
) -> list[float]:
    """Performs the dense OTOC simulation using the state caching strategy.

    Only :attr:`CircuitType.DENSE` is supported: every swap-network Trotter step reverses the
    spin-to-qubit layout, so swap-network circuits cannot be applied incrementally to a cached
    state. Use :func:`run_simulation` for swap-network circuits.

    Raises:
        ValueError: If ``sim_params.circuit_type`` is not :attr:`CircuitType.DENSE`.
    """
    if sim_params.circuit_type != CircuitType.DENSE:
        msg = (
            "run_simulation_cached supports only CircuitType.DENSE; use run_simulation for "
            "swap-network circuits."
        )
        raise ValueError(msg)
    _require_qubit0_x_measurement(experiment)
    num_spins = sim_params.num_spins
    n_haar = sim_params.n_haar
    n_times = len(sim_params.t_range)

    with log_duration(
        f"  Worker simulation (cached) for N={num_spins} iavg={iavg} ({name})", logger
    ):
        logger.info("  Simulating (cached) N=%d iavg=%d (%s)...", num_spins, iavg, name)
        options = qsimcirq.QSimOptions(cpu_threads=sim_params.n_thread)
        simulator = qsimcirq.QSimSimulator(qsim_options=options)

        qubits = cirq.LineQubit.range(num_spins)
        circuits = _cached_simulation_circuits(experiment, sim_params, qubits, sub_folder, iavg)

        otoc_accum = [0.0] * n_times
        rng_sim = np.random.default_rng()  # for haar state sampling

        ti = time.perf_counter()
        for _ in range(n_haar):
            psi = np.kron(np.array((1, 0)), generate_haar_state(num_spins - 1, rng_sim))
            psi_complex = np.complex64(psi)

            prev_state = None
            for step_id, (can_incremental, delta_forward, eval_circuit) in enumerate(circuits):
                initial_state = (
                    prev_state if can_incremental and prev_state is not None else psi_complex
                )
                sim_result = simulator.simulate(
                    delta_forward, initial_state=initial_state, qubit_order=qubits
                )

                state_vec = getattr(sim_result, "final_state_vector", None)
                if state_vec is None:
                    state_vec = sim_result.state_vector()
                prev_state = np.complex64(state_vec).copy()

                val = simulator.simulate_expectation_values(
                    eval_circuit,
                    observables=[cirq.Z(qubits[0])],
                    initial_state=prev_state,
                    qubit_order=qubits,
                )[0]
                otoc_accum[step_id] += float(np.real(val))
        tf = time.perf_counter()
        logger.debug("    Ran %d haar states over %d steps in %s s.", n_haar, n_times, tf - ti)

        otoc = [accum_val / n_haar for accum_val in otoc_accum]
        if np.isnan(otoc).sum() > 0:
            msg = "Found NaNs"
            raise RuntimeError(msg)
    return otoc


def _simulate_one(
    iavg: int,
    experiment: NMROTOCExperiment | None,
    sim_params: OTOCSimulation,
    sub_folder: Path,
    dist_name: str,
    *,
    verbose: bool,
) -> list[float]:
    """Simulates one averaging instance inside a parallel worker.

    Args:
        iavg: Index of the averaging instance.
        experiment: The experiment, or None to load it in the worker via
            :func:`create_experiment` (keeps loading lazy and avoids pickling the matrices).
        sim_params: Simulation parameters.
        sub_folder: Instance folder (data source when loading, and circuit output folder).
        dist_name: Name of the instance sub-folder (for logging).
        verbose: If True, enable debug-level logging in the worker.

    Returns:
        The OTOC values for every time point (NaNs if the instance has too few spins).
    """
    configure_logging(verbose=verbose)
    if experiment is None:
        experiment = create_experiment(iavg, sim_params, sub_folder, dist_name)
        if experiment is None:
            return [np.nan] * len(sim_params.t_range)
    if sim_params.circuit_type == CircuitType.SWAP_NETWORK:
        return run_simulation(experiment, sim_params, sub_folder, iavg, dist_name)
    return run_simulation_cached(experiment, sim_params, sub_folder, iavg, dist_name)


def run_parallel_simulation(
    sim_params: OTOCSimulation,
    sub_folder: Path,
    dist_name: str,
    experiments: list[NMROTOCExperiment] | None = None,
    n_avg: int | None = None,
    *,
    verbose: bool = False,
) -> list[list[float]]:
    """Runs the OTOC simulation in parallel, handling both dynamic and static experiments.

    Args:
        sim_params: Simulation parameters.
        sub_folder: Path to the folder containing data or for saving circuits.
        dist_name: Name of the distribution.
        experiments: List of pre-constructed experiments. If None, experiments
            will be loaded dynamically using n_avg inside parallel workers.
        n_avg: Number of averaging instances to load dynamically (required if
            experiments is None).
        verbose: If True, enable debug-level timing and logging.

    Returns:
        List of OTOC list results of shape (num_instances, num_times).
    """
    if experiments is None:
        if n_avg is None:
            msg = "n_avg must be specified if experiments is None."
            raise ValueError(msg)
        tasks: list[tuple[int, NMROTOCExperiment | None]] = [(iavg, None) for iavg in range(n_avg)]
    else:
        tasks = list(enumerate(experiments))

    return Parallel(n_jobs=sim_params.n_proc)(
        delayed(_simulate_one)(iavg, exp, sim_params, sub_folder, dist_name, verbose=verbose)
        for iavg, exp in tasks
    )
