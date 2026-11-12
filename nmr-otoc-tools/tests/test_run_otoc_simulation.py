# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for running OTOC simulations."""

import argparse
import copy
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import cirq
import numpy as np
import pytest
import qsimcirq
from nmrotoc.dataclasses import NMROTOCExperiment

from nmr_otoc_tools.run_otoc_simulation import (
    main,
    process_sub_instance,
    run_instance,
)
from nmr_otoc_tools.simulation import (
    CircuitType,
    OTOCSimulation,
    create_experiment,
    generate_haar_state,
    make_experiment_fixed_num_spins,
    make_mas_experiment_fixed_num_spins,
    make_otoc_circuit,
    make_otoc_circuit2,
    make_otoc_components,
    make_otoc_components2,
    make_phased_dq_components,
    run_parallel_simulation,
    run_simulation,
    run_simulation_cached,
    trotter_step_counts,
)


def test_generate_haar_state() -> None:
    """Test standard normalized state vector generation."""
    rng = np.random.default_rng(42)
    n_qubits = 2
    state = generate_haar_state(n_qubits, rng)
    expected_dim = 4
    assert state.shape == (expected_dim,)
    expected_norm = 1.0
    assert np.isclose(np.linalg.norm(state), expected_norm)


def test_make_experiment_fixed_num_spins() -> None:
    """Test generating a fixed spin experiment from full hamiltonian."""
    num_spins = 3
    expected_shape = (3, 3)
    indices_full = np.array([0, 1, 2, 3, 4])
    full_hamiltonian = np.ones((5, 5))

    experiment = make_experiment_fixed_num_spins(num_spins, indices_full, full_hamiltonian)

    assert experiment.homogeneous_coupling_mat.shape == expected_shape
    assert experiment.heterogeneous_coupling_mat.shape == expected_shape
    assert experiment.homogeneous_coupling_mat[2, 2] == 1.0
    assert experiment.heterogeneous_coupling_mat[2, 0] == 1.0

    # Check that indices slicing correctly took [0, 1, 2] part of the full hamiltonian
    assert np.all(experiment.homogeneous_coupling_mat[:2, :2] == 1.0)
    assert np.all(experiment.homogeneous_coupling_mat[2:, 2:] == 1.0)


def test_run_parallel_simulation_smoke(sample_spin_data: Path) -> None:
    """Smoke test to ensure parallel simulation functions without errors."""
    sim_params = OTOCSimulation(
        num_spins=2,
        t_range=(0.5, 1.0),
        n_trotter=1,
        n_haar=1,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
    )

    results = run_parallel_simulation(sim_params, sample_spin_data, "test_dist", n_avg=1)
    assert isinstance(results, list)
    assert len(results) == 1
    assert len(results[0]) == len(sim_params.t_range)


@patch("nmr_otoc_tools.run_otoc_simulation.run_parallel_simulation")
def test_process_sub_instance(mock_run_parallel: MagicMock, tmp_path: Path) -> None:
    """Test sub instance processing parallel pipeline."""
    mock_run_parallel.return_value = [[0.5, 0.4], [0.5, 0.4]]

    args = argparse.Namespace(
        pdb_id="test_pdb",
        n_avg=2,
        n_min=2,
        n_max=2,
        max_time=2.0,
        n_time=2,
        n_trotter=1,
        n_haar=1,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
        circuit_type="swap_network",
        eta_hom=1.0,
        eta_het=1.0,
    )

    process_sub_instance(tmp_path, args)

    mock_run_parallel.assert_called_once()

    # Verify the output JSON was created
    res_file = tmp_path / "otoc_N2_t1_h1.json"
    assert res_file.exists()

    with res_file.open() as f:
        data = json.load(f)

    expected_n = 2
    assert data["N"] == expected_n
    assert "times" in data
    assert len(data["otocs"]) == 2  # n_avg
    assert data["otocs"][0] == [0.5, 0.4]


@patch("nmr_otoc_tools.run_otoc_simulation.process_sub_instance")
def test_run_instance(mock_process_sub_instance: MagicMock) -> None:
    """Test full system instance run entrypoint."""
    args = argparse.Namespace(
        pdb_id="test_pdb",
        subfolder="sub1",
        n_avg=1,
        n_min=2,
        n_max=3,
        max_time=2.0,
        n_time=8,
        n_trotter=10,
        n_haar=10,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
        circuit_type="swap_network",
        eta_hom=1.0,
        eta_het=1.0,
    )
    run_instance(args)
    mock_process_sub_instance.assert_called_once()


def test_create_experiment_num_spins_exceeds(sample_spin_data: Path) -> None:
    """Test create_experiment handles cases where num_spins exceeds available indices.

    Verifies it returns None.
    """
    sim_params = OTOCSimulation(
        num_spins=10,
        t_range=(0.0, 1.0),
        n_trotter=1,
        n_haar=1,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
    )
    exp = create_experiment(0, sim_params, sample_spin_data, "test")
    assert exp is None


def test_run_parallel_simulation_save_circuit(sample_spin_data: Path) -> None:
    """Mock-free integration test.

    Verify run_parallel_simulation saves circuits when save_circuit=True.
    """
    sim_params = OTOCSimulation(
        num_spins=2,
        t_range=(0.5,),
        n_trotter=1,
        n_haar=1,
        n_thread=1,
        n_proc=1,
        save_circuit=True,
    )
    results = run_parallel_simulation(
        sim_params, sample_spin_data, "test", n_avg=1
    )
    assert len(results) == 1
    assert len(results[0]) == 1
    assert not np.isnan(results[0]).any()

    # Check if circuit file was saved
    circuit_files = list(sample_spin_data.glob("circ*.json"))
    assert len(circuit_files) == 1


@patch("nmr_otoc_tools.run_otoc_simulation.run_instance")
def test_main(mock_run: MagicMock) -> None:
    """Test main CLI entry point."""
    test_args = [
        "nmr-run-otoc-simulation",
        "--pdb_id",
        "test_pdb",
        "--n_avg",
        "1",
        "--n_min",
        "2",
        "--n_max",
        "2",
    ]
    with patch.object(sys, "argv", test_args):
        main()
    mock_run.assert_called_once()



def test_run_parallel_simulation_dense_circuit_smoke(sample_spin_data: Path) -> None:
    """Smoke test: run_parallel_simulation runs without error when circuit_type='dense'."""
    sim_params = OTOCSimulation(
        num_spins=2,
        t_range=(1.0,),
        n_trotter=1,
        n_haar=1,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
        circuit_type=CircuitType.DENSE,
    )
    results = run_parallel_simulation(
        sim_params, sample_spin_data, "test_dense", n_avg=1
    )
    assert isinstance(results, list)
    assert len(results) == 1
    assert len(results[0]) == len(sim_params.t_range)


def test_otoc_simulation_circuit_type_serialization() -> None:
    """Verify circuit_type round-trips correctly through OTOCSimulation.to_dict/from_dict."""
    for ct in ("swap_network", "dense"):
        params = OTOCSimulation(
            num_spins=3,
            t_range=(0.5, 1.0),
            n_trotter=2,
            n_haar=5,
            n_thread=1,
            n_proc=1,
            save_circuit=False,
            circuit_type=ct,
        )
        restored = OTOCSimulation.from_dict(params.to_dict())
        assert restored.circuit_type == ct


def test_eta_scaling_changes_experiment() -> None:
    """Verify that eta_hom/eta_het are correctly applied to the coupling matrices."""
    num_spins = 4
    indices_full = np.arange(num_spins)
    full_hamiltonian = np.ones((num_spins, num_spins))
    np.fill_diagonal(full_hamiltonian, 0.0)

    # Baseline with unit scaling
    exp_base = make_experiment_fixed_num_spins(
        num_spins, indices_full, full_hamiltonian, eta_hom=1.0, eta_het=1.0
    )

    # Scale homogeneous only
    exp_hom2 = make_experiment_fixed_num_spins(
        num_spins, indices_full, full_hamiltonian, eta_hom=2.0, eta_het=1.0
    )
    assert np.allclose(exp_hom2.homogeneous_coupling_mat, 2.0 * exp_base.homogeneous_coupling_mat)
    assert np.allclose(exp_hom2.heterogeneous_coupling_mat, exp_base.heterogeneous_coupling_mat)

    # Scale heterogeneous only
    exp_het3 = make_experiment_fixed_num_spins(
        num_spins, indices_full, full_hamiltonian, eta_hom=1.0, eta_het=3.0
    )
    assert np.allclose(exp_het3.homogeneous_coupling_mat, exp_base.homogeneous_coupling_mat)
    assert np.allclose(
        exp_het3.heterogeneous_coupling_mat, 3.0 * exp_base.heterogeneous_coupling_mat
    )

    # Zero out homogeneous coupling entirely
    exp_no_hom = make_experiment_fixed_num_spins(
        num_spins, indices_full, full_hamiltonian, eta_hom=0.0, eta_het=1.0
    )
    assert np.allclose(exp_no_hom.homogeneous_coupling_mat, 0.0)
    assert np.allclose(exp_no_hom.heterogeneous_coupling_mat, exp_base.heterogeneous_coupling_mat)


def test_eta_serialization() -> None:
    """Verify eta_hom and eta_het round-trip through OTOCSimulation.to_dict/from_dict."""
    params = OTOCSimulation(
        num_spins=3,
        t_range=(0.5, 1.0),
        n_trotter=2,
        n_haar=5,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
        eta_hom=0.5,
        eta_het=2.0,
    )
    restored = OTOCSimulation.from_dict(params.to_dict())
    assert restored.eta_hom == 0.5
    assert restored.eta_het == 2.0


# ---------------------------------------------------------------------------
# Parametrized circuit comparison tests
# ---------------------------------------------------------------------------


def _hamiltonian_all_ones(n: int) -> np.ndarray:
    """Uniform coupling — all off-diagonal entries equal 1."""
    ham = np.ones((n, n))
    np.fill_diagonal(ham, 0.0)
    return ham


def _hamiltonian_random(n: int, seed: int = 42) -> np.ndarray:
    """Random symmetric coupling matrix with a fixed seed for reproducibility."""
    rng = np.random.default_rng(seed)
    ham = rng.standard_normal((n, n))
    ham = (ham + ham.T) / 2
    np.fill_diagonal(ham, 0.0)
    return ham


def _hamiltonian_nearest_neighbor(n: int, coupling: float = 1.0) -> np.ndarray:
    """Only nearest-neighbor pairs are coupled (band-1 off-diagonal)."""
    ham = np.zeros((n, n))
    for i in range(n - 1):
        ham[i, i + 1] = ham[i + 1, i] = coupling
    return ham


_HAMILTONIAN_FACTORIES = {
    "all_ones": _hamiltonian_all_ones,
    "random_seeded": _hamiltonian_random,
    "nearest_neighbor": _hamiltonian_nearest_neighbor,
}


def _run_circuit_comparison(
    experiment: NMROTOCExperiment,
    num_spins: int,
    expt_time: float,
    num_steps: int,
) -> tuple[complex, complex]:
    """Simulate both OTOC circuits and return (val1, val2) for <Z(qubit[0])>.

    Args:
        experiment: The NMROTOCExperiment configuration.
        num_spins: Number of qubits.
        expt_time: Total evolution time.
        num_steps: Number of Trotter steps.

    Returns:
        Tuple of expectation values (swap_network_result, dense_result).
    """
    qubits = cirq.LineQubit.range(num_spins)
    circuit1 = make_otoc_circuit(experiment, expt_time, num_steps, qubits)
    circuit2 = make_otoc_circuit2(experiment, expt_time, num_steps, qubits)
    simulator = cirq.Simulator()
    state = np.zeros(2**num_spins, dtype=np.complex64)
    state[0] = 1.0
    val1 = simulator.simulate_expectation_values(
        circuit1, observables=[cirq.Z(qubits[0])], initial_state=state
    )[0]
    val2 = simulator.simulate_expectation_values(
        circuit2, observables=[cirq.Z(qubits[0])], initial_state=state
    )[0]
    return val1, val2


@pytest.mark.parametrize(("expt_time", "num_steps"), [(0.1, 32), (0.2, 64)])
@pytest.mark.parametrize("ham_type", ["all_ones", "random_seeded", "nearest_neighbor"])
@pytest.mark.parametrize("num_spins", [4, 5, 6])
def test_compare_otoc_circuits_parametrized(
    num_spins: int,
    ham_type: str,
    expt_time: float,
    num_steps: int,
) -> None:
    """Parametrized comparison of make_otoc_circuit vs make_otoc_circuit2.

    Covers 3 qubit counts x 3 Hamiltonian structures x 2 time/step pairs (18 cases total).
    """
    full_hamiltonian = _HAMILTONIAN_FACTORIES[ham_type](num_spins)
    indices_full = np.arange(num_spins)
    experiment = make_experiment_fixed_num_spins(num_spins, indices_full, full_hamiltonian)
    val1, val2 = _run_circuit_comparison(experiment, num_spins, expt_time, num_steps)
    assert np.isclose(val1, val2, atol=1e-3), (
        f"Circuit mismatch: swap_network={val1:.6f}, dense={val2:.6f} "
        f"(n={num_spins}, H={ham_type}, t={expt_time}, steps={num_steps})"
    )


def test_otoc_simulation_post_init_circuit_type() -> None:
    """Verify that circuit_type string is converted to CircuitType Enum in post_init."""
    sim = OTOCSimulation(
        num_spins=2,
        t_range=(0.0, 1.0),
        n_trotter=1,
        n_haar=1,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
        circuit_type="dense",
    )
    assert sim.circuit_type == CircuitType.DENSE


def test_otoc_simulation_dump() -> None:
    """Verify that dump logs the expected parameters."""
    sim = OTOCSimulation(
        num_spins=2,
        t_range=(0.0, 1.0),
        n_trotter=1,
        n_haar=1,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
    )
    mock_logger = MagicMock()
    sim.dump(mock_logger)
    assert mock_logger.info.call_count > 0
    # Verify at least one expected parameter was logged
    mock_logger.info.assert_any_call("%17s: %s", "num_spins", "2")


def test_make_experiment_too_few_spins() -> None:
    """Test make_experiment_fixed_num_spins raises ValueError for too few spins."""
    indices_full = np.array([0, 1])
    full_ham = np.ones((2, 2))
    # num_system_spins defaults to 2, so num_spins=1 is too few.
    with pytest.raises(ValueError, match="must be at least num_system_spins"):
        make_experiment_fixed_num_spins(
            num_spins=1,
            indices_full=indices_full,
            full_hamiltonian=full_ham,
            num_system_spins=2,
        )


def test_create_experiment_file_not_found(tmp_path: Path) -> None:
    """Test create_experiment raises FileNotFoundError when npy files are missing."""
    sim_params = OTOCSimulation(
        num_spins=2,
        t_range=(0.0, 1.0),
        n_trotter=1,
        n_haar=1,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
    )
    # Attempt to load from folder without dipolar_matrix0.npy
    with pytest.raises(FileNotFoundError, match="Required matrix or indices file not found"):
        create_experiment(iavg=0, sim_params=sim_params, sub_folder=tmp_path, name="test")


def test_create_experiment_nan_values(tmp_path: Path) -> None:
    """Test create_experiment raises ValueError when Hamiltonian contains NaN values."""
    sim_params = OTOCSimulation(
        num_spins=2,
        t_range=(0.0, 1.0),
        n_trotter=1,
        n_haar=1,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
    )
    iavg = 0
    # Create test files containing NaN values
    nan_ham = np.array([[np.nan, 1.0], [1.0, 0.0]])
    placeholder_indices = np.array([0, 1])
    np.save(tmp_path / f"dipolar_matrix{iavg}.npy", nan_ham)
    np.save(tmp_path / f"ordered_indices{iavg}.npy", placeholder_indices)

    with pytest.raises(ValueError, match="NaN values found in Hamiltonian"):
        create_experiment(iavg=iavg, sim_params=sim_params, sub_folder=tmp_path, name="test")


def test_create_experiment_complex_requires_dense(tmp_path: Path) -> None:
    """Complex (phased DQ) MAS couplings are rejected by the swap-network builder."""
    sim_params = OTOCSimulation(
        num_spins=2,
        t_range=(0.0, 1.0),
        n_trotter=1,
        n_haar=1,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
    )
    iavg = 0
    complex_ham = np.array([[0.0, 1.0 + 0.5j], [1.0 + 0.5j, 0.0]])
    np.save(tmp_path / f"dipolar_matrix{iavg}.npy", complex_ham)
    np.save(tmp_path / f"ordered_indices{iavg}.npy", np.array([0, 1]))

    with pytest.raises(NotImplementedError, match="--circuit_type dense"):
        create_experiment(iavg=iavg, sim_params=sim_params, sub_folder=tmp_path, name="test")


def test_create_experiment_complex_dense_accepted(tmp_path: Path) -> None:
    """With the dense builder, complex couplings keep their phases and eta is applied."""
    num_spins = 4
    full = _random_mas_matrix(num_spins, seed=1)
    np.save(tmp_path / "dipolar_matrix0.npy", full)
    np.save(tmp_path / "ordered_indices0.npy", np.arange(num_spins))
    sim_params = OTOCSimulation(
        num_spins=num_spins,
        t_range=(0.1,),
        n_trotter=1,
        n_haar=1,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
        circuit_type=CircuitType.DENSE,
        eta_hom=0.5,
        eta_het=2.0,
    )

    experiment = create_experiment(0, sim_params, tmp_path, "test")

    assert experiment is not None
    hom = experiment.homogeneous_coupling_mat
    het = experiment.heterogeneous_coupling_mat
    assert np.iscomplexobj(hom)
    assert not np.iscomplexobj(het)
    np.testing.assert_allclose(hom[:2, :2], 0.5 * full[:2, :2])
    np.testing.assert_allclose(hom[2:, 2:], 0.5 * full[2:, 2:])
    np.testing.assert_allclose(het[:2, 2:], 2.0 * full[:2, 2:].real)
    np.testing.assert_allclose(het[2:, :2], 2.0 * full[2:, :2].real)
    assert np.all(hom[:2, 2:] == 0.0)
    assert np.all(het[:2, :2] == 0.0)
    assert np.all(het[2:, 2:] == 0.0)


def test_run_parallel_simulation_skipped_experiment(sample_spin_data: Path) -> None:
    """Verify run_parallel_simulation returns NaNs when create_experiment returns None."""
    sim_params = OTOCSimulation(
        num_spins=10,  # Too many spins for 2 available indices
        t_range=(0.0, 1.0),
        n_trotter=1,
        n_haar=1,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
    )
    results = run_parallel_simulation(sim_params, sample_spin_data, "test", n_avg=1)
    assert len(results) == 1
    assert np.isnan(results[0]).all()
    assert len(results[0]) == len(sim_params.t_range)


def test_verbose_logging_with_timing(
    sample_spin_data: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that running simulation with verbose=True captures debug timing logs."""
    sim_params = OTOCSimulation(
        num_spins=2,
        t_range=(0.5,),
        n_trotter=1,
        n_haar=1,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
    )
    with caplog.at_level(logging.DEBUG):
        run_parallel_simulation(sim_params, sample_spin_data, "test", n_avg=1, verbose=True)

    debug_messages = [rec.message for rec in caplog.records if rec.levelno == logging.DEBUG]
    assert any("Trotter circuit construction" in msg for msg in debug_messages)
    assert any("completed in" in msg for msg in debug_messages)


def test_non_verbose_logging(sample_spin_data: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test that running simulation with verbose=False does not emit debug timing logs."""
    sim_params = OTOCSimulation(
        num_spins=2,
        t_range=(0.5,),
        n_trotter=1,
        n_haar=1,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
    )
    caplog.clear()
    run_parallel_simulation(sim_params, sample_spin_data, "test", n_avg=1, verbose=False)

    debug_records = [rec for rec in caplog.records if rec.levelno == logging.DEBUG]
    assert len(debug_records) == 0


# ---------------------------------------------------------------------------
# MAS (complex, phased double-quantum) couplings
# ---------------------------------------------------------------------------

_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _pauli_string(num_qubits: int, paulis: dict[int, np.ndarray]) -> np.ndarray:
    """Kronecker product with qubit 0 as the most significant factor (cirq's convention)."""
    out = np.eye(1, dtype=complex)
    for k in range(num_qubits):
        out = np.kron(out, paulis.get(k, np.eye(2, dtype=complex)))
    return out


def _mas_hamiltonian_matrix(hom: np.ndarray, het: np.ndarray) -> np.ndarray:
    """H = sum_{i<j} [Re d (Y_i Y_j - X_i X_j) + Im d (X_i Y_j + Y_i X_j) + c Z_i Z_j]."""
    n = hom.shape[0]
    x, y, z = _PAULI_X, _PAULI_Y, _PAULI_Z
    ham = np.zeros((2**n, 2**n), dtype=complex)
    for i in range(n):
        for j in range(i + 1, n):
            d = complex(hom[i, j])
            ham += d.real * (_pauli_string(n, {i: y, j: y}) - _pauli_string(n, {i: x, j: x}))
            ham += d.imag * (_pauli_string(n, {i: x, j: y}) + _pauli_string(n, {i: y, j: x}))
            ham += float(het[i, j]) * _pauli_string(n, {i: z, j: z})
    return ham


def _evolve(ham: np.ndarray, t: float) -> np.ndarray:
    """Returns exp(-i H t) for a Hermitian H."""
    evals, evecs = np.linalg.eigh(ham)
    return (evecs * np.exp(-1j * evals * t)) @ evecs.conj().T


def _overlap(u: np.ndarray, v: np.ndarray) -> float:
    """|tr(U^dag V)| / dim, which equals one iff U and V agree up to a global phase."""
    return float(abs(np.trace(u.conj().T @ v)) / u.shape[0])


def _random_mas_matrix(num_spins: int, seed: int, num_system_spins: int = 2) -> np.ndarray:
    """Complex symmetric couplings with complex DQ blocks and a real system-environment block."""
    rng = np.random.default_rng(seed)
    shape = (num_spins, num_spins)
    full = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    full = (full + full.T) / 2
    np.fill_diagonal(full, 0.0)
    s = num_system_spins
    full[:s, s:] = full[:s, s:].real
    full[s:, :s] = full[s:, :s].real
    return full


def test_make_mas_experiment_rejects_complex_heteronuclear() -> None:
    """Z_p Z_i couplings are Hermitian, so complex system-environment entries are rejected."""
    full = _random_mas_matrix(3, seed=3)
    full[0, 2] = full[2, 0] = 0.3 + 0.2j
    with pytest.raises(ValueError, match="must be real"):
        make_mas_experiment_fixed_num_spins(3, np.arange(3), full)


def test_make_mas_experiment_rejects_non_symmetric() -> None:
    """The stored MAS couplings satisfy d_ij = d_ji."""
    full = _random_mas_matrix(3, seed=4)
    full[0, 1] += 0.1j
    with pytest.raises(ValueError, match="symmetric"):
        make_mas_experiment_fixed_num_spins(3, np.arange(3), full)


def test_make_mas_experiment_too_few_spins() -> None:
    """The system-environment partition needs at least num_system_spins spins."""
    with pytest.raises(ValueError, match="must be at least num_system_spins"):
        make_mas_experiment_fixed_num_spins(1, np.arange(2), np.zeros((2, 2), dtype=complex))


@pytest.mark.parametrize("seed", range(5))
def test_phased_dq_pair_matches_exact_evolution(seed: int) -> None:
    """One step on a single pair is exactly exp(-i t [Re d (YY - XX) + Im d (XY + YX)])."""
    rng = np.random.default_rng(seed)
    d = complex(rng.standard_normal(), rng.standard_normal())
    t = float(rng.uniform(0.05, 1.0))
    hom = np.array([[0.0, d], [d, 0.0]])
    experiment = make_mas_experiment_fixed_num_spins(2, np.arange(2), hom)
    qubits = cirq.LineQubit.range(2)

    forward, _ = make_phased_dq_components(experiment, t, 1, qubits)

    exact = _evolve(_mas_hamiltonian_matrix(hom, np.zeros((2, 2))), t)
    assert np.isclose(_overlap(forward.unitary(qubit_order=qubits), exact), 1.0, atol=1e-10)


def test_phased_dq_components_follow_exp_minus_iht_convention() -> None:
    """The dense MAS circuit implements exp(-iHt), not exp(+iHt) or exp(-iH*t).

    Haar-averaged OTOCs are unchanged under H -> -H and H -> H*, so this sign and phase
    convention can only be checked on the unitary itself.
    """
    num_spins, t, num_steps = 4, 0.3, 100
    full = _random_mas_matrix(num_spins, seed=7)
    experiment = make_mas_experiment_fixed_num_spins(num_spins, np.arange(num_spins), full)
    qubits = cirq.LineQubit.range(num_spins)

    forward, _ = make_otoc_components2(experiment, t, num_steps, qubits)

    unitary = forward.unitary(qubit_order=qubits)
    ham = _mas_hamiltonian_matrix(
        experiment.homogeneous_coupling_mat, experiment.heterogeneous_coupling_mat
    )
    assert _overlap(unitary, _evolve(ham, t)) > 1 - 1e-4
    assert _overlap(unitary, _evolve(-ham, t)) < 0.9
    assert _overlap(unitary, _evolve(ham.conj(), t)) < 0.9


@pytest.mark.parametrize("num_system_spins", [2, 3])
def test_dense_builder_matches_exact_evolution_for_real_couplings(num_system_spins: int) -> None:
    """The dense forward circuit approximates exp(-iHt) with every coupling of the partition.

    With ``num_system_spins=3`` the system-system couplings beyond the (measurement, butterfly)
    pair and the couplings of spin 2 to the bath must all be applied.
    """
    num_spins, t, num_steps = 4, 0.3, 100
    full = _hamiltonian_random(num_spins, seed=11)
    experiment = make_experiment_fixed_num_spins(
        num_spins, np.arange(num_spins), full, num_system_spins=num_system_spins
    )
    qubits = cirq.LineQubit.range(num_spins)

    forward, _ = make_otoc_components2(experiment, t, num_steps, qubits)

    ham = _mas_hamiltonian_matrix(
        experiment.homogeneous_coupling_mat, experiment.heterogeneous_coupling_mat
    )
    assert _overlap(forward.unitary(qubit_order=qubits), _evolve(ham, t)) > 1 - 1e-4


def test_dense_builder_gate_sequence_for_default_partition() -> None:
    """Frozen gate sequence of one dense Trotter step for the default (2 + bath) partition."""
    num_spins, dt = 4, 0.05
    full = _hamiltonian_random(num_spins, seed=3)
    experiment = make_experiment_fixed_num_spins(num_spins, np.arange(num_spins), full)
    q = cirq.LineQubit.range(num_spins)

    step, butterfly = make_otoc_components2(experiment, dt, 1, q)

    def alpha(coupling: float, sign: int = 1) -> float:
        return 2 * sign * (coupling * dt) / np.pi

    expected = cirq.Circuit(
        cirq.Moment(
            cirq.XX(q[0], q[1]) ** alpha(full[0, 1], -1),
            cirq.XX(q[2], q[3]) ** alpha(full[2, 3], -1),
        ),
        cirq.Moment(
            cirq.YY(q[0], q[1]) ** alpha(full[0, 1]),
            cirq.YY(q[2], q[3]) ** alpha(full[2, 3]),
        ),
        cirq.Moment(cirq.ZZ(q[0], q[2]) ** alpha(full[0, 2])),
        cirq.Moment(
            cirq.ZZ(q[0], q[3]) ** alpha(full[0, 3]),
            cirq.ZZ(q[1], q[2]) ** alpha(full[1, 2]),
        ),
        cirq.Moment(cirq.ZZ(q[1], q[3]) ** alpha(full[1, 3])),
    )
    assert step == expected
    assert butterfly == cirq.Circuit(cirq.X(q[1]))
    # n steps are the step repeated n times (up to the rounding of (n * dt) / n).
    assert cirq.approx_eq(make_otoc_components2(experiment, 3 * dt, 3, q)[0], step * 3, atol=1e-12)


def test_dense_builder_complex_dtype_with_real_values_matches_real_builder() -> None:
    """A complex dtype with zero imaginary parts needs no rz gates and gives the same circuit."""
    num_spins = 5
    real_ham = _hamiltonian_random(num_spins)
    indices = np.arange(num_spins)
    exp_real = make_experiment_fixed_num_spins(num_spins, indices, real_ham)
    exp_complex = make_mas_experiment_fixed_num_spins(
        num_spins, indices, real_ham.astype(np.complex128)
    )
    qubits = cirq.LineQubit.range(num_spins)

    forward_real, butterfly_real = make_otoc_components2(exp_real, 0.2, 4, qubits)
    forward_complex, butterfly_complex = make_otoc_components2(exp_complex, 0.2, 4, qubits)

    assert np.iscomplexobj(exp_complex.homogeneous_coupling_mat)
    assert forward_complex == forward_real
    assert butterfly_complex == butterfly_real


def test_dense_and_swap_network_agree_for_three_system_spins() -> None:
    """Both builders apply the full partition when num_system_spins > 2."""
    num_spins = 5
    full = _hamiltonian_random(num_spins, seed=5)
    experiment = make_experiment_fixed_num_spins(
        num_spins, np.arange(num_spins), full, num_system_spins=3
    )
    val1, val2 = _run_circuit_comparison(experiment, num_spins, 0.2, 64)
    assert np.isclose(val1, val2, atol=1e-3)


def test_dense_builder_rejects_asymmetric_couplings() -> None:
    """Asymmetric coupling matrices raise a ValueError instead of an assertion error."""
    full = _hamiltonian_random(3, seed=2)
    experiment = make_experiment_fixed_num_spins(3, np.arange(3), full)
    experiment.homogeneous_coupling_mat[0, 1] += 0.5
    with pytest.raises(ValueError, match="symmetric"):
        make_otoc_components2(experiment, 0.1, 2, cirq.LineQubit.range(3))


def test_swap_network_builder_rejects_complex_couplings() -> None:
    """The swap network cannot represent phased DQ couplings and says so."""
    experiment = make_mas_experiment_fixed_num_spins(4, np.arange(4), _random_mas_matrix(4, seed=8))
    with pytest.raises(NotImplementedError, match="circuit_type='dense'"):
        make_otoc_components(experiment, 0.1, 2, cirq.LineQubit.range(4))


def test_mas_dense_circuit_qsim_matches_cirq() -> None:
    """The qsim simulator evaluates the rz-wrapped DQ gates exactly like cirq's reference one."""
    num_spins = 4
    full = _random_mas_matrix(num_spins, seed=9)
    experiment = make_mas_experiment_fixed_num_spins(num_spins, np.arange(num_spins), full)
    qubits = cirq.LineQubit.range(num_spins)
    circuit = make_otoc_circuit2(experiment, 0.3, 6, qubits)
    rng = np.random.default_rng(0)
    psi = np.kron(np.array((1, 0)), generate_haar_state(num_spins - 1, rng)).astype(np.complex64)
    observables = [cirq.Z(qubits[0])]

    val_qsim = qsimcirq.QSimSimulator().simulate_expectation_values(
        circuit, observables=observables, initial_state=psi, qubit_order=qubits
    )[0]
    val_cirq = cirq.Simulator().simulate_expectation_values(
        circuit, observables=observables, initial_state=psi, qubit_order=qubits
    )[0]

    assert np.isclose(val_qsim, val_cirq, atol=1e-4)


def test_run_parallel_simulation_mas_dense_end_to_end(tmp_path: Path) -> None:
    """Complex MAS instances run end to end with the dense (cached, incremental) builder."""
    num_spins, n_avg = 4, 2
    for iavg in range(n_avg):
        np.save(tmp_path / f"dipolar_matrix{iavg}.npy", _random_mas_matrix(num_spins, 10 + iavg))
        np.save(tmp_path / f"ordered_indices{iavg}.npy", np.arange(num_spins))
    sim_params = OTOCSimulation(
        num_spins=num_spins,
        t_range=(0.1, 0.2),
        n_trotter=2,
        n_haar=2,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
        circuit_type=CircuitType.DENSE,
        eta_hom=0.5,
        eta_het=0.5,
    )

    results = np.array(run_parallel_simulation(sim_params, tmp_path, "test_mas", n_avg=n_avg))

    assert results.shape == (n_avg, len(sim_params.t_range))
    assert np.all(np.isfinite(results))
    assert np.all(np.abs(results) <= 1.0 + 1e-5)


# ---------------------------------------------------------------------------
# Parameter validation, time grid, and measurement assumptions
# ---------------------------------------------------------------------------

_VALID_SIM_KWARGS = {
    "num_spins": 2,
    "t_range": (0.5, 1.0),
    "n_trotter": 1,
    "n_haar": 1,
    "n_thread": 1,
    "n_proc": 1,
    "save_circuit": False,
}


@pytest.mark.parametrize(
    ("field", "value"),
    [("n_trotter", 0), ("n_haar", 0), ("n_thread", 0), ("n_proc", 0), ("t_range", ())],
)
def test_otoc_simulation_rejects_invalid_parameters(field: str, value: object) -> None:
    """Invalid counts and an empty time grid are rejected at construction."""
    with pytest.raises(ValueError, match=field):
        OTOCSimulation(**{**_VALID_SIM_KWARGS, field: value})


def test_otoc_simulation_accepts_negative_n_proc() -> None:
    """Negative n_proc follows joblib (-1 = all cores)."""
    assert OTOCSimulation(**{**_VALID_SIM_KWARGS, "n_proc": -1}).n_proc == -1


def test_otoc_simulation_from_dict_does_not_mutate_input() -> None:
    """from_dict converts t_range to a tuple without modifying the caller's dictionary."""
    d = OTOCSimulation(**_VALID_SIM_KWARGS).to_dict()
    d["t_range"] = list(d["t_range"])
    original = copy.deepcopy(d)

    restored = OTOCSimulation.from_dict(d)

    assert d == original
    assert isinstance(d["t_range"], list)
    assert restored.t_range == (0.5, 1.0)


def test_trotter_step_counts_uniform_grid() -> None:
    """Time index k uses (k + 1) * n_trotter steps on the grid (Δ, 2Δ, ..., KΔ)."""
    assert trotter_step_counts((0.25, 0.5, 0.75), 4) == [4, 8, 12]
    assert trotter_step_counts(np.linspace(0.1, 1.0, 10), 2) == [2 * (k + 1) for k in range(10)]


@pytest.mark.parametrize(
    "t_range", [(), (0.0, 1.0), (-0.5, -1.0), (0.5, 1.5), (0.5, 1.0, 2.0)]
)
def test_trotter_step_counts_rejects_other_grids(t_range: tuple[float, ...]) -> None:
    """Grids that do not start at Δ > 0 or are not proportional to (1, ..., K) are rejected."""
    with pytest.raises(ValueError, match="uniform grid"):
        trotter_step_counts(t_range, 2)


@pytest.mark.parametrize("circuit_type", [CircuitType.SWAP_NETWORK, CircuitType.DENSE])
def test_simulators_reject_non_uniform_time_grid(tmp_path: Path, circuit_type: CircuitType) -> None:
    """Both simulators refuse a time grid their step counts do not describe."""
    sim_params = OTOCSimulation(
        **{**_VALID_SIM_KWARGS, "t_range": (0.0, 1.0), "circuit_type": circuit_type}
    )
    experiment = make_experiment_fixed_num_spins(2, np.arange(2), _hamiltonian_all_ones(2))
    simulate = run_simulation if circuit_type == CircuitType.SWAP_NETWORK else run_simulation_cached
    with pytest.raises(ValueError, match="uniform grid"):
        simulate(experiment, sim_params, tmp_path, 0, "test")


def test_run_simulation_cached_rejects_swap_network(tmp_path: Path) -> None:
    """Swap-network steps permute the qubit layout, so the cached path is dense-only."""
    sim_params = OTOCSimulation(**{**_VALID_SIM_KWARGS, "circuit_type": CircuitType.SWAP_NETWORK})
    experiment = make_experiment_fixed_num_spins(2, np.arange(2), _hamiltonian_all_ones(2))
    with pytest.raises(ValueError, match=r"supports only CircuitType\.DENSE"):
        run_simulation_cached(experiment, sim_params, tmp_path, 0, "test")


def test_cached_simulation_saves_the_directly_built_dense_circuit(tmp_path: Path) -> None:
    """The step-repeated circuits of the cached path equal the directly built dense circuits."""
    num_spins, n_trotter = 3, 2
    experiment = make_experiment_fixed_num_spins(
        num_spins, np.arange(num_spins), _hamiltonian_random(num_spins, seed=1)
    )
    sim_params = OTOCSimulation(
        **{
            **_VALID_SIM_KWARGS,
            "num_spins": num_spins,
            "t_range": (0.25, 0.5, 0.75),
            "n_trotter": n_trotter,
            "save_circuit": True,
            "circuit_type": CircuitType.DENSE,
        }
    )

    otoc = run_simulation_cached(experiment, sim_params, tmp_path, 0, "test")

    assert len(otoc) == len(sim_params.t_range)
    qubits = cirq.LineQubit.range(num_spins)
    step_counts = trotter_step_counts(sim_params.t_range, n_trotter)
    for step_id, (t, num_steps) in enumerate(zip(sim_params.t_range, step_counts, strict=True)):
        saved = cirq.read_json(tmp_path / f"circ0_N{num_spins}_t{n_trotter}_{step_id}.json")
        assert saved == make_otoc_circuit2(experiment, t, num_steps, qubits)


def _experiment_measuring(msmt_id: int, msmt_gate: cirq.Gate) -> NMROTOCExperiment:
    """A 2-spin experiment with the given measurement spin and gate."""
    ham = _hamiltonian_all_ones(2)
    return NMROTOCExperiment(
        bfly_ids=[1 - msmt_id],
        msmt_id=msmt_id,
        msmt_gate=msmt_gate,
        bfly_gate=cirq.X,
        homogeneous_coupling_mat=ham,
        heterogeneous_coupling_mat=np.zeros_like(ham),
    )


@pytest.mark.parametrize(("msmt_id", "msmt_gate"), [(1, cirq.X), (0, cirq.Y)])
@pytest.mark.parametrize("builder", [make_otoc_circuit, make_otoc_circuit2])
def test_otoc_circuits_reject_unsupported_measurement(
    builder: Callable[..., cirq.Circuit], msmt_id: int, msmt_gate: cirq.Gate
) -> None:
    """The OTOC circuits hard-code an X measurement of spin 0 and reject anything else."""
    experiment = _experiment_measuring(msmt_id, msmt_gate)
    with pytest.raises(ValueError, match="Only an X measurement of spin 0"):
        builder(experiment, 0.1, 2, cirq.LineQubit.range(2))


@pytest.mark.parametrize("circuit_type", [CircuitType.SWAP_NETWORK, CircuitType.DENSE])
def test_simulators_reject_unsupported_measurement(
    tmp_path: Path, circuit_type: CircuitType
) -> None:
    """The simulators check the measurement assumptions before simulating."""
    sim_params = OTOCSimulation(**{**_VALID_SIM_KWARGS, "circuit_type": circuit_type})
    experiment = _experiment_measuring(1, cirq.X)
    simulate = run_simulation if circuit_type == CircuitType.SWAP_NETWORK else run_simulation_cached
    with pytest.raises(ValueError, match="Only an X measurement of spin 0"):
        simulate(experiment, sim_params, tmp_path, 0, "test")
