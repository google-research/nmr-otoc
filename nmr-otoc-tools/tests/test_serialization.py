# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for NMR OTOC experiment and simulation serialization."""

import json
from pathlib import Path

import cirq
import numpy as np
import pytest
from nmrotoc.dataclasses import NMROTOCExperiment

from nmr_otoc_tools.serialization import (
    _deserialize_gate,
    _serialize_gate,
    dict_to_experiment,
    experiment_to_dict,
    load_experiment_json,
    load_simulation_config,
    save_experiment_json,
    save_simulation_config,
)
from nmr_otoc_tools.simulation import OTOCSimulation


def _config_inputs() -> tuple[OTOCSimulation, list[NMROTOCExperiment]]:
    """Returns simulation parameters and two small experiments for config round trips."""
    sim = OTOCSimulation(
        num_spins=2,
        t_range=(0.5, 1.0),
        n_trotter=2,
        n_haar=3,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
        circuit_type="dense",
    )
    hom = np.array([[0.0, 1.5], [1.5, 0.0]])
    experiments = [
        NMROTOCExperiment(
            bfly_ids=[1],
            msmt_id=0,
            msmt_gate=cirq.X,
            bfly_gate=cirq.X,
            homogeneous_coupling_mat=hom * scale,
            heterogeneous_coupling_mat=np.zeros((2, 2)),
        )
        for scale in (1.0, 2.0)
    ]
    return sim, experiments


def test_simulation_config_round_trip_relative(tmp_path: Path) -> None:
    """A sub_folder inside the working directory is stored relative to it and resolved back."""
    sim, experiments = _config_inputs()
    sub_folder = tmp_path / "data" / "generated_instances" / "pdb" / "d_5"
    sub_folder.mkdir(parents=True)
    path = sub_folder / "sim_config_N2.json"

    save_simulation_config(
        path,
        sim_params=sim,
        experiments=experiments,
        pdb_id="pdb",
        dist_name="d_5",
        sub_folder=sub_folder,
    )

    raw = json.loads(path.read_text())
    assert list(raw) == ["sim_params", "experiments", "pdb_id", "dist_name", "sub_folder"]
    assert raw["sub_folder"] == "data/generated_instances/pdb/d_5"
    config = load_simulation_config(path)
    assert config.sim_params == sim
    assert config.pdb_id == "pdb"
    assert config.dist_name == "d_5"
    assert config.sub_folder == sub_folder.resolve()
    assert len(config.experiments) == len(experiments)
    np.testing.assert_array_equal(
        config.experiments[1].homogeneous_coupling_mat, experiments[1].homogeneous_coupling_mat
    )


def test_simulation_config_absolute_sub_folder(tmp_path: Path) -> None:
    """A sub_folder outside the working directory is stored as an absolute path."""
    sim, experiments = _config_inputs()
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    path = tmp_path / "config.json"

    save_simulation_config(
        path,
        sim_params=sim,
        experiments=experiments,
        pdb_id="pdb",
        dist_name="outside",
        sub_folder=outside,
    )

    assert json.loads(path.read_text())["sub_folder"] == str(outside)
    assert load_simulation_config(path).sub_folder == outside.resolve()


def test_serialization_deserialization() -> None:
    """Test serializing and deserializing an NMROTOCExperiment to and from a dict."""
    # Create a sample experiment
    hom = np.array([[1.0, 2.0], [2.0, 1.0]])
    het = np.array([[0.5, 1.5], [1.5, 0.5]])
    exp = NMROTOCExperiment(
        bfly_ids=[1],
        msmt_id=0,
        msmt_gate=cirq.X,
        bfly_gate=cirq.X,
        homogeneous_coupling_mat=hom,
        heterogeneous_coupling_mat=het,
    )

    # Serialize to dict
    d = experiment_to_dict(exp)

    # Verify dict content
    assert d["bfly_ids"] == [1]
    assert d["msmt_id"] == 0
    assert d["msmt_gate"] == "X"
    assert d["bfly_gate"] == "X"
    assert d["homogeneous_coupling_mat"] == [[1.0, 2.0], [2.0, 1.0]]
    assert d["heterogeneous_coupling_mat"] == [[0.5, 1.5], [1.5, 0.5]]

    # Deserialize from dict
    exp2 = dict_to_experiment(d)

    # Verify deserialized object
    assert exp2.bfly_ids == exp.bfly_ids
    assert exp2.msmt_id == exp.msmt_id
    assert exp2.msmt_gate == exp.msmt_gate
    assert exp2.bfly_gate == exp.bfly_gate
    np.testing.assert_array_almost_equal(
        exp2.homogeneous_coupling_mat, exp.homogeneous_coupling_mat
    )
    np.testing.assert_array_almost_equal(
        exp2.heterogeneous_coupling_mat, exp.heterogeneous_coupling_mat
    )


def test_file_save_load(tmp_path: Path) -> None:
    """Test saving and loading an NMROTOCExperiment to and from a JSON file."""
    hom = np.array([[1.0, 2.0], [2.0, 1.0]])
    het = np.array([[0.5, 1.5], [1.5, 0.5]])
    exp = NMROTOCExperiment(
        bfly_ids=[1],
        msmt_id=0,
        msmt_gate=cirq.X,
        bfly_gate=cirq.X,
        homogeneous_coupling_mat=hom,
        heterogeneous_coupling_mat=het,
    )

    filepath = tmp_path / "test_exp.json"
    save_experiment_json(exp, filepath)

    assert filepath.exists()

    exp2 = load_experiment_json(filepath)

    assert exp2.bfly_ids == exp.bfly_ids
    assert exp2.msmt_id == exp.msmt_id
    assert exp2.msmt_gate == exp.msmt_gate
    assert exp2.bfly_gate == exp.bfly_gate
    np.testing.assert_array_almost_equal(
        exp2.homogeneous_coupling_mat, exp.homogeneous_coupling_mat
    )
    np.testing.assert_array_almost_equal(
        exp2.heterogeneous_coupling_mat, exp.heterogeneous_coupling_mat
    )


def test_otoc_simulation_serialization() -> None:
    """Test that OTOCSimulation settings round-trip.

    Verifies round-trip through to_dict/from_dict and to_json/from_json.
    """
    sim = OTOCSimulation(
        num_spins=2,
        t_range=(0.0, 1.0, 2.0),
        n_trotter=10,
        n_haar=20,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
    )

    # To dict
    d = sim.to_dict()
    assert d["num_spins"] == 2
    assert d["t_range"] == (0.0, 1.0, 2.0)
    assert d["n_trotter"] == 10

    # To JSON
    json_str = sim.to_json()

    # From JSON
    sim2 = OTOCSimulation.from_json(json_str)
    assert sim2.num_spins == sim.num_spins
    assert sim2.t_range == sim.t_range
    assert sim2.n_trotter == sim.n_trotter
    assert sim2.save_circuit == sim.save_circuit

    # From dict
    sim3 = OTOCSimulation.from_dict(d)
    assert sim3 == sim


def test_serialize_unsupported_gate() -> None:
    """Test that _serialize_gate raises ValueError for unsupported gates."""
    with pytest.raises(ValueError, match="Unsupported gate for simple serialization"):
        _serialize_gate(cirq.H)


def test_deserialize_unsupported_gate_str() -> None:
    """Test that _deserialize_gate raises ValueError for unsupported gate strings."""
    with pytest.raises(ValueError, match="Unsupported gate string"):
        _deserialize_gate("H")


def test_dict_to_experiment_missing_keys() -> None:
    """Test that dict_to_experiment raises ValueError when keys are missing."""
    invalid_dict = {
        "bfly_ids": [1],
        # "msmt_id" is missing
        "msmt_gate": "X",
        "bfly_gate": "X",
        "homogeneous_coupling_mat": [[1.0]],
        "heterogeneous_coupling_mat": [[1.0]],
    }
    with pytest.raises(ValueError, match="Missing required serialization key 'msmt_id'"):
        dict_to_experiment(invalid_dict)


def test_serialization_none_gates() -> None:
    """Test serializing and deserializing an experiment with None gates."""
    hom = np.array([[1.0, 2.0], [2.0, 1.0]])
    het = np.array([[0.5, 1.5], [1.5, 0.5]])
    exp = NMROTOCExperiment(
        bfly_ids=[1],
        msmt_id=0,
        msmt_gate=None,
        bfly_gate=None,
        homogeneous_coupling_mat=hom,
        heterogeneous_coupling_mat=het,
    )

    d = experiment_to_dict(exp)
    assert d["msmt_gate"] is None
    assert d["bfly_gate"] is None

    exp2 = dict_to_experiment(d)
    assert exp2.msmt_gate is None
    assert exp2.bfly_gate is None


def _make_experiment(hom: np.ndarray, het: np.ndarray) -> NMROTOCExperiment:
    """Builds a minimal experiment with the given coupling matrices."""
    return NMROTOCExperiment(
        bfly_ids=[1],
        msmt_id=0,
        msmt_gate=cirq.X,
        bfly_gate=cirq.X,
        homogeneous_coupling_mat=hom,
        heterogeneous_coupling_mat=het,
    )


def test_complex_coupling_round_trip(tmp_path: Path) -> None:
    """Complex (MAS phased-DQ) couplings survive dict, JSON and file serialization."""
    hom = np.array([[0.0, 1.0 + 0.5j], [1.0 + 0.5j, 0.0]])
    het = np.array([[0.0, 0.25], [0.25, 0.0]])
    exp = _make_experiment(hom, het)

    d = experiment_to_dict(exp)
    assert d["homogeneous_coupling_mat"] == {"real": hom.real.tolist(), "imag": hom.imag.tolist()}
    assert d["heterogeneous_coupling_mat"] == het.tolist()

    restored = dict_to_experiment(json.loads(json.dumps(d)))
    assert restored.homogeneous_coupling_mat.dtype == np.complex128
    np.testing.assert_array_equal(restored.homogeneous_coupling_mat, hom)
    assert restored.heterogeneous_coupling_mat.dtype == np.float64
    np.testing.assert_array_equal(restored.heterogeneous_coupling_mat, het)

    filepath = tmp_path / "mas_exp.json"
    save_experiment_json(exp, filepath)
    np.testing.assert_array_equal(load_experiment_json(filepath).homogeneous_coupling_mat, hom)


def test_complex_coupling_missing_part_rejected() -> None:
    """A complex encoding without both parts is rejected with a clear message."""
    d = experiment_to_dict(_make_experiment(np.eye(2) * 1j, np.zeros((2, 2))))
    del d["homogeneous_coupling_mat"]["imag"]
    with pytest.raises(ValueError, match="missing key"):
        dict_to_experiment(d)


def test_real_coupling_format_unchanged() -> None:
    """Real matrices are still written as plain nested lists and read back as float64."""
    hom = np.array([[0.0, 1.5], [1.5, 0.0]])
    het = np.array([[0.0, -0.5], [-0.5, 0.0]])
    d = experiment_to_dict(_make_experiment(hom, het))
    assert d["homogeneous_coupling_mat"] == [[0.0, 1.5], [1.5, 0.0]]
    assert d["heterogeneous_coupling_mat"] == [[0.0, -0.5], [-0.5, 0.0]]
    restored = dict_to_experiment(d)
    assert restored.homogeneous_coupling_mat.dtype == np.float64
    assert restored.heterogeneous_coupling_mat.dtype == np.float64


