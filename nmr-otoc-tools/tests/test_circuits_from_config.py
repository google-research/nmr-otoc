# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the circuits-from-config CLI."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import cirq

from nmr_otoc_tools.circuits_from_config import main as circuits_main
from nmr_otoc_tools.setup_simulation import main as setup_main


def _setup_config(data_dir: Path) -> Path:
    """Run setup on the sample instance folder to produce a config file.

    Args:
        data_dir: Instance folder from the ``sample_spin_data`` fixture.

    Returns:
        Path to the generated config file.
    """
    setup_args = [
        "nmr-setup-simulation",
        "--pdb_id",
        data_dir.name,
        "--n_avg",
        "1",
        "--n_min",
        "2",
        "--n_max",
        "2",
        "--n_trotter",
        "1",
        "--n_haar",
        "1",
        "--n_time",
        "2",
    ]

    with patch.object(sys, "argv", setup_args):
        setup_main()

    config_file = data_dir / "sim_config_N2.json"
    assert config_file.exists()
    return config_file


def test_circuits_from_config_default_output(sample_spin_data: Path) -> None:
    """Test that circuits are generated in the config's sub_folder by default."""
    config_file = _setup_config(sample_spin_data)

    with config_file.open() as f:
        config = json.load(f)

    expected_dir = Path(config["sub_folder"])

    args = [
        "nmr-circuits-from-config",
        "--config",
        str(config_file),
    ]

    with patch.object(sys, "argv", args):
        circuits_main()

    # 1 experiment x 2 time steps = 2 circuit files
    circuit_files = sorted(expected_dir.glob("circ*.json"))
    assert len(circuit_files) == 2

    # Verify each file is valid cirq JSON
    for cf in circuit_files:
        circuit = cirq.read_json(str(cf))
        assert isinstance(circuit, cirq.Circuit)
        assert len(circuit) > 0


def test_circuits_from_config_custom_output(tmp_path: Path, sample_spin_data: Path) -> None:
    """Test that circuits are generated in a custom output directory."""
    config_file = _setup_config(sample_spin_data)
    output_dir = tmp_path / "my_circuits"

    args = [
        "nmr-circuits-from-config",
        "--config",
        str(config_file),
        "--output_dir",
        str(output_dir),
    ]

    with patch.object(sys, "argv", args):
        circuits_main()

    assert output_dir.exists()
    circuit_files = sorted(output_dir.glob("circ*.json"))
    assert len(circuit_files) == 2

    for cf in circuit_files:
        circuit = cirq.read_json(str(cf))
        assert isinstance(circuit, cirq.Circuit)
