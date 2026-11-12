# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the run_core_from_config CLI."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from nmr_otoc_tools.run_core_from_config import main as run_core_main
from nmr_otoc_tools.setup_simulation import main as setup_main

_CORE_HAMILTONIAN = np.array(
    [
        [0.0, 1.0, 0.2, 0.1],
        [1.0, 0.0, 0.3, 0.15],
        [0.2, 0.3, 0.0, 0.25],
        [0.1, 0.15, 0.25, 0.0],
    ]
)


@pytest.fixture
def core_config(tmp_path: Path) -> Path:
    """Writes a 4-spin instance and returns the sim_config_N4.json that setup creates for it."""
    pdb_id = "test_pdb_core"
    data_dir = tmp_path / "data" / "generated_instances" / pdb_id
    data_dir.mkdir(parents=True)
    np.save(data_dir / "dipolar_matrix0.npy", _CORE_HAMILTONIAN)
    np.save(data_dir / "ordered_indices0.npy", np.arange(4))

    setup_args = [
        "nmr-setup-simulation",
        "--pdb_id",
        pdb_id,
        "--n_avg",
        "1",
        "--n_min",
        "4",
        "--n_max",
        "4",
        "--n_trotter",
        "1",
        "--n_haar",
        "1",
        "--n_time",
        "1",
    ]
    with patch.object(sys, "argv", setup_args):
        setup_main()

    config_file = data_dir / "sim_config_N4.json"
    assert config_file.exists()
    return config_file


def test_run_core_from_config_workflow(core_config: Path) -> None:
    """Test that run_core_from_config runs the simulations and generates expected output files."""
    data_dir = core_config.parent

    # Run simulation from config CLI with n_core = 2
    # This should run 4 - 2 = 2 simulations of 2 + 1 = 3 spins
    # Environment spins: k=2 and k=3
    run_args = [
        "nmr-run-core-from-config",
        "--config",
        str(core_config),
        "--n-core",
        "2",
    ]

    with patch.object(sys, "argv", run_args):
        run_core_main()

    # Verify both files are created
    res_file_env2 = data_dir / "otoc_N3_core2_env2_t1_h1.json"
    res_file_env3 = data_dir / "otoc_N3_core2_env3_t1_h1.json"

    assert res_file_env2.exists()
    assert res_file_env3.exists()

    expected_n = 3
    expected_n_core = 2
    expected_env_spin2 = 2
    expected_env_spin3 = 3
    expected_otoc_len = 1

    # Verify content of env2
    with res_file_env2.open() as f:
        res2 = json.load(f)
    assert res2["N"] == expected_n
    assert res2["n_core"] == expected_n_core
    assert res2["env_spin"] == expected_env_spin2
    assert len(res2["otocs"]) == expected_otoc_len

    # Verify content of env3
    with res_file_env3.open() as f:
        res3 = json.load(f)
    assert res3["N"] == expected_n
    assert res3["n_core"] == expected_n_core
    assert res3["env_spin"] == expected_env_spin3
    assert len(res3["otocs"]) == expected_otoc_len


@pytest.mark.parametrize(
    "n_core",
    [
        "4",  # n-core must be < n
        "1",  # min_required_core is 2
    ],
)
def test_run_core_from_config_validation(core_config: Path, n_core: str) -> None:
    """Test that validation fails when n-core is invalid."""
    run_args = [
        "nmr-run-core-from-config",
        "--config",
        str(core_config),
        "--n-core",
        n_core,
    ]

    with (
        patch.object(sys, "argv", run_args),
        pytest.raises(SystemExit),
    ):
        run_core_main()
