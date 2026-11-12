# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the split simulation workflow."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from nmr_otoc_tools.run_from_config import main as run_main
from nmr_otoc_tools.run_otoc_simulation import main as orig_main
from nmr_otoc_tools.setup_simulation import main as setup_main


def test_split_workflow(sample_spin_data: Path) -> None:
    """Test that setup and run from config produce expected results."""
    data_dir = sample_spin_data
    pdb_id = data_dir.name

    # 1. Run setup CLI
    setup_args = [
        "nmr-setup-simulation",
        "--pdb_id",
        pdb_id,
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
        "1",
    ]

    with patch.object(sys, "argv", setup_args):
        setup_main()

    config_file = data_dir / "sim_config_N2.json"
    assert config_file.exists()

    # 2. Run simulation from config CLI
    run_args = [
        "nmr-run-from-config",
        "--config",
        str(config_file),
    ]

    with patch.object(sys, "argv", run_args):
        run_main()

    result_file = data_dir / "otoc_N2_t1_h1.json"
    assert result_file.exists()

    with result_file.open() as f:
        result_split = json.load(f)

    assert result_split["N"] == 2
    assert len(result_split["otocs"]) == 1

    # 3. Run original simulation to compare structure
    orig_args = [
        "nmr-run-otoc-simulation",
        "--pdb_id",
        pdb_id,
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
        "1",
    ]

    # Delete result file to ensure it's recreated
    result_file.unlink()

    with patch.object(sys, "argv", orig_args):
        orig_main()

    assert result_file.exists()
    with result_file.open() as f:
        result_orig = json.load(f)

    # Compare structure and metadata
    assert result_orig["N"] == result_split["N"]
    assert result_orig["times"] == result_split["times"]
    assert result_orig["instance"] == result_split["instance"]
    assert result_orig["dist_folder"] == result_split["dist_folder"]
    # Values might differ slightly due to random Haar states, but should be of same type
    assert len(result_orig["otocs"]) == len(result_split["otocs"])
