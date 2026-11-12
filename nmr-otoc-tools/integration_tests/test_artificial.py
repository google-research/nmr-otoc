# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Integration tests for artificial simulations."""

import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

logger = logging.getLogger(__name__)

ARTIFICIAL_DIR = Path(__file__).resolve().parent / "artificial"
MAE_THRESHOLD = 0.05


def discover_test_cases() -> list[Any]:
    """Discover simulation test cases from config files."""
    test_cases = []
    random_dir = ARTIFICIAL_DIR / "random"
    for config_file in random_dir.glob("sim_config_N*.json"):
        match = re.fullmatch(r"sim_config_N(\d+)\.json", config_file.name)
        if not match:
            continue
        num_spins = int(match.group(1))

        # Expected reference file in the same directory
        ref_file = config_file.parent / f"otoc_N{num_spins}_ref.json"

        case_id = f"random-N{num_spins}"
        test_cases.append(
            pytest.param(config_file, ref_file, num_spins, id=case_id)
        )
    return test_cases


@pytest.mark.parametrize(
    ("config_path", "ref_path", "num_spins"),
    discover_test_cases(),
)
def test_artificial_simulation(
    config_path: Path,
    ref_path: Path,
    num_spins: int,
    tmp_path: Path,
) -> None:
    """Verify simulation output matches reference ground truth OTOC data."""
    # Validate configuration file exists
    assert config_path.exists(), f"Configuration file not found: {config_path}"

    # Handle missing reference files by skipping execution
    if not ref_path.exists():
        pytest.skip(f"Reference file not found: {ref_path}")

    # Load configuration and reference output
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    with ref_path.open("r", encoding="utf-8") as f:
        reference = json.load(f)

    # Extract simulation parameters
    sim_params_dict = config["sim_params"].copy()
    n_trotter = sim_params_dict["n_trotter"]
    n_haar = sim_params_dict["n_haar"]
    t_range = sim_params_dict["t_range"]

    # Validate simulation parameters match filename expectations
    assert sim_params_dict["num_spins"] == num_spins, (
        f"num_spins mismatch: config has {sim_params_dict['num_spins']} "
        f"but filename indicates {num_spins}"
    )

    # Verify time ranges match reference
    assert len(t_range) == len(reference["times"]), (
        f"Time range length mismatch: config has {len(t_range)} "
        f"vs reference has {len(reference['times'])}"
    )
    assert np.allclose(t_range, reference["times"]), (
        f"Time range mismatch between config ({config_path}) and reference ({ref_path})"
    )

    # Limit resources for testing
    sim_params_dict["n_proc"] = 3
    sim_params_dict["n_thread"] = 1

    # Create a temporary config file with modified parameters and sub_folder
    test_config = config.copy()
    test_config["sim_params"] = sim_params_dict
    test_config["sub_folder"] = str(tmp_path)

    tmp_config_path = tmp_path / "temp_config.json"
    with tmp_config_path.open("w", encoding="utf-8") as f:
        json.dump(test_config, f)

    # Call the run_from_config module directly with sys.executable
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nmr_otoc_tools.run_from_config",
            "--config",
            str(tmp_config_path),
            "--verbose",
        ],
        check=True,
    )

    # Load the computed results from the generated output file
    res_file = tmp_path / f"otoc_N{num_spins}_t{n_trotter}_h{n_haar}.json"
    assert res_file.exists(), f"Output file was not generated: {res_file}"

    with res_file.open("r", encoding="utf-8") as f:
        computed_result = json.load(f)

    computed_otocs = computed_result["otocs"]

    computed_arr = np.array(computed_otocs)
    ref_arr = np.array(reference["otocs"])

    # Assert output dimensions match
    assert computed_arr.shape == ref_arr.shape, (
        f"Shape mismatch: computed shape {computed_arr.shape} vs reference shape {ref_arr.shape}"
    )

    # Calculate error metrics
    abs_errors = np.abs(computed_arr - ref_arr)
    mae = np.mean(abs_errors)
    max_ae = np.max(abs_errors)

    # Log the comparison
    logger.info(
        "Validation for %s:\n"
        "  Computed: %s\n"
        "  Reference: %s\n"
        "  Mean Absolute Error: %.5f\n"
        "  Max Absolute Error: %.5f",
        ref_path.relative_to(ARTIFICIAL_DIR.parent),
        computed_arr.tolist(),
        ref_arr.tolist(),
        mae,
        max_ae,
    )

    # Assert tolerance checks
    assert mae <= MAE_THRESHOLD, (
        f"Mean Absolute Error {mae:.5f} exceeds threshold of {MAE_THRESHOLD}"
    )

