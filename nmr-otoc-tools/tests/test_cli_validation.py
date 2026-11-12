# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for CLI argument validation and simulation edge cases."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from nmr_otoc_tools.circuits_from_config import main as circuits_main
from nmr_otoc_tools.cli import add_shared_simulation_args, validate_simulation_args
from nmr_otoc_tools.generate_instances import main as generate_main
from nmr_otoc_tools.run_core_from_config import main as run_core_main
from nmr_otoc_tools.run_from_config import main as run_main
from nmr_otoc_tools.serialization import save_simulation_config
from nmr_otoc_tools.setup_simulation import main as setup_main
from nmr_otoc_tools.simulation import OTOCSimulation, make_experiment_fixed_num_spins


def test_validate_simulation_args_invalid_n() -> None:
    """Test validation errors for simulation parameters."""
    parser = argparse.ArgumentParser()

    # 1. Test n_min > n_max
    args_invalid_bounds = argparse.Namespace(n_min=5, n_max=3)
    with patch.object(parser, "error") as mock_error:
        mock_error.side_effect = SystemExit("Parser error called")
        with pytest.raises(SystemExit):
            validate_simulation_args(args_invalid_bounds, parser)
        mock_error.assert_called_once_with("--n_min (5) cannot be greater than --n_max (3)")

    # 2. Test n_min < 2
    args_too_small = argparse.Namespace(n_min=1, n_max=3)
    with patch.object(parser, "error") as mock_error:
        mock_error.side_effect = SystemExit("Parser error called")
        with pytest.raises(SystemExit):
            validate_simulation_args(args_too_small, parser)
        mock_error.assert_called_once_with("--n_min must be at least 2.")


def test_generate_instances_mismatched_cidx() -> None:
    """Test that generate-instances CLI fails when only one carbon index is provided."""
    test_args = ["nmr-generate-instances", "test_pdb", "1", "--cidx1", "0"]
    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit):
        generate_main()


def test_circuits_from_config_missing_file() -> None:
    """Test that circuits-from-config exits gracefully when config is missing."""
    test_args = ["nmr-circuits-from-config", "--config", "nonexistent_config.json"]
    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit) as exc_info:
        circuits_main()
    assert exc_info.value.code == 1


def test_run_from_config_missing_file() -> None:
    """Test that run-from-config exits gracefully when config is missing."""
    test_args = ["nmr-run-from-config", "--config", "nonexistent_config.json"]
    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit) as exc_info:
        run_main()
    assert exc_info.value.code == 1


def test_setup_simulation_main_missing_experiments() -> None:
    """Test setup_simulation.main when no experiments are created."""
    test_args = [
        "nmr-setup-simulation",
        "--pdb_id",
        "test_pdb",
        "--n_avg",
        "1",
        "--n_min",
        "2",
        "--n_max",
        "2",
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("nmr_otoc_tools.setup_simulation.create_experiment", return_value=None),
        patch("nmr_otoc_tools.setup_simulation.logger") as mock_logger,
    ):
        setup_main()
        mock_logger.warning.assert_called_once_with("No experiments created for N=%d", 2)


def _shared_args(*overrides: str) -> argparse.Namespace:
    """Parses a valid shared-simulation command line, with optional extra flags appended."""
    parser = argparse.ArgumentParser()
    add_shared_simulation_args(parser)
    base = ["--pdb_id", "p", "--n_avg", "1", "--n_min", "2", "--n_max", "2"]
    return parser.parse_args([*base, *overrides])


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--n_avg", "0"),
        ("--n_time", "0"),
        ("--n_trotter", "0"),
        ("--n_haar", "0"),
        ("--n_thread", "0"),
        ("--n_proc", "0"),
        ("--max_time", "0"),
        ("--max_time", "-1.5"),
    ],
)
def test_validate_simulation_args_rejects_non_positive(flag: str, value: str) -> None:
    """Counts must be positive (n_proc non-zero) and max_time must be positive."""
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        validate_simulation_args(_shared_args(flag, value), parser)


def test_validate_simulation_args_accepts_defaults_and_all_cores() -> None:
    """The defaults are valid, and n_proc = -1 (joblib: all cores) is allowed."""
    parser = argparse.ArgumentParser()
    validate_simulation_args(_shared_args(), parser)
    validate_simulation_args(_shared_args("--n_proc", "-1"), parser)


def _write_config(path: Path, sub_folder: Path, num_spins: int = 2) -> None:
    """Writes a minimal one-experiment simulation config."""
    sim = OTOCSimulation(
        num_spins=num_spins,
        t_range=(0.5, 1.0),
        n_trotter=1,
        n_haar=1,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
    )
    ham = np.full((num_spins, num_spins), 0.3)
    np.fill_diagonal(ham, 0.0)
    exp = make_experiment_fixed_num_spins(num_spins, np.arange(num_spins), ham)
    save_simulation_config(
        path,
        sim_params=sim,
        experiments=[exp],
        pdb_id="p",
        dist_name=sub_folder.name,
        sub_folder=sub_folder,
    )


def test_run_from_config_missing_output_dir_exits_before_simulating(tmp_path: Path) -> None:
    """A sub_folder that does not resolve to a directory is reported before simulating."""
    config_file = tmp_path / "config.json"
    _write_config(config_file, tmp_path / "missing")
    test_args = ["nmr-run-from-config", "--config", str(config_file)]
    with (
        patch.object(sys, "argv", test_args),
        patch("nmr_otoc_tools.run_from_config.run_parallel_simulation") as mock_run,
        pytest.raises(SystemExit) as exc_info,
    ):
        run_main()
    assert exc_info.value.code == 1
    mock_run.assert_not_called()


def test_run_from_config_output_dir_override(tmp_path: Path) -> None:
    """--output_dir replaces the config's sub_folder as the result location."""
    config_file = tmp_path / "config.json"
    _write_config(config_file, tmp_path / "missing")
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    test_args = [
        "nmr-run-from-config",
        "--config",
        str(config_file),
        "--output_dir",
        str(output_dir),
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("nmr_otoc_tools.run_from_config.run_parallel_simulation") as mock_run,
    ):
        mock_run.return_value = [[0.9, 0.8]]
        run_main()
    mock_run.assert_called_once()
    assert mock_run.call_args.args[1] == output_dir
    assert (output_dir / "otoc_N2_t1_h1.json").exists()


def test_run_core_from_config_missing_output_dir(tmp_path: Path) -> None:
    """nmr-run-core-from-config also fails fast on a missing output directory."""
    config_file = tmp_path / "config.json"
    _write_config(config_file, tmp_path / "missing", num_spins=3)
    test_args = ["nmr-run-core-from-config", "--config", str(config_file), "--n-core", "2"]
    with (
        patch.object(sys, "argv", test_args),
        patch("nmr_otoc_tools.run_core_from_config.run_parallel_simulation") as mock_run,
        pytest.raises(SystemExit) as exc_info,
    ):
        run_core_main()
    assert exc_info.value.code == 1
    mock_run.assert_not_called()


def test_circuits_from_config_missing_default_output_dir(tmp_path: Path) -> None:
    """Without --output_dir, the config's sub_folder must exist."""
    config_file = tmp_path / "config.json"
    _write_config(config_file, tmp_path / "missing")
    test_args = ["nmr-circuits-from-config", "--config", str(config_file)]
    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit) as exc_info:
        circuits_main()
    assert exc_info.value.code == 1
    assert not (tmp_path / "missing").exists()


def test_setup_simulation_absolute_subfolder(tmp_path: Path) -> None:
    """An absolute --subfolder outside the working directory is stored as an absolute path."""
    outside = tmp_path.parent / f"{tmp_path.name}_instances"
    outside.mkdir()
    np.save(outside / "dipolar_matrix0.npy", np.array([[0.0, 1.0], [1.0, 0.0]]))
    np.save(outside / "ordered_indices0.npy", np.array([0, 1]))
    test_args = [
        "nmr-setup-simulation",
        "--pdb_id",
        "p",
        "--n_avg",
        "1",
        "--n_min",
        "2",
        "--n_max",
        "2",
        "--n_time",
        "2",
        "--subfolder",
        str(outside),
    ]
    with patch.object(sys, "argv", test_args):
        setup_main()
    config_file = outside / "sim_config_N2.json"
    assert json.loads(config_file.read_text())["sub_folder"] == str(outside)
