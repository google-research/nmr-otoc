# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for compute_distances CLI tool."""

import argparse
from pathlib import Path

import numpy as np
import pytest

from nmr_otoc_tools.compute_distances import compute_distance_stats


@pytest.fixture
def multi_frame_pdb_file(tmp_path: Path) -> Path:
    """Creates a sample PDB file with 2 models/frames."""
    pdb_content = (
        "MODEL        1\n"
        "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CB  GLY A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "ENDMDL\n"
        "MODEL        2\n"
        "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CB  GLY A   1       3.000   0.000   0.000  1.00  0.00           C\n"
        "ENDMDL\n"
    )
    filepath = tmp_path / "multi_frame.pdb"
    filepath.write_text(pdb_content, encoding="utf-8")
    return filepath


def test_compute_distance_stats_indices(multi_frame_pdb_file: Path) -> None:
    """Test compute_distance_stats with integer indices."""
    stats = compute_distance_stats(multi_frame_pdb_file, "0", "1")
    assert stats["idx1"] == 0
    assert stats["idx2"] == 1
    assert stats["n_frames"] == 2
    assert np.isclose(stats["min"], 1.0)
    assert np.isclose(stats["max"], 3.0)
    assert np.isclose(stats["mean"], 2.0)
    assert np.isclose(stats["std"], 1.0)


def test_compute_distance_stats_specifiers(multi_frame_pdb_file: Path) -> None:
    """Test compute_distance_stats with string specifiers."""
    # CA is index 0, CB is index 1
    stats = compute_distance_stats(
        multi_frame_pdb_file, "A:GLY:1:CA", "A:GLY:1:CB"
    )
    assert stats["idx1"] == 0
    assert stats["idx2"] == 1
    assert stats["n_frames"] == 2
    assert np.isclose(stats["min"], 1.0)
    assert np.isclose(stats["max"], 3.0)
    assert np.isclose(stats["mean"], 2.0)
    assert np.isclose(stats["std"], 1.0)


def test_compute_distance_stats_mixed(multi_frame_pdb_file: Path) -> None:
    """Test compute_distance_stats with mixed specifiers (int and string)."""
    stats = compute_distance_stats(multi_frame_pdb_file, "0", "A:GLY:1:CB")
    assert stats["idx1"] == 0
    assert stats["idx2"] == 1


def test_compute_distance_stats_out_of_bounds(multi_frame_pdb_file: Path) -> None:
    """Test compute_distance_stats raises ValueError for out of bounds indices."""
    with pytest.raises(ValueError, match="Atom1 index 2 out of bounds"):
        compute_distance_stats(multi_frame_pdb_file, "2", "1")

    with pytest.raises(ValueError, match="Atom2 index 2 out of bounds"):
        compute_distance_stats(multi_frame_pdb_file, "0", "2")

    with pytest.raises(ValueError, match="Atom1 index -1 out of bounds"):
        compute_distance_stats(multi_frame_pdb_file, "-1", "1")


def test_compute_distance_stats_invalid_specifier(multi_frame_pdb_file: Path) -> None:
    """Test compute_distance_stats raises error for invalid specifiers."""
    # This should fail during resolution because it looks like a specifier but is invalid
    with pytest.raises(argparse.ArgumentTypeError, match="Atom specifier must be"):
        compute_distance_stats(multi_frame_pdb_file, "invalid_specifier", "1")

    with pytest.raises(ValueError, match="No atom matching specifier"):
        compute_distance_stats(multi_frame_pdb_file, "A:GLY:1:NONEXIST", "1")
