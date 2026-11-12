# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Pytest fixtures for NMR OTOC Tools test suite."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_spin_data(tmp_path: Path) -> Path:
    """Creates a sample instance folder containing Hamiltonian and indices matrices.

    Returns:
        The path to the generated sub_folder directory.
    """
    pdb_id = "test_pdb"
    data_dir = tmp_path / "data" / "generated_instances" / pdb_id
    data_dir.mkdir(parents=True, exist_ok=True)

    sample_hamiltonian = np.array([[0.0, 1.0], [1.0, 0.0]])
    sample_indices = np.array([0, 1])

    np.save(data_dir / "dipolar_matrix0.npy", sample_hamiltonian)
    np.save(data_dir / "ordered_indices0.npy", sample_indices)
    return data_dir


@pytest.fixture
def sample_pdb_file(tmp_path: Path) -> Path:
    """Creates a minimal valid raw PDB text file.

    Returns:
        The path to the created PDB file.
    """
    pdb_content = (
        "ATOM      1  N   GLY A   1      -0.923   1.481  -0.124  1.00  0.00           N\n"
        "ATOM      2  CA  GLY A   1       0.423   1.210  -0.636  1.00  0.00           C\n"
        "ATOM      3  C   GLY A   1       1.403   2.312  -0.134  1.00  0.00           C\n"
        "ATOM      4  O   GLY A   1       1.033   3.491  -0.124  1.00  0.00           O\n"
        "ATOM      5  H   GLY A   1      -1.623   0.810  -0.536  1.00  0.00           H\n"
    )
    filepath = tmp_path / "test_structure.pdb"
    filepath.write_text(pdb_content, encoding="utf-8")
    return filepath


@pytest.fixture
def sample_cif_file(tmp_path: Path) -> Path:
    """Creates a minimal valid raw MMCIF text file.

    Returns:
        The path to the created CIF file.
    """
    cif_content = (
        "data_test\n"
        "#\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_alt_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.pdbx_PDB_ins_code\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.occupancy\n"
        "_atom_site.B_iso_or_equiv\n"
        "_atom_site.pdbx_formal_charge\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.auth_comp_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_atom_id\n"
        "_atom_site.pdbx_PDB_model_num\n"
        "ATOM   1 N N . GLY A 1 . -0.923 1.481 -0.124 1.0 0.0 . 1 GLY A N 1\n"
        "ATOM   2 C CA . GLY A 1 . 0.423 1.210 -0.636 1.0 0.0 . 1 GLY A CA 1\n"
        "ATOM   3 C C . GLY A 1 . 1.403 2.312 -0.134 1.0 0.0 . 1 GLY A C 1\n"
        "ATOM   4 O O . GLY A 1 . 1.033 3.491 -0.124 1.0 0.0 . 1 GLY A O 1\n"
        "ATOM   5 H H . GLY A 1 . -1.623 0.810 -0.536 1.0 0.0 . 1 GLY A H 1\n"
    )
    filepath = tmp_path / "test_structure.cif"
    filepath.write_text(cif_content, encoding="utf-8")
    return filepath


@pytest.fixture(autouse=True)
def run_in_temp_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sets the current working directory to a temporary directory for every test.

    This prevents tests from generating files in the repository workspace. The package resolves
    working-directory-relative paths at call time, so changing directory is all that is needed.
    """
    monkeypatch.chdir(tmp_path)
