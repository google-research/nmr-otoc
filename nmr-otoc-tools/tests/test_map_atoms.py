# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for nmr-map-atoms CLI tool."""

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from nmr_otoc_tools.generate_instances import (
    generate_spin_system,
    get_dipolar_hamiltonian_hbulk_cc,
    parse_structure,
    save_mas_grid,
)
from nmr_otoc_tools.map_atoms import instance_file_id, main
from nmr_otoc_tools.mas import MASRecouplingModel
from nmr_otoc_tools.orientations import so3_grid


def setup_test_files(
    tmp_path: Path,
    saved_indices: list[int] | np.ndarray,
    field_direction: list[float] | np.ndarray | None = None,
    dipolar_matrix: np.ndarray | None = None,
    metadata: dict | None = None,
) -> Path:
    """Helper to set up test files in a temporary directory."""
    indices_file = tmp_path / "ordered_indices0.npy"
    np.save(indices_file, np.array(saved_indices))

    if metadata is None:
        metadata = {
            "indices": [0, 1],
            "config": {"ball_ratio": 10.0},
            "coupling": "dipolar",
        }
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")

    if field_direction is not None:
        np.save(tmp_path / "field_direction0.npy", np.array(field_direction))

    if dipolar_matrix is not None:
        np.save(tmp_path / "dipolar_matrix0.npy", dipolar_matrix)

    return indices_file


def test_unpruned_mapping(tmp_path: Path, sample_pdb_file: Path) -> None:
    """Test mapping atoms when no pruning has occurred."""
    indices_file = setup_test_files(tmp_path, saved_indices=[0, 1, 2])
    output_file = tmp_path / "output.json"

    test_args = [
        "nmr-map-atoms",
        str(indices_file),
        str(sample_pdb_file),
        "-o",
        str(output_file),
    ]

    with patch("sys.argv", test_args):
        main()

    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))

    assert len(data) == 3

    # The first model has CA, C, and H.
    # CA (global index 1, atom CA, element C, serial 2)
    assert data[0]["atom_name"] == "CA"
    assert data[0]["global_index"] == 1
    assert data[0]["element"] == "C"

    # C (global index 2, atom C, element C, serial 3)
    assert data[1]["atom_name"] == "C"
    assert data[1]["global_index"] == 2
    assert data[1]["element"] == "C"

    # H (global index 4, atom H, element H, serial 5)
    assert data[2]["atom_name"] == "H"
    assert data[2]["global_index"] == 4
    assert data[2]["element"] == "H"


def test_pruned_mapping(tmp_path: Path, sample_pdb_file: Path) -> None:
    """Test mapping atoms when pruning occurred."""
    # N = 2, M = 3. Sequential saved_indices implies pruning.
    indices_file = setup_test_files(
        tmp_path,
        saved_indices=[0, 1],
        field_direction=[0.0, 0.0, 1.0],
    )
    output_file = tmp_path / "output.json"

    test_args = [
        "nmr-map-atoms",
        str(indices_file),
        str(sample_pdb_file),
        "-o",
        str(output_file),
    ]

    with patch("sys.argv", test_args):
        main()

    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))

    # Should output N = 2 atoms
    assert len(data) == 2


def test_max_size_slicing(tmp_path: Path, sample_pdb_file: Path) -> None:
    """Test mapping with --max-size slicing."""
    indices_file = setup_test_files(tmp_path, saved_indices=[0, 1, 2])
    output_file = tmp_path / "output.json"

    test_args = [
        "nmr-map-atoms",
        str(indices_file),
        str(sample_pdb_file),
        "-o",
        str(output_file),
        "--max-size",
        "2",
    ]

    with patch("sys.argv", test_args):
        main()

    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))

    # Should slice output to 2 atoms
    assert len(data) == 2
    assert data[0]["atom_name"] == "CA"
    assert data[1]["atom_name"] == "C"


def test_verify_success(tmp_path: Path, sample_pdb_file: Path) -> None:
    """Test verification passes when reconstructed matrix matches saved matrix."""
    # To get the expected dipolar matrix, we can recompute it using the same method.
    structure = parse_structure(sample_pdb_file)
    model = structure[0]
    carbons = [atom for atom in model.get_atoms() if atom.element.upper() == "C"]
    hydrogens = [atom for atom in model.get_atoms() if atom.element.upper() == "H"]
    c_pos = np.array([atom.coord for atom in carbons])
    h_pos = np.array([atom.coord for atom in hydrogens])
    field_direction = np.array([0.0, 0.0, 1.0])

    # Reconstruct unpruned mean Hamiltonian
    ham = get_dipolar_hamiltonian_hbulk_cc(
        c_pos,
        h_pos,
        c_ind1=0,
        c_ind2=1,
        href_ind=0,
        field_ref_frame=field_direction,
        fluorine=False,
    )

    # Reconstruct matrix
    indices_file = setup_test_files(
        tmp_path,
        saved_indices=[0, 1, 2],
        field_direction=field_direction,
        dipolar_matrix=ham,
    )
    output_file = tmp_path / "output.json"

    test_args = [
        "nmr-map-atoms",
        str(indices_file),
        str(sample_pdb_file),
        "-o",
        str(output_file),
        "--verify",
    ]

    with patch("sys.argv", test_args):
        main()  # Should not raise any error

    assert output_file.exists()


def test_verify_failure(tmp_path: Path, sample_pdb_file: Path) -> None:
    """Test verification raises error when reconstructed matrix does not match."""
    # Provide a wrong/mismatched dipolar matrix (all zeros)
    wrong_matrix = np.zeros((3, 3))
    indices_file = setup_test_files(
        tmp_path,
        saved_indices=[0, 1, 2],
        field_direction=[0.0, 0.0, 1.0],
        dipolar_matrix=wrong_matrix,
    )
    output_file = tmp_path / "output.json"

    test_args = [
        "nmr-map-atoms",
        str(indices_file),
        str(sample_pdb_file),
        "-o",
        str(output_file),
        "--verify",
    ]

    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


def test_cif_structure_mapping(tmp_path: Path, sample_cif_file: Path) -> None:
    """Test mapping atoms with CIF structure file."""
    indices_file = setup_test_files(tmp_path, saved_indices=[0, 1, 2])
    output_file = tmp_path / "output.json"

    test_args = [
        "nmr-map-atoms",
        str(indices_file),
        str(sample_cif_file),
        "-o",
        str(output_file),
    ]

    with patch("sys.argv", test_args):
        main()

    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(data) == 3
    assert data[0]["atom_name"] == "CA"
    assert data[1]["atom_name"] == "C"
    assert data[2]["atom_name"] == "H"


def test_fluorine_mapping(tmp_path: Path) -> None:
    """Test mapping atoms using fluorine elements."""
    pdb_content = (
        "ATOM      1  F   GLY A   1      -0.923   1.481  -0.124  1.00  0.00           F\n"
        "ATOM      2  CA  GLY A   1       0.423   1.210  -0.636  1.00  0.00           C\n"
        "ATOM      3  C   GLY A   1       1.403   2.312  -0.134  1.00  0.00           C\n"
        "ATOM      4  F2  GLY A   1       1.033   3.491  -0.124  1.00  0.00           F\n"
        "ATOM      5  H   GLY A   1      -1.623   0.810  -0.536  1.00  0.00           H\n"
    )
    pdb_file = tmp_path / "fluorine_structure.pdb"
    pdb_file.write_text(pdb_content, encoding="utf-8")

    metadata = {
        "indices": [0, 1],
        "config": {"ball_ratio": 10.0},
        "coupling": "dipolar",
        "fluorine": True,
    }
    indices_file = setup_test_files(tmp_path, saved_indices=[0, 1, 2], metadata=metadata)
    output_file = tmp_path / "output.json"

    test_args = [
        "nmr-map-atoms",
        str(indices_file),
        str(pdb_file),
        "-o",
        str(output_file),
    ]

    with patch("sys.argv", test_args):
        main()

    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(data) == 3
    assert data[0]["atom_name"] == "F"
    assert data[0]["element"] == "F"
    assert data[1]["atom_name"] == "F2"
    assert data[1]["element"] == "F"
    assert data[2]["atom_name"] == "H"


def test_stdout_redirection(
    tmp_path: Path,
    sample_pdb_file: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """Test that output is written to stdout if output file is not specified."""
    indices_file = setup_test_files(tmp_path, saved_indices=[0, 1, 2])

    test_args = [
        "nmr-map-atoms",
        str(indices_file),
        str(sample_pdb_file),
    ]

    with patch("sys.argv", test_args):
        main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert len(data) == 3
    assert data[0]["atom_name"] == "CA"


def test_missing_indices_file(tmp_path: Path, sample_pdb_file: Path) -> None:
    """Test CLI exits with status 1 on missing indices file."""
    missing_file = tmp_path / "nonexistent_indices.npy"

    test_args = [
        "nmr-map-atoms",
        str(missing_file),
        str(sample_pdb_file),
    ]

    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


def test_missing_pdb_file(tmp_path: Path) -> None:
    """Test CLI exits with status 1 on missing PDB file."""
    indices_file = setup_test_files(tmp_path, saved_indices=[0, 1])
    missing_pdb = tmp_path / "nonexistent.pdb"

    test_args = [
        "nmr-map-atoms",
        str(indices_file),
        str(missing_pdb),
    ]

    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


def test_invalid_max_size_negative(tmp_path: Path, sample_pdb_file: Path) -> None:
    """Test CLI exits on zero or negative --max-size values."""
    indices_file = setup_test_files(tmp_path, saved_indices=[0, 1, 2])

    for invalid_size in ["0", "-1", "-5"]:
        test_args = [
            "nmr-map-atoms",
            str(indices_file),
            str(sample_pdb_file),
            "--max-size",
            invalid_size,
        ]

        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


def test_out_of_bounds_metadata_indices(tmp_path: Path, sample_pdb_file: Path) -> None:
    """Test CLI exits when metadata indices are out of bounds."""
    metadata = {
        "indices": [0, 5],
        "config": {"ball_ratio": 10.0},
        "coupling": "dipolar",
    }
    indices_file = setup_test_files(tmp_path, saved_indices=[0, 1], metadata=metadata)

    test_args = [
        "nmr-map-atoms",
        str(indices_file),
        str(sample_pdb_file),
    ]

    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


def test_invalid_field_direction_shape(tmp_path: Path, sample_pdb_file: Path) -> None:
    """Test CLI exits when field direction file does not have shape (3,)."""
    # N = 2, M = 3. Sequential saved_indices implies pruning.
    wrong_field = np.array([1.0, 0.0])  # shape (2,)
    indices_file = setup_test_files(
        tmp_path,
        saved_indices=[0, 1],
        field_direction=wrong_field,
    )

    test_args = [
        "nmr-map-atoms",
        str(indices_file),
        str(sample_pdb_file),
    ]

    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


def test_verify_success_non_sequential(tmp_path: Path, sample_pdb_file: Path) -> None:
    """Test verification passes on unpruned system with non-sequential indices."""
    structure = parse_structure(sample_pdb_file)
    model = structure[0]
    carbons = [atom for atom in model.get_atoms() if atom.element.upper() == "C"]
    hydrogens = [atom for atom in model.get_atoms() if atom.element.upper() == "H"]
    c_pos = np.array([atom.coord for atom in carbons])
    h_pos = np.array([atom.coord for atom in hydrogens])
    field_direction = np.array([0.0, 0.0, 1.0])

    # Reconstruct unpruned mean Hamiltonian
    ham = get_dipolar_hamiltonian_hbulk_cc(
        c_pos,
        h_pos,
        c_ind1=0,
        c_ind2=1,
        href_ind=0,
        field_ref_frame=field_direction,
        fluorine=False,
    )

    # Reconstruct matrix
    indices_file = setup_test_files(
        tmp_path,
        saved_indices=[2, 0, 1],
        field_direction=field_direction,
        dipolar_matrix=ham,
    )
    output_file = tmp_path / "output.json"

    test_args = [
        "nmr-map-atoms",
        str(indices_file),
        str(sample_pdb_file),
        "-o",
        str(output_file),
        "--verify",
    ]

    with patch("sys.argv", test_args):
        main()  # Should not raise any error

    assert output_file.exists()


# --------------------------------------------------------------------------------------
# Instances produced by nmr-generate-instances (dipolar and MAS, rotated and unrotated)
# --------------------------------------------------------------------------------------

_STRUCTURE_ATOMS = [
    ("N", "N1", (-1.2, 0.3, 0.0)),
    ("C", "C2", (0.0, 0.0, 0.0)),
    ("C", "C3", (1.5, 0.0, 0.0)),
    ("C", "C4", (2.2, 1.3, 0.0)),
    ("H", "H5", (0.1, 1.0, 0.3)),
    ("H", "H6", (1.4, -1.0, 0.2)),
    ("H", "H7", (-0.5, -0.6, 0.9)),
    ("H", "H8", (2.0, 0.4, -1.0)),
    ("H", "H9", (3.0, 1.9, 0.6)),
    ("H", "H10", (15.0, 15.0, 15.0)),  # too far to be an environment spin
    ("C", "C11", (0.7, 2.5, -0.4)),
    ("H", "H12", (0.9, 3.4, 0.2)),
]
_PAIR = (1, 2)  # global atom indices of C2 and C3
_ENVIRONMENT = {"H5", "H6", "H7", "H8", "H9", "H12"}
_FIELDS = [v / np.linalg.norm(v) for v in np.random.default_rng(3).normal(size=(3, 3))]


@pytest.fixture
def two_model_pdb(tmp_path: Path) -> Path:
    """A two-conformer structure with a carbon pair, six nearby hydrogens and a distant one."""
    lines = []
    for model in (1, 2):
        lines.append(f"MODEL     {model:4d}")
        for serial, (element, name, xyz) in enumerate(_STRUCTURE_ATOMS, start=1):
            shift = 0.0 if model == 1 else 0.05 * np.sin(serial)
            x, y, z = (coord + shift for coord in xyz)
            lines.append(
                f"ATOM  {serial:5d} {name:<4} LIG A{serial:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2}"
            )
        lines.append("ENDMDL")
    lines.append("END")
    path = tmp_path / "two_models.pdb"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _generate_instance(
    folder: Path,
    pdb_file: Path,
    coupling: str,
    *,
    use_unrotated_field: bool,
    max_size: int | None,
) -> Path:
    """Generates instances for _PAIR and writes the metadata.json that the CLI would write."""
    folder.mkdir()
    extra: dict[str, Any] = {}
    if coupling == "mas":
        extra = {
            "orientation_grid": so3_grid(6, 2),
            "mas_model": MASRecouplingModel.single_component(),
        }
    indices, _, _ = generate_spin_system(
        {"ball_ratio": 2},
        pdb_file,
        folder,
        _PAIR,
        _FIELDS,
        coupling_mode=coupling,
        use_unrotated_field=use_unrotated_field,
        max_size=max_size,
        **extra,
    )
    metadata: dict[str, Any] = {
        "indices": list(indices),
        "config": {"ball_ratio": 2},
        "coupling": coupling,
        "fluorine": False,
        "use_unrotated_field": use_unrotated_field,
    }
    if coupling == "mas":
        metadata["mas"] = save_mas_grid(folder, extra["orientation_grid"], extra["mas_model"])
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return folder


def _map_atoms(indices_file: Path, pdb_file: Path, *extra_args: str) -> list[dict[str, Any]]:
    """Runs nmr-map-atoms and returns the mapped atoms."""
    output_file = indices_file.parent / f"map_{indices_file.stem}.json"
    test_args = [
        "nmr-map-atoms",
        str(indices_file),
        str(pdb_file),
        "-o",
        str(output_file),
        *extra_args,
    ]
    with patch("sys.argv", test_args):
        main()
    return json.loads(output_file.read_text(encoding="utf-8"))


@pytest.mark.parametrize("coupling", ["dipolar", "mas"])
@pytest.mark.parametrize("use_unrotated_field", [False, True])
def test_mapping_of_generated_instances(
    tmp_path: Path, two_model_pdb: Path, coupling: str, *, use_unrotated_field: bool
) -> None:
    """Generated instances map back to the right atoms and pass --verify, pruned or not."""
    full = _generate_instance(
        tmp_path / "full",
        two_model_pdb,
        coupling,
        use_unrotated_field=use_unrotated_field,
        max_size=None,
    )
    pruned = _generate_instance(
        tmp_path / "pruned",
        two_model_pdb,
        coupling,
        use_unrotated_field=use_unrotated_field,
        max_size=4,
    )
    for i in (0, 2):
        full_map = _map_atoms(full / f"ordered_indices{i}.npy", two_model_pdb, "--verify")
        pruned_map = _map_atoms(pruned / f"ordered_indices{i}.npy", two_model_pdb, "--verify")

        assert [atom["atom_name"] for atom in full_map[:2]] == ["C2", "C3"]
        assert [atom["global_index"] for atom in full_map[:2]] == list(_PAIR)
        assert {atom["atom_name"] for atom in full_map[2:]} == _ENVIRONMENT
        assert pruned_map == full_map[:4]

        # Instances pruned before original_indices{i}.npy existed: the order is recomputed.
        (pruned / f"original_indices{i}.npy").unlink()
        legacy_map = _map_atoms(pruned / f"ordered_indices{i}.npy", two_model_pdb, "--verify")
        assert legacy_map == pruned_map


def test_mas_orientation_fallback_and_metadata(tmp_path: Path, two_model_pdb: Path) -> None:
    """Without euler_angles.npy the orientation comes from orientation{i}.npy (incl. beta = pi).

    A MAS instance cannot be recomputed without the "mas" block of its metadata.
    """
    folder = _generate_instance(
        tmp_path / "mas", two_model_pdb, "mas", use_unrotated_field=False, max_size=None
    )
    expected = _map_atoms(folder / "ordered_indices0.npy", two_model_pdb)
    (folder / "euler_angles.npy").unlink()
    for i in range(len(so3_grid(6, 2))):
        _map_atoms(folder / f"ordered_indices{i}.npy", two_model_pdb, "--verify")
    assert _map_atoms(folder / "ordered_indices0.npy", two_model_pdb, "--verify") == expected

    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    del metadata["mas"]
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(SystemExit) as exc_info:
        _map_atoms(folder / "ordered_indices0.npy", two_model_pdb, "--verify")
    assert exc_info.value.code == 1


def test_field_file_is_only_read_when_needed(tmp_path: Path, sample_pdb_file: Path) -> None:
    """A missing -f file only matters when the coupling matrix has to be recomputed."""
    indices_file = setup_test_files(tmp_path, saved_indices=[0, 1, 2])
    output_file = tmp_path / "output.json"
    test_args = [
        "nmr-map-atoms",
        str(indices_file),
        str(sample_pdb_file),
        "-o",
        str(output_file),
        "-f",
        str(tmp_path / "missing_field.npy"),
    ]
    with patch("sys.argv", test_args):
        main()
    assert len(json.loads(output_file.read_text(encoding="utf-8"))) == 3

    with patch("sys.argv", [*test_args, "--verify"]), pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ordered_indices12.npy", "12"),
        ("run3_ordered_indices7.npy", "7"),
        ("N4_t80_ordered_indices0.npy", "0"),
        ("indices_5.npy", "5"),
    ],
)
def test_instance_file_id(name: str, expected: str) -> None:
    """The instance number follows "ordered_indices" (or ends the stem of a renamed file)."""
    assert instance_file_id(Path(name)) == expected


def test_instance_file_id_without_digits(caplog: pytest.LogCaptureFixture) -> None:
    """A file name without an instance number falls back to 0 with a warning."""
    with caplog.at_level(logging.WARNING):
        assert instance_file_id(Path("ordered_indices.npy")) == "0"
    assert "assuming 0" in caplog.text
