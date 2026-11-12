# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for generating NMR instance topologies."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from Bio.PDB import PDBParser
from pyscf.dft.LebedevGrid import MakeAngularGrid

from nmr_otoc_tools.generate_instances import (
    AtomSpecifier,
    MASOrientation,
    build_mas_model,
    compute_pair_geometry,
    conformer_pair_geometries,
    extract_spin_coordinates,
    generate_spin_system,
    get_carbon_indices,
    get_dipolar_hamiltonian_hbulk_cc,
    get_fields_and_weights,
    get_instance_group_folder,
    get_mas_hamiltonian_hbulk_cc,
    load_spin_coordinates,
    main,
    mean_coupling_matrix,
    parse_args,
    parse_atom_specifier,
    parse_structure,
    prune_lebedev_grid,
    reference_hydrogen_index,
    resolve_atom_specifier,
    select_environment_hydrogen_indices,
    select_environment_hydrogens,
    spin_order,
)
from nmr_otoc_tools.mas import MASRecouplingModel
from nmr_otoc_tools.orientations import lebedev_directions, rotation_zyz, so3_grid
from nmr_otoc_tools.simulation import OTOCSimulation, create_experiment, run_parallel_simulation


def test_get_dipolar_hamiltonian_hbulk_cc() -> None:
    """Test creating a hamiltonian from geometric representations."""
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_frame.return_value = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        c_pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        h_pos = np.array(
            [
                [0.0, 1.0, 0.0],
            ]
        )
        c_ind1 = 0
        c_ind2 = 1
        href_ind = 0
        field_ref_frame = np.array([0.0, 0.0, 1.0])

        hamiltonian = get_dipolar_hamiltonian_hbulk_cc(
            c_pos=c_pos,
            h_pos=h_pos,
            c_ind1=c_ind1,
            c_ind2=c_ind2,
            href_ind=href_ind,
            field_ref_frame=field_ref_frame,
            fluorine=False,
        )

        expected_shape = (3, 3)
        # 2 Carbons + 1 Hydrogen -> 3x3 matrix
        assert hamiltonian.shape == expected_shape
        # The static dipolar model gives a real matrix.
        assert not np.iscomplexobj(hamiltonian)
        assert not np.isnan(hamiltonian).any()


def test_get_dipolar_hamiltonian_hbulk_cc_unrotated() -> None:
    """Test creating a hamiltonian without rotating the field."""
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        c_pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        h_pos = np.array([[0.0, 1.0, 0.0]])
        c_ind1 = 0
        c_ind2 = 1
        href_ind = 0
        field_ref_frame = np.array([0.0, 0.0, 1.0])

        hamiltonian = get_dipolar_hamiltonian_hbulk_cc(
            c_pos=c_pos,
            h_pos=h_pos,
            c_ind1=c_ind1,
            c_ind2=c_ind2,
            href_ind=href_ind,
            field_ref_frame=field_ref_frame,
            fluorine=False,
            use_unrotated_field=True,
        )

        assert hamiltonian.shape == (3, 3)
        mock_frame.assert_not_called()


def test_get_carbon_indices(sample_pdb_file: Path) -> None:
    """Test acquiring specific element indices from valid PDB structures."""
    c_ind1, c_ind2 = get_carbon_indices(sample_pdb_file, ind1=1, ind2=2, fluorine=False)
    # CA (global index 1) -> carbon index 0
    # C (global index 2) -> carbon index 1
    assert c_ind1 == 0
    assert c_ind2 == 1


@patch("nmr_otoc_tools.generate_instances.load_spin_coordinates")
@patch("nmr_otoc_tools.generate_instances.get_separated_pair")
@patch("nmr_otoc_tools.generate_instances.get_indices_sorted_by_classical_time_approx")
def test_generate_spin_system(
    mock_get_indices: MagicMock,
    mock_get_pair: MagicMock,
    mock_load: MagicMock,
    tmp_path: Path,
) -> None:
    """Test orchestrating spin system generation and indexing."""
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_frame.return_value = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        mock_load.return_value = (
            MagicMock(),
            {
                "C": [np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])],
                "F": [np.array([])],
                "H": [np.array([[0.0, 1.0, 0.0]])],
            },
        )
        mock_get_pair.return_value = (0, 1)
        expected_indices = np.array([0, 1, 2])
        expected_n_min = 3
        mock_get_indices.return_value = (None, expected_indices, expected_n_min)

        config = {
            "distance_min": 1.0,
            "distance_max": 3.0,
            "ball_ratio": 2.0,
        }
        rng = np.random.RandomState(42)
        fields = [np.array([0.0, 0.0, 1.0])]

        indices, n_min, distance = generate_spin_system(
            config=config,
            pdb_path=tmp_path / "placeholder.pdb",
            output_folder=tmp_path,
            idx_generator=rng,
            fields=fields,
            fluorine=False,
        )

        assert indices == (0, 1)
        assert n_min == expected_n_min
        expected_distance = 2.0
        assert np.isclose(distance, expected_distance)

        assert (tmp_path / "field_direction0.npy").exists()
        assert (tmp_path / "dipolar_matrix0.npy").exists()
        assert (tmp_path / "ordered_indices0.npy").exists()


@patch("nmr_otoc_tools.generate_instances.load_spin_coordinates")
@patch("nmr_otoc_tools.generate_instances.get_separated_pair")
def test_generate_spin_system_max_size(
    mock_get_pair: MagicMock,
    mock_load: MagicMock,
    tmp_path: Path,
) -> None:
    """Test generate_spin_system with max_size pruning."""
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_frame.return_value = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        # 2 Carbons, 4 Hydrogens -> total 6 spins
        mock_load.return_value = (
            MagicMock(),
            {
                "C": [np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])],
                "F": [np.array([])],
                "H": [
                    np.array(
                        [
                            [0.0, 1.0, 0.0],
                            [1.0, 1.0, 0.0],
                            [2.0, 1.0, 0.0],
                            [3.0, 1.0, 0.0],
                        ]
                    )
                ],
            },
        )
        mock_get_pair.return_value = (0, 1)

        config = {
            "distance_min": 1.0,
            "distance_max": 3.0,
            "ball_ratio": 10.0,  # large enough to keep all hydrogens
        }
        rng = np.random.RandomState(42)
        fields = [np.array([0.0, 0.0, 1.0])]

        # Case 1: max_size is None (no pruning)
        indices, _, _ = generate_spin_system(
            config=config,
            pdb_path=tmp_path / "placeholder.pdb",
            output_folder=tmp_path,
            idx_generator=rng,
            fields=fields,
            fluorine=False,
            max_size=None,
        )
        assert indices == (0, 1)
        matrix = np.load(tmp_path / "dipolar_matrix0.npy")
        ordered_idx = np.load(tmp_path / "ordered_indices0.npy")
        assert matrix.shape == (6, 6)
        assert ordered_idx.shape == (6,)
        assert not (tmp_path / "original_indices0.npy").exists()

        # Case 2: max_size = 4 (pruned to 4 spins)
        indices, _, _ = generate_spin_system(
            config=config,
            pdb_path=tmp_path / "placeholder.pdb",
            output_folder=tmp_path,
            idx_generator=rng,
            fields=fields,
            fluorine=False,
            max_size=4,
        )
        assert indices == (0, 1)
        matrix_pruned = np.load(tmp_path / "dipolar_matrix0.npy")
        ordered_idx_pruned = np.load(tmp_path / "ordered_indices0.npy")
        assert matrix_pruned.shape == (4, 4)
        assert ordered_idx_pruned.shape == (4,)
        np.testing.assert_array_equal(ordered_idx_pruned, np.array([0, 1, 2, 3]))

        # Check that the pruned matrix corresponds to the first 4 spins of the sorted indices
        # under the unpruned run, and that the kept spins are recorded for nmr-map-atoms.
        expected_matrix = matrix[np.ix_(ordered_idx[:4], ordered_idx[:4])]
        np.testing.assert_allclose(matrix_pruned, expected_matrix)
        original_idx = np.load(tmp_path / "original_indices0.npy")
        np.testing.assert_array_equal(original_idx, ordered_idx[:4])

        # Case 3: regenerating without pruning removes the now stale original_indices file.
        generate_spin_system(
            config=config,
            pdb_path=tmp_path / "placeholder.pdb",
            output_folder=tmp_path,
            idx_generator=rng,
            fields=fields,
            fluorine=False,
        )
        assert not (tmp_path / "original_indices0.npy").exists()


@patch("nmr_otoc_tools.generate_instances.load_spin_coordinates")
@patch("nmr_otoc_tools.generate_instances.get_separated_pair")
@patch("nmr_otoc_tools.generate_instances.get_indices_sorted_by_classical_time_approx")
def test_generate_spin_system_unrotated_field(
    mock_get_indices: MagicMock,
    mock_get_pair: MagicMock,
    mock_load: MagicMock,
    tmp_path: Path,
) -> None:
    """Test generate_spin_system with use_unrotated_field=True."""
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_load.return_value = (
            MagicMock(),
            {
                "C": [np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])],
                "F": [np.array([])],
                "H": [np.array([[0.0, 1.0, 0.0]])],
            },
        )
        mock_get_pair.return_value = (0, 1)
        mock_get_indices.return_value = (None, np.array([0, 1, 2]), 3)

        config = {
            "distance_min": 1.0,
            "distance_max": 3.0,
            "ball_ratio": 2.0,
        }
        rng = np.random.RandomState(42)
        fields = [np.array([0.0, 0.0, 1.0])]

        generate_spin_system(
            config=config,
            pdb_path=tmp_path / "placeholder.pdb",
            output_folder=tmp_path,
            idx_generator=rng,
            fields=fields,
            fluorine=False,
            use_unrotated_field=True,
        )

        mock_frame.assert_not_called()


@patch("nmr_otoc_tools.generate_instances.generate_spin_system")
def test_main_with_max_size(mock_gen_spin: MagicMock) -> None:
    """Test main function with --max-size flag."""
    mock_gen_spin.return_value = ((0, 1), 3, 5.0)

    test_args = [
        "nmr-generate-instances",
        "test_pdb",
        "1",
        "--cidx1",
        "0",
        "--cidx2",
        "1",
        "--output_name",
        "test_output",
        "--max-size",
        "10",
    ]
    with patch.object(sys, "argv", test_args):
        main()

    _args, kwargs = mock_gen_spin.call_args
    expected_max_size = 10
    assert kwargs["max_size"] == expected_max_size


def test_main_invalid_max_size() -> None:
    """Test main function raises SystemExit for invalid --max-size (< 2)."""
    test_args = [
        "nmr-generate-instances",
        "test_pdb",
        "1",
        "--max-size",
        "1",
    ]
    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit):
        main()


@patch("nmr_otoc_tools.generate_instances.generate_spin_system")
def test_main(mock_gen_spin: MagicMock) -> None:
    """Test generating instances as a cli script over multiple subroutines."""
    mock_gen_spin.return_value = ((0, 1), 3, 5.0)

    test_args = [
        "nmr-generate-instances",
        "test_pdb",
        "1",
        "--cidx1",
        "0",
        "--cidx2",
        "1",
        "--output_name",
        "test_output",
    ]
    with patch.object(sys, "argv", test_args):
        main()

    mock_gen_spin.assert_called_once()


def test_get_carbon_indices_invalid_suffix(tmp_path: Path) -> None:
    """Test get_carbon_indices raises ValueError for unknown suffix."""
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("placeholder")
    with pytest.raises(ValueError, match=r"Unrecognized suffix: \.txt"):
        get_carbon_indices(invalid_file, 0, 1, fluorine=False)


def test_get_carbon_indices_failed_parse(tmp_path: Path) -> None:
    """Test get_carbon_indices handles parsing failures."""
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text("placeholder")
    with patch("nmr_otoc_tools.generate_instances.PDBParser") as mock_parser:
        mock_instance = mock_parser.return_value
        mock_instance.get_structure.return_value = None
        with pytest.raises(RuntimeError, match="Failed to parse structure"):
            get_carbon_indices(pdb_file, 0, 1, fluorine=False)


@patch("nmr_otoc_tools.generate_instances.load_spin_coordinates")
@patch("nmr_otoc_tools.generate_instances.get_carbon_indices")
@patch("nmr_otoc_tools.generate_instances.get_indices_sorted_by_classical_time_approx")
def test_generate_spin_system_tuple_generator(
    mock_get_indices: MagicMock,
    mock_get_carbon: MagicMock,
    mock_load: MagicMock,
    tmp_path: Path,
) -> None:
    """Test generate_spin_system with a tuple as idx_generator."""
    mock_load.return_value = (
        MagicMock(),
        {
            "C": [np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])],
            "F": [np.array([])],
            "H": [np.array([[0.0, 1.0, 0.0]])],
        },
    )
    mock_get_carbon.return_value = (0, 1)
    mock_get_indices.return_value = (None, [0, 1, 2], 3)
    config = {"ball_ratio": 2.0, "distance_min": 1.0, "distance_max": 3.0}
    fields = [np.array([0, 0, 1])]

    generate_spin_system(config, Path("placeholder.pdb"), tmp_path, (0, 2), fields)
    mock_get_carbon.assert_called_once()


@patch("nmr_otoc_tools.generate_instances.load_spin_coordinates")
def test_generate_spin_system_unrecognized_generator(
    mock_load: MagicMock, tmp_path: Path
) -> None:
    """Test generate_spin_system raises TypeError for unknown generator type."""
    mock_load.return_value = (
        MagicMock(),
        {
            "C": [np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])],
            "F": [np.array([])],
            "H": [np.array([[0.0, 1.0, 0.0]])],
        },
    )
    config = {"ball_ratio": 2.0}
    with pytest.raises(TypeError, match="Unrecognized generator type"):
        generate_spin_system(config, Path("placeholder.pdb"), tmp_path, "invalid", [])


@patch("nmr_otoc_tools.generate_instances.load_spin_coordinates")
@patch("nmr_otoc_tools.generate_instances.get_separated_pair")
@patch("nmr_otoc_tools.generate_instances.get_indices_sorted_by_classical_time_approx")
def test_generate_spin_system_no_reduced_h(
    mock_get_indices: MagicMock,
    mock_get_pair: MagicMock,
    mock_load: MagicMock,
    tmp_path: Path,
) -> None:
    """Test generate_spin_system fallback when no hydrogens are in ball ratio."""
    # Case where no hydrogens are within ball radius
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_frame.return_value = np.eye(3)
        mock_load.return_value = (
            MagicMock(),
            {
                "C": [np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])],
                "F": [np.array([])],
                "H": [np.array([[100.0, 100.0, 100.0]])],  # Far away
            },
        )
        mock_get_pair.return_value = (0, 1)
        mock_get_indices.return_value = (None, [0, 1, 2], 3)

        config = {"ball_ratio": 0.01, "distance_min": 0.5, "distance_max": 1.5}
        fields = [np.array([0, 0, 1])]

        indices, n_min, distance = generate_spin_system(
            config, Path("placeholder.pdb"), tmp_path, np.random.RandomState(0), fields
        )

    # The closest hydrogen is kept as the only environment spin: 2 system spins + 1 H.
    assert indices == (0, 1)
    assert n_min == 3
    assert np.isclose(distance, 1.0)
    assert np.load(tmp_path / "dipolar_matrix0.npy").shape == (3, 3)
    np.testing.assert_array_equal(np.load(tmp_path / "ordered_indices0.npy"), [0, 1, 2])
    np.testing.assert_array_equal(np.load(tmp_path / "field_direction0.npy"), [0, 0, 1])


@patch("nmr_otoc_tools.generate_instances.load_spin_coordinates")
@patch("nmr_otoc_tools.generate_instances.get_separated_pair")
@patch("nmr_otoc_tools.generate_instances.get_indices_sorted_by_classical_time_approx")
def test_generate_spin_system_averages_conformers_for_each_field(
    mock_get_indices: MagicMock,
    mock_get_pair: MagicMock,
    mock_load: MagicMock,
    tmp_path: Path,
) -> None:
    """Each field direction gets one instance: the coupling matrix averaged over conformers."""
    c_pos_list = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]),
    ]
    h_pos_list = [np.array([[0.0, 1.0, 0.0]]), np.array([[0.0, 1.1, 0.0]])]
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_frame.return_value = np.eye(3)
        mock_load.return_value = (MagicMock(), {"C": c_pos_list, "F": [], "H": h_pos_list})
        mock_get_pair.return_value = (0, 1)
        mock_get_indices.return_value = (None, [0, 1, 2], 3)

        config = {"ball_ratio": 2.0, "distance_min": 0.5, "distance_max": 1.5}
        fields = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])]

        generate_spin_system(
            config, Path("placeholder.pdb"), tmp_path, np.random.RandomState(0), fields
        )

        for i, field in enumerate(fields):
            expected = sum(
                get_dipolar_hamiltonian_hbulk_cc(c_pos, h_pos, 0, 1, 0, field, fluorine=False)
                for c_pos, h_pos in zip(c_pos_list, h_pos_list, strict=True)
            ) / len(c_pos_list)
            np.testing.assert_allclose(np.load(tmp_path / f"dipolar_matrix{i}.npy"), expected)
            np.testing.assert_array_equal(np.load(tmp_path / f"field_direction{i}.npy"), field)
            np.testing.assert_array_equal(np.load(tmp_path / f"ordered_indices{i}.npy"), [0, 1, 2])
    assert not np.allclose(
        np.load(tmp_path / "dipolar_matrix0.npy"), np.load(tmp_path / "dipolar_matrix1.npy")
    )


def test_prune_lebedev_grid() -> None:
    """Test pruning of Lebedev grid."""
    fields = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    pruned_f, pruned_w = prune_lebedev_grid(fields, weights)
    assert len(pruned_f) == 2
    assert len(pruned_w) == 2
    assert np.isclose(np.sum(pruned_w), 1.0)
    # Check that weights are doubled
    assert np.allclose(pruned_w, [0.5, 0.5])
    # Check that for each pruned field, its inversion is not present
    for i in range(len(pruned_f)):
        for j in range(len(pruned_f)):
            assert not np.allclose(pruned_f[i], -pruned_f[j])

    # Test exception when no partner is found
    fields_no_partner = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    weights_no_partner = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="has no inversion-symmetric partner"):
        prune_lebedev_grid(fields_no_partner, weights_no_partner)


def test_get_fields_and_weights_nleb() -> None:
    """Test field and weight generation for Lebedev and random fields."""
    rng = np.random.RandomState(0)
    num_fields = 2

    # Case 1: Random fields (nleb is None)
    fields, weights = get_fields_and_weights(num_fields, rng, nleb=None)
    assert len(fields) == num_fields
    assert len(weights) == num_fields
    assert np.isclose(np.sum(weights), 1.0)

    # Case 2: Lebedev fields (nleb is set, no symmetry)
    nleb = 6
    fields, weights = get_fields_and_weights(num_fields, rng, nleb=nleb, use_symmetry=False)
    assert len(fields) == nleb
    assert len(weights) == nleb
    assert np.isclose(np.sum(weights), 1.0)

    # Case 3: Lebedev fields (nleb is set, with symmetry)
    fields, weights = get_fields_and_weights(num_fields, rng, nleb=nleb, use_symmetry=True)
    assert len(fields) == nleb // 2
    assert len(weights) == nleb // 2
    assert np.isclose(np.sum(weights), 1.0)


@patch("nmr_otoc_tools.generate_instances.generate_spin_system")
def test_main_lebedev(mock_gen_spin: MagicMock) -> None:
    """Test main function with Lebedev grid (instance_id < 0)."""
    mock_gen_spin.return_value = ((0, 1), 3, 5.0)
    test_args = ["nmr-generate-instances", "test_pdb", "-6", "--filename", "test.cif"]
    with patch.object(sys, "argv", test_args):
        main()
    # By default, use_symmetry is False, so fields should be the full Lebedev grid (6 points)
    args, _kwargs = mock_gen_spin.call_args
    assert len(args[4]) == 6


@patch("nmr_otoc_tools.generate_instances.generate_spin_system")
def test_main_use_symmetry(mock_gen_spin: MagicMock) -> None:
    """Test main function with --use-symmetry flag."""
    mock_gen_spin.return_value = ((0, 1), 3, 5.0)
    test_args = [
        "nmr-generate-instances",
        "test_pdb",
        "-6",
        "--filename",
        "test.cif",
        "--use-symmetry",
    ]
    with patch.object(sys, "argv", test_args):
        main()
    # With --use-symmetry, fields should be pruned (3 points)
    args, _kwargs = mock_gen_spin.call_args
    assert len(args[4]) == 3


@patch("nmr_otoc_tools.generate_instances.generate_spin_system")
def test_main_use_unrotated_field(mock_gen_spin: MagicMock) -> None:
    """Test main function with --use-unrotated-field flag."""
    mock_gen_spin.return_value = ((0, 1), 3, 5.0)
    test_args = [
        "nmr-generate-instances",
        "test_pdb",
        "1",
        "--use-unrotated-field",
    ]
    with patch.object(sys, "argv", test_args):
        main()
    _args, kwargs = mock_gen_spin.call_args
    assert kwargs["use_unrotated_field"] is True


@patch("nmr_otoc_tools.generate_instances.generate_spin_system")
def test_main_metadata_exists(mock_gen_spin: MagicMock, tmp_path: Path) -> None:
    """Test main function skips processing if metadata already exists."""
    pdb_id = "test_pdb"
    instance_id = 1
    folder = tmp_path / "data" / "generated_instances" / f"{pdb_id}_{instance_id}" / "d_5"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text("{}")
    test_args = ["nmr-generate-instances", pdb_id, str(instance_id)]
    with patch.object(sys, "argv", test_args):
        main()
    mock_gen_spin.assert_not_called()


def test_get_instance_group_folder() -> None:
    """Test instance group folder path generation."""
    path = get_instance_group_folder("pdb", 1, None)
    assert "pdb_1" in str(path)
    path = get_instance_group_folder("pdb", -6, 6)
    assert "pdb_L6" in str(path)


@patch("nmr_otoc_tools.generate_instances.generate_spin_system")
def test_main_fluorine(mock_gen_spin: MagicMock) -> None:
    """Test main function with fluorine flag."""
    mock_gen_spin.return_value = ((0, 1), 3, 5.0)
    test_args = ["nmr-generate-instances", "test_pdb", "1", "--fluorine"]
    with patch.object(sys, "argv", test_args):
        main()
    _args, kwargs = mock_gen_spin.call_args
    assert kwargs["fluorine"] is True


def test_get_carbon_indices_fluorine(tmp_path: Path) -> None:
    """Test get_carbon_indices with fluorine=True counts only the fluorines."""
    pdb_content = (
        "ATOM      1  F1  NUM A   1       0.000   0.000   0.000  1.00  0.00           F\n"
        "ATOM      2  C1  NUM A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  F2  NUM A   1       2.000   0.000   0.000  1.00  0.00           F\n"
    )
    filepath = tmp_path / "fluorine.pdb"
    filepath.write_text(pdb_content, encoding="utf-8")

    c1, c2 = get_carbon_indices(filepath, ind1=0, ind2=2, fluorine=True)
    assert c1 == 0
    assert c2 == 1
    with pytest.raises(ValueError, match="must be 'F'"):
        get_carbon_indices(filepath, ind1=0, ind2=1, fluorine=True)


def test_get_carbon_indices_cif(sample_cif_file: Path) -> None:
    """Test get_carbon_indices with CIF files."""
    c1, c2 = get_carbon_indices(sample_cif_file, ind1=1, ind2=2, fluorine=False)
    assert c1 == 0
    assert c2 == 1


def test_parse_atom_specifier_int() -> None:
    """Test that a plain integer string is parsed as an int."""
    result = parse_atom_specifier("42")
    assert result == 42
    assert isinstance(result, int)


def test_parse_atom_specifier_negative_int() -> None:
    """Test that a negative integer string is parsed as an int."""
    result = parse_atom_specifier("-1")
    assert result == -1
    assert isinstance(result, int)


def test_parse_atom_specifier_string() -> None:
    """Test that a colon-separated string is parsed as an AtomSpecifier."""
    result = parse_atom_specifier("A:ALA:10:CA")
    assert isinstance(result, AtomSpecifier)
    assert result.chain == "A"
    assert result.resname == "ALA"
    assert result.resnum == 10
    assert result.atomlabel == "CA"


def test_parse_atom_specifier_bad_format() -> None:
    """Test that a string with wrong number of colons raises ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError, match="Atom specifier must be"):
        parse_atom_specifier("A:ALA:10")


def test_parse_atom_specifier_bad_resnum() -> None:
    """Test that a non-integer residue number raises ArgumentTypeError."""
    with pytest.raises(argparse.ArgumentTypeError, match="Residue number must be an integer"):
        parse_atom_specifier("A:ALA:XYZ:CA")


def test_resolve_atom_specifier(sample_pdb_file: Path) -> None:
    """Test resolving an AtomSpecifier to a global atom index."""
    spec = AtomSpecifier(chain="A", resname="GLY", resnum=1, atomlabel="CA")
    result = resolve_atom_specifier(sample_pdb_file, spec)
    assert result == 1


def test_resolve_atom_specifier_not_found(tmp_path: Path) -> None:
    """Test that resolve_atom_specifier raises ValueError when no atom matches."""
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text("placeholder")

    with patch("nmr_otoc_tools.generate_instances.PDBParser") as mock_parser:
        mock_structure = MagicMock()
        mock_model = MagicMock()
        mock_model.get_atoms.return_value = []  # No atoms
        mock_structure.__getitem__ = MagicMock(return_value=mock_model)
        mock_parser.return_value.get_structure.return_value = mock_structure

        spec = AtomSpecifier(chain="A", resname="ALA", resnum=10, atomlabel="CA")
        with pytest.raises(ValueError, match="No atom matching specifier"):
            resolve_atom_specifier(pdb_file, spec)


@patch("nmr_otoc_tools.generate_instances.generate_spin_system")
@patch("nmr_otoc_tools.generate_instances.resolve_atom_specifier")
@patch("nmr_otoc_tools.generate_instances.parse_structure")
def test_main_with_string_specifiers(
    mock_parse: MagicMock, mock_resolve: MagicMock, mock_gen_spin: MagicMock
) -> None:
    """Test main function with string-based atom specifiers."""
    mock_gen_spin.return_value = ((0, 1), 3, 5.0)
    # resolve_atom_specifier returns global indices 5 and 12
    mock_resolve.side_effect = [5, 12]

    test_args = [
        "nmr-generate-instances",
        "test_pdb",
        "1",
        "--cidx1",
        "A:ALA:10:CA",
        "--cidx2",
        "A:ALA:11:CB",
        "--output_name",
        "test_output",
    ]
    with patch.object(sys, "argv", test_args):
        main()

    # The pair_generator passed to generate_spin_system should be the resolved tuple
    args, kwargs = mock_gen_spin.call_args
    assert args[3] == (5, 12)
    mock_gen_spin.assert_called_once()
    # The structure is parsed once and reused for both specifiers and the coordinates.
    mock_parse.assert_called_once()
    assert all(call.args[0] is mock_parse.return_value for call in mock_resolve.call_args_list)
    assert kwargs["structure"] is mock_parse.return_value


# --- Vectorization tests ---


def _make_mock_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper returning an identity reference-frame mock."""
    return (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )


# Golden values computed from the original scalar implementation
# (before vectorization) using the test geometry below.
_GOLDEN_C_POS = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
_GOLDEN_H_POS = np.array([[0.0, 1.0, 0.0], [0.5, 0.5, 1.0], [1.0, 0.0, 1.5]])
_GOLDEN_FIELD = np.array([0.3, 0.4, 0.5])

_GOLDEN_DIPOLAR = np.array(
    [
        [0.0, 0.51780403, 0.60421796, -15.54026988, -2.66937942],
        [0.51780403, 0.0, 2.56625332, 2.56606147, 0.51971175],
        [0.60421796, 2.56625332, 0.0, 6.21158771, 2.76616262],
        [-15.54026988, 2.56606147, 6.21158771, 0.0, 62.87860918],
        [-2.66937942, 0.51971175, 2.76616262, 62.87860918, 0.0],
    ]
)

_GOLDEN_MAS = np.array(
    [
        [0.0, -1.72985789, -28.18531208, -6.18146947, -4.81668399],
        [-1.72985789, 0.0, -0.40418217, -6.25303153, -6.92180884],
        [-28.18531208, -0.40418217, 0.0, -58.05666044, -10.94359876],
        [-6.18146947, -6.25303153, -58.05666044, 0.0, -114.17598175],
        [-4.81668399, -6.92180884, -10.94359876, -114.17598175, 0.0],
    ]
)

# coupling_mode="mas-simple" is deprecated but still supported. The tests below check its
# numerics, so its DeprecationWarning is expected (and ignored) there;
# test_mas_simple_emits_deprecation_warning asserts that the warning is raised.
_IGNORE_MAS_SIMPLE_DEPRECATION = pytest.mark.filterwarnings(
    "ignore:coupling_mode='mas-simple' is deprecated:DeprecationWarning"
)
_STATIC_COUPLING_MODES = [
    "dipolar",
    pytest.param("mas-simple", marks=_IGNORE_MAS_SIMPLE_DEPRECATION),
]


def test_vectorized_dipolar_matches_golden() -> None:
    """Verify vectorized dipolar coupling matches the original scalar golden values."""
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_frame.return_value = _make_mock_frame()
        result = get_dipolar_hamiltonian_hbulk_cc(
            c_pos=_GOLDEN_C_POS,
            h_pos=_GOLDEN_H_POS,
            c_ind1=0,
            c_ind2=1,
            href_ind=0,
            field_ref_frame=_GOLDEN_FIELD,
            fluorine=False,
            coupling_mode="dipolar",
        )
    np.testing.assert_allclose(result, _GOLDEN_DIPOLAR, rtol=1e-6, atol=1e-12)


@_IGNORE_MAS_SIMPLE_DEPRECATION
def test_vectorized_mas_simple_matches_golden() -> None:
    """Verify vectorized MAS-simple coupling matches scalar golden values within tolerance.

    The vectorized version uses the algebraic identity 2|cosθ|√(1-cos²θ) instead of
    arccos+sin, which produces O(1e-5) relative differences due to the original's
    1e-6 epsilon hack in the norm denominator.
    """
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_frame.return_value = _make_mock_frame()
        result = get_dipolar_hamiltonian_hbulk_cc(
            c_pos=_GOLDEN_C_POS,
            h_pos=_GOLDEN_H_POS,
            c_ind1=0,
            c_ind2=1,
            href_ind=0,
            field_ref_frame=_GOLDEN_FIELD,
            fluorine=False,
            coupling_mode="mas-simple",
        )
    np.testing.assert_allclose(result, _GOLDEN_MAS, rtol=1e-4, atol=1e-10)


@pytest.mark.parametrize("mode", _STATIC_COUPLING_MODES)
def test_coupling_matrix_symmetry(mode: str) -> None:
    """Verify the coupling matrix is symmetric for both modes."""
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_frame.return_value = _make_mock_frame()
        result = get_dipolar_hamiltonian_hbulk_cc(
            c_pos=_GOLDEN_C_POS,
            h_pos=_GOLDEN_H_POS,
            c_ind1=0,
            c_ind2=1,
            href_ind=0,
            field_ref_frame=_GOLDEN_FIELD,
            fluorine=False,
            coupling_mode=mode,
        )
    np.testing.assert_allclose(result, result.T, atol=1e-12)


@pytest.mark.parametrize("mode", _STATIC_COUPLING_MODES)
def test_coupling_matrix_zero_diagonal(mode: str) -> None:
    """Verify the coupling matrix has zero diagonal (no self-coupling)."""
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_frame.return_value = _make_mock_frame()
        result = get_dipolar_hamiltonian_hbulk_cc(
            c_pos=_GOLDEN_C_POS,
            h_pos=_GOLDEN_H_POS,
            c_ind1=0,
            c_ind2=1,
            href_ind=0,
            field_ref_frame=_GOLDEN_FIELD,
            fluorine=False,
            coupling_mode=mode,
        )
    assert np.all(np.diag(result) == 0.0)


@_IGNORE_MAS_SIMPLE_DEPRECATION
def test_get_dipolar_hamiltonian_mas_simple_mode() -> None:
    """Test Hamiltonian construction with MAS-simple coupling mode."""
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_frame.return_value = _make_mock_frame()
        c_pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        h_pos = np.array([[0.0, 1.0, 0.0]])
        hamiltonian = get_dipolar_hamiltonian_hbulk_cc(
            c_pos=c_pos,
            h_pos=h_pos,
            c_ind1=0,
            c_ind2=1,
            href_ind=0,
            field_ref_frame=np.array([0.0, 0.0, 1.0]),
            fluorine=False,
            coupling_mode="mas-simple",
        )
    assert hamiltonian.shape == (3, 3)
    assert not np.isnan(hamiltonian).any()
    assert np.all(np.diag(hamiltonian) == 0.0)


def test_invalid_coupling_mode() -> None:
    """Test that an invalid coupling mode raises ValueError."""
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_frame.return_value = _make_mock_frame()
        c_pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        h_pos = np.array([[0.0, 1.0, 0.0]])
        with pytest.raises(ValueError, match="Unknown coupling_mode"):
            get_dipolar_hamiltonian_hbulk_cc(
                c_pos=c_pos,
                h_pos=h_pos,
                c_ind1=0,
                c_ind2=1,
                href_ind=0,
                field_ref_frame=np.array([0.0, 0.0, 1.0]),
                fluorine=False,
                coupling_mode="invalid",
            )


def test_fluorine_vectorized() -> None:
    """Test vectorized Hamiltonian with fluorine labels."""
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_frame.return_value = _make_mock_frame()
        c_pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        h_pos = np.array([[0.0, 1.0, 0.0]])
        ham = get_dipolar_hamiltonian_hbulk_cc(
            c_pos=c_pos,
            h_pos=h_pos,
            c_ind1=0,
            c_ind2=1,
            href_ind=0,
            field_ref_frame=np.array([0.0, 0.0, 1.0]),
            fluorine=True,
        )
        assert ham.shape == (3, 3)
        assert not np.isnan(ham).any()
        assert np.all(np.diag(ham) == 0.0)
        # Fluorine has a much larger gyromagnetic ratio than carbon, so the
        # F-F coupling (top-left 2x2) should differ from the C-C case.
        ham_carbon = get_dipolar_hamiltonian_hbulk_cc(
            c_pos=c_pos,
            h_pos=h_pos,
            c_ind1=0,
            c_ind2=1,
            href_ind=0,
            field_ref_frame=np.array([0.0, 0.0, 1.0]),
            fluorine=False,
        )
        assert not np.allclose(ham[0, 1], ham_carbon[0, 1])


# --------------------------------------------------------------------------------------
# MAS coupling mode
# --------------------------------------------------------------------------------------


def _mas_protein_payload() -> tuple[MagicMock, dict]:
    """Returns a mocked load_spin_coordinates result with 2 carbons and 2 hydrogens."""
    return (
        MagicMock(),
        {
            "C": [np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])],
            "F": [np.array([])],
            "H": [np.array([[0.0, 1.0, 0.0], [1.0, 0.5, 0.5]])],
        },
    )


def test_compute_pair_geometry_matches_static_kernel() -> None:
    """The MAS geometry with a static model reproduces the 'dipolar' coupling matrix."""
    c_pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    h_pos = np.array([[0.0, 1.0, 0.0], [0.7, 0.3, 1.1]])
    euler = (0.4, 1.2, -0.6)
    rotation = rotation_zyz(*euler)
    # The rotor axis in the reference frame plays the role of the field direction.
    field_ref_frame = rotation.T @ np.array([0.0, 0.0, 1.0])

    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_frame.return_value = _make_mock_frame()
        static_matrix = get_dipolar_hamiltonian_hbulk_cc(
            c_pos=c_pos,
            h_pos=h_pos,
            c_ind1=0,
            c_ind2=1,
            href_ind=0,
            field_ref_frame=field_ref_frame,
            fluorine=False,
            coupling_mode="dipolar",
        )
        geometry = compute_pair_geometry(
            c_pos, h_pos, 0, 1, 0, fluorine=False, use_unrotated_field=False
        )

    mas_matrix = get_mas_hamiltonian_hbulk_cc(geometry, MASRecouplingModel.static(), euler)
    np.testing.assert_allclose(mas_matrix.real, static_matrix, atol=1e-10)
    np.testing.assert_allclose(mas_matrix.imag, np.zeros_like(static_matrix), atol=1e-12)


def test_compute_pair_geometry_species_mask() -> None:
    """Homonuclear pairs (C-C, H-H) are flagged; C-H pairs are not."""
    c_pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    h_pos = np.array([[0.0, 1.0, 0.0], [0.7, 0.3, 1.1]])
    geometry = compute_pair_geometry(
        c_pos, h_pos, 0, 1, 0, fluorine=False, use_unrotated_field=True
    )
    assert geometry.labels == ["C", "C", "H", "H"]
    assert geometry.same_species[0, 1]
    assert geometry.same_species[2, 3]
    assert not geometry.same_species[0, 2]
    np.testing.assert_allclose(np.diag(geometry.prefactors), np.zeros(4), atol=1e-15)


@patch("nmr_otoc_tools.generate_instances.load_spin_coordinates")
@patch("nmr_otoc_tools.generate_instances.get_separated_pair")
def test_generate_spin_system_mas_writes_complex_matrices(
    mock_get_pair: MagicMock,
    mock_load: MagicMock,
    tmp_path: Path,
) -> None:
    """MAS mode writes complex coupling matrices and the per-orientation rotations."""
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_frame.return_value = _make_mock_frame()
        mock_load.return_value = _mas_protein_payload()
        mock_get_pair.return_value = (0, 1)

        grid = so3_grid(6, 2)
        config = {"distance_min": 1.0, "distance_max": 3.0, "ball_ratio": 10.0}
        indices, _n_min, distance = generate_spin_system(
            config=config,
            pdb_path=tmp_path / "placeholder.pdb",
            output_folder=tmp_path,
            idx_generator=np.random.RandomState(0),
            fields=[],
            fluorine=False,
            coupling_mode="mas",
            orientation_grid=grid,
            mas_model=MASRecouplingModel.single_component(),
        )

    assert indices == (0, 1)
    assert np.isclose(distance, 2.0)
    for i in range(len(grid)):
        matrix = np.load(tmp_path / f"dipolar_matrix{i}.npy")
        assert np.iscomplexobj(matrix)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-15)
        np.testing.assert_allclose(np.diag(matrix), np.zeros(matrix.shape[0]), atol=1e-15)
        # Carbon-proton entries are real; the carbon-carbon entry carries a rotor phase.
        assert np.isclose(matrix[0, 2].imag, 0.0, atol=1e-15)
        assert (tmp_path / f"orientation{i}.npy").exists()
        assert (tmp_path / f"field_direction{i}.npy").exists()
        assert (tmp_path / f"ordered_indices{i}.npy").exists()
    phases = [
        np.angle(np.load(tmp_path / f"dipolar_matrix{i}.npy")[0, 1]) for i in range(len(grid))
    ]
    assert len(set(np.round(phases, 6))) > 1


@patch("nmr_otoc_tools.generate_instances.load_spin_coordinates")
@patch("nmr_otoc_tools.generate_instances.get_separated_pair")
def test_generated_mas_instances_run_through_dense_simulation(
    mock_get_pair: MagicMock,
    mock_load: MagicMock,
    tmp_path: Path,
) -> None:
    """Generated MAS matrices pass the simulation checks and run with the dense builder."""
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_frame.return_value = _make_mock_frame()
        mock_load.return_value = _mas_protein_payload()
        mock_get_pair.return_value = (0, 1)
        grid = so3_grid(6, 2)
        config = {"distance_min": 1.0, "distance_max": 3.0, "ball_ratio": 10.0}
        generate_spin_system(
            config=config,
            pdb_path=tmp_path / "placeholder.pdb",
            output_folder=tmp_path,
            idx_generator=np.random.RandomState(0),
            fields=[],
            fluorine=False,
            coupling_mode="mas",
            orientation_grid=grid,
            mas_model=MASRecouplingModel.single_component(),
        )

    sim_params = OTOCSimulation(
        num_spins=4,
        t_range=(0.05, 0.1),
        n_trotter=2,
        n_haar=2,
        n_thread=1,
        n_proc=1,
        save_circuit=False,
        circuit_type="dense",
    )
    for iavg in range(len(grid)):
        experiment = create_experiment(iavg, sim_params, tmp_path, "test")
        assert experiment is not None
        assert np.iscomplexobj(experiment.homogeneous_coupling_mat)
        assert not np.iscomplexobj(experiment.heterogeneous_coupling_mat)

    results = np.array(run_parallel_simulation(sim_params, tmp_path, "test", n_avg=2))
    assert results.shape == (2, len(sim_params.t_range))
    assert np.all(np.isfinite(results))


@patch("nmr_otoc_tools.generate_instances.generate_spin_system")
def test_main_mas_grid_and_metadata(mock_gen_spin: MagicMock, tmp_path: Path) -> None:
    """The mas CLI builds an SO(3) grid and records the model in the metadata."""
    mock_gen_spin.return_value = ((0, 1), 3, 5.0)
    test_args = [
        "nmr-generate-instances",
        "test_pdb",
        "-6",
        "--filename",
        "test.cif",
        "--coupling",
        "mas",
        "--n-twist",
        "4",
    ]
    with patch.object(sys, "argv", test_args):
        main()

    _args, kwargs = mock_gen_spin.call_args
    grid = kwargs["orientation_grid"]
    assert len(grid) == 24
    assert grid.n_lebedev == 6
    assert grid.n_twist == 4
    assert kwargs["mas_model"].eta_dq == {-1: 1.0 + 0.0j}

    dist_folder = tmp_path / "data" / "generated_instances" / "test_pdb_L6" / "d_5"
    weights = np.load(dist_folder / "weights.npy")
    assert len(weights) == 24
    assert np.isclose(weights.sum(), 1.0)
    assert np.load(dist_folder / "orientations.npy").shape == (24, 3, 3)
    assert np.load(dist_folder / "euler_angles.npy").shape == (24, 3)

    metadata = json.loads((dist_folder / "metadata.json").read_text())
    assert metadata["coupling"] == "mas"
    assert metadata["coupling_format"] == "complex"
    assert metadata["mas"]["n_orientations"] == 24
    assert metadata["mas"]["grid"] == "lebedev_x_uniform_twist_about_rotor_axis"
    assert metadata["mas"]["eta_dq"] == {"-1": [1.0, 0.0]}
    assert np.isclose(metadata["mas"]["rotor_angle_deg"], 54.735610317245346)
    assert "--eta_hom / --eta_het" in metadata["mas"]["storage_convention"]


@patch("nmr_otoc_tools.generate_instances.generate_spin_system")
def test_main_mas_weights_are_so3_quadrature(mock_gen_spin: MagicMock, tmp_path: Path) -> None:
    """MAS weights.npy holds the SO(3) weights w_i / N_T in Lebedev-major order.

    The 14-point Lebedev grid has unequal weights, so this separates w_i / N_T from uniform
    weights (which the 6-point grid cannot) and from the static S^2 weights (14 entries).
    """
    mock_gen_spin.return_value = ((0, 1), 3, 5.0)
    n_twist = 3
    test_args = [
        "nmr-generate-instances",
        "test_pdb",
        "-14",
        "--filename",
        "test.cif",
        "--coupling",
        "mas",
        "--n-twist",
        str(n_twist),
    ]
    with patch.object(sys, "argv", test_args):
        main()

    dist_folder = tmp_path / "data" / "generated_instances" / "test_pdb_L14" / "d_5"
    weights = np.load(dist_folder / "weights.npy")
    _, lebedev_weights = lebedev_directions(14)
    assert len(np.unique(np.round(lebedev_weights, 12))) > 1
    np.testing.assert_allclose(weights, np.repeat(lebedev_weights, n_twist) / n_twist)
    assert np.isclose(weights.sum(), 1.0)


@patch("nmr_otoc_tools.generate_instances.generate_spin_system")
def test_main_mas_random_orientations(mock_gen_spin: MagicMock) -> None:
    """A positive instance id draws that many random Haar orientations."""
    mock_gen_spin.return_value = ((0, 1), 3, 5.0)
    test_args = [
        "nmr-generate-instances",
        "test_pdb",
        "5",
        "--filename",
        "test.cif",
        "--coupling",
        "mas",
    ]
    with patch.object(sys, "argv", test_args):
        main()
    _args, kwargs = mock_gen_spin.call_args
    assert len(kwargs["orientation_grid"]) == 5


def test_main_mas_rejects_use_symmetry() -> None:
    """Inversion pruning is not a valid symmetry of the SO(3) grid."""
    test_args = [
        "nmr-generate-instances",
        "test_pdb",
        "-6",
        "--coupling",
        "mas",
        "--use-symmetry",
    ]
    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--n-twist", "4"],
        ["--rotor-angle-deg", "50.0"],
        ["--mas-dq-order", "2"],
        ["--mas-zz-order", "0"],
    ],
)
def test_mas_only_flags_require_mas_mode(extra_args: list[str]) -> None:
    """MAS-specific flags are rejected for the static coupling models."""
    test_args = ["nmr-generate-instances", "test_pdb", "-6", *extra_args]
    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize(
    "removed_flag",
    [
        ["--mas-eta-dq", "0.5"],
        ["--mas-eta-zz", "0.5"],
        ["--mas-eta-json", '{"eta_dq": {"-1": [1, 0]}}'],
    ],
)
def test_mas_eta_flags_are_removed(removed_flag: list[str]) -> None:
    """RF scaling is applied at simulation time (--eta_hom / --eta_het), not at generation."""
    test_args = ["nmr-generate-instances", "test_pdb", "-6", "--coupling", "mas", *removed_flag]
    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit):
        main()


def test_build_mas_model_uses_unit_weights() -> None:
    """The CLI model only selects the recoupled components (unit weights) and the rotor angle."""
    args = argparse.Namespace(rotor_angle_deg=50.0, mas_dq_order=2, mas_zz_order=1)
    model = build_mas_model(args)
    assert model.eta_dq == {-2: 1.0 + 0.0j}
    assert model.eta_zz == {1: 1.0 + 0.0j}
    assert np.isclose(np.degrees(model.rotor_angle), 50.0)


def test_mas_simple_emits_deprecation_warning() -> None:
    """The legacy mas-simple model warns that it is incomplete."""
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_frame.return_value = _make_mock_frame()
        with pytest.warns(DeprecationWarning, match="mas-simple"):
            get_dipolar_hamiltonian_hbulk_cc(
                c_pos=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
                h_pos=np.array([[0.0, 1.0, 0.0]]),
                c_ind1=0,
                c_ind2=1,
                href_ind=0,
                field_ref_frame=np.array([0.0, 0.0, 1.0]),
                fluorine=False,
                coupling_mode="mas-simple",
            )


# --------------------------------------------------------------------------------------
# Parsing once, shared helpers, validation
# --------------------------------------------------------------------------------------


def test_extract_spin_coordinates(sample_pdb_file: Path) -> None:
    """Coordinates are grouped by element per model, keeping Bio.PDB's float32 precision."""
    coords = extract_spin_coordinates(parse_structure(sample_pdb_file))
    assert coords["C"][0].shape == (2, 3)
    assert coords["C"][0].dtype == np.float32
    assert coords["H"][0].shape == (1, 3)
    assert coords["F"][0].size == 0
    np.testing.assert_allclose(coords["C"][0][0], [0.423, 1.210, -0.636], atol=1e-6)


def test_load_spin_coordinates_reuses_a_parsed_structure(sample_pdb_file: Path) -> None:
    """A structure that is already parsed is not parsed again."""
    structure = parse_structure(sample_pdb_file)
    with patch("nmr_otoc_tools.generate_instances.parse_structure") as mock_parse:
        returned, coords = load_spin_coordinates(sample_pdb_file, structure)
    mock_parse.assert_not_called()
    assert returned is structure
    assert coords["H"][0].shape == (1, 3)


def test_get_carbon_indices_rejects_invalid_atoms(sample_pdb_file: Path) -> None:
    """Wrong elements and out-of-range indices raise ValueError (not stripped assertions)."""
    with pytest.raises(ValueError, match="must be 'C'"):
        get_carbon_indices(sample_pdb_file, ind1=0, ind2=1, fluorine=False)
    with pytest.raises(ValueError, match="out of range"):
        get_carbon_indices(sample_pdb_file, ind1=1, ind2=99, fluorine=False)
    structure = parse_structure(sample_pdb_file)
    assert get_carbon_indices(structure, ind1=1, ind2=2, fluorine=False) == (0, 1)


# Spins 0 and 2, and 2 and 3, are uncoupled. The summed shortest 1/|H| path lengths from the
# system spins 0 and 1 are 0.5, 0.5, 8.5 and 2.5, so spin 3 is ordered before spin 2.
_UNCOUPLED_PAIRS_HAMILTONIAN = np.array(
    [
        [0.0, 2.0, 0.0, 1.0],
        [2.0, 0.0, 0.25, 0.5],
        [0.0, 0.25, 0.0, 0.0],
        [1.0, 0.5, 0.0, 0.0],
    ]
)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "hamiltonian",
    [_UNCOUPLED_PAIRS_HAMILTONIAN, _UNCOUPLED_PAIRS_HAMILTONIAN * np.exp(0.3j)],
    ids=["real", "complex"],
)
def test_spin_order_treats_zero_couplings_as_missing_edges(hamiltonian: np.ndarray) -> None:
    """Zero couplings (incl. the diagonal) are missing edges, not a divide-by-zero warning."""
    indices, num_spins_min = spin_order(hamiltonian)
    np.testing.assert_array_equal(indices, [0, 1, 3, 2])
    assert num_spins_min == 2


def _install_structure(pdb_file: Path, root: Path) -> None:
    """Copies a structure file to protein_files/TEST/test.pdb below ``root``."""
    pdb_dir = root / "protein_files" / "TEST"
    pdb_dir.mkdir(parents=True)
    shutil.copy(pdb_file, pdb_dir / "test.pdb")


@pytest.mark.parametrize(
    ("cidx1", "cidx2"),
    [("1", "2"), ("A:GLY:1:CA", "A:GLY:1:C"), ("1", "A:GLY:1:C")],
)
def test_main_parses_the_structure_once(
    sample_pdb_file: Path, tmp_path: Path, cidx1: str, cidx2: str
) -> None:
    """Index resolution, carbon indices and coordinates share a single parse of the file."""
    _install_structure(sample_pdb_file, tmp_path)
    test_args = [
        "nmr-generate-instances",
        "TEST",
        "1",
        "--filename",
        "test.pdb",
        "--cidx1",
        cidx1,
        "--cidx2",
        cidx2,
        "--output_name",
        "out",
    ]
    with (
        patch.object(sys, "argv", test_args),
        patch("nmr_otoc_tools.generate_instances.PDBParser", wraps=PDBParser) as mock_parser,
    ):
        main()

    assert mock_parser.call_count == 1
    out = tmp_path / "data" / "generated_instances" / "TEST_1" / "out"
    assert json.loads((out / "metadata.json").read_text())["indices"] == [0, 1]
    assert np.load(out / "dipolar_matrix0.npy").shape == (3, 3)


def test_main_rejects_two_arguments_naming_the_same_atom(
    sample_pdb_file: Path, tmp_path: Path
) -> None:
    """An index and a specifier for the same atom are rejected before any folder is created."""
    _install_structure(sample_pdb_file, tmp_path)
    test_args = [
        "nmr-generate-instances",
        "TEST",
        "1",
        "--filename",
        "test.pdb",
        "--cidx1",
        "1",
        "--cidx2",
        "A:GLY:1:CA",
    ]
    with patch.object(sys, "argv", test_args), pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("instance_id", ["-6", "-14", "-5810", "1", "12"])
def test_parse_args_accepts_valid_instance_ids(instance_id: str) -> None:
    """Positive counts and supported Lebedev grid sizes are accepted."""
    with patch.object(sys, "argv", ["nmr-generate-instances", "pdb", instance_id]):
        assert parse_args().instance_id == int(instance_id)


@pytest.mark.parametrize(
    "extra_args",
    [
        ["0"],
        ["-1"],
        ["-7"],
        ["1", "--cidx1", "3", "--cidx2", "3"],
        ["1", "--cidx1", "A:GLY:1:CA", "--cidx2", "A:GLY:1:CA"],
        ["1", "--cidx1", "-1", "--cidx2", "2"],
        ["1", "--cidx1", "A:GLY", "--cidx2", "2"],
    ],
)
def test_parse_args_rejects_invalid_values(extra_args: list[str], tmp_path: Path) -> None:
    """Invalid instance ids and atom pairs exit before any output folder is created."""
    with (
        patch.object(sys, "argv", ["nmr-generate-instances", "pdb", *extra_args]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()
    assert exc_info.value.code == 2
    assert not (tmp_path / "data").exists()


def test_coupling_kernels_reject_coincident_nuclei() -> None:
    """Coincident nuclei raise instead of silently producing NaN couplings."""
    c_pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    h_pos = np.array([[0.0, 1.0, 0.0]])
    field = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="system spins"):
        get_dipolar_hamiltonian_hbulk_cc(c_pos, h_pos, 0, 0, 0, field, fluorine=False)
    with pytest.raises(ValueError, match="system spins"):
        compute_pair_geometry(c_pos, h_pos, 1, 1, 0, fluorine=False)
    with pytest.raises(ValueError, match="system spins"):
        get_dipolar_hamiltonian_hbulk_cc(np.zeros((2, 3)), h_pos, 0, 1, 0, field, fluorine=False)
    with pytest.raises(ValueError, match="share the position"):
        get_dipolar_hamiltonian_hbulk_cc(
            c_pos, c_pos[1:], 0, 1, 0, field, fluorine=False, use_unrotated_field=True
        )


def test_coupling_kernel_rejects_zero_field() -> None:
    """A zero field direction raises instead of producing NaN couplings."""
    with pytest.raises(ValueError, match="non-zero"):
        get_dipolar_hamiltonian_hbulk_cc(
            _GOLDEN_C_POS,
            _GOLDEN_H_POS,
            0,
            1,
            0,
            np.zeros(3),
            fluorine=False,
            use_unrotated_field=True,
        )


def test_static_coupling_matrix_keeps_coordinate_precision() -> None:
    """Float32 (Bio.PDB) coordinates give a float32 matrix, as generated instances always had."""
    with patch("nmr_otoc_tools.generate_instances.get_molecule_reference_frame") as mock_frame:
        mock_frame.return_value = _make_mock_frame()
        result = get_dipolar_hamiltonian_hbulk_cc(
            _GOLDEN_C_POS.astype(np.float32),
            _GOLDEN_H_POS.astype(np.float32),
            0,
            1,
            0,
            _GOLDEN_FIELD,
            fluorine=False,
        )
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, _GOLDEN_DIPOLAR, rtol=1e-5)


def test_get_fields_and_weights_returns_copies() -> None:
    """Modifying the returned Lebedev grid cannot corrupt PySCF's cached grid."""
    cached = MakeAngularGrid(6)
    snapshot = cached.copy()
    fields, weights = get_fields_and_weights(1, np.random.RandomState(0), nleb=6)
    assert not np.shares_memory(fields, cached)
    assert not np.shares_memory(weights, cached)
    weights *= 2.0
    fields[:] = 0.0
    np.testing.assert_array_equal(MakeAngularGrid(6), snapshot)


@pytest.mark.parametrize(("instance_id", "nleb"), [(0, None), (-1, None), (1, 0), (1, 1), (1, 7)])
def test_get_fields_and_weights_rejects_invalid_sizes(instance_id: int, nleb: int | None) -> None:
    """Zero random fields and unsupported Lebedev sizes (including 0 and 1) raise ValueError."""
    with pytest.raises(ValueError, match=r"must be positive|Lebedev grid size"):
        get_fields_and_weights(instance_id, np.random.RandomState(0), nleb)


def test_environment_hydrogen_helpers_agree() -> None:
    """The index and position helpers select the same hydrogens; href is the closest one."""
    spin_pos = [np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])]
    h_pos = [np.array([[0.0, 1.0, 0.0], [20.0, 0.0, 0.0], [1.5, 1.2, 0.0]])]

    indices = select_environment_hydrogen_indices(h_pos, spin_pos, 0, 1, 2.0)

    assert indices == [0, 2]
    reduced = select_environment_hydrogens(h_pos, spin_pos, 0, 1, 2.0)
    np.testing.assert_array_equal(reduced[0], h_pos[0][[0, 2]])
    assert reference_hydrogen_index(reduced, spin_pos, 0) == 0
    assert reference_hydrogen_index(reduced, spin_pos, 1) == 1


def test_mean_coupling_matrix_mas_geometries_on_demand() -> None:
    """MAS conformer averages agree with and without precomputed pair geometries."""
    spin_pos = [
        np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [1.6, 0.0, 0.1]]),
    ]
    h_pos = [
        np.array([[0.0, 1.0, 0.0], [1.0, 0.5, 0.5]]),
        np.array([[0.0, 1.1, 0.0], [1.0, 0.6, 0.5]]),
    ]
    model = MASRecouplingModel.single_component()
    euler = np.array([0.3, 1.0, -0.4])
    geometries = conformer_pair_geometries(spin_pos, h_pos, 0, 1, 0, fluorine=False)
    common = {"fluorine": False, "coupling_mode": "mas"}

    with_geometries = mean_coupling_matrix(
        spin_pos, h_pos, 0, 1, 0, mas_orientation=MASOrientation(model, euler, geometries), **common
    )
    on_demand = mean_coupling_matrix(
        spin_pos, h_pos, 0, 1, 0, mas_orientation=MASOrientation(model, euler), **common
    )

    assert np.iscomplexobj(on_demand)
    np.testing.assert_array_equal(with_geometries, on_demand)
    with pytest.raises(ValueError, match="MAS orientation"):
        mean_coupling_matrix(spin_pos, h_pos, 0, 1, 0, fluorine=False, coupling_mode="mas")
    with pytest.raises(ValueError, match="field direction"):
        mean_coupling_matrix(spin_pos, h_pos, 0, 1, 0, fluorine=False, coupling_mode="dipolar")
