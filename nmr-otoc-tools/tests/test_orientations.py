# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for the SO(3) powder-averaging orientation grids."""

import numpy as np
import pytest
from pyscf.dft.LebedevGrid import MakeAngularGrid

from nmr_otoc_tools.mas import wigner_d2
from nmr_otoc_tools.orientations import (
    SUPPORTED_LEBEDEV_SIZES,
    OrientationGrid,
    euler_zyz_from_rotation,
    lebedev_directions,
    lebedev_grid,
    random_so3_orientations,
    rotation_y,
    rotation_z,
    rotation_zyz,
    so3_grid,
)


def test_rotation_helpers_are_rotations() -> None:
    """Rotation helpers return proper (orthogonal, det = +1) rotation matrices."""
    for mat in (rotation_z(0.4), rotation_y(-1.2), rotation_zyz(0.3, 1.1, -0.7)):
        np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(mat), 1.0)


def test_rotation_zyz_composition() -> None:
    """rotation_zyz composes the individual axis rotations in the documented order."""
    alpha, beta, gamma = 0.3, 1.1, -0.7
    expected = rotation_z(alpha) @ rotation_y(beta) @ rotation_z(gamma)
    np.testing.assert_allclose(rotation_zyz(alpha, beta, gamma), expected, atol=1e-15)


def test_lebedev_directions_normalized_and_weighted() -> None:
    """Lebedev directions are unit vectors whose weights sum to one."""
    directions, weights = lebedev_directions(26)
    assert directions.shape == (26, 3)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-12)
    assert np.isclose(weights.sum(), 1.0)


def test_so3_grid_shape_and_weights() -> None:
    """The tensor-product grid has n_lebedev * n_twist points and normalized weights."""
    grid = so3_grid(26, 4)
    assert isinstance(grid, OrientationGrid)
    assert len(grid) == 26 * 4
    assert grid.rotations.shape == (104, 3, 3)
    assert grid.euler_zyz.shape == (104, 3)
    assert np.isclose(grid.weights.sum(), 1.0)


def test_so3_grid_rotations_match_euler_angles() -> None:
    """Stored Euler angles reproduce the stored rotation matrices."""
    grid = so3_grid(14, 3)
    for rotation, euler in zip(grid.rotations, grid.euler_zyz, strict=True):
        np.testing.assert_allclose(rotation, rotation_zyz(*euler), atol=1e-12)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)


def test_so3_grid_rotor_axis_is_lebedev_direction() -> None:
    """The twist is about the rotor axis, so each twist block shares one Lebedev direction."""
    n_twist = 5
    directions, _ = lebedev_directions(26)
    grid = so3_grid(26, n_twist)
    np.testing.assert_allclose(grid.rotor_axes[::n_twist], directions, atol=1e-12)
    for i, direction in enumerate(directions):
        block = grid.rotor_axes[i * n_twist : (i + 1) * n_twist]
        np.testing.assert_allclose(block, np.tile(direction, (n_twist, 1)), atol=1e-12)
        # Q maps the Lebedev direction onto the rotor z-axis.
        for rotation in grid.rotations[i * n_twist : (i + 1) * n_twist]:
            np.testing.assert_allclose(rotation @ direction, [0.0, 0.0, 1.0], atol=1e-12)


def test_so3_grid_twist_angles_are_uniform() -> None:
    """Within a Lebedev block the rotations differ by uniform rotor-axis twists."""
    n_twist = 6
    grid = so3_grid(6, n_twist)
    base = grid.rotations[0]
    for j in range(n_twist):
        expected = rotation_z(2.0 * np.pi * j / n_twist) @ base
        np.testing.assert_allclose(grid.rotations[j], expected, atol=1e-12)


def test_so3_grid_integrates_wigner_d_exactly() -> None:
    """All non-trivial rank <= 2 Wigner D elements integrate to zero on the grid."""
    grid = so3_grid(26, 5)
    accumulator = np.zeros((5, 5), dtype=np.complex128)
    for weight, euler in zip(grid.weights, grid.euler_zyz, strict=True):
        accumulator += weight * wigner_d2(*euler)
    expected = np.zeros((5, 5), dtype=np.complex128)
    np.testing.assert_allclose(accumulator, expected, atol=1e-12)


def test_so3_grid_rejects_nonpositive_twist() -> None:
    """A non-positive twist count is rejected."""
    with pytest.raises(ValueError, match="n_twist must be a positive integer"):
        so3_grid(6, 0)


def test_random_so3_orientations() -> None:
    """Random orientations are proper rotations with equal weights."""
    rng = np.random.default_rng(3)
    grid = random_so3_orientations(32, rng)
    assert len(grid) == 32
    assert np.isclose(grid.weights.sum(), 1.0)
    assert np.allclose(grid.weights, 1.0 / 32)
    for rotation in grid.rotations:
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rotation), 1.0)


def test_random_so3_orientations_are_uniform() -> None:
    """Random rotor axes average to zero, as required by a Haar-distributed sample."""
    rng = np.random.default_rng(11)
    grid = random_so3_orientations(20000, rng)
    np.testing.assert_allclose(grid.rotor_axes.mean(axis=0), np.zeros(3), atol=0.03)


def test_euler_zyz_round_trip() -> None:
    """Euler angles recovered from a rotation reproduce that rotation."""
    rng = np.random.default_rng(5)
    for _ in range(10):
        angles = (
            rng.uniform(0, 2 * np.pi),
            np.arccos(rng.uniform(-1, 1)),
            rng.uniform(0, 2 * np.pi),
        )
        rotation = rotation_zyz(*angles)
        recovered = euler_zyz_from_rotation(rotation)
        np.testing.assert_allclose(rotation_zyz(*recovered), rotation, atol=1e-12)


def test_euler_zyz_round_trip_degenerate() -> None:
    """The gimbal-locked case is handled without NaNs."""
    rotation = rotation_z(0.8)
    recovered = euler_zyz_from_rotation(rotation)
    np.testing.assert_allclose(rotation_zyz(*recovered), rotation, atol=1e-12)


@pytest.mark.parametrize(("alpha", "gamma"), [(0.8, 0.0), (0.3, 1.1), (-2.0, 0.4)])
def test_euler_zyz_round_trip_beta_pi(alpha: float, gamma: float) -> None:
    """At beta = pi only alpha - gamma is defined; the recovered angles reproduce the rotation."""
    rotation = rotation_zyz(alpha, np.pi, gamma)
    recovered = euler_zyz_from_rotation(rotation)
    assert np.isclose(recovered[1], np.pi)
    np.testing.assert_allclose(rotation_zyz(*recovered), rotation, atol=1e-12)


def test_euler_zyz_round_trip_on_so3_grid() -> None:
    """Every grid orientation round-trips, including the 8 twists of the -z direction."""
    grid = so3_grid(6, 8)
    assert np.sum(np.isclose(grid.rotations[:, 2, 2], -1.0)) == 8
    for rotation in grid.rotations:
        recovered = euler_zyz_from_rotation(rotation)
        np.testing.assert_allclose(rotation_zyz(*recovered), rotation, atol=1e-12)


def test_supported_lebedev_sizes_exclude_the_fallback() -> None:
    """PySCF's 0/1-point fallback is not a grid on the sphere and is not supported."""
    assert SUPPORTED_LEBEDEV_SIZES[:4] == (6, 14, 26, 38)
    assert SUPPORTED_LEBEDEV_SIZES[-1] == 5810
    assert 1 not in SUPPORTED_LEBEDEV_SIZES


@pytest.mark.parametrize("nleb", SUPPORTED_LEBEDEV_SIZES)
def test_lebedev_grid_supported_sizes_are_valid(nleb: int) -> None:
    """Every supported grid has ``nleb`` distinct unit vectors whose weights sum to one."""
    grid = lebedev_grid(nleb)
    assert grid.shape == (nleb, 4)
    assert np.isclose(grid[:, 3].sum(), 1.0)
    directions = grid[:, :3]
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-12)
    assert len(np.unique(np.round(directions, 12), axis=0)) == nleb


@pytest.mark.parametrize("nleb", [0, 1, 7, -6, 6.0])
def test_lebedev_directions_reject_unsupported_sizes(nleb: float) -> None:
    """Unsupported grid sizes raise instead of returning a zero vector (NaN directions)."""
    with pytest.raises(ValueError, match="Unsupported Lebedev grid size"):
        lebedev_directions(nleb)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported Lebedev grid size"):
        so3_grid(nleb, 2)  # type: ignore[arg-type]


def test_lebedev_grid_returns_copies() -> None:
    """The returned arrays do not share memory with PySCF's cached grid."""
    cached = MakeAngularGrid(14)
    snapshot = cached.copy()
    grid = lebedev_grid(14)
    directions, weights = lebedev_directions(14)
    for array in (grid, directions, weights):
        assert not np.shares_memory(array, cached)
    grid[:] = 0.0
    weights *= 2.0
    np.testing.assert_array_equal(MakeAngularGrid(14), snapshot)
    assert np.isclose(lebedev_directions(14)[1].sum(), 1.0)


def test_random_so3_orientations_rejects_empty_grid() -> None:
    """At least one orientation is needed (the weights are 1 / num_orientations)."""
    with pytest.raises(ValueError, match="at least 1"):
        random_so3_orientations(0, np.random.default_rng(0))
