# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

r"""Orientation grids for powder averaging.

Static (non-spinning) simulations only need to sample the direction of the magnetic field
on the unit sphere, which the Lebedev grids in PySCF's :mod:`pyscf.dft.LebedevGrid` provide.

Magic-angle-spinning (MAS) simulations additionally depend on the rotation of the molecule
*about* the rotor axis (the rotor phase), so the powder average has to run over the full
rotation group ``SO(3)``.  As described in the "Magic Angle Spinning (MAS)" section of the
README, such a grid is built as the tensor product of a Lebedev grid on ``S^2`` with a uniform
("twist") grid on ``S^1``:

.. math::

    \int_{SO(3)} f(R)\, dR \approx \sum_i \sum_j w_i \frac{1}{N_T} f(R_{ij}).

The twist is applied about the **rotor axis** rather than about the molecular z-axis.  Both
choices tile ``SO(3)`` uniformly with identical Haar weights, but this one places the
rotor-axis direction in the molecular frame -- the quantity that fixes the polar angle of
every spin pair -- on the accurate Lebedev grid, and leaves only the uniform azimuthal shift
on the trapezoidal grid.  It also makes the twist exactly redundant in the static limit, so
the grid reduces to the plain Lebedev grid used by the non-spinning coupling models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pyscf.dft.LebedevGrid import LEBEDEV_NGRID, MakeAngularGrid

if TYPE_CHECKING:
    from numpy.typing import NDArray

SUPPORTED_LEBEDEV_SIZES: tuple[int, ...] = tuple(int(n) for n in LEBEDEV_NGRID if n > 1)
"""Lebedev grid sizes accepted by :func:`lebedev_grid`.

PySCF's :func:`~pyscf.dft.LebedevGrid.MakeAngularGrid` also accepts 0 and 1 points, but returns
a single zero vector for them, which is not a grid on the unit sphere.
"""


def rotation_z(angle: float) -> NDArray:
    """Returns the 3x3 active rotation matrix about the z-axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        The rotation matrix ``Rz(angle)``.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_y(angle: float) -> NDArray:
    """Returns the 3x3 active rotation matrix about the y-axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        The rotation matrix ``Ry(angle)``.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_zyz(alpha: float, beta: float, gamma: float) -> NDArray:
    """Returns the ZYZ Euler rotation ``Rz(alpha) Ry(beta) Rz(gamma)``.

    Args:
        alpha: First (outermost) z rotation angle in radians.
        beta: y rotation angle in radians.
        gamma: Last (innermost) z rotation angle in radians.

    Returns:
        The composed 3x3 rotation matrix.
    """
    return rotation_z(alpha) @ rotation_y(beta) @ rotation_z(gamma)


def lebedev_grid(nleb: int) -> NDArray:
    """Returns a copy of the ``nleb``-point Lebedev grid.

    Args:
        nleb: Number of Lebedev points; one of :data:`SUPPORTED_LEBEDEV_SIZES`.

    Returns:
        Array of shape ``(nleb, 4)`` with rows ``(x, y, z, weight)``. It is a copy, so modifying
        it cannot corrupt the cache of PySCF's :func:`~pyscf.dft.LebedevGrid.MakeAngularGrid`.

    Raises:
        ValueError: If ``nleb`` is not a supported Lebedev grid size.
    """
    if not isinstance(nleb, (int, np.integer)) or nleb not in SUPPORTED_LEBEDEV_SIZES:
        sizes = ", ".join(str(n) for n in SUPPORTED_LEBEDEV_SIZES)
        msg = f"Unsupported Lebedev grid size {nleb!r}; use one of {sizes}."
        raise ValueError(msg)
    return np.array(MakeAngularGrid(nleb), dtype=np.float64)


def lebedev_directions(nleb: int) -> tuple[NDArray, NDArray]:
    """Returns the Lebedev grid directions and weights for a requested grid size.

    Args:
        nleb: Number of Lebedev points (must be a supported Lebedev grid size).

    Returns:
        A tuple ``(directions, weights)`` where ``directions`` has shape ``(nleb, 3)``
        (unit vectors) and ``weights`` has shape ``(nleb,)`` and sums to one. Both are new
        arrays that do not share memory with the cached grid.

    Raises:
        ValueError: If ``nleb`` is not a supported Lebedev grid size.
    """
    grid = lebedev_grid(nleb)
    directions = grid[:, :3]
    weights = grid[:, 3]
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / norms, weights


@dataclass(frozen=True)
class OrientationGrid:
    """A weighted set of molecular-frame to rotor-frame orientations.

    Attributes:
        rotations: Rotation matrices ``Q`` of shape ``(N, 3, 3)``.  ``Q`` maps the components
            of a vector in the molecular (or reference) frame to its components in the rotor
            frame: ``u_rotor = Q @ v_molecular``.
        euler_zyz: ZYZ Euler angles of shape ``(N, 3)`` such that
            ``Q = Rz(alpha) Ry(beta) Rz(gamma)`` for ``(alpha, beta, gamma)``.
        weights: Normalized quadrature weights of shape ``(N,)``, summing to one.
        n_lebedev: Number of Lebedev points used for the spherical part.
        n_twist: Number of uniform twist angles about the rotor axis.
    """

    rotations: NDArray
    euler_zyz: NDArray
    weights: NDArray
    n_lebedev: int
    n_twist: int

    def __len__(self) -> int:
        """Returns the total number of orientations in the grid."""
        return int(self.rotations.shape[0])

    @property
    def rotor_axes(self) -> NDArray:
        """Rotor (spinning) axis expressed in the molecular frame, shape ``(N, 3)``.

        This equals ``Q^T @ z_hat``, i.e. the third row of each rotation matrix.  For the
        static coupling models this vector plays exactly the role of the magnetic field
        direction, which keeps the on-disk ``field_direction*.npy`` files meaningful.
        """
        return np.ascontiguousarray(self.rotations[:, 2, :])


def so3_grid(nleb: int, n_twist: int) -> OrientationGrid:
    """Builds a Lebedev x uniform-twist quadrature grid over ``SO(3)``.

    For each Lebedev direction ``n_hat`` (polar angles ``theta``, ``phi``) with weight
    ``w_i``, and each twist angle ``psi_j = 2 pi j / n_twist`` with weight ``1 / n_twist``,
    the molecular-to-rotor rotation is

    ``Q_ij = Rz(psi_j) @ Ry(-theta_i) @ Rz(-phi_i)``

    which satisfies ``Q_ij @ n_hat_i = z_hat``: the Lebedev direction is the rotor axis as
    seen in the molecular frame, and ``psi_j`` is a rotation about the rotor axis.

    Args:
        nleb: Number of Lebedev points (supported Lebedev grid size).
        n_twist: Number of uniform twist angles (must be positive).

    Returns:
        The combined :class:`OrientationGrid` with ``nleb * n_twist`` orientations.

    Raises:
        ValueError: If ``n_twist`` is not positive or ``nleb`` is not a supported Lebedev grid
            size.
    """
    if n_twist < 1:
        msg = f"n_twist must be a positive integer, got {n_twist}."
        raise ValueError(msg)

    directions, sphere_weights = lebedev_directions(nleb)
    thetas = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    phis = np.arctan2(directions[:, 1], directions[:, 0])
    twists = 2.0 * np.pi * np.arange(n_twist) / n_twist

    rotations = []
    euler = []
    weights = []
    for theta, phi, weight in zip(thetas, phis, sphere_weights, strict=True):
        for psi in twists:
            rotations.append(rotation_zyz(psi, -theta, -phi))
            euler.append((psi, -theta, -phi))
            weights.append(weight / n_twist)

    return OrientationGrid(
        rotations=np.array(rotations),
        euler_zyz=np.array(euler),
        weights=np.array(weights),
        n_lebedev=len(directions),
        n_twist=n_twist,
    )


def euler_zyz_from_rotation(rotation: NDArray) -> tuple[float, float, float]:
    """Recovers ZYZ Euler angles from a rotation matrix.

    Args:
        rotation: A 3x3 rotation matrix ``Q = Rz(alpha) Ry(beta) Rz(gamma)``.

    Returns:
        The angles ``(alpha, beta, gamma)`` in radians, with ``beta`` in ``[0, pi]``.
    """
    q = np.asarray(rotation, dtype=np.float64)
    beta = float(np.arccos(np.clip(q[2, 2], -1.0, 1.0)))
    if np.isclose(np.sin(beta), 0.0, atol=1e-12):
        # Degenerate (gimbal) case: only alpha + gamma (beta = 0) or alpha - gamma (beta = pi)
        # is defined; put it all in alpha. The first column is cos(beta) times
        # (cos(alpha -+ gamma), sin(alpha -+ gamma)), so undo the sign of cos(beta) = q[2, 2].
        sign = 1.0 if q[2, 2] >= 0.0 else -1.0
        return float(np.arctan2(sign * q[1, 0], sign * q[0, 0])), beta, 0.0
    alpha = float(np.arctan2(q[1, 2], q[0, 2]))
    gamma = float(np.arctan2(q[2, 1], -q[2, 0]))
    return alpha, beta, gamma


def random_so3_orientations(
    num_orientations: int, rng: np.random.Generator | np.random.RandomState
) -> OrientationGrid:
    """Draws uniformly (Haar) distributed orientations with equal weights.

    This is the ``SO(3)`` analogue of the random field directions used by the static coupling
    models when no Lebedev grid is requested.

    Args:
        num_orientations: Number of orientations to draw (at least 1).
        rng: Random number generator.

    Returns:
        An :class:`OrientationGrid` with ``num_orientations`` equally weighted rotations.

    Raises:
        ValueError: If ``num_orientations`` is less than 1.
    """
    if num_orientations < 1:
        msg = f"num_orientations must be at least 1, got {num_orientations}."
        raise ValueError(msg)
    # Uniform ZYZ angles with cos(beta) uniform reproduce the Haar measure sin(beta) da db dg.
    alphas = rng.uniform(0.0, 2.0 * np.pi, size=num_orientations)
    betas = np.arccos(rng.uniform(-1.0, 1.0, size=num_orientations))
    gammas = rng.uniform(0.0, 2.0 * np.pi, size=num_orientations)
    rotations = np.array(
        [rotation_zyz(a, b, g) for a, b, g in zip(alphas, betas, gammas, strict=True)]
    )
    return OrientationGrid(
        rotations=rotations,
        euler_zyz=np.stack([alphas, betas, gammas], axis=1),
        weights=np.full(num_orientations, 1.0 / num_orientations),
        n_lebedev=0,
        n_twist=0,
    )
