# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Optional
import numpy as np
from nmrotoc.chem.molecule import Molecule, StaticVector
from nmrotoc.chem.hamiltonians import make_coupling_matrices
from nmrotoc.math.graphs import dijkstra


def submolecule_from_point(
    molecule: Molecule,
    point: np.ndarray,
    num_atoms: int,
    use_average_position: Optional[bool] = False,
) -> Molecule:
    """
    Generates a submolecule containing the K atoms closest to a fixed point.

    Args:
        molecule: The Molecule object.
        point: The fixed point in 3D space (NumPy array).
        num_atoms: The number of closest atoms to include in the submolecule.

    Returns:
        A new Molecule object containing the K closest atoms.
        Returns the molecule itself if K is larger than the size of the molecule

    Raises:
        TypeError: If input types are incorrect.
        ValueError: if k is non-positive
    """

    if not isinstance(molecule, Molecule):
        raise TypeError("molecule must be an nmrotoc.chem.molecule.Molecule object")
    if not isinstance(point, np.ndarray):
        raise TypeError("point must be a NumPy array")
    if point.shape != (3,):
        raise ValueError("point must be a 3D vector")
    if not isinstance(num_atoms, int) or num_atoms <= 0:
        raise ValueError("k must be a positive integer")

    if num_atoms > len(molecule.coordinates):
        return molecule

    distances = []
    for label in molecule:
        if use_average_position:
            pos = molecule.coordinates[label].mean
        else:
            pos = molecule[label]
        distance = np.linalg.norm(pos - point)
        distances.append((label, distance))

    distances.sort(key=lambda x: x[1])  # Sort by distance

    closest_labels = [label for label, _ in distances[:num_atoms]]

    submolecule_coordinates = {
        label: molecule.coordinates[label] for label in closest_labels
    }
    if molecule.parameters:
        submolecule_parameters = {
            label: molecule.parameters[label]
            for label in closest_labels
            if label in molecule.parameters
        }
    else:
        submolecule_parameters = None

    submolecule = Molecule(submolecule_coordinates, submolecule_parameters)
    return submolecule


def submolecule_from_pathing(
    molecule: Molecule,
    label1: str,
    label2: str,
    num_spins: int,
    order_parameters: Optional[np.ndarray] = np.array([-1, -1, 2])
):
    """Generates a submolecule by choosing H atoms that generate the shortest paths between C atoms
    
    Args:
        molecule: Input molecule to take submolecule from
        label1: 1st atom
        label2: 2nd atom
        num_spins: num spins to truncate to
        order_parameters: Saupe tensor for liquid crystals
    """
    labels = [label1, label2] + [key for key in molecule if key not in [label1, label2]]
    hom, het = make_coupling_matrices(
        molecule=molecule, labels=labels, order_parameters=order_parameters
    )
    # We take a classical approach; assume the time for a spin cluster to expand from j to k
    # is proportional to 1/h_{jk}, and assume that this occurs discretely (i.e. spin is either)
    # not on site k or it is. Then, the time it takes to propagate from j to k is equal
    # to the shortest weighted path on the graph defined by A_{jk} = 1 / h_{jk}. We thus want
    # to calculate the shortest path from the two starting vertices. 
    weight_mat = 1 / (hom + het)
    paths_left = dijkstra(weight_mat, 0)
    paths_right = dijkstra(weight_mat, 1)
    paths_full = paths_left + paths_right
    indices = np.argsort(paths_full)[:num_spins]
    reduced_labels = [labels[index] for index in indices]
    submolecule_coordinates = {
        label: molecule.coordinates[label] for label in reduced_labels
    }
    if molecule.parameters:
        submolecule_parameters = {
            label: molecule.parameters[label]
            for label in reduced_labels
            if label in molecule.parameters
        }
    else:
        submolecule_parameters = None

    submolecule = Molecule(submolecule_coordinates, submolecule_parameters)
    return submolecule
    


def submolecule_from_spheroid(
    molecule: Molecule,
    x1: np.ndarray,
    x2: np.ndarray,
    k: int,
    use_average_position: bool = False,
) -> Molecule:
    """
    Generates a submolecule containing the K atoms closest to a prolate spheroid defined by two points, satisfying the equation

    (x^2 + y^2) / a^2 + z^2 / (a + d/2)^2 = 1,

    where d is the distance between the points, z is the coordinate
    along the vector connecting the two points, and x and y are the
    coordinates in the perpendicular direction.

    Args:
        molecule: The Molecule object.
        x1: The first point defining the spheroid axis.
        x2: The second point defining the spheroid axis.
        k: The number of closest atoms to include in the submolecule.
        use_average_position: Whether to use the average position for dynamic atoms.

    Returns:
        A new Molecule object containing the K closest atoms.
        Returns the original molecule if k is greater than the number of atoms.

    Raises:
        TypeError: If input types are incorrect.
        ValueError: If k is non-positive.
    """

    if not isinstance(molecule, Molecule):
        raise TypeError("molecule must be an nmrotoc.chem.molecule.Molecule object")
    if not isinstance(x1, np.ndarray) or not isinstance(x2, np.ndarray):
        raise TypeError("x1 and x2 must be NumPy arrays")
    if x1.shape != (3,) or x2.shape != (3,):
        raise ValueError("x1 and x2 must be 3D vectors")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    if k > len(molecule.coordinates):
        return molecule

    center = (x1 + x2) / 2
    d = np.linalg.norm(x1 - x2)
    v_axis = (x2 - x1) / np.linalg.norm(x2 - x1)

    distances = []
    for label in molecule:
        if use_average_position:
            pos = molecule.coordinates[label].mean
        else:
            pos = molecule[label]

        v_atom = pos - center
        z = np.dot(v_atom, v_axis)
        x = np.linalg.norm(v_atom - z * v_axis)

        # We solve the following equation for a
        # x^2 / a^2 + z^2 / (a + d/2) ^ 2 = 1.
        # Unfortunately this yields a fourth-order polynomial
        # so we have to invoke root finding in numpy.
        size = max(
            [
                x
                for x in np.polynomial.Polynomial(
                    (-0.25 * d**2 * x**2, -d * x**2, (d**2 / 4 - x**2 - z**2), d, 1)
                ).roots()
                if abs(np.imag(x)) < 1e-12 and np.real(x) >= 0
            ]
        )

        distances.append((label, size))

    distances.sort(key=lambda x: x[1])  # Sort by required_size

    closest_labels = [label for label, _ in distances[:k]]

    submolecule_coordinates = {
        label: molecule.coordinates[label] for label in closest_labels
    }
    if molecule.parameters:
        submolecule_parameters = {
            label: molecule.parameters[label]
            for label in closest_labels
            if label in molecule.parameters
        }
    else:
        submolecule_parameters = None

    submolecule = Molecule(submolecule_coordinates, submolecule_parameters)
    return submolecule


def displace_molecule(
    molecule: Molecule,
    width: float,
    min_distance: float,
    rng: np.random.RandomState,
    max_attempts: Optional[int] = 1000,
) -> Molecule:
    """Displaces atoms in a molecule randomly within a specified width, ensuring a minimum distance between atoms.

    Args:
        molecule: The input Molecule object.
        width: The maximum displacement in each dimension.
        min_distance: The minimum allowed distance between any two atoms.
        rng: A NumPy random number generator.

    Returns:
        A new Molecule object with displaced coordinates.
        Returns None if it fails to generate a valid configuration after maximum_attempts.

    Raises:
        TypeError: If input types are incorrect.
        ValueError: If width or min_distance are non-positive.
    """

    if not isinstance(molecule, Molecule):
        raise TypeError("molecule must be a Molecule object.")
    if not isinstance(width, (int, float)) or width <= 0:
        raise ValueError("width must be a positive number.")
    if not isinstance(min_distance, (int, float)) or min_distance <= 0:
        raise ValueError("min_distance must be a positive number.")

    coordinates = {label: molecule[label] for label in molecule}
    displaced_coordinates = {}

    for label, original_pos in coordinates.items():
        for _ in range(max_attempts):
            displacement = rng.normal(0, width, size=3)
            new_pos = original_pos + displacement

            valid_position = True
            for other_label, other_pos in displaced_coordinates.items():
                distance = np.linalg.norm(new_pos - other_pos())
                if distance < min_distance:
                    valid_position = False
                    break

            if valid_position:
                displaced_coordinates[label] = StaticVector(*new_pos)
                break  # Move to the next atom if a valid position is found

        else:
            raise ValueError(
                "Exceeded {} attempts to find a valid position.".format(max_attempts)
            )
            # No valid position found after max_attempts for this atom.
            return None  # Could return the molecule as-is here, raise an exception, or implement another strategy.

    return Molecule(displaced_coordinates)


def get_separated_pair(
    positions: list[np.ndarray], 
    rng: np.random.RandomState,
    distance_min: float,
    distance_max: float,
    ntries_max: int = 100
):
    """Selects two atoms at random within a range of distances from each other
    
    Args:
        positions: list of atom co-ordinates to choose from
        rng: random number generator
        distance_min, distance_max: limits on the distance between atoms
        ntries_max: prevents infinite loop in case no pairs of atoms
            lie at a fixed distance from each other.
    """
    ind1 = rng.randint(len(positions))
    ind2_list = [
        ind for ind in range(len(positions))
        if np.linalg.norm(positions[ind] - positions[ind1]) >= distance_min
        and np.linalg.norm(positions[ind] - positions[ind1]) <= distance_max
    ]

    # Fix edge case where there is no C13 within the appropriate distance bounds
    # of the first C13 chosen.
    n_tries = 0
    while len(ind2_list) == 0:
        n_tries += 1
        if n_tries > ntries_max:
            raise ValueError(f"Failed to find valid pair after {ntries_max} attempts."
                             f" Would suggest changing distance limits.")
        ind1 = rng.randint(len(positions))
        ind2_list = [
            ind for ind in range(len(positions))
            if np.linalg.norm(positions[ind] - positions[ind1]) >= distance_min
            and np.linalg.norm(positions[ind] - positions[ind1]) <= distance_max
        ]

    ind2 = rng.choice(ind2_list)
    return ind1, ind2

def get_molecule_reference_frame(p1, p2, p3):
    """Generates a consistent set of vectors to define the molecule reference frame
    
    Given points p1, p2, p3, we fix the z-axis to be p1-p2, then use Gram-Schmidt on
    p1-p3 to generate a x-axis, and finally take the cross product to generate a y-axis.

    Args:
        p1, p2, p3: Input vectors
    """
    z_hat = p1 - p2
    z_hat /= np.linalg.norm(z_hat)
    x_hat = p1 - p3
    x_hat -= z_hat * np.dot(x_hat, z_hat)
    x_hat /= np.linalg.norm(x_hat)
    y_hat = np.cross(x_hat, z_hat)
    return x_hat, y_hat, z_hat
