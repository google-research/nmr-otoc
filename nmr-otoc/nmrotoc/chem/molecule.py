# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np
import json
import re
from typing import List, Hashable, Optional, Callable, Tuple, Dict
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


element_names = {
    17: "Cl",
    15: "P",
    7: "N",
    8: "O",
    6: "C",
    9: "F",
    1: "H",
}

# Define a color map for different elements
cmap_colors = ["magenta", "purple", "green", "orange", "blue", "cyan", "red"]
cmap = {
    element_names[element]: color for element, color in zip(element_names, cmap_colors)
}
cmap["default_color"] = "black"


class Vector:
    """Represents a potentially non-static callable 3D vector"""

    @property
    def mean(self, num_samples: Optional[int] = 10000) -> np.ndarray:
        """Estimate the mean value of this vector"""

        if self.isstatic:
            return self()
        meanval = np.zeros(3, dtype=float)
        for _ in range(num_samples):
            meanval += self()
        return meanval

    @property
    def isstatic(self) -> bool:
        """Boolean telling whether __call__ returns the same result each time"""
        raise NotImplementedError

    def __call__(self) -> np.ndarray:
        """Calling this returns a vector"""
        raise NotImplementedError


class StaticVector(Vector):
    """Represents a static 3D vector."""

    @property
    def isstatic(self):
        return True

    def __init__(self, x: float, y: float, z: float):
        self._coords = np.array([x, y, z], dtype=float)

    def __call__(self) -> np.ndarray:
        return self._coords

    def __eq__(self, other):
        return self._coords == other._coords

    def copy(self):
        return StaticVector(*self._coords)

    @property
    def norm(self):
        return np.linalg.norm(self._coords)

    @property
    def mean(self):
        return self._coords

    def __add__(self, other):
        """Overloads the + operator for StaticVector.

        Args:
            other: Another StaticVector or a NumPy array to add to self.coords.

        Returns:
            A new StaticVector instance with the result of the addition.

        Raises:
            TypeError: If 'other' is not a StaticVector or a NumPy array.
            ValueError: If 'other' is a NumPy array with incorrect dimensions.
        """
        if isinstance(other, StaticVector):
            return StaticVector(*(self._coords + other._coords))
        elif isinstance(other, np.ndarray):
            if other.shape != (3,):
                raise ValueError("NumPy array must be 3-dimensional.")
            return StaticVector(*(self._coords + other))
        else:
            return NotImplemented

    def __sub__(self, other):
        return (-1 * self) + other

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return StaticVector(*(self._coords * other))
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __iadd__(self, other):
        """Overloads the += operator for StaticVector.

        Args:
            other: Another StaticVector or a NumPy array to add to self.coords.

        Returns:
            self: The updated StaticVector instance.

        Raises:
            TypeError: If 'other' is not a StaticVector or a NumPy array.
            ValueError: If 'other' is a NumPy array with incorrect dimensions.
        """
        if isinstance(other, StaticVector):
            self._coords += other._coords
        elif isinstance(other, np.ndarray):
            if other.shape != (3,):
                raise ValueError("NumPy array must be 3-dimensional.")
            self._coords += other
        else:
            return NotImplemented  # Indicate that the operation is not supported
        return self

    def __repr__(self):
        return f"Vector({self._coords[0]}, {self._coords[1]}, {self._coords[2]})"


class RandomVariable:
    """Represents a randomly generated variable between 0 and 1."""

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        self.value = self.rng.rand()

    def reroll(self):
        """Generates a new random value."""
        self.value = self.rng.rand()

    def __call__(self) -> float:
        return self.value


class ContinuousIntegrator(RandomVariable):

    def __init__(self, points: np.ndarray):
        self.points = points
        self.index = 0
        self.value = self.points[self.index]

    def reroll(self):
        self.index += 1
        self.value = self.points[self.index % len(self.points)]

    def reset(self):
        self.index = 0
        self.value = self.points[self.index]


class CircleVector(Vector):
    """Represents a dynamic 3D vector rotating around a defined circle.

    Attributes:
        initial_vector (np.ndarray): The initial vector to be rotated.
        center (np.ndarray): The center of the circle.
        second_point (np.ndarray): A second point on the circle defining the plane of rotation.
        angle (RandomVariable): A RandomVariable object providing the rotation angle.
    """

    isstatic = False

    def __init__(
        self,
        initial_vector: np.ndarray,
        center: np.ndarray,
        second_point: np.ndarray,
        angle: RandomVariable,
    ):
        self.initial_vector = initial_vector.copy()
        self.center = center.copy()
        self.second_point = second_point.copy()
        self.angle = angle
        self.rotation_axis = self.get_rotation_axis(
            center, initial_vector, second_point
        )

    def copy(self):
        return CircleVector(
            self.initial_vector, self.center, self.second_point, self.angle
        )

    @property
    def mean(self):
        return self.center

    @staticmethod
    def get_rotation_axis(
        center, initial_vector, second_point, EQ_tol: Optional[float] = 1e-9
    ):
        # Calculate a rotation axis (normal to the plane defined by the three points)
        rotation_axis = np.cross(second_point - center, initial_vector - center)
        rotation_axis = rotation_axis / np.linalg.norm(
            rotation_axis
        )  # Normalize, and fix to positive half-plane.
        if rotation_axis[0] < -EQ_tol:
            rotation_axis = -rotation_axis
        elif abs(rotation_axis[0]) < EQ_tol and rotation_axis[1] <= -EQ_tol:
            rotation_axis = -rotation_axis
        elif abs(rotation_axis[0]) < EQ_tol and abs(rotation_axis[1]) < EQ_tol:
            rotation_axis = np.array([0, 0, 1])
        return rotation_axis

    def __call__(self) -> np.ndarray:
        """Returns the rotated vector."""

        # Use Rodrigues' rotation formula
        rotated_vector = (
            np.cos(self.angle() * 2 * np.pi) * self.initial_vector
            + np.sin(self.angle() * 2 * np.pi)
            * np.cross(self.rotation_axis, self.initial_vector)
            + (1 - np.cos(self.angle() * 2 * np.pi))
            * self.rotation_axis
            * np.dot(self.rotation_axis, self.initial_vector)
        )
        return rotated_vector

    def __add__(self, other):
        """Overloads the + operator for CircleVector.

        Args:
            other: A StaticVector or a NumPy array to add to the CircleVector's points.

        Returns:
            A new CircleVector instance with updated points.

        Raises:
            TypeError: If 'other' is not a StaticVector or a NumPy array.
            ValueError: If 'other' is a NumPy array with incorrect dimensions.
        """

        if isinstance(other, (np.ndarray, StaticVector)):
            other_coords = other() if isinstance(other, StaticVector) else other
            if other_coords.shape != (3,):
                raise ValueError("NumPy array or StaticVector must be 3-dimensional.")

            new_initial = self.initial_vector + other_coords
            new_center = self.center + other_coords
            new_second = self.second_point + other_coords

            return CircleVector(new_initial, new_center, new_second, self.angle)
        else:
            return NotImplemented

    def __iadd__(self, other):
        """Overloads the += operator for CircleVector.

        Args:
            other: A StaticVector or a NumPy array to add to the CircleVector's points.

        Returns:
            self: The updated CircleVector instance.

        Raises:
            TypeError: If 'other' is not a StaticVector or a NumPy array.
            ValueError: If 'other' is a NumPy array with incorrect dimensions.
        """

        if isinstance(other, (np.ndarray, StaticVector)):
            other_coords = other() if isinstance(other, StaticVector) else other
            if other_coords.shape != (3,):
                raise ValueError("NumPy array or StaticVector must be 3-dimensional.")

            self.initial_vector += other_coords
            self.center += other_coords
            self.second_point += other_coords

            # Recalculate the rotation axis
            self.rotation_axis = self.get_rotation_axis(
                self.center, self.initial_vector, self.second_point
            )  # Assumes this method exists.

            return self  # Return self for in-place modification
        else:
            return NotImplemented


class Molecule:
    """Stores and manipulates dynamic molecular coordinates.

    Attributes:
        coordinates (dict): A dictionary storing atomic coordinates. Keys are atom
                            labels (e.g., 'H1', 'C2') and values are callables
                            returning a NumPy array of coordinates (x, y, z).
        parameters List(RandomVariable):  A dictionary of random variables
                                                    used to parameterize the coordinates.
        rng (np.random.RandomState): Random number generator.
    """

    def __init__(
        self,
        coordinates: Optional[Dict[Hashable, Callable]] = None,
        parameters: Optional[List[RandomVariable]] = None,
    ):

        self.coordinates = coordinates if coordinates is not None else {}
        self.parameters = parameters if parameters is not None else []

    @classmethod
    def from_pdb(cls, path_to_file: str):
        """
        Creates a Molecule object from a PDB file.

        Args:
            path_to_file (str): Path to the PDB file.

        Returns:
            Molecule: A Molecule object initialized with coordinates from the file.
        """

        coordinates: Dict[str, Callable[[], np.ndarray]] = {}
        with open(path_to_file, "r") as infile:
            for line in infile:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom_name = line[12:16].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coordinates[atom_name] = StaticVector(x, y, z)

        return cls(coordinates=coordinates)

    @classmethod
    def from_pubchem(cls, path_to_file: str, rng=np.random.RandomState()):
        """Import a molecule from a pubchem-format json file.
        Initializes all coordinates as static StaticVectors.

        Args:
            path_to_file (str): Path to the PubChem JSON file.

        Returns:
            Molecule: A Molecule object initialized with coordinates from the file.
        """

        coordinates = {}
        with open(path_to_file, "r") as infile:
            chem_data = json.load(infile)

        conformer = chem_data["PC_Compounds"][0]["coords"][0]["conformers"][0]
        if "z" in conformer:
            for i, (x, y, z, a) in enumerate(
                zip(
                    conformer["x"],
                    conformer["y"],
                    conformer["z"],
                    chem_data["PC_Compounds"][0]["atoms"]["element"],
                )
            ):
                if a in element_names:
                    key = element_names[a] + str(i)
                    coordinates[key] = StaticVector(x, y, z)
        else:
            for i, (x, y, a) in enumerate(
                zip(
                    conformer["x"],
                    conformer["y"],
                    chem_data["PC_Compounds"][0]["atoms"]["element"],
                )
            ):
                if a in element_names:
                    key = element_names[a] + str(i)
                    coordinates[key] = StaticVector(x, y, 0)  # Default z to 0
        return cls(coordinates=coordinates)

    def plot(self, axes: Tuple[int, int] = (0, 1)):
        """Plots the molecule along specified axes.

        Args:
            axes (tuple, optional):  A tuple of two integers specifying the axes to plot. Defaults to (0, 1) (x, y).
                                     Valid values are 0, 1, or 2 representing the x, y, and z coordinates respectively.
        """

        if not all(0 <= ax <= 2 for ax in axes):
            raise ValueError("Axis indices must be 0 (x), 1 (y), or 2 (z).")
        if len(axes) != 2 or len(set(axes)) != 2:
            raise ValueError("axes must be a tuple with two distinct integers.")
        plt.figure()
        for key, coord_callable in self.coordinates.items():
            coords = coord_callable()
            mol_name = key[: re.search(r"\d", key).start()]
            color = cmap.get(mol_name, cmap["default_color"])
            plt.plot(coords[axes[0]], coords[axes[1]], ".", color=color, markersize=15)
            plt.text(coords[axes[0]], coords[axes[1]], key)
        plt.xlabel(f"Axis {axes[0]}")
        plt.ylabel(f"Axis {axes[1]}")
        plt.show()

    def reroll(self):
        """Rerolls all random parameters."""
        for param in self.parameters:
            param.reroll()

    def subset(self, keys: List[Hashable], copy: bool = True):
        """Creates a new Molecule object containing only the specified atoms.

        Args:
            keys: A list of atom labels to include in the subset.

        Returns:
            Molecule: A new Molecule object containing only the specified atoms.

        """
        if copy is True:
            subset_coordinates = {key: self.coordinates[key].copy() for key in keys}
        else:
            subset_coordinates = {key: self.coordinates[key] for key in keys}
        subset_parameters = self.parameters
        return Molecule(coordinates=subset_coordinates, parameters=subset_parameters)

    def __setitem__(self, key: Hashable, item: Callable):
        if isinstance(item, np.ndarray):
            item = StaticVector(*item)
        self.coordinates[key] = item

    def __getitem__(self, key: Hashable) -> Callable:
        return self.coordinates[key]()

    def keys(self):
        return self.coordinates.keys()

    def values(self):
        return self.coordinates.values()

    def items(self):
        return self.coordinates.items()

    def __len__(self):
        return len(self.coordinates)

    def __iter__(self):
        return iter(self.coordinates)
