# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

from dataclasses import dataclass, field
import hashlib
from typing import List, Optional, Callable, Tuple, Union
import numpy as np
import pickle

import cirq


def hash_ndarray(array):
    """Hashes a NumPy array using SHA256.

    Args:
        array: The NumPy array to hash.

    Returns:
        str: The hexadecimal representation of the SHA256 hash.
    """

    # Convert the array to bytes. The 'C' order ensures consistent hashing across different architectures.
    array_bytes = array.tobytes("C")

    # Create a SHA256 hash object
    sha256 = hashlib.sha256()

    # Update the hash object with the array bytes
    sha256.update(array_bytes)

    # Return the hexadecimal representation of the hash
    return sha256.hexdigest()


def hash_circuit(circuit):
    """Hashes a cirq.Circuit using the freeze function"""

    return hash(cirq.align_left(circuit).freeze())


class SelfSavingClass:

    def save(self, filename: str):
        """Saves the OTOCCircuitObj object to a file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str):
        """Loads an OTOCCircuitObj object from a file."""
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        return obj


@dataclass
class OTOCCircuitObj(SelfSavingClass):
    """Dataclass for an OTOC circuit."""

    unitary: cirq.Circuit = None  # type: ignore
    """The unitary circuit to evolve the system."""

    inverse_unitary: Optional[cirq.Circuit] = None  # type: ignore
    """The unitary to evolve the system backward, if different from cirq.inverse(unitary)"""

    butterfly: cirq.Circuit = None  # type: ignore
    """The butterfly circuit to apply the perturbation."""

    measurement_diagonalization: cirq.Gate = None  # type: ignore
    """Gate to diagonalize the measurement operator"""

    measurement_qubit: cirq.Qid = None  # type: ignore
    """The qubit to measure for the OTOC."""

    qubits: List[cirq.Qid] = None  # type: ignore
    """The qubits that the experiment acts on"""

    prep_circuit: cirq.Circuit = None  # type: ignore
    """Circuit to prepare state that the OTOC is performed on."""


@dataclass
class ProductFormulaObj(SelfSavingClass):
    """Dataclass for general product formula parameters."""

    num_steps: int = None  # type: ignore
    """Number of Trotter steps."""

    generating_function: Callable = None  # type: ignore
    """Function that consumes this class and a list of qubits
    and generates a cirq.Circuit and a qubit relabeling dictionary"""

    concatenate: bool = True
    """Determines whether to compile circuit using Circuit.append or
    nmrotoc.circuits.compilation.safe_concatenate_circuits. The latter
    minimizes the single-qubit gate count by merging subsequent single
    qubit gates."""


@dataclass
class NestedProductFormulaObj(ProductFormulaObj):
    """Dataclass for product formula defined by a list of ProductFormulaObj"""

    sub_terms: List[ProductFormulaObj] = None  # type: ignore
    """List of terms in product formula"""

    qid_list: List[List[int]] = None  # type: ignore
    """List of qubit indices to pass into each sub_term"""

    num_steps: int = 1
    """List of times to iterate over sub_terms"""


@dataclass
class OrderedProductFormulaObj(ProductFormulaObj):
    """Dataclass for product formula defined by an ordered list of gates in each step"""

    gate_ordering: List[List[Tuple]] = None  # type: ignore
    """List for each step of gates to implement during that step, defined as a list of
    tuples of indices"""


@dataclass
class SwapNetworkProductFormulaObj(ProductFormulaObj):
    """Wrapper for a product formula based on a swap network"""

    alternate_order: Optional[bool] = False
    """Flag for whether to perform interactions in the same order (given by
    the swap network, or in reversed order every second step)"""

    reverse: Optional[bool] = False
    """Flag for whether to start the product formula in reverse order
    (by switching the order of the layers of gates)"""

    permutations: Optional[List[List[int]]] = None
    """List of permutations of qubits defining different swap networks for different
    product formula steps (e.g. default is range(len(qubits)))"""

    compress: Optional[bool] = True
    """Whether to run cirq functions to compress the output circuit or not."""


@dataclass
class DoubleQuantumObj(ProductFormulaObj):
    """Wrapper for a product formula for a double quantum Hamiltonian"""

    homogeneous_step_mats: List[np.ndarray] = None  # type: ignore
    """List of homogeneous coupling matrices for individual steps in an
    all-to-all product formula."""

    heterogeneous_step_mats: List[np.ndarray] = None  # type: ignore
    """List of heterogeneous coupling matrices for individual steps in an
    all-to-all product formula."""

    z_step_mats: List[np.ndarray] = None  # type: ignore
    """List of single-qubit phases to be applied to individual steps in an
    all-to-all product formula."""


@dataclass
class DoubleQuantumSwapNetworkObj(DoubleQuantumObj, SwapNetworkProductFormulaObj):
    """Wrapper for a swap network product formula that uses a double quantum Hamiltonian"""

    xyyx_flag: bool = False


@dataclass
class DoubleQuantumOrderedProductObj(DoubleQuantumObj, OrderedProductFormulaObj):
    """Wrapper for an ordered product formula that uses a double quantum
    Hamiltonian"""

    pass


@dataclass
class DoubleQuantumXZYObj(DoubleQuantumObj):
    """Wrapper for an XZY product formula that uses a double quantum Hamiltonian"""

    alternate_order: Optional[bool] = False
    """Flag for whether to perform the XZY layers in reversed (YZX) order every second step."""

    reverse: Optional[bool] = False
    """Flag for whether to start the product formula in reversed (YZX) order."""



@dataclass
class SimulatorWrapper:
    """Wrapper for a cirq simulator with simulation details"""

    rng: np.random.RandomState = None  # type: ignore
    """Random number generator for this experiment"""

    simulator: cirq.SimulatesExpectationValues = None  # type: ignore
    """The simulator to run the circuit."""

    num_states_to_average: int = None  # type: ignore
    """The number of states to average over for the infinite temperature OTOC."""

    noise_model: Callable = None  # type: ignore
    """Noise to apply to a circuit before simulation"""


@dataclass
class NMROTOCExperiment(SelfSavingClass):
    """Dataclass for an NMR OTOC experiment."""

    t_range: np.ndarray = None  # type: ignore
    """Times to evolve the OTOC for"""

    bfly_ids: List[int] = field(default_factory=list)
    """Indexes for qubits corresponding to butterflies."""

    bfly_gate: cirq.Gate = None  # type: ignore
    """Gate to apply for the butterfly."""

    bfly_gate_list: List[cirq.Gate] = None  # type: ignore
    """Gates to apply to individual qubits for the case where these are different.
    Overrides NMROTOCExperimnet.bfly_gate when present."""

    msmt_id: int = None  # type: ignore
    """Index for measurement spin."""

    msmt_gate: cirq.Gate = None  # type: ignore
    """Operator to measure as a cirq.Gate"""

    onsite_vector: np.ndarray = None  # type: ignore
    """Vector containing any onsite interactions to be added"""

    xyyx_flag: bool = False
    """Flag whether homogeneous evolution is by an XX - YY (DREAM-2) or XY + YX
    (DREAM-1) term"""

    homogeneous_coupling_mat: np.ndarray = None  # type: ignore
    """Matrix for same-spin dipolar couplings. These will couple with an exp(idt * h *
    [XX-YY]) interaction."""

    heterogeneous_coupling_mat: np.ndarray = None  # type: ignore
    """Matrix for different-spin dipolar couplings. Due to the strong background magnetic
    field (chemical shifts between different spins are usually many MHz) these
    experience almost exclusively a ZZ interaction exp(idt * g * ZZ)."""

    homogeneous_J_coupling_mat: np.ndarray = None  # type: ignore
    """Matrix for same-spin J-couplings. These couple with a Heisenberg-like interaction"""

    heterogeneous_J_coupling_mat: np.ndarray = None  # type: ignore
    """Matrix for different-spin J-couplings. These couple with an Ising-like interaction"""
