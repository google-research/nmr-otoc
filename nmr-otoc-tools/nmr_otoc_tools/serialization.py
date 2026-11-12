# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""JSON (de)serialization of OTOC experiments and simulation configuration files."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cirq
import numpy as np
from nmrotoc.dataclasses import NMROTOCExperiment

from nmr_otoc_tools.simulation import OTOCSimulation

GATE_MAP = {
    "X": cirq.X,
    "Y": cirq.Y,
    "Z": cirq.Z,
}
REVERSE_GATE_MAP = {v: k for k, v in GATE_MAP.items()}


def _serialize_gate(gate: cirq.Gate) -> str:
    """Serializes a simple Cirq gate to a string."""
    if gate in REVERSE_GATE_MAP:
        return REVERSE_GATE_MAP[gate]
    msg = f"Unsupported gate for simple serialization: {gate}"
    raise ValueError(msg)


def _deserialize_gate(gate_str: str) -> cirq.Gate:
    """Deserializes a string to a simple Cirq gate."""
    if gate_str in GATE_MAP:
        return GATE_MAP[gate_str]
    msg = f"Unsupported gate string: {gate_str}"
    raise ValueError(msg)


def _encode_matrix(mat: np.ndarray | None) -> list | dict[str, list] | None:
    """Encodes a coupling matrix for JSON.

    Real matrices are written as nested lists (the original format). Complex matrices, such as
    the MAS phased double-quantum couplings, are written as ``{"real": ..., "imag": ...}``
    because JSON has no complex number type.
    """
    if mat is None:
        return None
    if np.iscomplexobj(mat):
        return {"real": np.real(mat).tolist(), "imag": np.imag(mat).tolist()}
    return mat.tolist()


def _decode_matrix(raw: list | dict[str, list] | None) -> np.ndarray | None:
    """Decodes a coupling matrix written by :func:`_encode_matrix`.

    Nested lists become float64 arrays (the original behaviour); ``{"real", "imag"}`` mappings
    become complex128 arrays.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        missing = {"real", "imag"} - raw.keys()
        if missing:
            msg = f"Complex matrix encoding is missing key(s): {sorted(missing)}"
            raise ValueError(msg)
        real = np.array(raw["real"], dtype=np.float64)
        return real + 1j * np.array(raw["imag"], dtype=np.float64)
    return np.array(raw, dtype=np.float64)


def experiment_to_dict(exp: NMROTOCExperiment) -> dict[str, Any]:
    """Serializes an NMROTOCExperiment to a dictionary with restricted fields.

    Only fields necessary for the OTOC experiment are serialized:
    - bfly_ids
    - msmt_id
    - msmt_gate
    - bfly_gate
    - homogeneous_coupling_mat
    - heterogeneous_coupling_mat
    """
    return {
        "bfly_ids": exp.bfly_ids,
        "msmt_id": exp.msmt_id,
        "msmt_gate": _serialize_gate(exp.msmt_gate) if exp.msmt_gate is not None else None,
        "bfly_gate": _serialize_gate(exp.bfly_gate) if exp.bfly_gate is not None else None,
        "homogeneous_coupling_mat": _encode_matrix(exp.homogeneous_coupling_mat),
        "heterogeneous_coupling_mat": _encode_matrix(exp.heterogeneous_coupling_mat),
    }


def dict_to_experiment(d: dict[str, Any]) -> NMROTOCExperiment:
    """Deserializes a dictionary to an NMROTOCExperiment."""
    required_keys = [
        "bfly_ids",
        "msmt_id",
        "msmt_gate",
        "bfly_gate",
        "homogeneous_coupling_mat",
        "heterogeneous_coupling_mat",
    ]
    for key in required_keys:
        if key not in d:
            msg = f"Missing required serialization key '{key}' in experiment dictionary."
            raise ValueError(msg)

    msmt_gate = d["msmt_gate"]
    bfly_gate = d["bfly_gate"]
    return NMROTOCExperiment(
        bfly_ids=d["bfly_ids"],
        msmt_id=d["msmt_id"],
        msmt_gate=_deserialize_gate(msmt_gate) if msmt_gate is not None else None,
        bfly_gate=_deserialize_gate(bfly_gate) if bfly_gate is not None else None,
        homogeneous_coupling_mat=_decode_matrix(d["homogeneous_coupling_mat"]),
        heterogeneous_coupling_mat=_decode_matrix(d["heterogeneous_coupling_mat"]),
    )


def save_experiment_json(exp: NMROTOCExperiment, filepath: str | Path) -> None:
    """Saves a single NMROTOCExperiment to a JSON file."""
    d = experiment_to_dict(exp)
    with Path(filepath).open("w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def load_experiment_json(filepath: str | Path) -> NMROTOCExperiment:
    """Loads a single NMROTOCExperiment from a JSON file."""
    with Path(filepath).open(encoding="utf-8") as f:
        d = json.load(f)
    return dict_to_experiment(d)


@dataclass(frozen=True)
class SimulationConfig:
    """A parsed ``sim_config_N*.json`` file, as written by ``nmr-setup-simulation``.

    Attributes:
        sim_params: The simulation parameters.
        experiments: The experiments (one per averaging instance) to simulate.
        pdb_id: Identifier of the instance group the experiments were built from.
        dist_name: Name of the instance sub-folder (used in result files and logs).
        sub_folder: The instance folder. The stored value is relative to the working directory
            of ``nmr-setup-simulation`` (or absolute if that folder lies outside it); here it is
            resolved against the current working directory.
    """

    sim_params: OTOCSimulation
    experiments: list[NMROTOCExperiment]
    pdb_id: str
    dist_name: str
    sub_folder: Path


def save_simulation_config(
    path: Path,
    *,
    sim_params: OTOCSimulation,
    experiments: list[NMROTOCExperiment],
    pdb_id: str,
    dist_name: str,
    sub_folder: Path,
) -> None:
    """Writes a simulation configuration file.

    ``sub_folder`` is stored relative to the current working directory when it lies inside it,
    and as an absolute path otherwise.

    Args:
        path: Destination of the JSON file.
        sim_params: The simulation parameters.
        experiments: The experiments to serialize.
        pdb_id: Identifier of the instance group.
        dist_name: Name of the instance sub-folder.
        sub_folder: The instance folder.
    """
    cwd = Path.cwd().resolve()
    stored = str(sub_folder.relative_to(cwd)) if sub_folder.is_relative_to(cwd) else str(sub_folder)
    payload = {
        "sim_params": sim_params.to_dict(),
        "experiments": [experiment_to_dict(exp) for exp in experiments],
        "pdb_id": pdb_id,
        "dist_name": dist_name,
        "sub_folder": stored,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_simulation_config(path: Path) -> SimulationConfig:
    """Reads a simulation configuration file written by :func:`save_simulation_config`.

    Args:
        path: The JSON file to read.

    Returns:
        The parsed configuration, with ``sub_folder`` resolved against the current working
        directory.
    """
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return SimulationConfig(
        sim_params=OTOCSimulation.from_dict(raw["sim_params"]),
        experiments=[dict_to_experiment(d) for d in raw["experiments"]],
        pdb_id=raw["pdb_id"],
        dist_name=raw["dist_name"],
        sub_folder=(Path.cwd().resolve() / Path(raw["sub_folder"])).resolve(),
    )
