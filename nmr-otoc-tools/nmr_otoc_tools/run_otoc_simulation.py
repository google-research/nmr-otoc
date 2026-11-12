# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Run an OTOC simulation."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from nmr_otoc_tools.cli import (
    add_shared_simulation_args,
    add_verbose_arg,
    configure_logging,
    validate_simulation_args,
)
from nmr_otoc_tools.simulation import (
    OTOCSimulation,
    run_parallel_simulation,
)

logger = logging.getLogger(__name__)


def process_sub_instance(
    sub_folder: Path,
    args: argparse.Namespace,
) -> None:
    """Runs the OTOC simulation for a specific sub-instance (distance/configuration).

    Args:
        sub_folder: Path to the sub-folder containing instance data.
        args: Parsed command-line arguments (see :func:`run_instance`).
    """
    pdb_id = args.pdb_id
    n_avg = args.n_avg
    n_min = args.n_min
    n_max = args.n_max
    max_time = args.max_time
    n_time = args.n_time

    dist_name = sub_folder.name
    logger.info("--- Processing %s in Instance %s ---", dist_name, pdb_id)

    t_range = np.linspace(max_time / n_time, max_time, n_time)

    for num_spins in range(n_min, n_max + 1):
        sim_params = OTOCSimulation(
            num_spins=num_spins,
            t_range=tuple(t_range),
            n_trotter=args.n_trotter,
            n_haar=args.n_haar,
            n_thread=args.n_thread,
            n_proc=args.n_proc,
            save_circuit=args.save_circuit,
            circuit_type=args.circuit_type,
            eta_hom=args.eta_hom,
            eta_het=args.eta_het,
        )
        sim_params.dump(logger)
        verbose = getattr(args, "verbose", False)
        otoc_vs_time = run_parallel_simulation(
            sim_params, sub_folder, dist_name, n_avg=n_avg, verbose=verbose
        )

        result = {
            "N": num_spins,
            "times": t_range.tolist(),
            "otocs": otoc_vs_time,
            "instance": pdb_id,
            "dist_folder": dist_name,
        }
        res_file = sub_folder / (
            f"otoc_N{num_spins}_t{sim_params.n_trotter}_h{sim_params.n_haar}.json"
        )
        res_file.write_text(json.dumps(result))


def run_instance(
    args: argparse.Namespace,
) -> None:
    """Sets up paths and triggers the simulation for a given instance.

    Args:
        args: Parsed command-line arguments. Requires the fields defined by
            :func:`nmr_otoc_tools.cli.add_shared_simulation_args` (``pdb_id``, ``n_avg``,
            ``n_min``, ``n_max``, ``max_time``, ``subfolder``, ``n_trotter``, ``n_haar``,
            ``n_thread``, ``n_proc``, ``n_time``, ``circuit_type``, ``eta_hom``, ``eta_het``)
            and ``save_circuit``; ``verbose`` (from
            :func:`nmr_otoc_tools.cli.add_verbose_arg`) is optional.
    """
    instance_root = Path.cwd().resolve() / "data" / "generated_instances" / args.pdb_id
    sub_folder = instance_root / args.subfolder
    process_sub_instance(
        sub_folder,
        args,
    )


def main() -> None:
    """Main entry point for running NMR OTOC simulations via command line."""
    parser = argparse.ArgumentParser()
    add_shared_simulation_args(parser)
    add_verbose_arg(parser)
    parser.add_argument("--save_circuit", required=False, default=False, action="store_true")
    args = parser.parse_args()
    validate_simulation_args(args, parser)
    configure_logging(verbose=args.verbose)
    logger.info(
        "Starting job with instance_id=%s, n_min=%d, n_max=%d, max_time=%s",
        args.pdb_id,
        args.n_min,
        args.n_max,
        args.max_time,
    )
    run_instance(args)


if __name__ == "__main__":
    main()
