# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Setup OTOC simulation by serializing configurations."""

import argparse
import logging
from pathlib import Path

import numpy as np

from nmr_otoc_tools.cli import (
    add_shared_simulation_args,
    add_verbose_arg,
    configure_logging,
    validate_simulation_args,
)
from nmr_otoc_tools.serialization import save_simulation_config
from nmr_otoc_tools.simulation import OTOCSimulation, create_experiment

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for setting up simulation configs."""
    parser = argparse.ArgumentParser()
    add_shared_simulation_args(parser)
    add_verbose_arg(parser)
    args = parser.parse_args()
    validate_simulation_args(args, parser)

    configure_logging(verbose=args.verbose)

    instance_root = Path.cwd().resolve() / "data" / "generated_instances" / args.pdb_id
    sub_folder = instance_root / args.subfolder
    dist_name = sub_folder.name

    t_range = np.linspace(args.max_time / args.n_time, args.max_time, args.n_time)

    for num_spins in range(args.n_min, args.n_max + 1):
        sim_params = OTOCSimulation(
            num_spins=num_spins,
            t_range=tuple(t_range),
            n_trotter=args.n_trotter,
            n_haar=args.n_haar,
            n_thread=args.n_thread,
            n_proc=args.n_proc,
            save_circuit=False,
            circuit_type=args.circuit_type,
            eta_hom=args.eta_hom,
            eta_het=args.eta_het,
        )

        experiments = []
        for i in range(args.n_avg):
            exp = create_experiment(i, sim_params, sub_folder, dist_name)
            if exp is not None:
                experiments.append(exp)

        if not experiments:
            logger.warning("No experiments created for N=%d", num_spins)
            continue

        config_file = sub_folder / f"sim_config_N{num_spins}.json"
        save_simulation_config(
            config_file,
            sim_params=sim_params,
            experiments=experiments,
            pdb_id=args.pdb_id,
            dist_name=dist_name,
            sub_folder=sub_folder,
        )
        logger.info("Saved config to %s", config_file)


if __name__ == "__main__":
    main()
