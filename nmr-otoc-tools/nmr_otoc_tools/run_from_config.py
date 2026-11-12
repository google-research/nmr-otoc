# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Run OTOC simulation from a serialized configuration file."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from nmr_otoc_tools.cli import add_verbose_arg, configure_logging, exit_if_missing_dir
from nmr_otoc_tools.serialization import load_simulation_config
from nmr_otoc_tools.simulation import run_parallel_simulation

logger = logging.getLogger(__name__)

SUB_FOLDER_HINT = (
    "The config's 'sub_folder' is relative to the directory nmr-setup-simulation was run from: "
    "run from that directory or pass --output_dir."
)


def main() -> None:
    """Main entry point for running simulation from config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to simulation config JSON")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for the result file. Defaults to the config's sub_folder.",
    )
    add_verbose_arg(parser)
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    config = load_simulation_config(config_path)
    output_dir = Path(args.output_dir) if args.output_dir else config.sub_folder
    exit_if_missing_dir(output_dir, hint=SUB_FOLDER_HINT)
    sim_params = config.sim_params

    logger.info("Running simulation from config for N=%d", sim_params.num_spins)

    t_range = np.asarray(sim_params.t_range)

    otoc_vs_time = run_parallel_simulation(
        sim_params,
        output_dir,
        config.dist_name,
        experiments=config.experiments,
        verbose=args.verbose,
    )

    result = {
        "N": sim_params.num_spins,
        "times": t_range.tolist(),
        "otocs": otoc_vs_time,
        "instance": config.pdb_id,
        "dist_folder": config.dist_name,
    }

    res_file = (
        output_dir
        / f"otoc_N{sim_params.num_spins}_t{sim_params.n_trotter}_h{sim_params.n_haar}.json"
    )
    res_file.write_text(json.dumps(result))
    logger.info("Saved results to %s", res_file)


if __name__ == "__main__":
    main()
