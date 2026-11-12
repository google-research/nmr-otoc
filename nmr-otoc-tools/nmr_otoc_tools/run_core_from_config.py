# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Run partition simulations of n_core + 1 spins from a configuration file."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from nmrotoc.dataclasses import NMROTOCExperiment

from nmr_otoc_tools.cli import add_verbose_arg, configure_logging, exit_if_missing_dir
from nmr_otoc_tools.run_from_config import SUB_FOLDER_HINT
from nmr_otoc_tools.serialization import load_simulation_config
from nmr_otoc_tools.simulation import OTOCSimulation, run_parallel_simulation

logger = logging.getLogger(__name__)


def slice_experiment(exp: NMROTOCExperiment, n_core: int, k: int) -> NMROTOCExperiment:
    """Slices the experiment coupling matrices to keep only n_core spins and environment spin k.

    Args:
        exp: The original NMROTOCExperiment.
        n_core: Number of core spins.
        k: The index of the environment spin to include.

    Returns:
        A new NMROTOCExperiment containing only the sliced system.
    """
    keep_indices = [*range(n_core), k]

    new_hom = (
        exp.homogeneous_coupling_mat[np.ix_(keep_indices, keep_indices)]
        if exp.homogeneous_coupling_mat is not None
        else None
    )
    new_het = (
        exp.heterogeneous_coupling_mat[np.ix_(keep_indices, keep_indices)]
        if exp.heterogeneous_coupling_mat is not None
        else None
    )

    return NMROTOCExperiment(
        bfly_ids=exp.bfly_ids,
        msmt_id=exp.msmt_id,
        msmt_gate=exp.msmt_gate,
        bfly_gate=exp.bfly_gate,
        homogeneous_coupling_mat=new_hom,
        heterogeneous_coupling_mat=new_het,
    )


def main() -> None:
    """Main entry point for running core partition simulations from config."""
    parser = argparse.ArgumentParser(
        description="Run OTOC simulations of n_core + 1 spins from config."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to simulation config JSON")
    parser.add_argument(
        "--n-core",
        dest="n_core",
        type=int,
        required=True,
        help="Number of core spins (n_core < n)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for the result files. Defaults to the config's sub_folder.",
    )
    add_verbose_arg(parser)
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    config = load_simulation_config(config_path)
    sim_params = config.sim_params
    experiments = config.experiments
    pdb_id = config.pdb_id
    dist_name = config.dist_name

    n = sim_params.num_spins
    n_core = args.n_core

    # Validation
    if n_core >= n:
        parser.error(
            f"--n-core ({n_core}) must be strictly less than the number of spins "
            f"in the config ({n})."
        )

    all_indices = []
    for exp in experiments:
        all_indices.extend(exp.bfly_ids)
        all_indices.append(exp.msmt_id)
    min_required_core = max(all_indices) + 1

    if n_core < min_required_core:
        parser.error(
            f"--n-core ({n_core}) must be at least {min_required_core} "
            f"to contain all measurement and butterfly spins in the config."
        )

    output_dir = Path(args.output_dir) if args.output_dir else config.sub_folder
    exit_if_missing_dir(output_dir, hint=SUB_FOLDER_HINT)

    logger.info(
        "Running %d partition simulations of size %d (core size %d + 1 env spin) from config",
        n - n_core,
        n_core + 1,
        n_core,
    )

    t_range = np.asarray(sim_params.t_range)

    # Reconstruct sim_params with num_spins = n_core + 1
    sim_dict = sim_params.to_dict()
    sim_dict["num_spins"] = n_core + 1
    sub_sim_params = OTOCSimulation.from_dict(sim_dict)

    for k in range(n_core, n):
        logger.info("Running simulation for environment spin k=%d...", k)
        sub_experiments = [slice_experiment(exp, n_core, k) for exp in experiments]

        otoc_vs_time = run_parallel_simulation(
            sub_sim_params,
            output_dir,
            dist_name,
            experiments=sub_experiments,
            verbose=args.verbose,
        )

        result = {
            "N": n_core + 1,
            "n_core": n_core,
            "env_spin": k,
            "times": t_range.tolist(),
            "otocs": otoc_vs_time,
            "instance": pdb_id,
            "dist_folder": dist_name,
        }

        filename = (
            f"otoc_N{n_core + 1}_core{n_core}_env{k}_"
            f"t{sim_params.n_trotter}_h{sim_params.n_haar}.json"
        )
        res_file = output_dir / filename
        res_file.write_text(json.dumps(result))
        logger.info("Saved partition simulation results to %s", res_file)


if __name__ == "__main__":
    main()
