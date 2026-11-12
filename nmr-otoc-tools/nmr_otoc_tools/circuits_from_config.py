# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Generate Cirq circuits from a serialized simulation configuration file."""

import argparse
import logging
import sys
from pathlib import Path

import cirq
import numpy as np

from nmr_otoc_tools.cli import (
    add_verbose_arg,
    configure_logging,
    exit_if_missing_dir,
    log_duration,
)
from nmr_otoc_tools.run_from_config import SUB_FOLDER_HINT
from nmr_otoc_tools.serialization import SimulationConfig, load_simulation_config
from nmr_otoc_tools.simulation import (
    CircuitType,
    make_otoc_circuit,
    make_otoc_circuit2,
    trotter_step_counts,
)

logger = logging.getLogger(__name__)


def generate_circuits(
    config: SimulationConfig,
    output_dir: Path,
) -> None:
    """Generate and save Cirq circuits for all experiments and time steps.

    Args:
        config: The loaded simulation configuration.
        output_dir: Directory to save the generated circuit JSON files.
    """
    sim_params = config.sim_params
    experiments = config.experiments
    t_range = np.asarray(sim_params.t_range)
    step_counts = trotter_step_counts(t_range, sim_params.n_trotter)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing circuits to %s", str(output_dir))

    num_spins = sim_params.num_spins
    qubits = cirq.LineQubit.range(num_spins)

    circuit_builder = (
        make_otoc_circuit2 if sim_params.circuit_type == CircuitType.DENSE else make_otoc_circuit
    )

    total = 0
    for iavg, experiment in enumerate(experiments):
        for step_id, (expt_time, num_steps) in enumerate(zip(t_range, step_counts, strict=True)):
            with log_duration(
                f"    Trotter construction (N={num_spins}, steps={num_steps})", logger
            ):
                circuit = circuit_builder(experiment, expt_time, num_steps, qubits)

            circuit_path = output_dir / (
                f"circ{iavg}_N{num_spins}_t{sim_params.n_trotter}_{step_id}.json"
            )
            logger.info("Writing circuit %d, %d (avg, time)", iavg, step_id)
            circuit_path.write_text(cirq.to_json(circuit))
            total += 1

    logger.info(
        "Generated %d circuits (%d experiments x %d time steps) in %s",
        total,
        len(experiments),
        len(t_range),
        output_dir,
    )


def main() -> None:
    """Main entry point for generating circuits from config."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Cirq circuits from a simulation config without running the simulation."
        ),
    )
    parser.add_argument("--config", type=str, required=True, help="Path to simulation config JSON")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default=None,
        help="Directory to save generated circuits (created if needed). "
        "Defaults to the config's sub_folder, which must exist.",
    )
    add_verbose_arg(parser)
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    config = load_simulation_config(config_path)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.sub_folder
        exit_if_missing_dir(output_dir, hint=SUB_FOLDER_HINT)

    generate_circuits(config, output_dir)


if __name__ == "__main__":
    main()
