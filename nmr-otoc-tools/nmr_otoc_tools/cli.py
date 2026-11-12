# Copyright 2026 Google LLC
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Common CLI parsing tools and validators for NMR OTOC Tools."""

import argparse
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_SPINS = 2
"""Smallest system size: the two system spins (measurement and butterfly)."""


def add_shared_simulation_args(parser: argparse.ArgumentParser) -> None:
    """Adds shared simulation arguments to the provided parser.

    Args:
        parser: The argparse.ArgumentParser to populate.
    """
    parser.add_argument("--pdb_id", type=str, required=True)
    parser.add_argument("--n_avg", type=int, required=True)
    parser.add_argument("--n_min", type=int, required=True)
    parser.add_argument("--n_max", type=int, required=True)
    parser.add_argument("--max_time", type=float, default=2.0)
    parser.add_argument("--subfolder", type=str, default="")
    parser.add_argument("--n_trotter", type=int, required=False, default=20)
    parser.add_argument("--n_haar", type=int, required=False, default=20)
    parser.add_argument("--n_thread", type=int, required=False, default=1)
    parser.add_argument("--n_proc", type=int, required=False, default=1)
    parser.add_argument("--n_time", type=int, required=False, default=8)
    parser.add_argument(
        "--circuit_type",
        choices=["swap_network", "dense"],
        default="swap_network",
        help="Circuit construction: 'swap_network' (default) or 'dense' (make_otoc_circuit2).",
    )
    parser.add_argument(
        "--eta_hom",
        type=float,
        default=1.0,
        help="Scale factor for the homogeneous (same-spin) coupling matrix. Default: 1.0.",
    )
    parser.add_argument(
        "--eta_het",
        type=float,
        default=1.0,
        help="Scale factor for the heterogeneous (cross-spin) coupling matrix. Default: 1.0.",
    )


def validate_simulation_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validates the simulation arguments, raising parser.error if bounds are violated.

    Args:
        args: The parsed arguments namespace.
        parser: The parser instance to raise errors on.
    """
    if args.n_min > args.n_max:
        parser.error(f"--n_min ({args.n_min}) cannot be greater than --n_max ({args.n_max})")
    if args.n_min < MIN_SPINS:
        parser.error(f"--n_min must be at least {MIN_SPINS}.")
    for name in ("n_avg", "n_time", "n_trotter", "n_haar", "n_thread"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1.")
    if args.n_proc == 0:
        parser.error("--n_proc must be non-zero (negative values follow joblib: -1 = all cores).")
    if args.max_time <= 0:
        parser.error("--max_time must be positive.")


def add_verbose_arg(parser: argparse.ArgumentParser) -> None:
    """Adds the --verbose / -v argument to the provided parser."""
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging and detailed execution timing.",
    )


def configure_logging(*, verbose: bool = False) -> None:
    """Configures the root logger level and stream.

    Args:
        verbose: If True, sets logging level to DEBUG. Otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)


def exit_if_missing_dir(directory: Path, *, hint: str) -> None:
    """Exits with an error before any work is done if an output directory does not exist.

    Args:
        directory: The directory that must already exist.
        hint: Explanation appended to the error message (e.g. how to fix the path).
    """
    if not directory.is_dir():
        logger.error("Output directory does not exist: %s. %s", directory, hint)
        sys.exit(1)


@contextmanager
def log_duration(
    activity_name: str, in_logger: logging.Logger, log_level: int = logging.DEBUG
) -> Iterator[None]:
    """Context manager to log duration of execution for high-value segments.

    Args:
        activity_name: Label describing the timed task/segment.
        in_logger: The caller logger instance.
        log_level: Log level to write to (defaults to DEBUG).
    """
    start_time = time.perf_counter()
    yield
    duration = time.perf_counter() - start_time
    in_logger.log(log_level, "%s completed in %.4f seconds.", activity_name, duration)
