from __future__ import annotations

import argparse
import urllib.error

from gvns_smtwt_sdst import load_cicirello_instance
from metaheuristic_common import MetaheuristicResult


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "instance",
        help="Local .instance path, raw/GitHub URL, or Cicirello number 1-120.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=10.0,
        help="Wall-clock limit in seconds; 0 disables it (default: 10).",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=20_000_000,
        help="Objective-evaluation limit; 0 disables it (default: 20000000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print the ATCS initial solution and search diagnostics.",
    )
    return parser


def load_instance_or_exit(source: str):
    try:
        return load_cicirello_instance(source)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise SystemExit(f"Error while loading instance: {exc}") from exc


def print_result(result: MetaheuristicResult, *, details: bool = False) -> None:
    one_based = [job + 1 for job in result.sequence]
    print(f"Algorithm: {result.algorithm}")
    print(f"Final job sequence (1-based): {one_based}")
    print(f"Total weighted tardiness: {result.objective}")
    print(f"Runtime: {result.runtime_seconds:.6f} seconds")

    if details:
        initial_one_based = [job + 1 for job in result.initial_sequence]
        print(f"ATCS initial sequence (1-based): {initial_one_based}")
        print(f"ATCS initial objective: {result.initial_objective}")
        print(f"Objective evaluations: {result.objective_evaluations}")
        print(f"Iterations / generations: {result.iterations}")
