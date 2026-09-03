from __future__ import annotations

import argparse
import csv
from pathlib import Path

from aco_smtwt_sdst import solve_aco
from dde_smtwt_sdst import solve_dde
from dpso_smtwt_sdst import solve_dpso
from ga_smtwt_sdst import solve_ga
from run_common import load_instance_or_exit
from sa_smtwt_sdst import solve_sa
from ts_smtwt_sdst import solve_ts


SOLVERS = {
    "SA": solve_sa,
    "GA": solve_ga,
    "TS": solve_ts,
    "ACO": solve_aco,
    "DPSO": solve_dpso,
    "DDE": solve_dde,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple metaheuristics on the same SMTWT-SDST instance."
    )
    parser.add_argument(
        "instance",
        help="Local .instance path, raw/GitHub URL, or Cicirello number 1-120.",
    )
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--max-evaluations", type=int, default=20_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=list(SOLVERS),
        default=list(SOLVERS),
        help="Algorithms to run (default: all six).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path for a CSV summary.",
    )
    args = parser.parse_args()

    instance = load_instance_or_exit(args.instance)
    results = []

    for name in args.algorithms:
        solver = SOLVERS[name]
        result = solver(
            instance,
            time_limit=args.time_limit,
            max_evaluations=args.max_evaluations,
            seed=args.seed,
        )
        results.append(result)

    print()
    print(
        f"{'Algorithm':<10} {'Objective':>14} {'Runtime(s)':>12} "
        f"{'Evaluations':>14} {'Iterations':>12}"
    )
    print("-" * 68)
    for result in sorted(results, key=lambda r: r.objective):
        print(
            f"{result.algorithm:<10} {result.objective:>14} "
            f"{result.runtime_seconds:>12.4f} "
            f"{result.objective_evaluations:>14} "
            f"{result.iterations:>12}"
        )

    if args.csv is not None:
        with args.csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "algorithm",
                    "objective",
                    "runtime_seconds",
                    "objective_evaluations",
                    "iterations",
                    "initial_objective",
                    "sequence_1_based",
                ]
            )
            for result in results:
                writer.writerow(
                    [
                        result.algorithm,
                        result.objective,
                        result.runtime_seconds,
                        result.objective_evaluations,
                        result.iterations,
                        result.initial_objective,
                        " ".join(str(job + 1) for job in result.sequence),
                    ]
                )
        print(f"\nCSV saved to: {args.csv}")


if __name__ == "__main__":
    main()
