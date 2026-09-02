from __future__ import annotations

import argparse
import csv
from pathlib import Path

from aco_smtwt_sdst import solve_aco
from dataset_io import ReducedInstance, load_reduced_dataset
from dde_smtwt_sdst import solve_dde
from dpso_smtwt_sdst import solve_dpso
from ga_smtwt_sdst import solve_ga
from gvns_smtwt_sdst import solve_gvns
from sa_smtwt_sdst import solve_sa
from ts_smtwt_sdst import solve_ts

SOLVERS = {
    "SA": solve_sa,
    "GA": solve_ga,
    "TS": solve_ts,
    "ACO": solve_aco,
    "DPSO": solve_dpso,
    "DDE": solve_dde,
    "GVNS": solve_gvns,
}
RAW_FIELDS = [
    "algorithm", "instance", "tau", "R", "eta", "seed",
    "runtime_sec", "best_twt", "improvement_vs_ATCS",
]
CONV_FIELDS = [
    "algorithm", "instance", "tau", "R", "eta", "seed", "elapsed_sec", "best_twt",
]


def _read_atcs(path: Path) -> dict[str, int]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["instance"]: int(row["ATCS_twt"]) for row in rows}
    if len(result) != 40:
        raise ValueError(f"Expected 40 ATCS baselines, found {len(result)} in {path}.")
    return result


def _completed_keys(raw_path: Path, conv_path: Path) -> set[tuple[str, str, int]]:
    raw: set[tuple[str, str, int]] = set()
    conv_counts: dict[tuple[str, str, int], int] = {}

    if raw_path.exists():
        with raw_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                raw.add((row["algorithm"], row["instance"], int(row["seed"])))
    if conv_path.exists():
        with conv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                key = (row["algorithm"], row["instance"], int(row["seed"]))
                conv_counts[key] = conv_counts.get(key, 0) + 1

    return {key for key in raw if conv_counts.get(key) == 11}


def _ensure_header(path: Path, fields: list[str], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not path.exists():
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()


def _forward_fill(history: tuple[tuple[float, int], ...], fallback: int) -> list[int]:
    """Sample incumbent-best history at integer seconds 0..10 by forward fill."""
    events = sorted(history, key=lambda x: x[0])
    values: list[int] = []
    index = 0
    current = int(events[0][1]) if events else int(fallback)
    for second in range(11):
        while index < len(events) and events[index][0] <= second + 1e-12:
            current = int(events[index][1])
            index += 1
        values.append(current)
    return values


def _run_one(item: ReducedInstance, algorithm: str, seed: int, time_limit: float):
    solver = SOLVERS[algorithm]
    # Fairness: the shared wall-clock budget is the active stopping condition.
    # max_evaluations=None avoids a fast implementation stopping before 10 seconds.
    return solver(item.problem, time_limit=time_limit, max_evaluations=None, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all stochastic SMTWT-SDST metaheuristics and create raw/convergence CSV files."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/reduced_dataset.json"))
    parser.add_argument("--baselines", type=Path, default=Path("results/baseline_results.csv"))
    parser.add_argument("--raw-output", type=Path, default=Path("results/results_raw.csv"))
    parser.add_argument("--convergence-output", type=Path, default=Path("results/convergence.csv"))
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
        help="Common random seeds for every algorithm/instance (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--algorithms", nargs="+", choices=list(SOLVERS), default=list(SOLVERS),
        help="Algorithms to run (default: SA GA TS ACO DPSO DDE GVNS).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Discard existing output files instead of resuming completed runs.",
    )
    args = parser.parse_args()

    if args.time_limit <= 0:
        raise ValueError("--time-limit must be positive.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")

    instances = load_reduced_dataset(args.dataset)
    atcs = _read_atcs(args.baselines)

    if args.overwrite:
        completed: set[tuple[str, str, int]] = set()
    else:
        completed = _completed_keys(args.raw_output, args.convergence_output)

    _ensure_header(args.raw_output, RAW_FIELDS, args.overwrite)
    _ensure_header(args.convergence_output, CONV_FIELDS, args.overwrite)

    total = len(instances) * len(args.algorithms) * len(args.seeds)
    done = len(completed)
    print(f"Planned runs: {total}; already complete: {done}; remaining: {total - done}")

    with args.raw_output.open("a", newline="", encoding="utf-8-sig") as raw_handle, \
         args.convergence_output.open("a", newline="", encoding="utf-8-sig") as conv_handle:
        raw_writer = csv.DictWriter(raw_handle, fieldnames=RAW_FIELDS)
        conv_writer = csv.DictWriter(conv_handle, fieldnames=CONV_FIELDS)

        for item in instances:
            atcs_twt = atcs[item.instance]
            for algorithm in args.algorithms:
                for seed in args.seeds:
                    key = (algorithm, item.instance, seed)
                    if key in completed:
                        continue

                    result = _run_one(item, algorithm, seed, args.time_limit)
                    best_twt = int(result.objective)
                    improvement = 0.0 if atcs_twt == 0 else (atcs_twt - best_twt) / atcs_twt * 100.0

                    raw_writer.writerow(
                        {
                            "algorithm": algorithm,
                            "instance": item.instance,
                            "tau": item.tau,
                            "R": item.R,
                            "eta": item.eta,
                            "seed": seed,
                            "runtime_sec": f"{result.runtime_seconds:.6f}",
                            "best_twt": best_twt,
                            "improvement_vs_ATCS": f"{improvement:.8f}",
                        }
                    )
                    raw_handle.flush()

                    sampled = _forward_fill(result.history, best_twt)
                    for second, value in enumerate(sampled):
                        conv_writer.writerow(
                            {
                                "algorithm": algorithm,
                                "instance": item.instance,
                                "tau": item.tau,
                                "R": item.R,
                                "eta": item.eta,
                                "seed": seed,
                                "elapsed_sec": second,
                                "best_twt": value,
                            }
                        )
                    conv_handle.flush()

                    done += 1
                    print(
                        f"[{done}/{total}] {algorithm:4s} {item.instance:12s} seed={seed:<4d} "
                        f"best={best_twt} runtime={result.runtime_seconds:.3f}s"
                    )

    print(f"Saved raw results to {args.raw_output}")
    print(f"Saved convergence data to {args.convergence_output}")


if __name__ == "__main__":
    main()
