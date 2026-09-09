from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from statistics import mean, stdev

from algorithms.aco_smtwt_sdst import solve_aco
from algorithms.dde_smtwt_sdst import solve_dde
from algorithms.dpso_smtwt_sdst import solve_dpso
from algorithms.ga_smtwt_sdst import solve_ga
from algorithms.sa_smtwt_sdst import solve_sa
from algorithms.ts_smtwt_sdst import solve_ts
from dataset_io import ReducedInstance, load_dataset


# -----------------------------------------------------------------------------
# 1. Solvers
# -----------------------------------------------------------------------------

SOLVERS = {
    "ACO": solve_aco,
    "SA": solve_sa,
    "GA": solve_ga,
    "TS": solve_ts,
    "DDE": solve_dde,
    "DPSO": solve_dpso,
}


# -----------------------------------------------------------------------------
# 2. Small, practical tuning grids
#
# Total configurations
#   SA   : 3
#   GA   : 2 x 2 x 2 = 8
#   TS   : 3 x 2     = 6
#   DDE  : 2 x 3     = 6
#   DPSO : 2 x 3     = 6
#   ACO  : 2 x 2 x 2 = 8 (8 instances x 3 seeds = 192 runs)
#
# With 8 tuning instances, 3 seeds, and 10 sec/run:
# one configuration = 8 x 3 x 10 sec = about 4 minutes of search time.
# -----------------------------------------------------------------------------

PARAM_GRIDS: dict[str, list[dict[str, int | float]]] = {
    "ACO": [
        {"alpha": alpha, "beta": beta, "evaporation_rate": evaporation_rate}
        for alpha, beta, evaporation_rate in product((1, 2), (2, 4), (0.1, 0.3))
    ],
    "SA": [
        {"cooling_rate": cooling_rate}
        for cooling_rate in (0.90, 0.95, 0.99)
    ],

    "GA": [
        {
            "population_size": population_size,
            "mutation_rate": mutation_rate,
            "tournament_size": tournament_size,
        }
        for population_size, mutation_rate, tournament_size in product(
            (30, 50),
            (0.10, 0.20),
            (2, 4),
        )
    ],

    "TS": [
        {
            "tabu_tenure": tabu_tenure,
            "neighborhood_sample": neighborhood_sample,
        }
        for tabu_tenure, neighborhood_sample in product(
            (5, 10, 20),
            (150, 300),
        )
    ],

    "DDE": [
        {
            "population_size": population_size,
            # F / ff in the tuning plan
            "differential_weight": differential_weight,
        }
        for population_size, differential_weight in product(
            (30, 50),
            (0.40, 0.60, 0.80),
        )
    ],

    "DPSO": [
        {
            "swarm_size": swarm_size,
            # w / ww in the tuning plan
            "inertia_probability": inertia_probability,
        }
        for swarm_size, inertia_probability in product(
            (30, 50),
            (0.30, 0.50, 0.70),
        )
    ],
}


# Parameters not being tuned remain at the defaults already defined
# inside each solver, e.g. GA crossover_rate=0.90, DDE crossover_rate=0.90,
# DPSO cognitive/social/mutation probabilities, etc.

PARAM_COLUMNS = [
    "cooling_rate",
    "population_size",
    "mutation_rate",
    "tournament_size",
    "tabu_tenure",
    "neighborhood_sample",
    "differential_weight",
    "swarm_size",
    "inertia_probability",
    "alpha",
    "beta",
    "evaporation_rate",
]

RAW_FIELDS = [
    "algorithm",
    "config_id",
    "instance",
    "class_id",
    "tau",
    "R",
    "eta",
    "seed",
    *PARAM_COLUMNS,
    "runtime_sec",
    "ATCS_twt",
    "best_twt",
    "improvement_vs_ATCS",
]

SUMMARY_FIELDS = [
    "algorithm",
    "config_id",
    *PARAM_COLUMNS,
    "n_runs",
    "mean_improvement_vs_ATCS",
    "std_improvement_vs_ATCS",
    "mean_best_twt",
    "mean_runtime_sec",
]


# -----------------------------------------------------------------------------
# 3. Helpers
# -----------------------------------------------------------------------------


def _value_token(value: int | float) -> str:
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


def _config_id(algorithm: str, params: dict[str, int | float]) -> str:
    short_names = {
        "cooling_rate": "CR",
        "population_size": "P",
        "mutation_rate": "M",
        "tournament_size": "T",
        "tabu_tenure": "TT",
        "neighborhood_sample": "CS",
        "differential_weight": "F",
        "swarm_size": "S",
        "inertia_probability": "W",
        "alpha": "A",
        "beta": "B",
        "evaporation_rate": "ER",
    }

    pieces = [algorithm]
    for key, value in params.items():
        pieces.append(f"{short_names[key]}{_value_token(value)}")
    return "_".join(pieces)


def _ensure_header(path: Path, fields: list[str], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not path.exists():
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()
    else:
        # Upgrade older tuning CSVs before appending rows with new parameter columns.
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames == fields:
                return
            if set(reader.fieldnames or []) - set(fields):
                raise ValueError(f"Unexpected tuning CSV columns in {path}")
            rows = list(reader)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def _completed_keys(path: Path) -> set[tuple[str, str, int]]:
    """Return completed (config_id, instance, seed) triples."""
    if not path.exists():
        return set()

    completed: set[tuple[str, str, int]] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            completed.add(
                (
                    row["config_id"],
                    row["instance"],
                    int(row["seed"]),
                )
            )
    return completed


def _parameter_cells(params: dict[str, int | float]) -> dict[str, int | float | str]:
    return {
        column: params.get(column, "")
        for column in PARAM_COLUMNS
    }


def _run_one(
    item: ReducedInstance,
    algorithm: str,
    params: dict[str, int | float],
    seed: int,
    time_limit: float,
):
    solver = SOLVERS[algorithm]

    # Wall-clock time is the active budget for fairness.
    # Disabling max_evaluations prevents a configuration from stopping
    # before the common time limit merely because it evaluates solutions faster.
    return solver(
        item.problem,
        time_limit=time_limit,
        max_evaluations=None,
        seed=seed,
        **params,
    )


def _read_relevant_rows(
    raw_path: Path,
    algorithm: str,
    valid_config_ids: set[str],
    valid_instances: set[str],
    valid_seeds: set[int],
) -> list[dict[str, str]]:
    if not raw_path.exists():
        return []

    rows: list[dict[str, str]] = []
    with raw_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["algorithm"] != algorithm:
                continue
            if row["config_id"] not in valid_config_ids:
                continue
            if row["instance"] not in valid_instances:
                continue
            if int(row["seed"]) not in valid_seeds:
                continue
            rows.append(row)
    return rows


def _build_summary_and_best(
    *,
    algorithm: str,
    instances: list[ReducedInstance],
    seeds: list[int],
    raw_path: Path,
    summary_path: Path,
    best_params_path: Path,
) -> None:
    configs = PARAM_GRIDS[algorithm]
    config_map = {
        _config_id(algorithm, params): params
        for params in configs
    }

    rows = _read_relevant_rows(
        raw_path=raw_path,
        algorithm=algorithm,
        valid_config_ids=set(config_map),
        valid_instances={item.instance for item in instances},
        valid_seeds=set(seeds),
    )

    grouped: dict[str, list[dict[str, str]]] = {
        config_id: [] for config_id in config_map
    }
    for row in rows:
        grouped[row["config_id"]].append(row)

    expected_runs = len(instances) * len(seeds)
    summary_rows: list[dict[str, object]] = []

    for config_id, params in config_map.items():
        group = grouped[config_id]
        if not group:
            continue

        improvements = [float(row["improvement_vs_ATCS"]) for row in group]
        best_twts = [int(row["best_twt"]) for row in group]
        runtimes = [float(row["runtime_sec"]) for row in group]

        summary_rows.append(
            {
                "algorithm": algorithm,
                "config_id": config_id,
                **_parameter_cells(params),
                "n_runs": len(group),
                "mean_improvement_vs_ATCS": mean(improvements),
                "std_improvement_vs_ATCS": (
                    stdev(improvements) if len(improvements) >= 2 else 0.0
                ),
                "mean_best_twt": mean(best_twts),
                "mean_runtime_sec": mean(runtimes),
            }
        )

    # Best configurations first. Incomplete configs are still written to the
    # summary, but are never eligible to be selected as best.
    summary_rows.sort(
        key=lambda row: (
            -int(row["n_runs"] == expected_runs),
            -float(row["mean_improvement_vs_ATCS"]),
            float(row["std_improvement_vs_ATCS"]),
            str(row["config_id"]),
        )
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            output = dict(row)
            output["mean_improvement_vs_ATCS"] = (
                f"{float(row['mean_improvement_vs_ATCS']):.8f}"
            )
            output["std_improvement_vs_ATCS"] = (
                f"{float(row['std_improvement_vs_ATCS']):.8f}"
            )
            output["mean_best_twt"] = f"{float(row['mean_best_twt']):.6f}"
            output["mean_runtime_sec"] = f"{float(row['mean_runtime_sec']):.6f}"
            writer.writerow(output)

    complete_rows = [
        row for row in summary_rows
        if int(row["n_runs"]) == expected_runs
    ]

    if not complete_rows:
        print(
            f"No complete {algorithm} configuration yet; "
            "summary was saved but best_params.json was not updated."
        )
        return

    best_row = max(
        complete_rows,
        key=lambda row: (
            float(row["mean_improvement_vs_ATCS"]),
            -float(row["std_improvement_vs_ATCS"]),
        ),
    )
    best_config_id = str(best_row["config_id"])
    best_params = config_map[best_config_id]

    if best_params_path.exists():
        try:
            all_best_params = json.loads(
                best_params_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            all_best_params = {}
    else:
        all_best_params = {}

    all_best_params[algorithm] = best_params
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    best_params_path.write_text(
        json.dumps(all_best_params, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Best {algorithm} config: {best_config_id} | "
        f"mean improvement vs ATCS = "
        f"{float(best_row['mean_improvement_vs_ATCS']):.4f}%"
    )
    print(f"Best parameters: {best_params}")


# -----------------------------------------------------------------------------
# 4. Run one algorithm
# -----------------------------------------------------------------------------


def _run_algorithm(
    *,
    algorithm: str,
    instances: list[ReducedInstance],
    seeds: list[int],
    time_limit: float,
    results_dir: Path,
    overwrite: bool,
) -> None:
    configs = PARAM_GRIDS[algorithm]

    raw_path = results_dir / f"tuning_{algorithm}_raw.csv"
    summary_path = results_dir / f"tuning_{algorithm}_summary.csv"
    best_params_path = results_dir / "best_params.json"

    _ensure_header(raw_path, RAW_FIELDS, overwrite)
    completed = set() if overwrite else _completed_keys(raw_path)

    total_runs = len(configs) * len(instances) * len(seeds)
    relevant_completed = sum(
        1
        for params in configs
        for item in instances
        for seed in seeds
        if (_config_id(algorithm, params), item.instance, seed) in completed
    )

    print()
    print("=" * 78)
    print(f"TUNING {algorithm}")
    print("=" * 78)
    print(f"Configurations : {len(configs)}")
    print(f"Instances      : {len(instances)}")
    print(f"Seeds          : {seeds}")
    print(f"Time limit     : {time_limit:g} sec/run")
    print(f"Planned runs   : {total_runs}")
    print(f"Already done   : {relevant_completed}")
    print(f"Remaining      : {total_runs - relevant_completed}")
    print(
        f"Approx. remaining search time: "
        f"{(total_runs - relevant_completed) * time_limit / 60:.1f} min"
    )
    print(f"Raw output     : {raw_path}")
    print()

    interrupted = False

    try:
        with raw_path.open("a", newline="", encoding="utf-8-sig") as raw_handle:
            writer = csv.DictWriter(raw_handle, fieldnames=RAW_FIELDS)

            progress = relevant_completed

            for config_index, params in enumerate(configs, start=1):
                config_id = _config_id(algorithm, params)

                print(
                    f"[{algorithm}] config {config_index}/{len(configs)}: "
                    f"{config_id} -> {params}"
                )

                for item in instances:
                    for seed in seeds:
                        key = (config_id, item.instance, seed)
                        if key in completed:
                            continue

                        result = _run_one(
                            item=item,
                            algorithm=algorithm,
                            params=params,
                            seed=seed,
                            time_limit=time_limit,
                        )

                        atcs_twt = int(result.initial_objective)
                        best_twt = int(result.objective)

                        improvement = (
                            0.0
                            if atcs_twt == 0
                            else (atcs_twt - best_twt) / atcs_twt * 100.0
                        )

                        writer.writerow(
                            {
                                "algorithm": algorithm,
                                "config_id": config_id,
                                "instance": item.instance,
                                "class_id": item.class_id,
                                "tau": item.tau,
                                "R": item.R,
                                "eta": item.eta,
                                "seed": seed,
                                **_parameter_cells(params),
                                "runtime_sec": f"{result.runtime_seconds:.6f}",
                                "ATCS_twt": atcs_twt,
                                "best_twt": best_twt,
                                "improvement_vs_ATCS": f"{improvement:.8f}",
                            }
                        )
                        raw_handle.flush()

                        completed.add(key)
                        progress += 1

                        print(
                            f"  [{progress}/{total_runs}] "
                            f"{item.instance:12s} seed={seed} "
                            f"ATCS={atcs_twt} best={best_twt} "
                            f"impr={improvement:.3f}% "
                            f"time={result.runtime_seconds:.3f}s"
                        )

    except KeyboardInterrupt:
        interrupted = True
        print("\nTuning interrupted by user. Completed rows have already been saved.")

    finally:
        # Rebuild summary from whatever has been completed so far.
        _build_summary_and_best(
            algorithm=algorithm,
            instances=instances,
            seeds=seeds,
            raw_path=raw_path,
            summary_path=summary_path,
            best_params_path=best_params_path,
        )
        print(f"Summary saved to: {summary_path}")
        print(f"Best params file: {best_params_path}")

    if interrupted:
        raise KeyboardInterrupt


# -----------------------------------------------------------------------------
# 5. CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune SMTWT-SDST metaheuristics on tuning_dataset.json. "
            "Each algorithm can be run independently and resumed later."
        )
    )

    parser.add_argument(
        "--algorithm",
        required=True,
        choices=[*SOLVERS.keys(), "ALL"],
        help="Algorithm to tune: ACO, SA, GA, TS, DDE, DPSO, or ALL.",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/tuning_dataset.json"),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )

    parser.add_argument(
        "--time-limit",
        type=float,
        default=10.0,
        help="Wall-clock search budget per run in seconds (default: 10).",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Common random seeds; default is three runs: 0 1 2.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Discard the selected algorithm's existing raw tuning file "
            "instead of resuming it."
        ),
    )

    args = parser.parse_args()

    if args.time_limit <= 0:
        raise ValueError("--time-limit must be positive.")
    if len(args.seeds) != 3:
        raise ValueError(
            "This tuning design uses exactly 3 seeds. "
            "Provide exactly three values with --seeds."
        )
    if len(set(args.seeds)) != 3:
        raise ValueError("The three seeds must be unique.")

    instances = load_dataset(args.dataset)

    if len(instances) != 8:
        raise ValueError(
            f"Expected exactly 8 tuning instances, found {len(instances)} "
            f"in {args.dataset}."
        )

    if len({item.class_id for item in instances}) != 8:
        raise ValueError(
            "The tuning dataset must contain exactly one instance from each "
            "of the 8 coefficient classes."
        )

    algorithms = (
        list(SOLVERS.keys())
        if args.algorithm == "ALL"
        else [args.algorithm]
    )

    for algorithm in algorithms:
        try:
            _run_algorithm(
                algorithm=algorithm,
                instances=instances,
                seeds=args.seeds,
                time_limit=args.time_limit,
                results_dir=args.results_dir,
                overwrite=args.overwrite,
            )
        except KeyboardInterrupt:
            print("Stopped. Re-run the same command later to resume.")
            break


if __name__ == "__main__":
    main()
