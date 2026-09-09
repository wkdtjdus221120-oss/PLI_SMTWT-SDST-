from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from algorithms.aco_smtwt_sdst import solve_aco
from algorithms.dde_smtwt_sdst import solve_dde
from algorithms.dpso_smtwt_sdst import solve_dpso
from algorithms.ga_smtwt_sdst import solve_ga
from algorithms.gvns_smtwt_sdst import solve_gvns
from algorithms.sa_smtwt_sdst import solve_sa
from algorithms.ts_smtwt_sdst import solve_ts
from dataset_io import ReducedInstance, load_dataset
# -----------------------------------------------------------------------------
# 1. Solver registry
# -----------------------------------------------------------------------------

SOLVERS = {
    "SA": solve_sa,
    "GA": solve_ga,
    "TS": solve_ts,
    "ACO": solve_aco,
    "DPSO": solve_dpso,
    "DDE": solve_dde,
    "GVNS": solve_gvns,
}

# These algorithms are tuned by run_tuning.py.
# GVNS uses the defaults already defined in its solver file.
TUNED_ALGORITHMS = {
    "ACO",
    "SA",
    "GA",
    "TS",
    "DDE",
    "DPSO",
}


# -----------------------------------------------------------------------------
# 2. Output schemas
# -----------------------------------------------------------------------------

RAW_FIELDS = [
    "algorithm",
    "instance",
    "tau",
    "R",
    "eta",
    "seed",
    "runtime_sec",
    "best_twt",
    "improvement_vs_ATCS",
]

CONV_FIELDS = [
    "algorithm",
    "instance",
    "tau",
    "R",
    "eta",
    "seed",
    "elapsed_sec",
    "best_twt",
]

# Experimental design:
# incumbent histories are sampled at 0, 1, ..., 10 sec.
CONVERGENCE_SECONDS = tuple(range(11))


# -----------------------------------------------------------------------------
# 3. Input validation / loading
# -----------------------------------------------------------------------------

def _validate_final_dataset(instances: list[ReducedInstance]) -> None:
    """
    Validate the fixed final-evaluation dataset.

    Expected design:
      8 coefficient classes x 5 instances = 40 instances.
    """

    if len(instances) != 40:
        raise ValueError(
            f"Final evaluation requires exactly 40 instances, "
            f"but {len(instances)} were loaded."
        )

    counts = Counter(item.class_id for item in instances)

    if len(counts) != 8:
        raise ValueError(
            f"Expected 8 coefficient classes, found {len(counts)}: {dict(counts)}"
        )

    invalid = {
        class_id: count
        for class_id, count in counts.items()
        if count != 5
    }

    if invalid:
        raise ValueError(
            "Each coefficient class must contain exactly 5 final-test instances. "
            f"Invalid class counts: {invalid}"
        )

    instance_names = [item.instance for item in instances]
    if len(instance_names) != len(set(instance_names)):
        raise ValueError("Duplicate instance names were found in the final dataset.")


def _read_atcs(path: Path) -> dict[str, int]:
    """
    Read ATCS baselines produced by run_baselines.py.

    Required columns:
      instance, ATCS_twt
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Baseline file not found: {path}\n"
            "Run run_baselines.py before the final experiment."
        )

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No baseline rows found in {path}.")

    required = {"instance", "ATCS_twt"}
    missing_columns = required - set(rows[0])

    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing_columns)}"
        )

    result: dict[str, int] = {}

    for row in rows:
        instance = str(row["instance"])
        if instance in result:
            raise ValueError(
                f"Duplicate ATCS baseline found for instance {instance}."
            )
        result[instance] = int(row["ATCS_twt"])

    if len(result) != 40:
        raise ValueError(
            f"Expected 40 ATCS baselines, found {len(result)} in {path}."
        )

    return result


def _load_best_params(path: Path) -> dict[str, dict[str, int | float]]:
    """
    Load tuned parameters produced by run_tuning.py.

    Example:
    {
        "SA": {"cooling_rate": 0.95},
        "GA": {
            "population_size": 50,
            "mutation_rate": 0.10,
            "tournament_size": 4
        }
    }
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Tuned parameter file not found: {path}\n"
            "Complete run_tuning.py first so that best_params.json is created."
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in tuned parameter file: {path}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "best_params.json must contain a JSON object at the top level."
        )

    result: dict[str, dict[str, int | float]] = {}

    for algorithm, params in data.items():
        if algorithm not in SOLVERS:
            # Ignore unrelated entries rather than failing.
            continue

        if not isinstance(params, dict):
            raise ValueError(
                f"Parameters for {algorithm} must be stored as a JSON object."
            )

        result[algorithm] = params

    return result


def _validate_tuned_params(
    best_params: dict[str, dict[str, int | float]],
    selected_algorithms: list[str],
) -> None:
    """
    Prevent an accidental final experiment with untuned defaults.

    Only SA, GA, TS, DDE, and DPSO are required to have tuned entries.
    GVNS intentionally keeps its solver defaults.
    """

    required = [
        algorithm
        for algorithm in selected_algorithms
        if algorithm in TUNED_ALGORITHMS
    ]

    missing = [
        algorithm
        for algorithm in required
        if algorithm not in best_params
    ]

    if missing:
        raise ValueError(
            "The following requested algorithms do not yet have tuned parameters "
            f"in best_params.json: {missing}\n"
            "Finish their tuning runs before the final evaluation."
        )


# -----------------------------------------------------------------------------
# 4. Resume / CSV helpers
# -----------------------------------------------------------------------------

def _ensure_header(
    path: Path,
    fields: list[str],
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not path.exists():
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(
                handle,
                fieldnames=fields,
            ).writeheader()


def _completed_keys(
    raw_path: Path,
    conv_path: Path,
) -> set[tuple[str, str, int]]:
    """
    A run is considered complete only when:
      1) one raw result exists, and
      2) convergence contains all 11 sampled seconds (0..10).

    Key:
      (algorithm, instance, seed)
    """

    raw_keys: set[tuple[str, str, int]] = set()
    conv_seconds: dict[
        tuple[str, str, int],
        set[int],
    ] = {}

    if raw_path.exists():
        with raw_path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            for row in csv.DictReader(handle):
                try:
                    key = (
                        row["algorithm"],
                        row["instance"],
                        int(row["seed"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue

                raw_keys.add(key)

    if conv_path.exists():
        with conv_path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            for row in csv.DictReader(handle):
                try:
                    key = (
                        row["algorithm"],
                        row["instance"],
                        int(row["seed"]),
                    )
                    second = int(row["elapsed_sec"])
                except (KeyError, TypeError, ValueError):
                    continue

                conv_seconds.setdefault(
                    key,
                    set(),
                ).add(second)

    required_seconds = set(CONVERGENCE_SECONDS)

    return {
        key
        for key in raw_keys
        if conv_seconds.get(key, set()) >= required_seconds
    }


def _planned_keys(
    instances: list[ReducedInstance],
    algorithms: list[str],
    seeds: list[int],
) -> set[tuple[str, str, int]]:
    """
    Build the exact set of runs requested in the current command.

    This is important when algorithms are run separately on different days:
    completed SA runs must not be counted as completed GA runs.
    """

    return {
        (algorithm, item.instance, seed)
        for item in instances
        for algorithm in algorithms
        for seed in seeds
    }


# -----------------------------------------------------------------------------
# 5. Convergence history
# -----------------------------------------------------------------------------

def _forward_fill(
    history: tuple[tuple[float, int], ...],
    fallback: int,
) -> list[int]:
    """
    Sample incumbent-best history at integer seconds 0..10.

    Example:
      history = ((0.0, 1000), (2.4, 900), (6.7, 800))

    gives approximately:
      sec 0-2  -> 1000
      sec 3-6  -> 900
      sec 7-10 -> 800
    """

    events = sorted(
        history,
        key=lambda event: event[0],
    )

    values: list[int] = []
    index = 0

    current = (
        int(events[0][1])
        if events
        else int(fallback)
    )

    for second in CONVERGENCE_SECONDS:
        while (
            index < len(events)
            and events[index][0] <= second + 1e-12
        ):
            current = int(events[index][1])
            index += 1

        values.append(current)

    return values


# -----------------------------------------------------------------------------
# 6. Run one algorithm / instance / seed
# -----------------------------------------------------------------------------

def _run_one(
    item: ReducedInstance,
    algorithm: str,
    seed: int,
    time_limit: float,
    best_params: dict[str, dict[str, int | float]],
):
    solver = SOLVERS[algorithm]

    # Tuned algorithms receive the parameter combination selected by
    # run_tuning.py. GVNS has no entry and therefore uses defaults.
    params = best_params.get(
        algorithm,
        {},
    )

    # Fair comparison:
    # wall-clock time is the common active stopping criterion.
    # max_evaluations=None prevents one implementation from terminating
    # earlier simply because it reaches an evaluation-count limit faster.
    return solver(
        item.problem,
        time_limit=time_limit,
        max_evaluations=None,
        seed=seed,
        **params,
    )


# -----------------------------------------------------------------------------
# 7. Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final SMTWT-SDST metaheuristic experiment on the fixed "
            "40-instance test dataset using tuned parameters from best_params.json."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/reduced_dataset.json"),
        help=(
            "Fixed 40-instance final evaluation dataset "
            "(default: data/reduced_dataset.json)."
        ),
    )

    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("results/baseline_results.csv"),
        help=(
            "EDD/ATCS baseline CSV produced by run_baselines.py "
            "(default: results/baseline_results.csv)."
        ),
    )

    parser.add_argument(
        "--best-params",
        type=Path,
        default=Path("results/best_params.json"),
        help=(
            "Tuned parameter JSON produced by run_tuning.py "
            "(default: results/best_params.json)."
        ),
    )

    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("results/results_raw.csv"),
        help="Final raw result CSV.",
    )

    parser.add_argument(
        "--convergence-output",
        type=Path,
        default=Path("results/convergence.csv"),
        help="Forward-filled 0..10 sec convergence CSV.",
    )

    parser.add_argument(
        "--time-limit",
        type=float,
        default=10.0,
        help="Common wall-clock search limit per run (default: 10 sec).",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help=(
            "Common final-experiment seeds for every algorithm/instance "
            "(default: 0 1 2 3 4). "
            "These are independent of the 3-seed tuning design."
        ),
    )

    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=list(SOLVERS),
        default=list(SOLVERS),
        help=(
            "Algorithms to run. You may run them separately, e.g. "
            "--algorithms GA or --algorithms SA TS. "
            "Default: SA GA TS ACO DPSO DDE GVNS."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Discard the existing results_raw.csv and convergence.csv before running. "
            "WARNING: this removes results from all algorithms, not only the algorithms "
            "selected in the current command."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Basic argument checks
    # -------------------------------------------------------------------------

    if args.time_limit <= 0:
        raise ValueError("--time-limit must be positive.")

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique.")

    if len(set(args.algorithms)) != len(args.algorithms):
        raise ValueError("Algorithms must not be duplicated.")

    # -------------------------------------------------------------------------
    # Load experimental inputs
    # -------------------------------------------------------------------------

    instances = load_dataset(args.dataset)
    _validate_final_dataset(instances)

    atcs = _read_atcs(args.baselines)

    dataset_names = {
        item.instance
        for item in instances
    }

    baseline_names = set(atcs)

    if dataset_names != baseline_names:
        missing_baselines = sorted(
            dataset_names - baseline_names
        )
        extra_baselines = sorted(
            baseline_names - dataset_names
        )

        raise ValueError(
            "The baseline CSV and reduced dataset do not contain the same "
            "40 instances.\n"
            f"Missing baselines: {missing_baselines}\n"
            f"Extra baselines: {extra_baselines}"
        )

    best_params = _load_best_params(
        args.best_params
    )

    _validate_tuned_params(
        best_params,
        args.algorithms,
    )

    # -------------------------------------------------------------------------
    # Show exactly which parameters will be used
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL EXPERIMENT SETTINGS")
    print("=" * 78)
    print(f"Dataset        : {args.dataset}")
    print(f"Instances      : {len(instances)} (8 classes x 5)")
    print(f"Algorithms     : {args.algorithms}")
    print(f"Seeds          : {args.seeds}")
    print(f"Time limit     : {args.time_limit:g} sec/run")
    print(f"Best params    : {args.best_params}")
    print()

    for algorithm in args.algorithms:
        if algorithm in best_params:
            print(
                f"{algorithm:4s} -> tuned params: "
                f"{best_params[algorithm]}"
            )
        else:
            print(
                f"{algorithm:4s} -> solver defaults "
                "(not tuned in this experiment)"
            )

    print("=" * 78)
    print()

    # -------------------------------------------------------------------------
    # Resume state
    # -------------------------------------------------------------------------

    _ensure_header(
        args.raw_output,
        RAW_FIELDS,
        args.overwrite,
    )

    _ensure_header(
        args.convergence_output,
        CONV_FIELDS,
        args.overwrite,
    )

    planned = _planned_keys(
        instances,
        args.algorithms,
        args.seeds,
    )

    if args.overwrite:
        completed: set[
            tuple[str, str, int]
        ] = set()
    else:
        completed = (
            _completed_keys(
                args.raw_output,
                args.convergence_output,
            )
            & planned
        )

    total = len(planned)
    done = len(completed)

    print(
        f"Planned runs   : {total}"
    )
    print(
        f"Already done   : {done}"
    )
    print(
        f"Remaining      : {total - done}"
    )
    print(
        f"Approx. remaining search time: "
        f"{(total - done) * args.time_limit / 60:.1f} min"
    )
    print()

    # -------------------------------------------------------------------------
    # Main experiment
    # -------------------------------------------------------------------------

    with (
        args.raw_output.open(
            "a",
            newline="",
            encoding="utf-8-sig",
        ) as raw_handle,
        args.convergence_output.open(
            "a",
            newline="",
            encoding="utf-8-sig",
        ) as conv_handle,
    ):
        raw_writer = csv.DictWriter(
            raw_handle,
            fieldnames=RAW_FIELDS,
        )

        conv_writer = csv.DictWriter(
            conv_handle,
            fieldnames=CONV_FIELDS,
        )

        for item in instances:
            atcs_twt = atcs[
                item.instance
            ]

            for algorithm in args.algorithms:
                for seed in args.seeds:

                    key = (
                        algorithm,
                        item.instance,
                        seed,
                    )

                    if key in completed:
                        continue

                    result = _run_one(
                        item=item,
                        algorithm=algorithm,
                        seed=seed,
                        time_limit=args.time_limit,
                        best_params=best_params,
                    )

                    best_twt = int(
                        result.objective
                    )

                    improvement = (
                        0.0
                        if atcs_twt == 0
                        else (
                            atcs_twt
                            - best_twt
                        )
                        / atcs_twt
                        * 100.0
                    )

                    # ---------------------------------------------------------
                    # Save one raw result immediately
                    # ---------------------------------------------------------

                    raw_writer.writerow(
                        {
                            "algorithm":
                                algorithm,

                            "instance":
                                item.instance,

                            "tau":
                                item.tau,

                            "R":
                                item.R,

                            "eta":
                                item.eta,

                            "seed":
                                seed,

                            "runtime_sec":
                                f"{result.runtime_seconds:.6f}",

                            "best_twt":
                                best_twt,

                            "improvement_vs_ATCS":
                                f"{improvement:.8f}",
                        }
                    )

                    raw_handle.flush()

                    # ---------------------------------------------------------
                    # Convert incumbent history to 0..10 sec forward-fill data
                    # ---------------------------------------------------------

                    sampled = _forward_fill(
                        result.history,
                        best_twt,
                    )

                    for second, value in zip(
                        CONVERGENCE_SECONDS,
                        sampled,
                    ):
                        conv_writer.writerow(
                            {
                                "algorithm":
                                    algorithm,

                                "instance":
                                    item.instance,

                                "tau":
                                    item.tau,

                                "R":
                                    item.R,

                                "eta":
                                    item.eta,

                                "seed":
                                    seed,

                                "elapsed_sec":
                                    second,

                                "best_twt":
                                    value,
                            }
                        )

                    conv_handle.flush()

                    # Mark complete in-memory as soon as both files are written.
                    completed.add(key)
                    done += 1

                    print(
                        f"[{done}/{total}] "
                        f"{algorithm:4s} "
                        f"{item.instance:12s} "
                        f"seed={seed:<4d} "
                        f"ATCS={atcs_twt} "
                        f"best={best_twt} "
                        f"impr={improvement:8.3f}% "
                        f"runtime={result.runtime_seconds:.3f}s"
                    )

    print()
    print("=" * 78)
    print("FINAL EXPERIMENT COMPLETED")
    print("=" * 78)
    print(f"Raw results      : {args.raw_output}")
    print(f"Convergence data : {args.convergence_output}")


if __name__ == "__main__":
    main()
