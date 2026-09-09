from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Iterable

import gurobipy as gp
from gurobipy import GRB

from algorithms.gvns_smtwt_sdst import (
    WTSDSInstance,
    generate_atcs_sequence,
    total_weighted_tardiness,
)
from dataset_io import ReducedInstance, load_dataset


# =============================================================================
# Output schema
# =============================================================================

RESULT_FIELDS = [
    "mode",
    "original_instance",
    "subinstance",
    "source_number",
    "class_id",
    "tau",
    "R",
    "eta",
    "n_jobs",
    "status",
    "optimal_proven",
    "runtime_sec",
    "best_twt",
    "best_bound",
    "absolute_gap",
    "mip_gap",
    "node_count",
    "num_vars",
    "num_binary_vars",
    "num_constraints",
    "big_M",
    "atcs_start_twt",
    "verified_twt",
    "solution_verified",
    "sequence_1based",
]


# =============================================================================
# Basic helpers
# =============================================================================

def _instance_number(name: str) -> int:
    numbers = re.findall(r"\d+", str(name))
    if not numbers:
        return 10**9
    return int(numbers[-1])


def _source_number(item: ReducedInstance) -> int:
    value = getattr(item, "source_number", None)

    if value is not None:
        return int(value)

    return _instance_number(item.instance)


def _status_name(status: int) -> str:
    mapping = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INPROGRESS: "INPROGRESS",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }

    return mapping.get(
        status,
        f"STATUS_{status}",
    )


def _safe_float(value: float | None) -> str:
    if value is None:
        return ""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""

    if not math.isfinite(number):
        return ""

    return f"{number:.10f}"


# =============================================================================
# Representative instances
# =============================================================================

def _select_first_instance_per_class(
    instances: list[ReducedInstance],
) -> list[ReducedInstance]:
    """
    From reduced_dataset.json, choose the first final-test instance
    in each (tau, R, eta) class.
    """

    grouped: dict[str, list[ReducedInstance]] = {}

    for item in instances:
        grouped.setdefault(
            item.class_id,
            [],
        ).append(item)

    if len(grouped) != 8:
        raise ValueError(
            f"Expected 8 classes in the reduced dataset, "
            f"but found {len(grouped)}."
        )

    selected: list[ReducedInstance] = []

    for class_id in sorted(grouped):
        class_items = sorted(
            grouped[class_id],
            key=_source_number,
        )

        selected.append(
            class_items[0]
        )

    return selected


def _filter_classes(
    selected: list[ReducedInstance],
    class_ids: list[str],
) -> list[ReducedInstance]:
    """
    class_ids:
      ["C01"]          -> one representative instance
      ["C01","C02"]    -> selected classes
      ["ALL"]          -> all eight classes
    """

    normalized = [
        value.upper()
        for value in class_ids
    ]

    if "ALL" in normalized:
        return selected

    valid = {
        item.class_id
        for item in selected
    }

    unknown = sorted(
        set(normalized)
        - valid
    )

    if unknown:
        raise ValueError(
            f"Unknown class_id(s): {unknown}. "
            f"Available classes: {sorted(valid)}"
        )

    wanted = set(normalized)

    return [
        item
        for item in selected
        if item.class_id in wanted
    ]


# =============================================================================
# Build a smaller n-job sub-instance from an original 60-job instance
# =============================================================================

def make_prefix_subinstance(
    original: WTSDSInstance,
    n_jobs: int,
) -> WTSDSInstance:
    """
    Create a nested prefix sub-instance using jobs 1..n_jobs
    from the original 60-job benchmark instance.

    Example
    -------
    n=10 uses original jobs 1..10
    n=20 uses original jobs 1..20

    Thus the sequence of test problems is nested:
        10-job problem ⊂ 20-job problem ⊂ ... ⊂ 60-job problem

    The original dummy-initial-setup row is preserved.

    Important
    ---------
    This is a scalability sub-instance, not an original Cicirello benchmark
    instance. The stored tau/R/eta remain the generator labels of the
    original 60-job instance; empirical characteristics of the subset can
    differ somewhat.
    """

    original_n = original.n_jobs

    if not (1 <= n_jobs <= original_n):
        raise ValueError(
            f"n_jobs must be between 1 and {original_n}, "
            f"but received {n_jobs}."
        )

    if n_jobs == original_n:
        return original

    selected = list(
        range(n_jobs)
    )

    old_dummy = original.dummy_job

    processing_times = tuple(
        original.processing_times[j]
        for j in selected
    )

    weights = tuple(
        original.weights[j]
        for j in selected
    )

    due_dates = tuple(
        original.due_dates[j]
        for j in selected
    )

    # Actual predecessor-job rows.
    setup_rows = [
        tuple(
            original.setup_times[i][j]
            for j in selected
        )
        for i in selected
    ]

    # Initial dummy predecessor row.
    dummy_row = tuple(
        original.setup_times[old_dummy][j]
        for j in selected
    )

    setup_times = tuple(
        setup_rows
        + [dummy_row]
    )

    generator_parameters = dict(
        getattr(
            original,
            "generator_parameters",
            {},
        )
        or {}
    )

    generator_parameters[
        "subinstance_n_jobs"
    ] = n_jobs

    generator_parameters[
        "original_n_jobs"
    ] = original_n

    return WTSDSInstance(
        name=f"{original.name}_n{n_jobs}",
        n_jobs=n_jobs,
        processing_times=processing_times,
        weights=weights,
        due_dates=due_dates,
        setup_times=setup_times,
        generator_parameters=generator_parameters,
    )


# =============================================================================
# Big-M and horizon
# =============================================================================

def _schedule_horizon(
    instance: WTSDSInstance,
) -> int:
    """
    Safe upper bound for every completion time.
    """

    n = instance.n_jobs
    dummy = instance.dummy_job

    max_initial_setup = max(
        instance.setup_times[dummy][j]
        for j in range(n)
    )

    pair_setup_values = [
        instance.setup_times[i][j]
        for i in range(n)
        for j in range(n)
        if i != j
    ]

    max_pair_setup = (
        max(pair_setup_values)
        if pair_setup_values
        else 0
    )

    horizon = (
        sum(instance.processing_times)
        + max_initial_setup
        + max(0, n - 1)
        * max_pair_setup
    )

    return int(horizon)


def _big_m_value(
    instance: WTSDSInstance,
    horizon: int,
) -> int:
    """
    Safe M for the paper's completion-time implication.
    """

    n = instance.n_jobs

    transitions = [
        instance.setup_times[i][j]
        + instance.processing_times[j]
        for i in range(n)
        for j in range(n)
        if i != j
    ]

    max_transition = (
        max(transitions)
        if transitions
        else max(
            instance.processing_times
        )
    )

    return int(
        horizon
        + max_transition
    )


# =============================================================================
# ATCS warm start
# =============================================================================

def _apply_atcs_warm_start(
    instance: WTSDSInstance,
    x: gp.tupledict,
    y: gp.tupledict,
    completion: gp.tupledict,
    tardiness: gp.tupledict,
) -> int:
    """
    Supply ATCS as a feasible MIP start.

    This does not affect exactness; it only gives Gurobi an incumbent.
    """

    sequence, _ = generate_atcs_sequence(
        instance
    )

    n = instance.n_jobs
    dummy = instance.dummy_job

    position_of = {
        job: k
        for k, job in enumerate(sequence)
    }

    for job in range(n):
        for k in range(n):
            x[job, k].Start = (
                1.0
                if position_of[job] == k
                else 0.0
            )

    for key in y.keys():
        y[key].Start = 0.0

    for k in range(1, n):
        i = sequence[k - 1]
        j = sequence[k]

        if (i, j, k) in y:
            y[i, j, k].Start = 1.0

    current_time = 0
    previous = dummy

    for job in sequence:
        current_time += (
            instance.setup_times[previous][job]
            + instance.processing_times[job]
        )

        completion[job].Start = float(
            current_time
        )

        tardiness[job].Start = float(
            max(
                current_time
                - instance.due_dates[job],
                0,
            )
        )

        previous = job

    return int(
        total_weighted_tardiness(
            sequence,
            instance,
        )
    )


# =============================================================================
# Paper MILP formulation
# =============================================================================

def build_smtwt_sdst_mip(
    instance: WTSDSInstance,
    *,
    time_limit: float | None,
    mip_gap: float,
    log_file: Path | None,
    use_atcs_warm_start: bool,
) -> tuple[
    gp.Model,
    gp.tupledict,
    gp.tupledict,
    gp.tupledict,
    gp.tupledict,
    int,
    int | None,
]:
    """
    Position-based MILP for:

        1 | s_ij | sum_j w_j T_j

    Variables
    ---------
    x[j,k]     : job j is at position k
    y[i,j,k]   : i is at k-1 and j is at k
    C[j]       : completion time
    T[j]       : tardiness
    """

    n = instance.n_jobs
    jobs = range(n)
    positions = range(n)
    dummy = instance.dummy_job

    horizon = _schedule_horizon(
        instance
    )

    big_m = _big_m_value(
        instance,
        horizon,
    )

    model = gp.Model(
        f"SMTWT_SDST_{instance.name}"
    )

    # ---------------------------------------------------------------------
    # Solver settings
    # ---------------------------------------------------------------------

    # Scalability mode supplies a finite limit.
    # Optimal mode supplies None, so Gurobi's default infinite TimeLimit
    # is left untouched.
    if time_limit is not None:
        model.Params.TimeLimit = float(
            time_limit
        )

    model.Params.MIPGap = float(
        mip_gap
    )

    model.Params.Seed = 0

    if log_file is not None:
        log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.Params.LogFile = str(
            log_file
        )

    # ---------------------------------------------------------------------
    # Variables
    # ---------------------------------------------------------------------

    x = model.addVars(
        jobs,
        positions,
        vtype=GRB.BINARY,
        name="x",
    )

    y_index = [
        (i, j, k)
        for k in range(1, n)
        for i in jobs
        for j in jobs
        if i != j
    ]

    y = model.addVars(
        y_index,
        vtype=GRB.BINARY,
        name="y",
    )

    completion = model.addVars(
        jobs,
        lb=0.0,
        ub=float(horizon),
        vtype=GRB.CONTINUOUS,
        name="C",
    )

    tardiness = model.addVars(
        jobs,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="T",
    )

    # ---------------------------------------------------------------------
    # Objective
    # ---------------------------------------------------------------------

    model.setObjective(
        gp.quicksum(
            instance.weights[j]
            * tardiness[j]
            for j in jobs
        ),
        GRB.MINIMIZE,
    )

    # ---------------------------------------------------------------------
    # (1) Each job exactly once.
    # ---------------------------------------------------------------------

    model.addConstrs(
        (
            gp.quicksum(
                x[j, k]
                for k in positions
            )
            == 1
            for j in jobs
        ),
        name="job_once",
    )

    # ---------------------------------------------------------------------
    # (2) Each position exactly once.
    # ---------------------------------------------------------------------

    model.addConstrs(
        (
            gp.quicksum(
                x[j, k]
                for j in jobs
            )
            == 1
            for k in positions
        ),
        name="position_once",
    )

    # ---------------------------------------------------------------------
    # (3) Completion time of first-position job.
    # ---------------------------------------------------------------------

    model.addConstrs(
        (
            completion[j]
            >= (
                instance.setup_times[dummy][j]
                + instance.processing_times[j]
            )
            * x[j, 0]
            for j in jobs
        ),
        name="first_completion",
    )

    # ---------------------------------------------------------------------
    # (4) Consecutive completion propagation.
    # ---------------------------------------------------------------------

    model.addConstrs(
        (
            completion[j]
            >= completion[i]
            - big_m
            * (
                1
                - y[i, j, k]
            )
            + instance.setup_times[i][j]
            + instance.processing_times[j]
            for k in range(1, n)
            for i in jobs
            for j in jobs
            if i != j
        ),
        name="completion_link",
    )

    # ---------------------------------------------------------------------
    # (5) Tardiness definition.
    # ---------------------------------------------------------------------

    model.addConstrs(
        (
            tardiness[j]
            >= completion[j]
            - instance.due_dates[j]
            for j in jobs
        ),
        name="tardiness_def",
    )

    # ---------------------------------------------------------------------
    # (6) Link x and y.
    # ---------------------------------------------------------------------

    model.addConstrs(
        (
            x[i, k - 1]
            + x[j, k]
            <= y[i, j, k]
            + 1
            for k in range(1, n)
            for i in jobs
            for j in jobs
            if i != j
        ),
        name="xy_link",
    )

    model.update()

    atcs_start_twt: int | None = None

    if use_atcs_warm_start:
        atcs_start_twt = (
            _apply_atcs_warm_start(
                instance,
                x,
                y,
                completion,
                tardiness,
            )
        )

    return (
        model,
        x,
        y,
        completion,
        tardiness,
        big_m,
        atcs_start_twt,
    )


# =============================================================================
# Solve one sub-instance
# =============================================================================

def solve_one(
    item: ReducedInstance,
    *,
    n_jobs: int,
    mode: str,
    time_limit: float | None,
    mip_gap: float,
    log_dir: Path,
    use_atcs_warm_start: bool,
    write_lp: bool,
) -> dict[str, object]:

    original = item.problem

    instance = make_prefix_subinstance(
        original,
        n_jobs,
    )

    log_path = (
        log_dir
        / mode
        / item.class_id
        / f"{item.instance}_n{n_jobs}.log"
    )

    (
        model,
        x,
        y,
        completion,
        tardiness,
        big_m,
        atcs_start_twt,
    ) = build_smtwt_sdst_mip(
        instance,
        time_limit=time_limit,
        mip_gap=mip_gap,
        log_file=log_path,
        use_atcs_warm_start=(
            use_atcs_warm_start
        ),
    )

    if write_lp:
        lp_dir = (
            log_dir.parent
            / "gurobi_models"
            / mode
            / item.class_id
        )

        lp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.write(
            str(
                lp_dir
                / f"{item.instance}_n{n_jobs}.lp"
            )
        )

    print()
    print("=" * 82)
    print(
        f"{mode.upper()} | "
        f"{item.class_id} | "
        f"{item.instance} -> "
        f"n={n_jobs}"
    )
    print(
        f"tau={item.tau}, "
        f"R={item.R}, "
        f"eta={item.eta}, "
        f"M={big_m:,}"
    )

    if time_limit is None:
        print(
            "Time limit: NONE "
            "(run until optimality is proven)"
        )
    else:
        print(
            f"Time limit: {time_limit:g} sec"
        )

    if atcs_start_twt is not None:
        print(
            f"ATCS warm start TWT: "
            f"{atcs_start_twt}"
        )

    print("=" * 82)

    model.optimize()

    status = _status_name(
        model.Status
    )

    has_solution = (
        model.SolCount > 0
    )

    best_twt: float | None = None
    best_bound: float | None = None
    mip_gap_value: float | None = None
    absolute_gap: float | None = None
    verified_twt: int | None = None
    solution_verified = False
    sequence: list[int] = []

    try:
        best_bound = float(
            model.ObjBound
        )
    except (AttributeError, gp.GurobiError):
        best_bound = None

    if has_solution:
        best_twt = float(
            model.ObjVal
        )

        try:
            mip_gap_value = float(
                model.MIPGap
            )
        except (AttributeError, gp.GurobiError):
            mip_gap_value = None

        if (
            best_bound is not None
            and best_twt is not None
        ):
            absolute_gap = (
                best_twt
                - best_bound
            )

        # Recover sequence.
        for k in range(
            instance.n_jobs
        ):
            jobs_here = [
                j
                for j in range(
                    instance.n_jobs
                )
                if x[j, k].X > 0.5
            ]

            if len(jobs_here) != 1:
                raise RuntimeError(
                    f"Could not recover unique job "
                    f"at position {k + 1}: "
                    f"{jobs_here}"
                )

            sequence.append(
                jobs_here[0]
            )

        verified_twt = (
            total_weighted_tardiness(
                sequence,
                instance,
            )
        )

        solution_verified = (
            abs(
                verified_twt
                - best_twt
            )
            <= 1e-5
        )

    optimal_proven = (
        model.Status == GRB.OPTIMAL
    )

    sequence_1based = (
        " ".join(
            str(job + 1)
            for job in sequence
        )
        if sequence
        else ""
    )

    return {
        "mode":
            mode,

        "original_instance":
            item.instance,

        "subinstance":
            f"{item.instance}_n{n_jobs}",

        "source_number":
            _source_number(item),

        "class_id":
            item.class_id,

        "tau":
            item.tau,

        "R":
            item.R,

        "eta":
            item.eta,

        "n_jobs":
            n_jobs,

        "status":
            status,

        "optimal_proven":
            int(optimal_proven),

        "runtime_sec":
            f"{model.Runtime:.6f}",

        "best_twt":
            _safe_float(best_twt),

        "best_bound":
            _safe_float(best_bound),

        "absolute_gap":
            _safe_float(absolute_gap),

        "mip_gap":
            _safe_float(mip_gap_value),

        "node_count":
            _safe_float(model.NodeCount),

        "num_vars":
            model.NumVars,

        "num_binary_vars":
            model.NumBinVars,

        "num_constraints":
            model.NumConstrs,

        "big_M":
            big_m,

        "atcs_start_twt":
            (
                atcs_start_twt
                if atcs_start_twt
                is not None
                else ""
            ),

        "verified_twt":
            (
                verified_twt
                if verified_twt
                is not None
                else ""
            ),

        "solution_verified":
            (
                int(solution_verified)
                if has_solution
                else ""
            ),

        "sequence_1based":
            sequence_1based,
    }


# =============================================================================
# Resume helpers
# =============================================================================

def _result_key(
    row: dict[str, str],
) -> tuple[str, str, int]:
    return (
        row["mode"],
        row["original_instance"],
        int(row["n_jobs"]),
    )


def _read_completed(
    output_path: Path,
) -> set[
    tuple[str, str, int]
]:
    if not output_path.exists():
        return set()

    completed = set()

    with output_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        for row in csv.DictReader(
            handle
        ):
            try:
                completed.add(
                    _result_key(row)
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

    return completed


def _prepare_output(
    output_path: Path,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        overwrite
        or not output_path.exists()
    ):
        with output_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            csv.DictWriter(
                handle,
                fieldnames=RESULT_FIELDS,
            ).writeheader()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper MILP in one of two modes: "
            "(1) finite-time scalability analysis over gradually "
            "increasing job sizes, or "
            "(2) no-time-limit exact optimization until optimality "
            "is proven."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "scalability",
            "optimal",
        ],
        required=True,
        help=(
            "scalability: use a finite time limit and compare "
            "bounds/gaps as n increases; "
            "optimal: no time limit and solve until Gurobi "
            "proves optimality."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/reduced_dataset.json"
        ),
    )

    parser.add_argument(
        "--class-id",
        nargs="+",
        default=["C01"],
        help=(
            "Representative class(es) to run. "
            "Examples: --class-id C01 ; "
            "--class-id C01 C02 ; "
            "--class-id ALL. "
            "Default: C01."
        ),
    )

    parser.add_argument(
        "--job-sizes",
        nargs="+",
        type=int,
        default=[
            10,
            20,
            30,
            40,
            60,
        ],
        help=(
            "Nested prefix sub-instance sizes. "
            "Default: 10 20 30 40 60."
        ),
    )

    parser.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help=(
            "Per-run time limit in scalability mode. "
            "Ignored in optimal mode. "
            "Default: 300 sec."
        ),
    )

    parser.add_argument(
        "--mip-gap",
        type=float,
        default=0.0,
        help=(
            "Relative MIP gap tolerance. "
            "Default 0.0 requests proof to zero gap."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output CSV path. "
            "Defaults to results/exact_scalability.csv "
            "or results/exact_optimal.csv."
        ),
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(
            "results/gurobi_logs"
        ),
    )

    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help=(
            "Disable the ATCS MIP start."
        ),
    )

    parser.add_argument(
        "--write-lp",
        action="store_true",
        help=(
            "Also save LP files for inspection."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite the selected output CSV."
        ),
    )

    args = parser.parse_args()

    if args.time_limit <= 0:
        raise ValueError(
            "--time-limit must be positive."
        )

    if args.mip_gap < 0:
        raise ValueError(
            "--mip-gap cannot be negative."
        )

    job_sizes = sorted(
        set(args.job_sizes)
    )

    invalid_sizes = [
        n
        for n in job_sizes
        if n < 1 or n > 60
    ]

    if invalid_sizes:
        raise ValueError(
            f"All job sizes must lie in [1,60]. "
            f"Invalid values: {invalid_sizes}"
        )

    if args.output is None:
        args.output = Path(
            "results/exact_scalability.csv"
            if args.mode
            == "scalability"
            else
            "results/exact_optimal.csv"
        )

    # ---------------------------------------------------------------------
    # Load the fixed 40-instance dataset and select representative instances.
    # ---------------------------------------------------------------------

    all_instances = load_dataset(
        args.dataset
    )

    representatives = (
        _select_first_instance_per_class(
            all_instances
        )
    )

    representatives = _filter_classes(
        representatives,
        args.class_id,
    )

    # ---------------------------------------------------------------------
    # Resolve time-limit behavior.
    # ---------------------------------------------------------------------

    if args.mode == "scalability":
        effective_time_limit: (
            float | None
        ) = float(
            args.time_limit
        )

    else:
        # Do not set Gurobi's TimeLimit parameter.
        effective_time_limit = None

        print()
        print(
            "WARNING: optimal mode has NO time limit."
        )
        print(
            "Gurobi will continue until it proves optimality, "
            "is interrupted manually, or encounters another "
            "solver termination condition."
        )

    # ---------------------------------------------------------------------
    # Experimental plan.
    # ---------------------------------------------------------------------

    planned = [
        (
            item,
            n_jobs,
        )
        for item in representatives
        for n_jobs in job_sizes
    ]

    print()
    print("=" * 82)
    print(
        "EXACT GUROBI EXPERIMENT PLAN"
    )
    print("=" * 82)
    print(
        f"Mode       : {args.mode}"
    )
    print(
        "Classes    : "
        + ", ".join(
            item.class_id
            for item in representatives
        )
    )
    print(
        "Job sizes  : "
        + ", ".join(
            str(n)
            for n in job_sizes
        )
    )

    if effective_time_limit is None:
        print(
            "Time limit : NONE"
        )
    else:
        print(
            f"Time limit : "
            f"{effective_time_limit:g} sec/run"
        )

    print(
        f"Output     : {args.output}"
    )
    print("=" * 82)

    for item in representatives:
        print(
            f"{item.class_id}: "
            f"{item.instance} "
            f"(source={_source_number(item)}, "
            f"tau={item.tau}, "
            f"R={item.R}, "
            f"eta={item.eta})"
        )

    print("=" * 82)
    print()

    # ---------------------------------------------------------------------
    # Resume state.
    # ---------------------------------------------------------------------

    _prepare_output(
        args.output,
        args.overwrite,
    )

    completed = (
        set()
        if args.overwrite
        else _read_completed(
            args.output
        )
    )

    pending = [
        (
            item,
            n_jobs,
        )
        for item, n_jobs in planned
        if (
            args.mode,
            item.instance,
            n_jobs,
        )
        not in completed
    ]

    print(
        f"Planned runs : {len(planned)}"
    )
    print(
        f"Completed    : "
        f"{len(planned) - len(pending)}"
    )
    print(
        f"Remaining    : {len(pending)}"
    )

    if (
        effective_time_limit
        is not None
    ):
        print(
            "Maximum remaining solver time: "
            f"{len(pending) * effective_time_limit / 60:.1f} min"
        )

    print()

    if not pending:
        print(
            "Nothing to solve. "
            "Use --overwrite to rerun."
        )
        return

    # ---------------------------------------------------------------------
    # Solve runs.
    # ---------------------------------------------------------------------

    with args.output.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULT_FIELDS,
        )

        for index, (
            item,
            n_jobs,
        ) in enumerate(
            pending,
            start=1,
        ):
            print(
                f"\n[{index}/{len(pending)}] "
                f"{item.class_id} "
                f"{item.instance} "
                f"n={n_jobs}"
            )

            row = solve_one(
                item,
                n_jobs=n_jobs,
                mode=args.mode,
                time_limit=(
                    effective_time_limit
                ),
                mip_gap=args.mip_gap,
                log_dir=args.log_dir,
                use_atcs_warm_start=(
                    not args.no_warm_start
                ),
                write_lp=args.write_lp,
            )

            writer.writerow(
                row
            )

            # Preserve every fully completed run immediately.
            handle.flush()

            print()
            print(
                f"Completed: "
                f"n={n_jobs}, "
                f"status={row['status']}, "
                f"best={row['best_twt']}, "
                f"bound={row['best_bound']}, "
                f"gap={row['mip_gap']}, "
                f"runtime={row['runtime_sec']}s"
            )

            if (
                row["solution_verified"]
                == 1
            ):
                print(
                    "Sequence objective verification: OK"
                )

    print()
    print("=" * 82)
    print(
        "EXPERIMENT FINISHED"
    )
    print("=" * 82)
    print(
        f"Results : {args.output}"
    )
    print(
        f"Logs    : {args.log_dir}"
    )


if __name__ == "__main__":
    main()
