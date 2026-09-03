from __future__ import annotations

import argparse
import math
import random
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

DEFAULT_RAW_URL = (
    "https://raw.githubusercontent.com/cicirello/"
    "scheduling-benchmarks/master/wtsds/wt_sds_{number}.instance"
)


@dataclass(frozen=True)
class WTSDSInstance:

    name: str
    n_jobs: int
    processing_times: tuple[int, ...]
    weights: tuple[int, ...]
    due_dates: tuple[int, ...]
    setup_times: tuple[tuple[int, ...], ...]
    generator_parameters: dict[str, float]

    @property
    def dummy_job(self) -> int:
        return self.n_jobs


@dataclass(frozen=True)
class ATCSParameters:
    tau: float
    due_date_range: float
    eta: float
    mean_processing_time: float
    mean_setup_time: float
    k1: float
    k2: float


@dataclass(frozen=True)
class SearchResult:
    sequence: tuple[int, ...]
    objective: int
    initial_sequence: tuple[int, ...]
    initial_objective: int
    runtime_seconds: float
    objective_evaluations: int
    atcs_parameters: ATCSParameters
    history: tuple[tuple[float, int], ...]


@dataclass(frozen=True)
class NeighborhoodResult:
    sequence: list[int]
    objective: int
    completed: bool


def _read_source(source: str) -> tuple[str, str]:
    """Return ``(display_name, text)`` from a path, URL, or instance number."""

    if source.isdigit():
        number = int(source)
        if not 1 <= number <= 120:
            raise ValueError("Cicirello instance number must be between 1 and 120.")
        source = DEFAULT_RAW_URL.format(number=number)

    if source.startswith(("http://", "https://")):
        # Convert a normal GitHub blob URL to a raw URL when possible.
        source = source.replace(
            "https://github.com/cicirello/scheduling-benchmarks/blob/",
            "https://raw.githubusercontent.com/cicirello/scheduling-benchmarks/",
        )
        source = source.replace("/refs/heads/", "/")
        with urllib.request.urlopen(source, timeout=30) as response:
            text = response.read().decode("utf-8")
        return source.rsplit("/", 1)[-1], text

    path = Path(source)
    return path.name, path.read_text(encoding="utf-8")


def parse_cicirello_instance(text: str, name: str = "instance") -> WTSDSInstance:
    """Parse the plain-text format in the Cicirello ``wtsds`` directory."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    def value_after(prefix: str) -> str:
        for line in lines:
            if line.lower().startswith(prefix.lower()):
                return line.split(":", 1)[1].strip()
        raise ValueError(f"Missing required field: {prefix}")

    n = int(value_after("Problem Size:"))

    try:
        generator_start = lines.index("Begin Generator Parameters") + 1
        generator_end = lines.index("End Generator Parameters")
    except ValueError as exc:
        raise ValueError("Generator-parameter section was not found.") from exc

    parameters: dict[str, float] = {}
    for line in lines[generator_start:generator_end]:
        if ":" in line:
            key, value = line.split(":", 1)
            parameters[key.strip().lower()] = float(value.strip())

    def read_integer_section(header: str) -> tuple[int, ...]:
        try:
            start = lines.index(header) + 1
        except ValueError as exc:
            raise ValueError(f"Section was not found: {header}") from exc
        values = tuple(int(lines[start + index]) for index in range(n))
        if len(values) != n:
            raise ValueError(f"{header} must contain {n} values.")
        return values

    processing_times = read_integer_section("Process Times:")
    weights = read_integer_section("Weights:")
    due_dates = read_integer_section("Duedates:")

    try:
        setup_start = lines.index("Setup Times:") + 1
        setup_end = lines.index("End Problem Specification")
    except ValueError as exc:
        raise ValueError("Setup-time section was not found.") from exc

    setup = [[0] * n for _ in range(n + 1)]
    observed_arcs: set[tuple[int, int]] = set()
    for line in lines[setup_start:setup_end]:
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"Invalid setup-time row: {line!r}")
        previous, job, setup_time = map(int, fields)
        if previous == -1:
            row = n
        elif 0 <= previous < n:
            row = previous
        else:
            raise ValueError(f"Invalid predecessor job: {previous}")
        if not 0 <= job < n:
            raise ValueError(f"Invalid successor job: {job}")
        setup[row][job] = setup_time
        observed_arcs.add((row, job))

    required_arcs = {(n, job) for job in range(n)}
    required_arcs.update(
        (previous, job) for previous in range(n) for job in range(n) if previous != job
    )
    missing = required_arcs - observed_arcs
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"Setup-time data are incomplete; missing arcs include {sample}."
        )

    if any(value <= 0 for value in processing_times):
        raise ValueError("All processing times must be positive.")
    if any(value < 0 for value in weights):
        raise ValueError("Weights cannot be negative.")

    return WTSDSInstance(
        name=name,
        n_jobs=n,
        processing_times=processing_times,
        weights=weights,
        due_dates=due_dates,
        setup_times=tuple(tuple(row) for row in setup),
        generator_parameters=parameters,
    )


def load_cicirello_instance(source: str | Path) -> WTSDSInstance:
    """Load an instance from a local path, URL, or number from 1 through 120."""

    name, text = _read_source(str(source))
    return parse_cicirello_instance(text, name)


def total_weighted_tardiness(sequence: Sequence[int], instance: WTSDSInstance) -> int:
    """Evaluate ``sum(w_j * max(C_j - d_j, 0))`` for a job sequence."""

    if len(sequence) != instance.n_jobs or set(sequence) != set(range(instance.n_jobs)):
        raise ValueError("A solution must be a permutation of every job exactly once.")

    return _objective_unchecked(sequence, instance)


def _objective_unchecked(sequence: Sequence[int], instance: WTSDSInstance) -> int:
    """Fast objective evaluation for permutations created inside the search."""

    current_time = 0
    objective = 0
    previous = instance.dummy_job

    for job in sequence:
        current_time += (
            instance.setup_times[previous][job] + instance.processing_times[job]
        )
        tardiness = max(current_time - instance.due_dates[job], 0)
        objective += instance.weights[job] * tardiness
        previous = job

    return objective


def _atcs_parameters(instance: WTSDSInstance) -> ATCSParameters:
    """Calculate the ATCS scaling parameters used by Kirlik and Oguz."""

    params = instance.generator_parameters
    p_bar = params.get("p_bar", statistics.fmean(instance.processing_times))

    setup_values = [
        instance.setup_times[previous][job]
        for previous in range(instance.n_jobs + 1)
        for job in range(instance.n_jobs)
        if previous == instance.dummy_job or previous != job
    ]
    s_bar = params.get("s_bar", statistics.fmean(setup_values))

    # Cicirello files provide C_max, Tau, R, and Eta. The fallbacks make the
    # parser usable for compatible files that omit some generator metadata.
    estimated_cmax = params.get(
        "c_max", sum(instance.processing_times) + 0.3 * instance.n_jobs * s_bar
    )
    tau = params.get("tau", 1.0 - statistics.fmean(instance.due_dates) / estimated_cmax)
    due_date_range = params.get(
        "r", (max(instance.due_dates) - min(instance.due_dates)) / estimated_cmax
    )
    eta = params.get("eta", s_bar / p_bar)

    k1 = 4.5 + due_date_range if due_date_range <= 0.5 else 6.0 - 2.0 * due_date_range
    k2 = tau / (2.0 * math.sqrt(eta)) if eta > 0 else 1.0

    return ATCSParameters(
        tau=tau,
        due_date_range=due_date_range,
        eta=eta,
        mean_processing_time=p_bar,
        mean_setup_time=s_bar,
        k1=max(k1, 1e-12),
        k2=max(k2, 1e-12),
    )


def generate_atcs_sequence(
    instance: WTSDSInstance,
) -> tuple[list[int], ATCSParameters]:
    """Construct an initial sequence using the ATCS priority index.

    Log priorities are compared instead of evaluating the exponential form;
    this is mathematically equivalent and avoids floating-point underflow.
    """

    params = _atcs_parameters(instance)
    unscheduled = set(range(instance.n_jobs))
    sequence: list[int] = []
    current_time = 0
    previous = instance.dummy_job

    while unscheduled:

        def log_priority(job: int) -> tuple[float, int]:
            if instance.weights[job] == 0:
                # Zero-weight jobs have zero ATCS priority. They are still
                # scheduled after all positive-priority jobs.
                return -math.inf, -job
            slack = max(
                instance.due_dates[job] - instance.processing_times[job] - current_time,
                0,
            )
            priority = (
                math.log(instance.weights[job] / instance.processing_times[job])
                - slack / (params.k1 * params.mean_processing_time)
                - (
                    instance.setup_times[previous][job]
                    / (params.k2 * params.mean_setup_time)
                    if params.mean_setup_time > 0
                    else 0.0
                )
            )
            # A smaller job index breaks an exact tie deterministically.
            return priority, -job

        job = max(unscheduled, key=log_priority)
        sequence.append(job)
        unscheduled.remove(job)
        current_time += (
            instance.setup_times[previous][job] + instance.processing_times[job]
        )
        previous = job

    return sequence, params


def swap_move(sequence: Sequence[int], first: int, second: int) -> list[int]:
    """Exchange the jobs at two positions."""

    candidate = list(sequence)
    candidate[first], candidate[second] = candidate[second], candidate[first]
    return candidate


def insertion_move(sequence: Sequence[int], source: int, destination: int) -> list[int]:
    """Remove one job and insert it at an index in the shortened sequence."""

    candidate = list(sequence)
    job = candidate.pop(source)
    candidate.insert(destination, job)
    return candidate


def edge_insertion_move(
    sequence: Sequence[int], edge_start: int, destination: int
) -> list[int]:
    """Remove two consecutive jobs and insert the pair elsewhere"""

    candidate = list(sequence)
    edge = candidate[edge_start : edge_start + 2]
    del candidate[edge_start : edge_start + 2]
    candidate[destination:destination] = edge
    return candidate


class SearchControl:
    """Shared runtime and objective-evaluation limits."""

    def __init__(
        self,
        instance: WTSDSInstance,
        deadline: float | None,
        max_evaluations: int | None,
        *,
        start_time: float | None = None,
    ) -> None:
        self.instance = instance
        self.deadline = deadline
        self.max_evaluations = max_evaluations
        self.start_time = time.perf_counter() if start_time is None else start_time
        self.evaluations = 0
        self.best_objective: int | None = None
        self.history: list[tuple[float, int]] = []

    def exhausted(self) -> bool:
        time_exhausted = (
            self.deadline is not None and time.perf_counter() >= self.deadline
        )
        evaluations_exhausted = (
            self.max_evaluations is not None
            and self.evaluations >= self.max_evaluations
        )
        return time_exhausted or evaluations_exhausted

    def evaluate(self, sequence: Sequence[int]) -> int:
        self.evaluations += 1
        objective = _objective_unchecked(sequence, self.instance)
        if self.best_objective is None or objective < self.best_objective:
            self.best_objective = objective
            elapsed = 0.0 if not self.history else time.perf_counter() - self.start_time
            self.history.append((elapsed, int(objective)))
        return objective


BestMove = Callable[
    [Sequence[int], WTSDSInstance, SearchControl, int | None],
    NeighborhoodResult,
]


def best_swap_move(
    sequence: Sequence[int],
    instance: WTSDSInstance,
    control: SearchControl,
    current_objective: int | None = None,
) -> NeighborhoodResult:
    """Return the best solution in the swap neighborhood."""

    best_sequence = list(sequence)
    best_objective = (
        control.evaluate(sequence) if current_objective is None else current_objective
    )

    for first in range(instance.n_jobs - 1):
        for second in range(first + 1, instance.n_jobs):
            if control.exhausted():
                return NeighborhoodResult(best_sequence, best_objective, False)
            candidate = swap_move(sequence, first, second)
            objective = control.evaluate(candidate)
            if objective < best_objective:
                best_sequence, best_objective = candidate, objective

    return NeighborhoodResult(best_sequence, best_objective, True)


def best_insertion_move(
    sequence: Sequence[int],
    instance: WTSDSInstance,
    control: SearchControl,
    current_objective: int | None = None,
) -> NeighborhoodResult:
    """Return the best solution in the single-job insertion neighborhood."""

    best_sequence = list(sequence)
    best_objective = (
        control.evaluate(sequence) if current_objective is None else current_objective
    )

    for source in range(instance.n_jobs):
        for destination in range(instance.n_jobs):
            if source == destination:
                continue
            if control.exhausted():
                return NeighborhoodResult(best_sequence, best_objective, False)
            candidate = insertion_move(sequence, source, destination)
            objective = control.evaluate(candidate)
            if objective < best_objective:
                best_sequence, best_objective = candidate, objective

    return NeighborhoodResult(best_sequence, best_objective, True)


def best_edge_insertion_move(
    sequence: Sequence[int],
    instance: WTSDSInstance,
    control: SearchControl,
    current_objective: int | None = None,
) -> NeighborhoodResult:
    """Return the best solution in the two-job edge-insertion neighborhood."""

    best_sequence = list(sequence)
    best_objective = (
        control.evaluate(sequence) if current_objective is None else current_objective
    )

    for edge_start in range(instance.n_jobs - 1):
        # After removing two jobs, there are n-1 insertion boundaries.
        for destination in range(instance.n_jobs - 1):
            if destination == edge_start:
                continue
            if control.exhausted():
                return NeighborhoodResult(best_sequence, best_objective, False)
            candidate = edge_insertion_move(sequence, edge_start, destination)
            objective = control.evaluate(candidate)
            if objective < best_objective:
                best_sequence, best_objective = candidate, objective

    return NeighborhoodResult(best_sequence, best_objective, True)


VND_NEIGHBORHOODS: tuple[BestMove, ...] = (
    best_swap_move,
    best_edge_insertion_move,
    best_insertion_move,
)


def variable_neighborhood_descent(
    sequence: Sequence[int],
    instance: WTSDSInstance,
    control: SearchControl,
    current_objective: int | None = None,
) -> NeighborhoodResult:
    """Run sequential VND in the paper's Case-7 neighborhood order."""

    current = list(sequence)
    objective = (
        control.evaluate(sequence) if current_objective is None else current_objective
    )
    neighborhood_index = 0

    while neighborhood_index < len(VND_NEIGHBORHOODS) and not control.exhausted():
        result = VND_NEIGHBORHOODS[neighborhood_index](
            current, instance, control, objective
        )
        if result.objective < objective:
            current = result.sequence
            objective = result.objective
            neighborhood_index = 0
        else:
            neighborhood_index += 1

        if not result.completed:
            return NeighborhoodResult(current, objective, False)

    return NeighborhoodResult(current, objective, not control.exhausted())


def random_insertion_move(sequence: Sequence[int], rng: random.Random) -> list[int]:
    """Random shaking move from the insertion neighborhood."""

    if len(sequence) < 2:
        return list(sequence)
    source, destination = rng.sample(range(len(sequence)), 2)
    return insertion_move(sequence, source, destination)


def random_edge_insertion_move(
    sequence: Sequence[int], rng: random.Random
) -> list[int]:
    """Random shaking move from the edge-insertion neighborhood."""

    if len(sequence) < 3:
        return list(sequence)
    edge_start = rng.randrange(len(sequence) - 1)
    destinations = [
        position for position in range(len(sequence) - 1) if position != edge_start
    ]
    destination = rng.choice(destinations)
    return edge_insertion_move(sequence, edge_start, destination)


# GVNS main loop
def solve_gvns(
    instance: WTSDSInstance,
    *,
    time_limit: float | None = 10.0,
    max_evaluations: int | None = 20_000_000,
    max_consecutive_nonimprovements: int | None = None,
    seed: int | None = None,
) -> SearchResult:
    """Solve one instance with GVNS and return the best-found schedule.

    The paper reports an upper limit of 20,000,000 objective evaluations and
    also uses consecutive non-improvement as a stopping criterion, but it does
    not state the latter threshold. This implementation therefore exposes both
    limits and additionally uses a practical, user-configurable wall-clock
    limit. Set ``time_limit=None`` to disable the time limit.
    """

    if instance.n_jobs < 1:
        raise ValueError("The instance must contain at least one job.")
    if time_limit is not None and time_limit <= 0:
        time_limit = None
    if max_evaluations is not None and max_evaluations < 1:
        max_evaluations = None
    if max_consecutive_nonimprovements is not None:
        if max_consecutive_nonimprovements < 1:
            max_consecutive_nonimprovements = None
    if time_limit is None and max_evaluations is None:
        raise ValueError("At least one stopping limit must be active.")

    start = time.perf_counter()
    deadline = start + time_limit if time_limit is not None else None
    control = SearchControl(instance, deadline, max_evaluations, start_time=start)
    rng = random.Random(seed)

    initial_sequence, atcs_parameters = generate_atcs_sequence(instance)
    initial_objective = control.evaluate(initial_sequence)
    incumbent = list(initial_sequence)
    incumbent_objective = initial_objective
    consecutive_nonimprovements = 0

    shaking_moves = (random_insertion_move, random_edge_insertion_move)

    while not control.exhausted():
        shaking_index = 0
        while shaking_index < len(shaking_moves) and not control.exhausted():
            shaken = shaking_moves[shaking_index](incumbent, rng)
            if control.exhausted():
                break
            shaken_objective = control.evaluate(shaken)
            local = variable_neighborhood_descent(
                shaken, instance, control, shaken_objective
            )

            if local.objective < incumbent_objective:
                incumbent = local.sequence
                incumbent_objective = local.objective
                shaking_index = 0
                consecutive_nonimprovements = 0
            else:
                shaking_index += 1
                consecutive_nonimprovements += 1

            if (
                max_consecutive_nonimprovements is not None
                and consecutive_nonimprovements >= max_consecutive_nonimprovements
            ):
                break

        if (
            max_consecutive_nonimprovements is not None
            and consecutive_nonimprovements >= max_consecutive_nonimprovements
        ):
            break

    runtime = time.perf_counter() - start
    return SearchResult(
        sequence=tuple(incumbent),
        objective=incumbent_objective,
        initial_sequence=tuple(initial_sequence),
        initial_objective=initial_objective,
        runtime_seconds=runtime,
        objective_evaluations=control.evaluations,
        atcs_parameters=atcs_parameters,
        history=tuple(control.history),
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve a Cicirello weighted-tardiness benchmark instance with GVNS."
        )
    )
    parser.add_argument(
        "instance",
        help=("Local .instance file, raw/GitHub URL, or an instance number 1-120."),
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=10.0,
        help="Wall-clock limit in seconds; use 0 to disable it (default: 10).",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=20_000_000,
        help="Maximum objective evaluations; use 0 to disable (default: 20000000).",
    )
    parser.add_argument(
        "--max-no-improve",
        type=int,
        default=0,
        help=(
            "Optional consecutive non-improving shaking steps; 0 disables it "
            "(the paper does not report the threshold)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible shaking moves.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Also print the ATCS starting solution and search diagnostics.",
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    try:
        instance = load_cicirello_instance(args.instance)
        result = solve_gvns(
            instance,
            time_limit=args.time_limit,
            max_evaluations=args.max_evaluations,
            max_consecutive_nonimprovements=args.max_no_improve,
            seed=args.seed,
        )
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    one_based_sequence = [job + 1 for job in result.sequence]
    print(f"Final job sequence (1-based): {one_based_sequence}")
    print(f"Total weighted tardiness: {result.objective}")
    print(f"Runtime: {result.runtime_seconds:.6f} seconds")

    if args.details:
        one_based_initial = [job + 1 for job in result.initial_sequence]
        print(f"ATCS initial sequence: {one_based_initial}")
        print(f"ATCS initial objective: {result.initial_objective}")
        print(f"Objective evaluations: {result.objective_evaluations}")
        print(
            "ATCS parameters: "
            f"tau={result.atcs_parameters.tau:.4f}, "
            f"R={result.atcs_parameters.due_date_range:.4f}, "
            f"eta={result.atcs_parameters.eta:.4f}, "
            f"k1={result.atcs_parameters.k1:.4f}, "
            f"k2={result.atcs_parameters.k2:.4f}"
        )


if __name__ == "__main__":
    main()
