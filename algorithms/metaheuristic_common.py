from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Sequence

from gvns_smtwt_sdst import (
    ATCSParameters,
    SearchControl,
    WTSDSInstance,
    generate_atcs_sequence,
)


@dataclass(frozen=True)
class MetaheuristicResult:
    algorithm: str
    sequence: tuple[int, ...]
    objective: int
    initial_sequence: tuple[int, ...]
    initial_objective: int
    runtime_seconds: float
    objective_evaluations: int
    iterations: int
    history: tuple[tuple[float, int], ...]


def initialize_search(
    instance: WTSDSInstance,
    *,
    time_limit: float | None,
    max_evaluations: int | None,
) -> tuple[
    float,
    SearchControl,
    list[int],
    int,
    ATCSParameters,
]:
    """Create a common search budget and the ATCS starting solution."""

    if instance.n_jobs < 1:
        raise ValueError("The instance must contain at least one job.")

    if time_limit is not None and time_limit <= 0:
        time_limit = None
    if max_evaluations is not None and max_evaluations < 1:
        max_evaluations = None
    if time_limit is None and max_evaluations is None:
        raise ValueError("At least one stopping limit must be active.")

    start = time.perf_counter()
    deadline = start + time_limit if time_limit is not None else None
    control = SearchControl(instance, deadline, max_evaluations, start_time=start)

    initial_sequence, atcs_parameters = generate_atcs_sequence(instance)
    initial_objective = control.evaluate(initial_sequence)

    return start, control, initial_sequence, initial_objective, atcs_parameters


def finish_result(
    algorithm: str,
    *,
    start: float,
    control: SearchControl,
    best_sequence: Sequence[int],
    best_objective: int,
    initial_sequence: Sequence[int],
    initial_objective: int,
    iterations: int,
) -> MetaheuristicResult:
    return MetaheuristicResult(
        algorithm=algorithm,
        sequence=tuple(best_sequence),
        objective=int(best_objective),
        initial_sequence=tuple(initial_sequence),
        initial_objective=int(initial_objective),
        runtime_seconds=time.perf_counter() - start,
        objective_evaluations=control.evaluations,
        iterations=iterations,
        history=tuple(control.history),
    )


def random_swap_neighbor(
    sequence: Sequence[int],
    rng: random.Random,
) -> list[int]:
    candidate = list(sequence)
    if len(candidate) < 2:
        return candidate
    i, j = rng.sample(range(len(candidate)), 2)
    candidate[i], candidate[j] = candidate[j], candidate[i]
    return candidate


def random_insertion_neighbor(
    sequence: Sequence[int],
    rng: random.Random,
) -> list[int]:
    candidate = list(sequence)
    if len(candidate) < 2:
        return candidate
    source, destination = rng.sample(range(len(candidate)), 2)
    job = candidate.pop(source)
    candidate.insert(destination, job)
    return candidate


def random_permutation(n_jobs: int, rng: random.Random) -> list[int]:
    permutation = list(range(n_jobs))
    rng.shuffle(permutation)
    return permutation


def order_crossover(
    parent1: Sequence[int],
    parent2: Sequence[int],
    rng: random.Random,
) -> list[int]:
    """Order crossover (OX), which always preserves a valid permutation."""

    n = len(parent1)
    if n != len(parent2):
        raise ValueError("Parents must have the same length.")
    if n < 2:
        return list(parent1)

    left, right = sorted(rng.sample(range(n), 2))
    right += 1

    child: list[int | None] = [None] * n
    child[left:right] = parent1[left:right]
    used = set(parent1[left:right])

    parent2_wrapped = list(parent2[right:]) + list(parent2[:right])
    fill_values = [job for job in parent2_wrapped if job not in used]
    fill_positions = list(range(right, n)) + list(range(0, left))
    for position, job in zip(fill_positions, fill_values):
        child[position] = job

    if any(job is None for job in child):
        raise RuntimeError("Order crossover failed to create a complete permutation.")
    return [int(job) for job in child]


def difference_position_swaps(
    source: Sequence[int],
    target: Sequence[int],
) -> list[tuple[int, int]]:
    """Return position swaps that transform source into target."""

    if len(source) != len(target) or set(source) != set(target):
        raise ValueError("source and target must be permutations of the same jobs.")

    work = list(source)
    position = {job: idx for idx, job in enumerate(work)}
    swaps: list[tuple[int, int]] = []

    for i, desired_job in enumerate(target):
        if work[i] == desired_job:
            continue
        j = position[desired_job]
        displaced = work[i]
        work[i], work[j] = work[j], work[i]
        position[desired_job] = i
        position[displaced] = j
        swaps.append((i, j))

    return swaps


def apply_position_swaps(
    sequence: Sequence[int],
    swaps: Sequence[tuple[int, int]],
) -> list[int]:
    result = list(sequence)
    for i, j in swaps:
        result[i], result[j] = result[j], result[i]
    return result


def difference_job_swaps(
    source: Sequence[int],
    target: Sequence[int],
) -> list[tuple[int, int]]:
    """Return job-label swap operators that transform source into target.

    Job-label swaps can be applied to a third permutation, which makes them
    useful as a discrete analogue of a differential vector in permutation DE.
    """

    if len(source) != len(target) or set(source) != set(target):
        raise ValueError("source and target must be permutations of the same jobs.")

    work = list(source)
    position = {job: idx for idx, job in enumerate(work)}
    swaps: list[tuple[int, int]] = []

    for i, desired_job in enumerate(target):
        current_job = work[i]
        if current_job == desired_job:
            continue

        j = position[desired_job]
        swaps.append((current_job, desired_job))

        work[i], work[j] = work[j], work[i]
        position[desired_job] = i
        position[current_job] = j

    return swaps


def apply_job_swaps(
    sequence: Sequence[int],
    swaps: Sequence[tuple[int, int]],
) -> list[int]:
    result = list(sequence)
    position = {job: idx for idx, job in enumerate(result)}

    for job_a, job_b in swaps:
        i, j = position[job_a], position[job_b]
        result[i], result[j] = result[j], result[i]
        position[job_a], position[job_b] = j, i

    return result


def position_based_crossover(
    target: Sequence[int],
    mutant: Sequence[int],
    crossover_rate: float,
    rng: random.Random,
) -> list[int]:
    """Permutation-safe binomial-style crossover for discrete DE.

    Positions selected by the crossover mask inherit jobs from the mutant.
    Remaining positions are filled in the order induced by the target parent.
    """

    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be in [0, 1].")
    if len(target) != len(mutant) or set(target) != set(mutant):
        raise ValueError("target and mutant must be permutations of the same jobs.")

    n = len(target)
    if n < 2:
        return list(target)

    selected = [rng.random() < crossover_rate for _ in range(n)]
    selected[rng.randrange(n)] = True  # DE-style forced mutant contribution.

    child: list[int | None] = [None] * n
    used: set[int] = set()

    for i, take_mutant in enumerate(selected):
        if take_mutant:
            child[i] = mutant[i]
            used.add(mutant[i])

    filler = (job for job in target if job not in used)
    for i in range(n):
        if child[i] is None:
            child[i] = next(filler)

    return [int(job) for job in child]
