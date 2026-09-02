from __future__ import annotations

import math
import random

from gvns_smtwt_sdst import WTSDSInstance
from metaheuristic_common import MetaheuristicResult, finish_result, initialize_search


def _weighted_log_choice(
    jobs: list[int],
    log_weights: list[float],
    rng: random.Random,
) -> int:
    finite = [value for value in log_weights if math.isfinite(value)]
    if not finite:
        return rng.choice(jobs)

    maximum = max(finite)
    weights = [
        math.exp(value - maximum) if math.isfinite(value) else 0.0
        for value in log_weights
    ]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        return rng.choice(jobs)

    draw = rng.random() * total
    cumulative = 0.0
    for job, weight in zip(jobs, weights):
        cumulative += weight
        if draw <= cumulative:
            return job
    return jobs[-1]


def _construct_ant(
    instance: WTSDSInstance,
    pheromone: list[list[float]],
    atcs,
    alpha: float,
    beta: float,
    rng: random.Random,
) -> list[int]:
    unscheduled = list(range(instance.n_jobs))
    sequence: list[int] = []
    current_time = 0
    previous = instance.dummy_job

    while unscheduled:
        heuristic_logs: list[float] = []
        for job in unscheduled:
            if instance.weights[job] <= 0:
                heuristic_log = -math.inf
            else:
                slack = max(
                    instance.due_dates[job]
                    - instance.processing_times[job]
                    - current_time,
                    0,
                )
                setup_denominator = (
                    atcs.k2 * atcs.mean_setup_time
                    if atcs.mean_setup_time > 0
                    else 1.0
                )
                heuristic_log = (
                    math.log(instance.weights[job] / instance.processing_times[job])
                    - slack / (atcs.k1 * atcs.mean_processing_time)
                    - instance.setup_times[previous][job] / setup_denominator
                )
            heuristic_logs.append(heuristic_log)

        # If every ATCS heuristic is zero (e.g., all weights are zero), fall
        # back to pheromone-only sampling instead of producing all -inf.
        all_heuristics_infinite = not any(
            math.isfinite(value) for value in heuristic_logs
        )

        log_weights = []
        for job, heuristic_log in zip(unscheduled, heuristic_logs):
            pheromone_term = alpha * math.log(max(pheromone[previous][job], 1e-300))
            heuristic_term = 0.0 if all_heuristics_infinite else beta * heuristic_log
            log_weights.append(pheromone_term + heuristic_term)

        job = _weighted_log_choice(unscheduled, log_weights, rng)
        sequence.append(job)
        unscheduled.remove(job)

        current_time += (
            instance.setup_times[previous][job]
            + instance.processing_times[job]
        )
        previous = job

    return sequence


def _deposit(
    pheromone: list[list[float]],
    sequence: list[int],
    amount: float,
    dummy_job: int,
    maximum_pheromone: float,
) -> None:
    previous = dummy_job
    for job in sequence:
        pheromone[previous][job] = min(
            maximum_pheromone,
            pheromone[previous][job] + amount,
        )
        previous = job


def solve_aco(
    instance: WTSDSInstance,
    *,
    time_limit: float | None = 10.0,
    max_evaluations: int | None = 20_000_000,
    seed: int | None = None,
    n_ants: int = 30,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation_rate: float = 0.20,
    elitist_weight: float = 2.0,
    initial_pheromone: float = 1.0,
    minimum_pheromone: float = 1e-8,
    maximum_pheromone: float = 1e6,
) -> MetaheuristicResult:
    """Ant Colony Optimization with predecessor-successor pheromone trails.

    The static/dynamic desirability term is the ATCS log-priority, so each ant
    constructs a full valid job permutation while considering due dates,
    weights, processing times, and sequence-dependent setup times.
    """

    if n_ants < 1:
        raise ValueError("n_ants must be at least 1.")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative.")
    if not 0.0 < evaporation_rate < 1.0:
        raise ValueError("evaporation_rate must be between 0 and 1.")
    if elitist_weight < 0:
        raise ValueError("elitist_weight must be non-negative.")
    if initial_pheromone <= 0 or minimum_pheromone <= 0:
        raise ValueError("pheromone values must be positive.")
    if maximum_pheromone < initial_pheromone:
        raise ValueError("maximum_pheromone must be >= initial_pheromone.")

    start, control, initial, initial_obj, atcs = initialize_search(
        instance,
        time_limit=time_limit,
        max_evaluations=max_evaluations,
    )
    rng = random.Random(seed)

    best = list(initial)
    best_obj = initial_obj

    n = instance.n_jobs
    pheromone = [
        [float(initial_pheromone) for _ in range(n)]
        for _ in range(n + 1)
    ]

    # Seed the colony with the ATCS path.
    _deposit(
        pheromone,
        best,
        amount=1.0,
        dummy_job=instance.dummy_job,
        maximum_pheromone=maximum_pheromone,
    )

    iterations = 0
    quality_scale = float(max(initial_obj, 1))

    while not control.exhausted() and best_obj > 0:
        iteration_best = None
        iteration_best_obj = None

        for _ in range(n_ants):
            if control.exhausted():
                break

            ant = _construct_ant(
                instance,
                pheromone,
                atcs,
                alpha,
                beta,
                rng,
            )
            ant_obj = control.evaluate(ant)

            if iteration_best_obj is None or ant_obj < iteration_best_obj:
                iteration_best = ant
                iteration_best_obj = ant_obj

            if ant_obj < best_obj:
                best = list(ant)
                best_obj = ant_obj
                if best_obj == 0:
                    break

        if iteration_best is None or iteration_best_obj is None:
            break

        retain = 1.0 - evaporation_rate
        for previous in range(n + 1):
            for job in range(n):
                pheromone[previous][job] = max(
                    minimum_pheromone,
                    pheromone[previous][job] * retain,
                )

        iteration_amount = min(
            10.0,
            quality_scale / max(iteration_best_obj, 1),
        )
        _deposit(
            pheromone,
            iteration_best,
            amount=iteration_amount,
            dummy_job=instance.dummy_job,
            maximum_pheromone=maximum_pheromone,
        )

        if elitist_weight > 0:
            global_amount = min(
                10.0,
                elitist_weight * quality_scale / max(best_obj, 1),
            )
            _deposit(
                pheromone,
                best,
                amount=global_amount,
                dummy_job=instance.dummy_job,
                maximum_pheromone=maximum_pheromone,
            )

        iterations += 1

    return finish_result(
        "ACO",
        start=start,
        control=control,
        best_sequence=best,
        best_objective=best_obj,
        initial_sequence=initial,
        initial_objective=initial_obj,
        iterations=iterations,
    )
