from __future__ import annotations

import random

from gvns_smtwt_sdst import WTSDSInstance
from metaheuristic_common import (
    MetaheuristicResult,
    apply_job_swaps,
    difference_job_swaps,
    finish_result,
    initialize_search,
    position_based_crossover,
    random_permutation,
)


def solve_dde(
    instance: WTSDSInstance,
    *,
    time_limit: float | None = 10.0,
    max_evaluations: int | None = 20_000_000,
    seed: int | None = None,
    population_size: int = 40,
    differential_weight: float = 0.60,
    crossover_rate: float = 0.90,
) -> MetaheuristicResult:
    """Discrete Differential Evolution for permutation schedules.

    Mutation:
        base = x_r1
        difference = swap operators that transform x_r2 into x_r3
        mutant = apply a fraction F of those operators to base

    Crossover:
        permutation-safe position-based crossover between target and mutant

    Selection:
        greedy DE replacement (trial replaces target if no worse)
    """

    if population_size < 4:
        raise ValueError("population_size must be at least 4.")
    if not 0.0 <= differential_weight <= 1.0:
        raise ValueError("differential_weight must be in [0, 1].")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be in [0, 1].")

    start, control, initial, initial_obj, _ = initialize_search(
        instance,
        time_limit=time_limit,
        max_evaluations=max_evaluations,
    )
    rng = random.Random(seed)

    population: list[list[int]] = [list(initial)]
    objectives: list[int] = [initial_obj]

    while len(population) < population_size and not control.exhausted():
        individual = random_permutation(instance.n_jobs, rng)
        objective = control.evaluate(individual)
        population.append(individual)
        objectives.append(objective)

    best_index = min(range(len(population)), key=lambda idx: objectives[idx])
    best = list(population[best_index])
    best_obj = objectives[best_index]
    generations = 0

    if best_obj == 0 or instance.n_jobs < 2 or len(population) < 4:
        return finish_result(
            "DDE",
            start=start,
            control=control,
            best_sequence=best,
            best_objective=best_obj,
            initial_sequence=initial,
            initial_objective=initial_obj,
            iterations=generations,
        )

    while not control.exhausted():
        next_population = [list(individual) for individual in population]
        next_objectives = list(objectives)

        for i, target in enumerate(population):
            if control.exhausted():
                break

            candidate_indices = [idx for idx in range(len(population)) if idx != i]
            if len(candidate_indices) < 3:
                break
            r1, r2, r3 = rng.sample(candidate_indices, 3)

            difference = difference_job_swaps(
                population[r2], population[r3]
            )

            selected_difference = [
                operation
                for operation in difference
                if rng.random() < differential_weight
            ]
            # If F > 0 and a difference exists, force at least one differential
            # operator so mutation is not accidentally identical to the base.
            if (
                differential_weight > 0
                and difference
                and not selected_difference
            ):
                selected_difference = [rng.choice(difference)]

            mutant = apply_job_swaps(
                population[r1], selected_difference
            )
            trial = position_based_crossover(
                target,
                mutant,
                crossover_rate,
                rng,
            )

            if control.exhausted():
                break
            trial_obj = control.evaluate(trial)

            if trial_obj <= objectives[i]:
                next_population[i] = trial
                next_objectives[i] = trial_obj

                if trial_obj < best_obj:
                    best = list(trial)
                    best_obj = trial_obj
                    if best_obj == 0:
                        break

        population = next_population
        objectives = next_objectives
        generations += 1

        if best_obj == 0:
            break

    return finish_result(
        "DDE",
        start=start,
        control=control,
        best_sequence=best,
        best_objective=best_obj,
        initial_sequence=initial,
        initial_objective=initial_obj,
        iterations=generations,
    )
