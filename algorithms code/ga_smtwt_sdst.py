from __future__ import annotations

import random

from gvns_smtwt_sdst import WTSDSInstance
from metaheuristic_common import (
    MetaheuristicResult,
    finish_result,
    initialize_search,
    order_crossover,
    random_permutation,
)


def _tournament_select(
    population: list[list[int]],
    objectives: list[int],
    tournament_size: int,
    rng: random.Random,
) -> list[int]:
    k = min(max(1, tournament_size), len(population))
    indices = rng.sample(range(len(population)), k)
    winner = min(indices, key=lambda idx: objectives[idx])
    return population[winner]


def _swap_mutation(
    chromosome: list[int],
    mutation_rate: float,
    rng: random.Random,
) -> None:
    if len(chromosome) >= 2 and rng.random() < mutation_rate:
        i, j = rng.sample(range(len(chromosome)), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]


def solve_ga(
    instance: WTSDSInstance,
    *,
    time_limit: float | None = 10.0,
    max_evaluations: int | None = 20_000_000,
    seed: int | None = None,
    population_size: int = 50,
    crossover_rate: float = 0.90,
    mutation_rate: float = 0.20,
    tournament_size: int = 3,
    elite_count: int = 2,
) -> MetaheuristicResult:
    """Permutation Genetic Algorithm using OX crossover and swap mutation."""

    if population_size < 2:
        raise ValueError("population_size must be at least 2.")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be in [0, 1].")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be in [0, 1].")
    if tournament_size < 1:
        raise ValueError("tournament_size must be at least 1.")
    if elite_count < 1:
        raise ValueError("elite_count must be at least 1.")

    start, control, initial, initial_obj, _ = initialize_search(
        instance,
        time_limit=time_limit,
        max_evaluations=max_evaluations,
    )
    rng = random.Random(seed)

    population: list[list[int]] = [list(initial)]
    objectives: list[int] = [initial_obj]

    while len(population) < population_size and not control.exhausted():
        chromosome = random_permutation(instance.n_jobs, rng)
        objective = control.evaluate(chromosome)
        population.append(chromosome)
        objectives.append(objective)

    best_index = min(range(len(population)), key=lambda idx: objectives[idx])
    best = list(population[best_index])
    best_obj = objectives[best_index]
    generations = 0

    if best_obj == 0 or instance.n_jobs < 2:
        return finish_result(
            "GA",
            start=start,
            control=control,
            best_sequence=best,
            best_objective=best_obj,
            initial_sequence=initial,
            initial_objective=initial_obj,
            iterations=generations,
        )

    while not control.exhausted() and len(population) >= 2:
        ranked = sorted(range(len(population)), key=lambda idx: objectives[idx])
        elites_to_keep = min(elite_count, len(population), population_size)
        new_population = [list(population[idx]) for idx in ranked[:elites_to_keep]]
        new_objectives = [objectives[idx] for idx in ranked[:elites_to_keep]]

        while len(new_population) < population_size and not control.exhausted():
            parent1 = _tournament_select(
                population, objectives, tournament_size, rng
            )
            parent2 = _tournament_select(
                population, objectives, tournament_size, rng
            )

            if rng.random() < crossover_rate:
                child = order_crossover(parent1, parent2, rng)
            else:
                child = list(parent1)

            _swap_mutation(child, mutation_rate, rng)

            if control.exhausted():
                break
            child_obj = control.evaluate(child)
            new_population.append(child)
            new_objectives.append(child_obj)

            if child_obj < best_obj:
                best = list(child)
                best_obj = child_obj
                if best_obj == 0:
                    break

        if len(new_population) < 2:
            break

        population = new_population
        objectives = new_objectives
        generations += 1

        if best_obj == 0:
            break

    return finish_result(
        "GA",
        start=start,
        control=control,
        best_sequence=best,
        best_objective=best_obj,
        initial_sequence=initial,
        initial_objective=initial_obj,
        iterations=generations,
    )
