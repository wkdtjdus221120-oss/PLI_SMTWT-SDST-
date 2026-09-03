from __future__ import annotations

import random
from dataclasses import dataclass

from .gvns_smtwt_sdst import WTSDSInstance
from .metaheuristic_common import (
    MetaheuristicResult,
    difference_position_swaps,
    finish_result,
    initialize_search,
    random_permutation,
)


@dataclass
class _Particle:
    position: list[int]
    objective: int
    best_position: list[int]
    best_objective: int
    velocity: list[tuple[int, int]]


def _apply_selected_swaps(
    position: list[int],
    swaps: list[tuple[int, int]],
    probability: float,
    rng: random.Random,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Apply a stochastic prefix of a swap sequence.

    Swap sequences are order-dependent.  Using a prefix preserves the meaning
    of a partial move toward a target permutation better than independently
    skipping arbitrary operators in the middle of the sequence.
    """

    result = list(position)
    if not swaps or probability <= 0.0:
        return result, []

    expected = probability * len(swaps)
    count = int(expected)
    if count < len(swaps) and rng.random() < (expected - count):
        count += 1
    count = min(count, len(swaps))

    applied = list(swaps[:count])
    for i, j in applied:
        result[i], result[j] = result[j], result[i]

    return result, applied


def solve_dpso(
    instance: WTSDSInstance,
    *,
    time_limit: float | None = 10.0,
    max_evaluations: int | None = 20_000_000,
    seed: int | None = None,
    swarm_size: int = 40,
    inertia_probability: float = 0.50,
    cognitive_probability: float = 0.60,
    social_probability: float = 0.70,
    mutation_probability: float = 0.05,
    max_velocity_length: int | None = None,
) -> MetaheuristicResult:
    """Discrete PSO for permutations using swap-sequence velocities.

    Velocity is represented as a sequence of position-swap operators.
    Personal-best and global-best differences are converted to swap sequences,
    then probabilistically applied to the current particle.
    """

    if swarm_size < 2:
        raise ValueError("swarm_size must be at least 2.")
    for name, value in (
        ("inertia_probability", inertia_probability),
        ("cognitive_probability", cognitive_probability),
        ("social_probability", social_probability),
        ("mutation_probability", mutation_probability),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")

    start, control, initial, initial_obj, _ = initialize_search(
        instance,
        time_limit=time_limit,
        max_evaluations=max_evaluations,
    )
    rng = random.Random(seed)

    particles: list[_Particle] = [
        _Particle(
            position=list(initial),
            objective=initial_obj,
            best_position=list(initial),
            best_objective=initial_obj,
            velocity=[],
        )
    ]

    while len(particles) < swarm_size and not control.exhausted():
        position = random_permutation(instance.n_jobs, rng)
        objective = control.evaluate(position)
        particles.append(
            _Particle(
                position=position,
                objective=objective,
                best_position=list(position),
                best_objective=objective,
                velocity=[],
            )
        )

    global_particle = min(particles, key=lambda p: p.best_objective)
    global_best = list(global_particle.best_position)
    global_best_obj = global_particle.best_objective

    if global_best_obj == 0 or instance.n_jobs < 2:
        return finish_result(
            "DPSO",
            start=start,
            control=control,
            best_sequence=global_best,
            best_objective=global_best_obj,
            initial_sequence=initial,
            initial_objective=initial_obj,
            iterations=0,
        )

    velocity_cap = max_velocity_length or max(1, 2 * instance.n_jobs)
    if velocity_cap < 1:
        raise ValueError("max_velocity_length must be at least 1.")

    iterations = 0

    while not control.exhausted():
        for particle in particles:
            if control.exhausted():
                break

            new_position = list(particle.position)
            new_velocity: list[tuple[int, int]] = []

            # Inertia: retain a prefix of the previous swap sequence.
            new_position, retained = _apply_selected_swaps(
                new_position,
                particle.velocity,
                inertia_probability,
                rng,
            )
            new_velocity.extend(retained)

            # Cognitive component: move toward personal best.
            cognitive_moves = difference_position_swaps(
                new_position, particle.best_position
            )
            new_position, applied = _apply_selected_swaps(
                new_position,
                cognitive_moves,
                cognitive_probability,
                rng,
            )
            new_velocity.extend(applied)

            # Social component: then move toward global best from the state
            # already modified by inertia/cognitive terms.
            social_moves = difference_position_swaps(
                new_position, global_best
            )
            new_position, applied = _apply_selected_swaps(
                new_position,
                social_moves,
                social_probability,
                rng,
            )
            new_velocity.extend(applied)

            # Small random perturbation prevents complete swarm collapse.
            if (
                instance.n_jobs >= 2
                and rng.random() < mutation_probability
            ):
                i, j = rng.sample(range(instance.n_jobs), 2)
                new_position[i], new_position[j] = (
                    new_position[j],
                    new_position[i],
                )
                new_velocity.append((i, j))

            if len(new_velocity) > velocity_cap:
                new_velocity = new_velocity[-velocity_cap:]

            if control.exhausted():
                break
            new_objective = control.evaluate(new_position)

            particle.position = new_position
            particle.objective = new_objective
            particle.velocity = new_velocity

            if new_objective < particle.best_objective:
                particle.best_position = list(new_position)
                particle.best_objective = new_objective

                if new_objective < global_best_obj:
                    global_best = list(new_position)
                    global_best_obj = new_objective
                    if global_best_obj == 0:
                        break

        iterations += 1
        if global_best_obj == 0:
            break

    return finish_result(
        "DPSO",
        start=start,
        control=control,
        best_sequence=global_best,
        best_objective=global_best_obj,
        initial_sequence=initial,
        initial_objective=initial_obj,
        iterations=iterations,
    )
