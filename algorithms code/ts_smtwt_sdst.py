from __future__ import annotations

import itertools
import random

from gvns_smtwt_sdst import WTSDSInstance
from metaheuristic_common import MetaheuristicResult, finish_result, initialize_search


def solve_ts(
    instance: WTSDSInstance,
    *,
    time_limit: float | None = 10.0,
    max_evaluations: int | None = 20_000_000,
    seed: int | None = None,
    tabu_tenure: int = 10,
    neighborhood_sample: int | None = 300,
) -> MetaheuristicResult:
    """Tabu Search using swap moves, aspiration, and a job-pair tabu list.

    A tabu attribute is the unordered pair of job IDs that were swapped.
    A tabu move is still allowed when it improves the global best solution
    (aspiration criterion).
    """

    if tabu_tenure < 1:
        raise ValueError("tabu_tenure must be at least 1.")
    if neighborhood_sample is not None and neighborhood_sample < 1:
        raise ValueError("neighborhood_sample must be at least 1 or None.")

    start, control, initial, initial_obj, _ = initialize_search(
        instance,
        time_limit=time_limit,
        max_evaluations=max_evaluations,
    )
    rng = random.Random(seed)

    current = list(initial)
    current_obj = initial_obj
    best = list(initial)
    best_obj = initial_obj

    if best_obj == 0 or instance.n_jobs < 2:
        return finish_result(
            "TS",
            start=start,
            control=control,
            best_sequence=best,
            best_objective=best_obj,
            initial_sequence=initial,
            initial_objective=initial_obj,
            iterations=0,
        )

    all_moves = list(itertools.combinations(range(instance.n_jobs), 2))
    tabu_until: dict[tuple[int, int], int] = {}
    iteration = 0

    while not control.exhausted():
        if neighborhood_sample is None or neighborhood_sample >= len(all_moves):
            sampled_moves = all_moves
        else:
            sampled_moves = rng.sample(all_moves, neighborhood_sample)

        def scan_moves(moves):
            candidate_best = None
            candidate_best_obj = None
            candidate_best_key = None

            for i, j in moves:
                if control.exhausted():
                    break

                job_i, job_j = current[i], current[j]
                key = (job_i, job_j) if job_i < job_j else (job_j, job_i)

                candidate = list(current)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                candidate_obj = control.evaluate(candidate)

                is_tabu = iteration < tabu_until.get(key, 0)
                aspiration = candidate_obj < best_obj
                if is_tabu and not aspiration:
                    continue

                if (
                    candidate_best_obj is None
                    or candidate_obj < candidate_best_obj
                ):
                    candidate_best = candidate
                    candidate_best_obj = candidate_obj
                    candidate_best_key = key

            return candidate_best, candidate_best_obj, candidate_best_key

        best_candidate, best_candidate_obj, best_key = scan_moves(sampled_moves)

        # If every sampled move is tabu, expand to the complete swap
        # neighborhood before doing anything more drastic.
        if (
            best_candidate is None
            and not control.exhausted()
            and sampled_moves is not all_moves
        ):
            best_candidate, best_candidate_obj, best_key = scan_moves(all_moves)

        # On very small instances it is possible for every swap to be tabu.
        # Clear the memory and rescan rather than terminating the search.
        if best_candidate is None and not control.exhausted():
            tabu_until.clear()
            best_candidate, best_candidate_obj, best_key = scan_moves(all_moves)

        if best_candidate is None or best_candidate_obj is None or best_key is None:
            break

        current = best_candidate
        current_obj = best_candidate_obj
        tabu_until[best_key] = iteration + tabu_tenure

        if current_obj < best_obj:
            best = list(current)
            best_obj = current_obj
            if best_obj == 0:
                iteration += 1
                break

        iteration += 1

        # Remove expired entries to keep the dictionary compact.
        if iteration % 100 == 0:
            tabu_until = {
                key: expiry
                for key, expiry in tabu_until.items()
                if iteration < expiry
            }

    return finish_result(
        "TS",
        start=start,
        control=control,
        best_sequence=best,
        best_objective=best_obj,
        initial_sequence=initial,
        initial_objective=initial_obj,
        iterations=iteration,
    )
